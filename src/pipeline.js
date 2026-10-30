// Browser-side video plumbing: probing, exact sequential frame grabbing, the worker client and the
// full processing job (decode -> segment -> worker render -> WebCodecs encode -> MP4).
import { Muxer, ArrayBufferTarget } from '../vendor/mp4-muxer.mjs';
import { MASK_W } from './engine/intent.js';
import { extractAudio, muxAudio } from './audio.js';

// ---------------------------------------------------------------- worker client
export class EngineClient {
  constructor() {
    this.worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    this.pending = new Map();
    this.seq = 0;
    this.worker.onmessage = (e) => {
      const m = e.data, p = this.pending.get(m.id ?? `job:${m.jobId}`);
      if (!p) return;
      this.pending.delete(m.id ?? `job:${m.jobId}`);
      if (m.type === 'error') p.reject(new Error(m.message)); else p.resolve(m);
    };
  }
  call(msg, transfer = []) {
    const id = ++this.seq;
    return new Promise((resolve, reject) => { this.pending.set(id, { resolve, reject }); this.worker.postMessage({ ...msg, id }, transfer); });
  }
  fit(frames, reference, opts = {}) { return this.call({ type: 'fit', frames, reference, ...opts }); }
  fitPair(src, ai, pairId) { // src, ai: ImageData of the same size
    const a = src.data.slice().buffer, b = ai.data.slice().buffer;
    return this.call({ type: 'fit-pair', src: a, ai: b, w: src.width, h: src.height, pairId }, [a, b]);
  }
  fitLocal(src, srcSeg, ref, refSeg, cands) { // src, ref: ImageData; seg: {w, h, skin, hair, clothes, background}
    const a = src.data.slice().buffer, b = ref.data.slice().buffer;
    return this.call({ type: 'fit-local', src: { data: a, w: src.width, h: src.height, seg: srcSeg }, ref: { data: b, w: ref.width, h: ref.height, seg: refSeg }, cands }, [a, b]);
  }
  clearPair() { return this.call({ type: 'clear-pair' }); }
  bake(look, intentJson) { return this.call({ type: 'bake', look, intent: intentJson }).then((m) => ({ size: m.size, data: new Float32Array(m.data) })); }
  startJob(jobId, intentJson, look, { noTrack = false } = {}) { // noTrack: photos (separate pictures, no camera motion to follow)
    return new Promise((resolve, reject) => { this.pending.set(`job:${jobId}`, { resolve, reject }); this.worker.postMessage({ type: 'start-job', jobId, intent: intentJson, look, noTrack }); });
  }
  /** background-blur selection: painted layer {data Float32 −1..1, w, h} or null, anchor luma {data, w, h} of the frame it was painted on */
  setBlurSelection(selId, layer, anchor) {
    const lb = layer ? layer.data.slice().buffer : null, ab = anchor ? anchor.data.slice().buffer : null;
    return this.call({ type: 'blur-selection', selId, layer: lb, w: layer && layer.w, h: layer && layer.h, anchor: anchor ? { data: ab, w: anchor.w, h: anchor.h } : null }, [lb, ab].filter(Boolean));
  }
  endJob(jobId) { this.worker.postMessage({ type: 'end-job', jobId }); }
  render(img, opts) { // img: ImageData. Returns {imageData, intentHash, localApplied}
    const buf = img.data.slice().buffer;
    return this.call({ type: 'render', rgba: buf, w: img.width, h: img.height, ...opts }, [buf]).then((m) => ({
      imageData: new ImageData(new Uint8ClampedArray(m.rgba), m.w, m.h), intentHash: m.intentHash, localApplied: m.localApplied, skinProtected: m.skinProtected, localGraded: m.localGraded, index: m.index, blur: m.blur, timing: m.timing,
    }));
  }
}

// ---------------------------------------------------------------- video helpers
const once = (el, ev) => new Promise((res, rej) => {
  const ok = () => { el.removeEventListener('error', bad); res(); };
  const bad = () => { el.removeEventListener(ev, ok); rej(new Error(el.error ? `影片解碼錯誤（code ${el.error.code}）` : '影片載入失敗')); };
  el.addEventListener(ev, ok, { once: true }); el.addEventListener('error', bad, { once: true });
});

// WebKit (Safari, and every browser on iPad / iPhone) differs in two ways that otherwise give black frames:
// a <video> outside the document may never decode a picture, and 'seeked' can fire before the new frame
// is ready to draw (Chromium does that too under load). So the work video is kept in the page (invisible)
// and every seek also waits for the frame to be presented.
const UA = typeof navigator !== 'undefined' ? navigator.userAgent : '';
export const WEBKIT = typeof window !== 'undefined' && window.__webkitMode !== undefined ? !!window.__webkitMode
  : /AppleWebKit/.test(UA) && (!/Chrome\/|Chromium|Edg\//.test(UA) || /CriOS|FxiOS|EdgiOS/.test(UA));
let workVideo = null;

/** Load a video file into a hidden <video> and measure size, duration and frame rate. */
export async function probeVideo(file) {
  if (workVideo) { workVideo.remove(); URL.revokeObjectURL(workVideo.src); }
  const video = workVideo = document.createElement('video');
  video.muted = true; video.playsInline = true; video.preload = 'auto';
  video.setAttribute('playsinline', ''); video.setAttribute('webkit-playsinline', ''); video.setAttribute('muted', '');
  video.style.cssText = 'position:fixed;left:0;top:0;width:4px;height:4px;opacity:0.01;pointer-events:none;z-index:-1';
  video.setAttribute('aria-hidden', 'true');
  document.body.appendChild(video);
  video.src = URL.createObjectURL(file);
  await once(video, 'loadeddata');
  const info = { video, url: video.src, width: video.videoWidth, height: video.videoHeight, duration: video.duration, fps: 30, fpsMeasured: false };
  if (!info.width || !isFinite(info.duration)) throw new Error('無法讀取影片尺寸或時長');
  if ('requestVideoFrameCallback' in video) { // estimate the frame rate from presented-frame media times
    const times = [];
    await new Promise((resolve) => {
      const stop = setTimeout(resolve, 1500);
      const cb = (_, meta) => { times.push(meta.mediaTime); if (times.length >= 12) { clearTimeout(stop); resolve(); } else video.requestVideoFrameCallback(cb); };
      video.requestVideoFrameCallback(cb);
      video.play().catch(() => resolve());
    });
    video.pause();
    const d = times.slice(1).map((t, i) => t - times[i]).filter((x) => x > 0.001).sort((a, b) => a - b);
    if (d.length >= 3) { info.fps = snapFps(1 / d[Math.floor(d.length / 2)]); info.fpsMeasured = true; }
  }
  else { await video.play().catch(() => {}); video.pause(); } // iOS decodes pictures only after a play
  info.frameCount = frameCountOf(info.duration, info.fps);
  return info;
}

/** Frames in a clip: the browser gives the duration to the microsecond (8.708333 s × 24 = 208.99999 for 209
 * frames), so a hundredth of a frame of tolerance; a partial last frame (dog: 289.63) is not counted. */
export const frameCountOf = (duration, fps) => Math.max(1, Math.floor(duration * fps + 0.01));

// measured rates are snapped to the standard ones (exact NTSC fractions: 29.97 is 30000/1001)
const STANDARD_FPS = [24000 / 1001, 24, 25, 30000 / 1001, 30, 48, 50, 60000 / 1001, 60];
export function snapFps(f) {
  const s = STANDARD_FPS.find((x) => Math.abs(x - f) < 0.02);
  return s ?? Math.round(f * 100) / 100;
}
/** MP4 timescale for a frame rate: the rate itself when it is an integer, else 1000 × the NTSC base
 *  (30000 for 29.97) so that every frame lands on an exact tick; 90000 otherwise. */
export function timescaleFor(fps) {
  if (Number.isInteger(fps)) return fps;
  const k = Math.round(fps * 1.001);
  return Math.abs(fps - (k * 1000) / 1001) < 1e-6 ? k * 1000 : 90000;
}

/** Time of the centre of frame i (seeking there always lands on exactly that frame). */
export const frameTime = (meta, i) => Math.min(meta.duration - 1e-3, (Math.min(meta.frameCount - 1, Math.max(0, i)) + 0.5) / meta.fps);

const presented = (video, ms) => new Promise((res) => { // next presented frame, or give up after ms
  const timer = setTimeout(res, ms);
  if ('requestVideoFrameCallback' in video) video.requestVideoFrameCallback(() => { clearTimeout(timer); res(); });
  else requestAnimationFrame(() => requestAnimationFrame(() => { clearTimeout(timer); res(); }));
});

export async function seek(video, t) {
  if (Math.abs(video.currentTime - t) < 1e-4 && video.readyState >= 2) return;
  const p = once(video, 'seeked');
  // wait for the new frame to be presented, in every browser: 'seeked' can fire before it is decoded (WebKit
  // always, Chromium under load), and drawing then grabs the previous frame (non-deterministic fits,
  // duplicated frames in the export)
  const shown = presented(video, 400);
  video.currentTime = t;
  await p;
  await shown;
}

function isBlank(d) { // every sampled pixel black (or fully transparent): the frame was not decoded yet
  for (let i = 0; i < d.length; i += 4 * 97) if (d[i] > 3 || d[i + 1] > 3 || d[i + 2] > 3) return false;
  return true;
}
export const canvas2d = (w, h) => { // OffscreenCanvas where it is dependable, else a detached <canvas>
  if (!WEBKIT && typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(w, h);
  const c = document.createElement('canvas'); c.width = w; c.height = h; return c;
};

const locks = new WeakMap(); // one seek+draw at a time per <video> (analysis, dialog and job share it)

/** Grab the frame at time t, scaled to w×h, as ImageData. A photo set's frame t is photo ⌊t⌋. */
export function grabFrame(video, t, w, h, canvas = canvas2d(w, h)) {
  if (video instanceof PhotoSet) return video.frame(t, w, h, canvas);
  const run = async () => {
    await seek(video, t);
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    for (let attempt = 0; ; attempt++) {
      ctx.drawImage(video, 0, 0, w, h);
      const img = ctx.getImageData(0, 0, w, h);
      // a genuinely black frame stays black; an undecoded one fills in after a nudge
      if (!isBlank(img.data) || attempt >= 3) return img;
      if (attempt === 1) { await video.play().catch(() => {}); video.pause(); video.currentTime = t; await once(video, 'seeked'); }
      await presented(video, 300);
    }
  };
  const p = (locks.get(video) || Promise.resolve()).then(run, run);
  locks.set(video, p.catch(() => {}));
  return p;
}

/**
 * Photos as the target: one or more image files, decoded only when a frame is needed (60 phone photos
 * decoded at once would not fit in a tablet's memory). Frame i = photo i, so the batch has the same meta
 * shape as a video (fps 1): analysis samples across the set, the Lighting Intent dialog browses it.
 */
export class PhotoSet {
  constructor(files, sizes) { this.files = files; this.sizes = sizes; }
  get length() { return this.files.length; }
  index(t) { return Math.min(this.files.length - 1, Math.max(0, Math.floor(t))); }
  async draw(i, w, h, canvas, fit = 'stretch') {
    const bmp = await createImageBitmap(this.files[i], { imageOrientation: 'from-image' });
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (fit === 'contain' && Math.abs(bmp.width / bmp.height - w / h) > 0.02) { // another aspect in a mixed set: letterbox
      const k = Math.min(w / bmp.width, h / bmp.height), dw = Math.round(bmp.width * k), dh = Math.round(bmp.height * k);
      ctx.fillStyle = '#000'; ctx.fillRect(0, 0, w, h); ctx.drawImage(bmp, (w - dw) >> 1, (h - dh) >> 1, dw, dh);
    } else ctx.drawImage(bmp, 0, 0, w, h);
    bmp.close();
    return ctx.getImageData(0, 0, w, h);
  }
  /** photo ⌊t⌋ in a w×h frame (previews, dialog): letterboxed if its aspect differs from the frame */
  frame(t, w, h, canvas = canvas2d(w, h)) { return this.draw(this.index(t), w, h, canvas, 'contain'); }
  /** photo i at its own aspect, long side ≤ maxSide (analysis and processing) */
  frameAt(i, maxSide) { const [W, H] = this.sizes[i], { w, h } = fitSize(W, H, maxSide); return this.draw(i, w, h, canvas2d(w, h)); }
}
export const PHOTO_MAX = 60; // photos per batch

/** One or more photos as the target (EXIF orientation applied). meta: like a video with fps 1, one frame per photo. */
export async function probeImages(files) {
  const sizes = [];
  for (const f of files) { const b = await createImageBitmap(f, { imageOrientation: 'from-image' }); sizes.push([b.width, b.height]); b.close(); }
  const set = new PhotoSet(files, sizes);
  return { kind: 'image', video: set, url: URL.createObjectURL(files[0]), width: sizes[0][0], height: sizes[0][1], duration: files.length, fps: 1, fpsMeasured: true, frameCount: files.length, count: files.length };
}
export const probeImage = (file) => probeImages([file]);

/**
 * Grade photo `index` of the set: the same worker job as one video frame (confirmed intent, look, AI
 * segmentation for the local grade and skin protection), at up to maxSide on the long side, encoded as JPEG
 * (quality 0.95). Returns {blob, width, height, intentHash, lookHash, jobRegions}; with keepPixels also
 * {imageData, source} (single photos: the wipe compare draws from them).
 */
export async function processPhoto({ engine, meta, intentJson, look, needsSeg, segmentFn, maskH, maxSide = 4096, index = 0, keepPixels = true }) {
  const img = await meta.video.frameAt(index, maxSide), w = img.width, h = img.height;
  const c = canvas2d(w, h); c.getContext('2d').putImageData(img, 0, 0);
  let seg = null;
  if (needsSeg) { const sc = canvas2d(256, Math.max(2, Math.round(256 * h / w))); sc.getContext('2d').drawImage(c, 0, 0, sc.width, sc.height); seg = segmentFn(sc, MASK_W, maskH); }
  const jobId = `photo${Date.now().toString(36)}_${index}`;
  const started = await engine.startJob(jobId, intentJson, look, { noTrack: true });
  try {
    const r = await engine.render(img, { jobId, index: 0, seg, grade: true });
    if (r.intentHash !== started.intentHash) throw new Error('處理引擎收到的設定與確認的設定不一致');
    c.getContext('2d').putImageData(r.imageData, 0, 0);
    const blob = c.convertToBlob ? await c.convertToBlob({ type: 'image/jpeg', quality: 0.95 }) : await new Promise((res) => c.toBlob(res, 'image/jpeg', 0.95));
    return { blob, width: w, height: h, intentHash: started.intentHash, lookHash: started.lookHash, jobRegions: started.regions, ...(keepPixels ? { imageData: r.imageData, source: img } : {}) };
  } finally { engine.endJob(jobId); }
}

export function fitSize(w, h, maxSide) {
  const s = Math.min(1, maxSide / Math.max(w, h));
  return { w: Math.max(2, Math.round(w * s / 2) * 2), h: Math.max(2, Math.round(h * s / 2) * 2) };
}

export async function loadImage(file) {
  const bmp = await createImageBitmap(file);
  return bmp;
}

export function imageDataOf(bitmap, maxSide) {
  const { w, h } = fitSize(bitmap.width, bitmap.height, maxSide);
  const c = canvas2d(w, h), ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

async function pickCodec(w, h, fps) {
  const level = w * h <= 414720 ? '1e' : w * h <= 921600 ? '1f' : '28';
  for (const codec of [`avc1.4200${level}`, `avc1.4d00${level}`, 'avc1.640028']) {
    const cfg = { codec, width: w, height: h, bitrate: Math.round(w * h * fps * 0.5), framerate: fps, avc: { format: 'avc' } };
    try { if ((await VideoEncoder.isConfigSupported(cfg)).supported) return cfg; } catch { /* try next */ }
  }
  throw new Error('此瀏覽器不支援 H.264 影片編碼（請使用最新版 Chrome 或 Edge）');
}

/**
 * Process the whole video with the CONFIRMED intent. Frames are grabbed in order, rendered by the
 * worker (which received the intent via startJob), encoded with WebCodecs and muxed into MP4.
 * Returns {blob, frames, fps, width, height, codec, intentHash}.
 */
// segmentation workers kept between jobs (the model loads once): two on a 4-core machine, up to three on more
let segPool = null;
async function segWorkers() {
  if (segPool) return segPool;
  const n = Math.max(1, Math.min(3, (navigator.hardwareConcurrency || 2) - 2));
  const { startSegWorker } = await import('./segment.js');
  const ws = (await Promise.all(Array.from({ length: n }, () => startSegWorker()))).filter(Boolean);
  segPool = ws.length ? ws : null;
  return segPool;
}

/**
 * Process the whole clip: decode → (AI segmentation) → grade → encode, as a pipeline. Frames are decoded front to
 * back with WebCodecs (vdecode.js; seeking the <video> element as the fallback), segmentation runs in worker
 * threads and grading in the render worker, so the stages work on different frames at the same time (up to
 * IN_FLIGHT frames). Frames reach the render worker strictly in order (its mask smoothing and blur tracking follow
 * the frame order) and are encoded in order. segPrepare(canvas): the 256 px frame made ready for the model (LOG
 * restoration), segmentFn: main-thread segmentation when no worker can run.
 */
const IN_FLIGHT = 4;
export async function processVideo({ engine, meta, intentJson, look, needsSeg, segmentFn, segPrepare = null, maskH, maxSide = 1280, onProgress, isCancelled, keepFrames = [], stream: useStream = true, segWorker: useSegWorker = true }) {
  const kept = {}, blurTrack = { frames: 0, lost: 0, chained: 0 };
  // where the time goes (ms, summed over frames): grab = decode + draw + read the frame (main thread), seg = AI
  // segmentation (in its workers, overlapped), render = grading in the render worker (its own split in .worker),
  // wait = main thread waiting for finished frames, encode = waiting for the encoder queue
  const timing = { grab: 0, seg: 0, render: 0, wait: 0, encode: 0, worker: {} }, now = () => performance.now();
  if (typeof VideoEncoder === 'undefined') throw new Error('此瀏覽器不支援 WebCodecs，無法輸出影片（請使用最新版 Chrome 或 Edge）');
  const { w, h } = fitSize(meta.width, meta.height, maxSide);
  const fps = meta.fps, n = meta.frameCount;
  const cfg = await pickCodec(w, h, fps);
  const bytes = await (await fetch(meta.url)).arrayBuffer();
  // the source's audio is carried over as it is (AAC samples copied, not re-encoded)
  let audio;
  try { audio = await extractAudio(bytes); } catch (e) { audio = { skipped: `讀取音訊失敗：${e.message}` }; }
  let stream = null;
  if (useStream && !(meta.video instanceof PhotoSet)) { try { const { openFrameStream } = await import('./vdecode.js'); stream = await openFrameStream(bytes); } catch { stream = null; } }
  const workers = needsSeg && useSegWorker && segPrepare ? await segWorkers() : null;
  timing.decoder = stream ? 'webcodecs' : 'seek'; timing.segWorkers = workers ? workers.length : 0;
  const muxer = new Muxer({ target: new ArrayBufferTarget(), video: { codec: 'avc', width: w, height: h, frameRate: timescaleFor(fps) }, fastStart: 'in-memory', // (the muxer's frameRate is the track timescale)
    ...(audio.track ? { audio: { codec: 'aac', numberOfChannels: audio.track.numberOfChannels, sampleRate: audio.track.sampleRate } } : {}) });
  let encErr = null;
  const encoder = new VideoEncoder({ output: (chunk, md) => muxer.addVideoChunk(chunk, md), error: (e) => { encErr = e; } });
  encoder.configure(cfg);
  const jobId = `job${Date.now().toString(36)}`;
  const started = await engine.startJob(jobId, intentJson, look);
  const canvas = canvas2d(w, h);
  const segCanvas = canvas2d(256, Math.round(256 * h / w));
  const t0 = now();
  const inflight = [];
  let posted = Promise.resolve(), done = 0, segCalls = 0;
  const finish = async (job) => { // in frame order: wait for the graded frame, then encode it
    let tt = now();
    const r = await job.render;
    timing.wait += now() - tt;
    for (const [k, v] of Object.entries(r.timing || {})) timing.worker[k] = (timing.worker[k] || 0) + v;
    timing.render += Object.values(r.timing || {}).reduce((a, b) => a + b, 0);
    if (r.blur && r.blur.tracked) { blurTrack.frames++; if (!r.blur.ok) blurTrack.lost++; if (r.blur.chained) blurTrack.chained++; }
    if (keepFrames.includes(job.i)) kept[job.i] = { source: job.img, rendered: r.imageData };
    if (r.intentHash !== started.intentHash) throw new Error('處理引擎收到的設定與確認的設定不一致');
    const frame = new VideoFrame(r.imageData.data, { format: 'RGBA', codedWidth: w, codedHeight: h, timestamp: Math.round(job.i * 1e6 / fps), duration: Math.round(1e6 / fps) });
    encoder.encode(frame, { keyFrame: job.i % Math.max(1, Math.round(fps * 2)) === 0 });
    frame.close();
    tt = now();
    while (encoder.encodeQueueSize > 4) await new Promise((res) => setTimeout(res, 5));
    timing.encode += now() - tt;
    done++;
    onProgress && onProgress({ done, total: n, elapsed: (now() - t0) / 1000, preview: job.i % 10 === 0 ? r.imageData : null });
  };
  try {
    for (let i = 0; i < n; i++) {
      if (isCancelled && isCancelled()) throw new Error('已取消');
      if (encErr) throw encErr;
      const t = frameTime(meta, i);
      let tt = now();
      const img = stream ? await stream.frameAt(t, w, h, canvas) : await grabFrame(meta.video, t, w, h, canvas);
      timing.grab += now() - tt;
      let segP = Promise.resolve(null);
      if (needsSeg) {
        segCanvas.getContext('2d').drawImage(canvas, 0, 0, segCanvas.width, segCanvas.height);
        if (workers) {
          const sc = segPrepare(segCanvas), sImg = sc.getContext('2d').getImageData(0, 0, sc.width, sc.height), ts = now();
          segP = workers[segCalls++ % workers.length].segment(sImg, MASK_W, maskH).then((g) => { timing.seg += now() - ts; return g; });
        } else { tt = now(); segP = Promise.resolve(segmentFn(segCanvas, MASK_W, maskH)); timing.seg += now() - tt; }
      }
      // post to the render worker in frame order (after this frame's masks and the previous frame's post)
      const step = posted.then(() => segP).then((seg) => ({ r: engine.render(img, { jobId, index: i, seg, grade: true }) }));
      posted = step.then(() => {}, () => {});
      const render = step.then((o) => o.r);
      render.catch(() => {}); // (handled when finished, in order)
      inflight.push({ i, img, render });
      if (inflight.length >= IN_FLIGHT) await finish(inflight.shift());
    }
    while (inflight.length) await finish(inflight.shift());
    let tf = now();
    await encoder.flush();
    timing.encode += now() - tf;
    timing.total = now() - t0;
    if (encErr) throw encErr;
    if (audio.track) audio.written = muxAudio(muxer, audio, (n * 1e6) / fps);
    muxer.finalize();
  } finally {
    if (stream) stream.close();
    engine.endJob(jobId);
    if (encoder.state !== 'closed') encoder.close();
  }
  return { blob: new Blob([muxer.target.buffer], { type: 'video/mp4' }), frames: n, fps, width: w, height: h, codec: cfg.codec, intentHash: started.intentHash, lookHash: started.lookHash, jobRegions: started.regions, kept, blurTrack, timing,
    audio: audio.track ? { kept: true, codec: audio.track.codec, samples: audio.written } : { kept: false, reason: audio.skipped } };
}
