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
  /** 依深度分層匹配: src / ref ImageData with their AI person grids and depth grids → {fit, src, ref} */
  fitLayers(src, srcSeg, srcDepth, ref, refSeg, refDepth, look, intentJson) {
    const a = src.data.slice().buffer, b = ref.data.slice().buffer;
    return this.call({ type: 'fit-layers', src: { data: a, w: src.width, h: src.height, seg: srcSeg, depth: srcDepth }, ref: { data: b, w: ref.width, h: ref.height, seg: refSeg, depth: refDepth }, look, intent: intentJson }, [a, b]);
  }
  fitLocal(src, srcSeg, ref, refSeg, cands) { // src, ref: ImageData; seg: {w, h, skin, hair, clothes, background}
    const a = src.data.slice().buffer, b = ref.data.slice().buffer;
    return this.call({ type: 'fit-local', src: { data: a, w: src.width, h: src.height, seg: srcSeg }, ref: { data: b, w: ref.width, h: ref.height, seg: refSeg }, cands }, [a, b]);
  }
  clearPair() { return this.call({ type: 'clear-pair' }); }
  gpuBench(img, look, intentJson) { const d = img.data.slice(); return this.call({ type: 'gpu-bench', data: d.buffer, w: img.width, h: img.height, look, intent: intentJson }, [d.buffer]).then((m) => m.res); }
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
  render(img, opts, { give = false } = {}) { // img: ImageData. Returns {imageData, intentHash, localApplied}
    // give: the caller no longer needs img, so its pixels are moved to the worker instead of copied (one frame-sized
    // buffer less per frame while processing)
    const buf = give ? img.data.buffer : img.data.slice().buffer;
    return this.call({ type: 'render', rgba: buf, w: img.width, h: img.height, ...opts }, [buf]).then((m) => ({
      imageData: new ImageData(new Uint8ClampedArray(m.rgba), m.w, m.h), intentHash: m.intentHash, localApplied: m.localApplied, dof: m.dof, skinProtected: m.skinProtected, localGraded: m.localGraded, index: m.index, blur: m.blur, timing: m.timing,
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
// iPad / iPhone (every browser there is WebKit) and devices reporting ≤ 4 GB: a browser tab gets far less memory
// (an iPad tab is stopped at roughly 1–1.5 GB), so processing keeps less in flight there: AI segmentation on the
// main thread with the model already loaded for the preview (no extra copy of it per worker), 2 frames in flight
// instead of 4, 2 decoded frames ahead instead of 6. Same graded frames, slower.
const IOS = /iPad|iPhone|iPod/.test(UA) || (typeof navigator !== 'undefined' && navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
export const lowMemory = () => (typeof window !== 'undefined' && window.__lowMemory !== undefined ? !!window.__lowMemory
  : IOS || (typeof navigator !== 'undefined' && navigator.deviceMemory > 0 && navigator.deviceMemory <= 4));
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
  const info = { video, file, url: video.src, width: video.videoWidth, height: video.videoHeight, duration: video.duration, fps: 30, fpsMeasured: false };
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
  // the file's own frame rate where it can be read (MP4 / MOV index: the median frame duration): playback-based
  // measurement drops frames on a heavy clip (4K, 50 / 60 fps) on a busy computer and then reads half the rate,
  // which halved the frames of the export
  try {
    const { openMp4 } = await import('./mp4index.js');
    const idx = await Promise.race([openMp4(file), new Promise((res) => setTimeout(() => res(null), 8000))]);
    const tr = idx && idx.info.videoTracks && idx.info.videoTracks[0], ss = tr && idx.file.getTrackById(tr.id).samples;
    if (ss && ss.length > 1) {
      const d = ss.map((x) => x.duration).filter((x) => x > 0).sort((a, b) => a - b)[ss.length >> 1];
      const f = d ? snapFps(ss[0].timescale / d) : 0;
      if (f > 1 && f <= 240) { if (info.fpsMeasured && Math.abs(f - info.fps) > 0.05) info.fpsPlayback = info.fps; info.fps = f; info.fpsMeasured = true; info.fpsSource = 'container'; }
    }
  } catch { /* not MP4 / MOV: keep the measured rate */ }
  if (!info.fpsSource) info.fpsSource = info.fpsMeasured ? 'playback' : 'default';
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

const presented = (video, ms) => new Promise((res) => { // next presented frame (its media time), or give up after ms
  const timer = setTimeout(() => res(null), ms);
  if ('requestVideoFrameCallback' in video) video.requestVideoFrameCallback((_, m) => { clearTimeout(timer); res(m ? m.mediaTime : null); });
  else requestAnimationFrame(() => requestAnimationFrame(() => { clearTimeout(timer); res(null); }));
});

// 'seeked', or false after ms: iPad Safari sometimes never fires it, which used to leave processing at 正在處理 forever
const seekedWithin = (video, ms) => Promise.race([once(video, 'seeked').then(() => true), new Promise((res) => setTimeout(() => res(false), ms))]);
export const seekStats = { timeouts: 0 };

/** A moment of play (muted) and pause: iOS gives the video a decoder again / decodes pictures only after a play. */
export async function primeVideo(video) {
  if (!video || !video.getAttribute || !video.getAttribute('src')) return;
  try { await Promise.race([video.play(), new Promise((r) => setTimeout(r, 1500))]); } catch { /* (not allowed: carry on) */ }
  video.pause();
}
export async function seek(video, t) {
  if (Math.abs(video.currentTime - t) < 1e-4 && video.readyState >= 2) return;
  // wait for the new frame to be presented, in every browser: 'seeked' can fire before it is decoded (WebKit
  // always, Chromium under load), and drawing then grabs the previous frame (non-deterministic fits,
  // duplicated frames in the export)
  for (const [attempt, ms] of [3000, 6000, 15000].entries()) { // (a slow 4K seek on a tablet gets time; a lost event is asked again)
    const p = seekedWithin(video, ms);
    // the frame must really be on screen before it is drawn: a heavy clip (4K camera file) can take far longer than a
    // moment to present it, and drawing earlier grabbed the previous frame (duplicated frames = a choppy export).
    // Up to 3 s (it returns as soon as the frame is presented, so a fast machine waits no longer than before).
    // a retry first wakes the decoder with a moment of play: on an iPad, playing another <video> (a reference clip is played
    // once so that it can be read) takes the decoder from this one, and its seeks then never answer
    if (attempt) { await primeVideo(video); }
    const shown = presented(video, 3000);
    video.currentTime = attempt ? t + attempt * 1e-3 : t; // (a retry asks again, a hair later: same frame)
    const ok = await p;
    await shown;
    if (ok) return;
    seekStats.timeouts++;
  }
  throw new Error('瀏覽器沒有回應影片跳轉（逐格讀取影片失敗）。請重新整理頁面再試；如果仍然失敗，請把影片剪短或轉成 1080p。');
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
      if (attempt === 1) { await video.play().catch(() => {}); video.pause(); video.currentTime = t; await seekedWithin(video, 4000); }
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
export async function processPhoto({ engine, meta, intentJson, look, needsSeg, segmentFn, needsDepth = false, depthFn = null, maskH, maxSide = 4096, index = 0, keepPixels = true }) {
  const img = await meta.video.frameAt(index, maxSide), w = img.width, h = img.height;
  const c = canvas2d(w, h); c.getContext('2d').putImageData(img, 0, 0);
  let seg = null;
  if (needsSeg) { const sc = canvas2d(256, Math.max(2, Math.round(256 * h / w))); sc.getContext('2d').drawImage(c, 0, 0, sc.width, sc.height); seg = segmentFn(sc, MASK_W, maskH); }
  const depth = needsDepth && depthFn ? await depthFn(c, w, h) : null; // 對焦（景深）
  const jobId = `photo${Date.now().toString(36)}_${index}`;
  const started = await engine.startJob(jobId, intentJson, look, { noTrack: true });
  try {
    const r = await engine.render(img, { jobId, index: 0, seg, depth, grade: true });
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
  // each worker holds its own copy of the model (~150 MB); none on low-memory devices (lowMemory(), see above)
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
const IN_FLIGHT = 4, IN_FLIGHT_LOW = 2;
// a stage that stops answering ends the job with a message instead of 正在處理 forever
const withTimeout = (p, ms, what) => { let timer; return Promise.race([p, new Promise((_, rej) => { timer = setTimeout(() => rej(new Error(`${what}，處理已停止。請重新整理頁面再試。`)), ms); })]).finally(() => clearTimeout(timer)); };
export async function processVideo({ engine, meta, intentJson, look, needsSeg, segmentFn, needsDepth = false, depthFn = null, segPrepare = null, maskH, maxSide = 1280, onProgress, onStage = () => {}, isCancelled, keepFrames = [], stream: useStream = true, segWorker: useSegWorker = true }) {
  const kept = {}, blurTrack = { frames: 0, lost: 0, chained: 0 };
  // where the time goes (ms, summed over frames): grab = decode + draw + read the frame (main thread), seg = AI
  // segmentation (in its workers, overlapped), render = grading in the render worker (its own split in .worker),
  // wait = main thread waiting for finished frames, encode = waiting for the encoder queue
  const timing = { grab: 0, seg: 0, depth: 0, render: 0, wait: 0, encode: 0, worker: {} }, now = () => performance.now();
  if (typeof VideoEncoder === 'undefined') throw new Error('此瀏覽器不支援 WebCodecs，無法輸出影片（請使用最新版 Chrome 或 Edge）');
  const { w, h } = fitSize(meta.width, meta.height, maxSide);
  const fps = meta.fps, n = meta.frameCount;
  onStage('準備影片編碼器…');
  const cfg = await pickCodec(w, h, fps);
  const low = lowMemory();
  // the source is read from the file as needed (its index first, sample data in windows): never the whole file in memory
  const blob = meta.video instanceof PhotoSet ? null : (meta.file || await (await fetch(meta.url)).blob());
  let index = null;
  onStage('讀取影片檔…');
  if (blob) { try { const { openMp4 } = await import('./mp4index.js'); index = await withTimeout(openMp4(blob), 20000, '讀取影片索引'); } catch { index = null; } }
  // the source's audio is carried over as it is (AAC samples copied, not re-encoded)
  let audio;
  onStage('讀取原片音軌…');
  if (!blob) audio = { skipped: '照片沒有音軌' };
  else { try { audio = await withTimeout(extractAudio(blob, index), 30000, '讀取音訊'); } catch (e) { audio = { skipped: `讀取音訊失敗：${e.message}` }; } }
  let stream = null;
  onStage('開啟解碼器…');
  // WebKit (iPad / iPhone / Safari): its WebCodecs decoder failed on users' clips ("Decoder failure"), so the proven
  // seeking path is used there; elsewhere a decoder error mid-clip falls back to seeking (below)
  if (useStream && blob && index && !WEBKIT) { try { const { openFrameStream } = await import('./vdecode.js'); stream = await withTimeout(openFrameStream(blob, index, { lookahead: low ? 2 : 6 }), 10000, '開啟解碼器'); } catch { stream = null; } }
  if (needsSeg && useSegWorker && segPrepare && !low) onStage('載入 AI 人物分割模型…');
  const workers = needsSeg && useSegWorker && segPrepare && !low ? await segWorkers() : null;
  const seekT0 = seekStats.timeouts;
  timing.decoder = stream ? 'webcodecs' : 'seek'; timing.segWorkers = workers ? workers.length : 0; timing.lowMemory = low;
  if (needsSeg) { try { const { segmenterInfo } = await import('./segment.js'); timing.segInfo = segmenterInfo(); } catch { /* not loaded */ } } // CPU or GPU, as measured on this device
  try { timing.gradeInfo = (await engine.call({ type: 'gpu-info' })).info; } catch { /* (older worker) */ } // the grade: graphics card or processor
  const muxer = new Muxer({ target: new ArrayBufferTarget(), video: { codec: 'avc', width: w, height: h, frameRate: timescaleFor(fps) }, fastStart: 'in-memory', // (the muxer's frameRate is the track timescale)
    ...(audio.track ? { audio: { codec: 'aac', numberOfChannels: audio.track.numberOfChannels, sampleRate: audio.track.sampleRate } } : {}) });
  let encErr = null;
  const encoder = new VideoEncoder({ output: (chunk, md) => muxer.addVideoChunk(chunk, md), error: (e) => { encErr = e; } });
  encoder.configure(cfg);
  const jobId = `job${Date.now().toString(36)}`;
  onStage('準備調色…');
  const started = await engine.startJob(jobId, intentJson, look);
  const canvas = canvas2d(w, h);
  const segCanvas = canvas2d(256, Math.round(256 * h / w));
  const t0 = now();
  onStage(`處理第 1 / ${n} 格…`);
  const inflight = [];
  let posted = Promise.resolve(), done = 0, segCalls = 0;
  const finish = async (job) => { // in frame order: wait for the graded frame, then encode it
    let tt = now();
    const r = await withTimeout(job.render, 120000, `調色第 ${job.i + 1} 格超過 2 分鐘沒有回應`);
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
      let img = null;
      if (stream) {
        try {
          if (window.__failDecodeAt === i) throw new Error('Decoder failure (test)');
          // a decoder that silently stops giving frames (seen on a desktop with a camera .mov) must not hang the job
          const frame = window.__stallDecodeAt === i ? new Promise(() => {}) : stream.frameAt(t, w, h, canvas);
          img = await withTimeout(frame, window.__decodeTimeoutMs || 15000, `解碼器 ${((window.__decodeTimeoutMs || 15000) / 1000)} 秒沒有輸出第 ${i + 1} 格`);
        } catch (e) { // the browser's decoder gave up (seen on iPad: "Decoder failure"): the rest by seeking, same frames
          timing.decoderFallback = { at: i, message: String(e && e.message || e) };
          onStage(`解碼器沒有回應，改用逐格讀取（第 ${i + 1} / ${n} 格）…`);
          try { stream.close(); } catch { /* already closed */ }
          stream = null;
        }
      }
      if (!img) img = await withTimeout(grabFrame(meta.video, t, w, h, canvas), 60000, `讀取第 ${i + 1} 格影像超過 60 秒沒有回應`);
      timing.grab += now() - tt;
      let segP = Promise.resolve(null);
      if (needsSeg) {
        segCanvas.getContext('2d').drawImage(canvas, 0, 0, segCanvas.width, segCanvas.height);
        if (workers) {
          const sc = segPrepare(segCanvas), sImg = sc.getContext('2d').getImageData(0, 0, sc.width, sc.height), ts = now();
          segP = workers[segCalls++ % workers.length].segment(sImg, MASK_W, maskH).then((g) => { timing.seg += now() - ts; return g; });
        } else { tt = now(); segP = Promise.resolve(segmentFn(segCanvas, MASK_W, maskH)); timing.seg += now() - tt; }
      }
      // 對焦（景深）: this frame's AI depth (read from the canvas now, estimated in the depth worker meanwhile)
      const depP = needsDepth && depthFn ? depthFn(canvas, w, h).then((d) => { timing.depth += d.ms || 0; return d; }) : Promise.resolve(null);
      // post to the render worker in frame order (after this frame's masks and depth and the previous frame's post)
      const step = posted.then(() => Promise.all([segP, depP])).then(([seg, depth]) => ({ r: engine.render(img, { jobId, index: i, seg, depth, grade: true }, { give: !keepFrames.includes(i) }) }));
      posted = step.then(() => {}, () => {});
      const render = step.then((o) => o.r);
      render.catch(() => {}); // (handled when finished, in order)
      inflight.push({ i, img, render });
      if (inflight.length >= (low ? IN_FLIGHT_LOW : IN_FLIGHT)) await finish(inflight.shift());
    }
    while (inflight.length) await finish(inflight.shift());
    let tf = now();
    await encoder.flush();
    timing.encode += now() - tf;
    timing.total = now() - t0;
    timing.seekTimeouts = seekStats.timeouts - seekT0; // 'seeked' that never came and was asked for again
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
