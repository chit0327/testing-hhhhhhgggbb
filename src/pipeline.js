// Browser-side video plumbing: probing, exact sequential frame grabbing, the worker client and the
// full processing job (decode -> segment -> worker render -> WebCodecs encode -> MP4).
import { Muxer, ArrayBufferTarget } from '../vendor/mp4-muxer.mjs';
import { MASK_W } from './engine/intent.js';

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
  fit(frames, reference) { return this.call({ type: 'fit', frames, reference }); }
  fitPair(src, ai, pairId) { // src, ai: ImageData of the same size
    const a = src.data.slice().buffer, b = ai.data.slice().buffer;
    return this.call({ type: 'fit-pair', src: a, ai: b, w: src.width, h: src.height, pairId }, [a, b]);
  }
  clearPair() { return this.call({ type: 'clear-pair' }); }
  startJob(jobId, intentJson, look) {
    return new Promise((resolve, reject) => { this.pending.set(`job:${jobId}`, { resolve, reject }); this.worker.postMessage({ type: 'start-job', jobId, intent: intentJson, look }); });
  }
  endJob(jobId) { this.worker.postMessage({ type: 'end-job', jobId }); }
  render(img, opts) { // img: ImageData. Returns {imageData, intentHash, localApplied}
    const buf = img.data.slice().buffer;
    return this.call({ type: 'render', rgba: buf, w: img.width, h: img.height, ...opts }, [buf]).then((m) => ({
      imageData: new ImageData(new Uint8ClampedArray(m.rgba), m.w, m.h), intentHash: m.intentHash, localApplied: m.localApplied, index: m.index,
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
// is ready to draw. So the work video is kept in the page (invisible) and, on WebKit, every seek also
// waits for the frame to be presented.
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
    if (d.length >= 3) { info.fps = Math.round(1 / d[Math.floor(d.length / 2)] * 100) / 100; info.fpsMeasured = true; }
  }
  else { await video.play().catch(() => {}); video.pause(); } // iOS decodes pictures only after a play
  info.frameCount = Math.max(1, Math.floor(info.duration * info.fps + 1e-6));
  return info;
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
  const shown = WEBKIT ? presented(video, 400) : null;
  video.currentTime = t;
  await p;
  if (shown) await shown;
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

/** Grab the frame at time t, scaled to w×h, as ImageData. */
export function grabFrame(video, t, w, h, canvas = canvas2d(w, h)) {
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
export async function processVideo({ engine, meta, intentJson, look, needsSeg, segmentFn, maskH, maxSide = 1280, onProgress, isCancelled, keepFrames = [] }) {
  const kept = {};
  if (typeof VideoEncoder === 'undefined') throw new Error('此瀏覽器不支援 WebCodecs，無法輸出影片（請使用最新版 Chrome 或 Edge）');
  const { w, h } = fitSize(meta.width, meta.height, maxSide);
  const fps = meta.fps, n = meta.frameCount;
  const cfg = await pickCodec(w, h, fps);
  const muxer = new Muxer({ target: new ArrayBufferTarget(), video: { codec: 'avc', width: w, height: h, frameRate: fps }, fastStart: 'in-memory' });
  let encErr = null;
  const encoder = new VideoEncoder({ output: (chunk, md) => muxer.addVideoChunk(chunk, md), error: (e) => { encErr = e; } });
  encoder.configure(cfg);
  const jobId = `job${Date.now().toString(36)}`;
  const started = await engine.startJob(jobId, intentJson, look);
  const canvas = canvas2d(w, h);
  const segCanvas = canvas2d(256, Math.round(256 * h / w));
  const t0 = performance.now();
  try {
    for (let i = 0; i < n; i++) {
      if (isCancelled && isCancelled()) throw new Error('已取消');
      if (encErr) throw encErr;
      const t = frameTime(meta, i);
      const img = await grabFrame(meta.video, t, w, h, canvas);
      let seg = null;
      if (needsSeg) {
        segCanvas.getContext('2d').drawImage(canvas, 0, 0, segCanvas.width, segCanvas.height);
        seg = segmentFn(segCanvas, MASK_W, maskH);
      }
      const r = await engine.render(img, { jobId, index: i, seg, grade: true });
      if (keepFrames.includes(i)) kept[i] = { source: img, rendered: r.imageData };
      if (r.intentHash !== started.intentHash) throw new Error('處理引擎收到的設定與確認的設定不一致');
      const frame = new VideoFrame(r.imageData.data, { format: 'RGBA', codedWidth: w, codedHeight: h, timestamp: Math.round(i * 1e6 / fps), duration: Math.round(1e6 / fps) });
      encoder.encode(frame, { keyFrame: i % Math.max(1, Math.round(fps * 2)) === 0 });
      frame.close();
      while (encoder.encodeQueueSize > 4) await new Promise((res) => setTimeout(res, 5));
      onProgress && onProgress({ done: i + 1, total: n, elapsed: (performance.now() - t0) / 1000, preview: i % 10 === 0 ? r.imageData : null });
    }
    await encoder.flush();
    if (encErr) throw encErr;
    muxer.finalize();
  } finally {
    engine.endJob(jobId);
    if (encoder.state !== 'closed') encoder.close();
  }
  return { blob: new Blob([muxer.target.buffer], { type: 'video/mp4' }), frames: n, fps, width: w, height: h, codec: cfg.codec, intentHash: started.intentHash, lookHash: started.lookHash, jobRegions: started.regions, kept };
}
