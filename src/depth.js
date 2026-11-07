// 對焦（景深）: the client of the depth worker (src/depthworker.js). One worker for the page, started on first use (the
// model is ~27 MB: loaded only when the focus effect is used). depthOf(drawable) draws the picture at the model's input
// size (long side 364 px, multiples of 14) and resolves to {data: Float32Array (relative inverse depth), w, h, ms}.
let worker = null, ready = null, seq = 0, lastInfo = null;
const pending = new Map();

/** the model's input size for a w×h picture */
export function depthSize(w, h) {
  const L = 364, s = Math.max(14, Math.round(L * Math.min(w, h) / Math.max(w, h) / 14) * 14);
  return w >= h ? { w: L, h: s } : { w: s, h: L };
}
export const depthInfo = () => lastInfo;
export const depthLoaded = () => !!worker && !!lastInfo;

function call(msg, transfer = []) {
  return new Promise((resolve, reject) => { const id = ++seq; pending.set(id, { resolve, reject }); worker.postMessage({ ...msg, id }, transfer); });
}
/** start the worker and load the model (once); resolves to its info */
export function initDepth({ gpu = globalThis.__depthGpu !== false } = {}) {
  if (ready) return ready;
  worker = new Worker(new URL('./depthworker.js', import.meta.url), { type: 'module' });
  worker.onmessage = (e) => { const m = e.data, p = pending.get(m.id); if (!p) return; pending.delete(m.id); if (m.info) lastInfo = m.info; m.type === 'error' ? p.reject(new Error(m.message)) : p.resolve(m); };
  worker.onerror = (e) => { for (const p of pending.values()) p.reject(new Error(e.message || '深度模型執行失敗')); pending.clear(); };
  ready = call({ type: 'init', gpu }).then((m) => m.info).catch((err) => { worker.terminate(); worker = null; ready = null; throw err; });
  return ready;
}
let canvas = null;
/** the picture at the model's input size, read NOW (the caller may draw the next frame into the same canvas next) */
export function depthInput(src, sw, sh) {
  const { w, h } = depthSize(sw, sh);
  if (!canvas) canvas = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(w, h) : document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(src, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}
/** depth of a picture (canvas, video, image, ImageBitmap — anything drawImage takes): read now, estimated in the worker */
export function depthOf(src, sw, sh) { return estimateDepth(depthInput(src, sw, sh)); }
export async function estimateDepth(img) {
  await initDepth();
  const d = img.data.buffer;
  const m = await call({ type: 'depth', data: d, w: img.width, h: img.height }, [d]);
  return { data: new Float32Array(m.data), w: m.w, h: m.h, ms: m.ms };
}
