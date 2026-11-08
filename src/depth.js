// 對焦（景深）: the client of the depth worker (src/depthworker.js). One worker for the page, started on first use (the
// model is ~27 MB: loaded only when the focus effect is used). depthOf(drawable) draws the picture at the model's input
// size (long side 364 px, multiples of 14) and resolves to {data: Float32Array (relative inverse depth), w, h, ms}.
let worker = null, ready = null, seq = 0, lastInfo = null, stage = '';
const pending = new Map(), listeners = new Set();
const STAGES = { runtime: '載入執行程式', 'runtime-wasm': '下載 AI 執行程式', 'runtime-wasm-cached': '讀取已儲存的 AI 執行程式', download: '下載 AI 深度模型', 'download-cached': '讀取已儲存的 AI 深度模型', session: '準備 AI 深度模型', gpu: '測試顯示卡' };
/** what loading is doing now, as text (for the page while it waits) */
export const depthStage = () => stage;
export function onDepthProgress(fn) { listeners.add(fn); return () => listeners.delete(fn); }

/** the model's input size for a w×h picture */
export function depthSize(w, h) {
  const L = 364, s = Math.max(14, Math.round(L * Math.min(w, h) / Math.max(w, h) / 14) * 14);
  return w >= h ? { w: L, h: s } : { w: s, h: L };
}
export const depthInfo = () => lastInfo;
export const depthLoaded = () => !!worker && !!lastInfo;

function call(msg, transfer = [], ms = 0, what = '') { // ms: give up after (a worker that died or hangs must not leave the page waiting)
  return new Promise((resolve, reject) => {
    const id = ++seq, timer = ms ? setTimeout(() => { if (pending.delete(id)) reject(new Error(`${what}超過 ${Math.round(ms / 1000)} 秒沒有回應${stage ? `（停在「${stage}」）` : ''}`)); }, ms) : null;
    pending.set(id, { resolve: (v) => { clearTimeout(timer); resolve(v); }, reject: (e) => { clearTimeout(timer); reject(e); } });
    worker.postMessage({ ...msg, id }, transfer);
  });
}
const MODEL_URL = new URL('../vendor/models/depth_anything_v2_small_q8.onnx', import.meta.url).href;
/** the model is already kept in this browser (Cache Storage): no need to ask about downloading it */
export async function depthStored() {
  try { return !!(await (await caches.open('zhaoyang-depth-v1')).match(MODEL_URL)); } catch { return false; }
}
function startWorker() {
  if (worker) return;
  worker = new Worker(new URL('./depthworker.js', import.meta.url), { type: 'module' });
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === 'progress') { stage = `${STAGES[m.stage] || m.stage}${m.total ? ` ${(m.loaded / 1e6).toFixed(0)} / ${(m.total / 1e6).toFixed(0)} MB` : ''}`; for (const fn of listeners) fn(stage, m); return; }
    const p = pending.get(m.id); if (!p) return; pending.delete(m.id); if (m.info) lastInfo = m.info; m.type === 'error' ? p.reject(new Error(m.message)) : p.resolve(m);
  };
  worker.onerror = (e) => { e.preventDefault && e.preventDefault(); for (const p of pending.values()) p.reject(new Error(`深度模型執行失敗：${e.message || '這個瀏覽器不能執行'}`)); pending.clear(); };
}
let prefetching = null;
/** download the model and runtime and keep them (no model in memory yet); resolves when stored */
export function prefetchDepth({ gpu = globalThis.__depthGpu !== false } = {}) {
  if (prefetching) return prefetching;
  startWorker();
  prefetching = call({ type: 'prefetch', gpu }, [], 600000, '下載 AI 深度模型').then(() => { stage = ''; }).catch((err) => { prefetching = null; stage = ''; throw err; });
  return prefetching;
}
/** start the worker and load the model (once); resolves to its info */
export function initDepth({ gpu = globalThis.__depthGpu !== false } = {}) {
  if (ready) return ready;
  startWorker();
  ready = (prefetching || Promise.resolve()).catch(() => {}).then(() => call({ type: 'init', gpu }, [], 300000, '載入 AI 深度模型')) // (a download in progress first: not twice).then((m) => { stage = ''; return m.info; }).catch((err) => { stage = ''; ready = null; throw err; });
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
  const m = await call({ type: 'depth', data: d, w: img.width, h: img.height }, [d], 120000, '計算深度');
  return { data: new Float32Array(m.data), w: m.w, h: m.h, ms: m.ms };
}
