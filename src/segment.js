// AI region proposals with MediaPipe's multi-class selfie segmenter (runs locally in the browser, ~16 MB).
// Classes: 0 background, 1 hair, 2 body-skin, 3 face-skin, 4 clothes, 5 others (accessories).
// Output grids are PROPOSALS: the dialog shows them and the user must confirm before they are used.
import { resize } from './engine/masks.js';

let segmenter = null, loading = null;

export function segmenterReady() { return !!segmenter; }

const MODEL = new URL('../vendor/models/selfie_multiclass_256x256.tflite', import.meta.url).href;

/** The model file, or its parts (.partN binary or .partN.b64.txt) where a host limits file size or type. */
async function loadModel() {
  const whole = await fetch(MODEL);
  if (whole.ok) return new Uint8Array(await whole.arrayBuffer());
  const parts = [];
  for (let i = 0; ; i++) {
    const r = await fetch(`${MODEL}.part${i}`);
    if (!r.ok) break;
    parts.push(new Uint8Array(await r.arrayBuffer()));
  }
  for (let i = 0; !parts.length || i > 0; i++) { // base64 text parts, for hosts that serve only text/web types
    const r = await fetch(`${MODEL}.part${i}.b64.txt`);
    if (!r.ok) break;
    const bin = atob((await r.text()).trim());
    const u8 = new Uint8Array(bin.length);
    for (let k = 0; k < bin.length; k++) u8[k] = bin.charCodeAt(k);
    parts.push(u8);
  }
  if (!parts.length) throw new Error('找不到 AI 分割模型（請先執行 npm run setup）');
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

// CPU or GPU: on first use both are timed on a few small frames and the faster is kept (the GPU only when clearly faster,
// ≥ 1.3×; any GPU failure keeps the CPU). The GPU version can be several times faster on a real graphics card; on
// software rendering (no graphics card) it is slower, so the choice is measured, never assumed. Masks from the two
// differ only by floating-point rounding. Override: globalThis.__segDelegate = 'CPU' | 'GPU'.
let info = { delegate: 'CPU', cpuMs: null, gpuMs: null };
export function segmenterInfo() { return { ...info }; }
// timed as used: segment and read every mask back (on the GPU the call only queues the work; reading pays for it)
function runOnce(seg, img) { const r = seg.segment(img); try { for (const m of r.confidenceMasks) m.getAsFloat32Array(); } finally { r.close(); } }
async function benchMs(seg, img) { runOnce(seg, img); const t = performance.now(); for (let i = 0; i < 3; i++) runOnce(seg, img); return (performance.now() - t) / 3; }
function benchImage() { // a small frame with a face-like blob (what the model sees: 256 px wide)
  const w = 256, h = 144, d = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const i = 4 * (y * w + x), f = Math.hypot(x - 128, y - 60) < 30; d[i] = f ? 200 : 60 + x / 4; d[i + 1] = f ? 150 : 70 + y / 3; d[i + 2] = f ? 110 : 90; d[i + 3] = 255; }
  return new ImageData(d, w, h);
}
function glCanvas() { return typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(1, 1) : Object.assign(document.createElement('canvas'), { width: 1, height: 1 }); }

export async function initSegmenter() {
  if (segmenter) return segmenter;
  if (!loading) {
    loading = (async () => {
      const { FilesetResolver, ImageSegmenter } = await import('../vendor/mediapipe/vision_bundle.mjs');
      const files = await FilesetResolver.forVisionTasks(new URL('../vendor/mediapipe/wasm', import.meta.url).href);
      const model = await loadModel();
      const make = (delegate) => ImageSegmenter.createFromOptions(files, {
        baseOptions: { modelAssetBuffer: model, delegate }, ...(delegate === 'GPU' ? { canvas: glCanvas() } : {}),
        runningMode: 'IMAGE', outputConfidenceMasks: true, outputCategoryMask: false,
      });
      const want = globalThis.__segDelegate;
      if (want === 'CPU' || want === 'GPU') {
        segmenter = await make(want).catch(() => (want === 'GPU' ? make('CPU') : Promise.reject(new Error('segmenter failed'))));
        info = { delegate: want, cpuMs: null, gpuMs: null };
        return segmenter;
      }
      const cpu = await make('CPU'), img = benchImage();
      const cpuMs = await benchMs(cpu, img);
      let gpu = null, gpuMs = null;
      try {
        gpu = await make('GPU'); runOnce(gpu, img); // (first run: shader compilation)
        const t = performance.now(); runOnce(gpu, img); gpuMs = performance.now() - t;
        if (gpuMs < cpuMs) gpuMs = await benchMs(gpu, img); // only a promising GPU gets the full timing (a slow one costs startup time)
      } catch { gpu = null; }
      if (gpu && gpuMs * 1.3 < cpuMs) { cpu.close(); segmenter = gpu; info = { delegate: 'GPU', cpuMs, gpuMs }; }
      else { if (gpu) gpu.close(); segmenter = cpu; info = { delegate: 'CPU', cpuMs, gpuMs }; }
      return segmenter;
    })();
    loading.catch(() => { loading = null; });
  }
  return loading;
}

/** source: canvas / ImageData / video frame. Returns {person, skin, background, hair, clothes} Float32 grids (mw×mh). */
export function segment(source, mw, mh) {
  if (!segmenter) throw new Error('segmenter not initialised');
  const res = segmenter.segment(source);
  try {
    const masks = res.confidenceMasks;
    const w = masks[0].width, h = masks[0].height;
    const c = masks.map((m) => m.getAsFloat32Array());
    const n = w * h, person = new Float32Array(n), skin = new Float32Array(n), background = new Float32Array(n);
    const hair = new Float32Array(n), clothes = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      background[i] = c[0][i];
      person[i] = 1 - c[0][i];
      skin[i] = Math.min(1, c[2][i] + c[3][i]);
      hair[i] = c[1][i];
      clothes[i] = Math.min(1, c[4][i] + c[5][i]);
    }
    const r = (g) => resize(g, w, h, mw, mh);
    return { person: r(person), skin: r(skin), background: r(background), hair: r(hair), clothes: r(clothes) };
  } finally {
    res.close();
  }
}

/**
 * Segmentation in a worker thread (for processing: it then overlaps decoding and grading). Resolves to
 * {segment(imageData, mw, mh) → Promise<grids>, close()}, or null when the worker cannot start here (the caller
 * segments on the main thread instead). Results are the same model on the same pixels as segment().
 */
export async function startSegWorker() {
  let w;
  try { w = new Worker(new URL('./segworker.js', import.meta.url)); } catch { return null; }
  const pending = new Map(); let seq = 0;
  w.onmessage = (e) => { const m = e.data, p = pending.get(m.id); if (!p) return; pending.delete(m.id); m.type === 'error' ? p.reject(new Error(m.message)) : p.resolve(m); };
  w.onerror = (e) => { for (const p of pending.values()) p.reject(new Error(e.message || 'segmentation worker failed')); pending.clear(); };
  const call = (msg, transfer = []) => new Promise((resolve, reject) => { const id = ++seq; pending.set(id, { resolve, reject }); w.postMessage({ ...msg, id }, transfer); });
  try { await Promise.race([call({ type: 'init', delegate: segmenter ? info.delegate : null }), new Promise((_, rej) => setTimeout(() => rej(new Error('timeout')), 60000))]); } // the page's measured choice: workers do not time again
  catch { w.terminate(); return null; }
  return {
    segment(img, mw, mh) { const buf = img.data.slice().buffer; return call({ type: 'seg', data: buf, w: img.width, h: img.height, mw, mh }, [buf]).then((m) => m.r); },
    close() { w.terminate(); },
  };
}
