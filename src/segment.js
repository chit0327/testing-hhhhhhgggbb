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

export async function initSegmenter() {
  if (segmenter) return segmenter;
  if (!loading) {
    loading = (async () => {
      const { FilesetResolver, ImageSegmenter } = await import('../vendor/mediapipe/vision_bundle.mjs');
      const files = await FilesetResolver.forVisionTasks(new URL('../vendor/mediapipe/wasm', import.meta.url).href);
      segmenter = await ImageSegmenter.createFromOptions(files, {
        baseOptions: { modelAssetBuffer: await loadModel(), delegate: 'CPU' },
        runningMode: 'IMAGE', outputConfidenceMasks: true, outputCategoryMask: false,
      });
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
  try { await Promise.race([call({ type: 'init' }), new Promise((_, rej) => setTimeout(() => rej(new Error('timeout')), 60000))]); }
  catch { w.terminate(); return null; }
  return {
    segment(img, mw, mh) { const buf = img.data.slice().buffer; return call({ type: 'seg', data: buf, w: img.width, h: img.height, mw, mh }, [buf]).then((m) => m.r); },
    close() { w.terminate(); },
  };
}
