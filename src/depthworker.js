// 對焦（景深）: the AI depth model (Depth Anything V2 Small, Apache-2.0, 8-bit ONNX, vendor/models) in its own thread, with
// ONNX Runtime Web. On the graphics card (WebGPU) where the browser has it AND it is measured to give the same depth map
// and be faster (both run once on the first frame), else on the processor (WebAssembly) — like the grade's choice.
// Safari / iPad (WebKit): the processor build only (a smaller runtime, ~15 MB instead of ~29 MB) — the first version
// tried the WebGPU session there while loading and never got past "loading" on an iPad.
// Messages: {type: 'init', gpu} → {type: 'progress', stage, loaded?, total?}… then {type: 'ready', info};
// {type: 'depth', data (RGBA), w, h} → {type: 'depth', data (Float32 relative inverse depth, larger = nearer), w, h, ms}.
const MODEL = new URL('../vendor/models/depth_anything_v2_small_q8.onnx', import.meta.url).href;
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
let ort = null, cpu = null, gpu = null, use = null;
const info = { ep: null, cpuMs: null, gpuMs: null, diff: null, why: '', loadMs: null, runtime: null };
const progress = (stage, extra = {}) => self.postMessage({ type: 'progress', stage, ...extra });
const timeout = (p, ms, what) => Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error(`${what}超過 ${Math.round(ms / 1000)} 秒沒有回應`)), ms))]);

function tensor(rgba, w, h) {
  const t = new Float32Array(3 * w * h), n = w * h;
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) t[c * n + i] = (rgba[4 * i + c] / 255 - MEAN[c]) / STD[c];
  return new ort.Tensor('float32', t, [1, 3, h, w]);
}
const run = async (s, rgba, w, h) => { const o = await s.run({ [s.inputNames[0]]: tensor(rgba, w, h) }); return o[s.outputNames[0]].data; };
const norm = (d) => { let lo = Infinity, hi = -Infinity; for (const v of d) { if (v < lo) lo = v; if (v > hi) hi = v; } const k = 1 / Math.max(1e-6, hi - lo); return Float32Array.from(d, (v) => (v - lo) * k); };
// The large files (model, runtime): kept in the browser's Cache Storage after the first download (a static host's HTTP
// cache is short and Safari drops large entries: every visit downloaded ~40 MB again), downloaded as the gzip copy
// (.gz, unpacked here: the runtime 14 MB → ~3 MB) where the browser can unpack (DecompressionStream), with progress.
const CACHE = 'zhaoyang-depth-v1';
async function cached(url) {
  try { const c = await caches.open(CACHE), hit = await c.match(url); if (hit) return new Uint8Array(await hit.arrayBuffer()); } catch { /* (no Cache Storage: download) */ }
  return null;
}
async function store(url, bytes) { try { const c = await caches.open(CACHE); await c.put(url, new Response(bytes, { headers: { 'Content-Type': 'application/octet-stream' } })); } catch { /* (storage full or unavailable: next time it downloads again) */ } }
async function readAll(res, what, totalGuess) {
  const total = +res.headers.get('content-length') || totalGuess;
  if (!res.body || !res.body.getReader) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader(), parts = []; let loaded = 0, last = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; parts.push(value); loaded += value.length; if (loaded - last > 5e5) { last = loaded; progress(what, { loaded, total }); } }
  const out = new Uint8Array(loaded); let o = 0; for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}
async function download(url, what, sizeGuess) {
  const hit = await cached(url);
  if (hit) { progress(`${what}-cached`); return hit; }
  let bytes = null;
  if (typeof DecompressionStream !== 'undefined') {
    try {
      const r = await fetch(`${url}.gz`);
      if (r.ok) { const gz = await readAll(r, what, sizeGuess * 0.6); bytes = new Uint8Array(await new Response(new Blob([gz]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer()); }
    } catch { bytes = null; }
  }
  if (!bytes) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(what === 'download' ? '找不到 AI 深度模型（請先執行 npm run setup）' : '找不到 AI 執行程式');
    bytes = await readAll(r, what, sizeGuess);
  }
  await store(url, bytes);
  return bytes;
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'prefetch') { // download and keep the files only (no model in memory): the page asks for it on opening
      const webkit = /AppleWebKit/.test(navigator.userAgent) && !/Chrome|Chromium|Edg\//.test(navigator.userAgent) || /iPad|iPhone|iPod/.test(navigator.userAgent);
      const tryGpu = m.gpu !== false && !!(self.navigator && self.navigator.gpu) && !webkit;
      await download(new URL(`../vendor/ort/ort-wasm-simd-threaded${tryGpu ? '.asyncify' : ''}.wasm`, import.meta.url).href, 'runtime-wasm', tryGpu ? 28.7e6 : 14.8e6);
      await download(MODEL, 'download', 27258801);
      self.postMessage({ type: 'prefetched', id: m.id });
    } else if (m.type === 'init') {
      const t0 = performance.now(), webkit = /AppleWebKit/.test(navigator.userAgent) && !/Chrome|Chromium|Edg\//.test(navigator.userAgent) || /iPad|iPhone|iPod/.test(navigator.userAgent);
      const tryGpu = m.gpu !== false && !!(self.navigator && self.navigator.gpu) && !webkit;
      info.runtime = tryGpu ? 'webgpu' : 'wasm';
      if (!tryGpu) info.why = webkit ? 'Safari／iPad 使用處理器版本' : m.gpu === false ? '已設定使用處理器' : '這個瀏覽器沒有 WebGPU';
      progress('runtime');
      ort = await timeout(import(tryGpu ? '../vendor/ort/ort.webgpu.min.mjs' : '../vendor/ort/ort.wasm.min.mjs'), 60000, '載入執行程式');
      ort.env.wasm.wasmPaths = new URL('../vendor/ort/', import.meta.url).href;
      ort.env.wasm.numThreads = 1; // (no cross-origin isolation on a static host: one thread)
      const wasmUrl = new URL(`../vendor/ort/ort-wasm-simd-threaded${tryGpu ? '.asyncify' : ''}.wasm`, import.meta.url).href;
      ort.env.wasm.wasmBinary = await download(wasmUrl, 'runtime-wasm', tryGpu ? 28.7e6 : 14.8e6); // (handed to the runtime: cached / compressed like the model)
      const model = await download(MODEL, 'download', 27258801);
      progress('session');
      cpu = await timeout(ort.InferenceSession.create(model, { executionProviders: ['wasm'] }), 120000, '準備 AI 深度模型');
      if (tryGpu) {
        progress('gpu');
        try { gpu = await timeout(ort.InferenceSession.create(model, { executionProviders: ['webgpu'] }), 20000, '顯示卡'); }
        catch (err) { info.why = /adapter/i.test(err.message) ? '這部裝置的瀏覽器沒有提供顯示卡（WebGPU）' : /沒有回應/.test(err.message) ? '顯示卡準備超時' : '顯示卡不能載入模型'; info.error = String(err.message).slice(0, 200); gpu = null; }
      }
      info.loadMs = Math.round(performance.now() - t0);
      self.postMessage({ type: 'ready', id: m.id, info: { ...info } });
    } else if (m.type === 'depth') {
      const rgba = new Uint8ClampedArray(m.data), t0 = performance.now();
      let out;
      if (!use) { // first frame: decide (both warmed up once, then timed)
        if (gpu) {
          try {
            await run(cpu, rgba, m.w, m.h); await timeout(run(gpu, rgba, m.w, m.h), 30000, '顯示卡');
            let t = performance.now(); const a = await run(cpu, rgba, m.w, m.h); info.cpuMs = Math.round(performance.now() - t);
            t = performance.now(); const b = await timeout(run(gpu, rgba, m.w, m.h), 30000, '顯示卡'); info.gpuMs = Math.round(performance.now() - t);
            const na = norm(a), nb = norm(b); let s = 0; for (let i = 0; i < na.length; i++) s += Math.abs(na[i] - nb[i]); info.diff = +(s / na.length).toFixed(4);
            use = info.diff < 0.02 && info.gpuMs * 1.2 < info.cpuMs ? 'gpu' : 'cpu';
            if (use === 'cpu') info.why = info.diff >= 0.02 ? `顯示卡結果不同（平均差 ${info.diff}）` : '顯示卡沒有比處理器快';
            out = use === 'gpu' ? b : a;
          } catch (err) { use = 'cpu'; info.why = `顯示卡出錯：${err.message}`; }
        } else use = 'cpu';
        info.ep = use;
      }
      if (!out) {
        try { out = await run(use === 'gpu' ? gpu : cpu, rgba, m.w, m.h); }
        catch (err) { if (use !== 'gpu') throw err; use = info.ep = 'cpu'; info.why = `顯示卡出錯：${err.message}`; out = await run(cpu, rgba, m.w, m.h); }
      }
      const d = Float32Array.from(out);
      self.postMessage({ type: 'depth', id: m.id, data: d.buffer, w: m.w, h: m.h, ms: Math.round(performance.now() - t0), info: { ...info } }, [d.buffer]);
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: m.id, message: String(err && err.message || err) });
  }
};
