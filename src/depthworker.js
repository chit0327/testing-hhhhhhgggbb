// 對焦（景深）: the AI depth model (Depth Anything V2 Small, Apache-2.0, 8-bit ONNX, vendor/models) in its own thread, with
// ONNX Runtime Web. On the graphics card (WebGPU) where the browser has it AND it is measured to give the same depth map
// and be faster (both run once on the first frame), else on the processor (WebAssembly) — like the grade's choice.
// Messages: {type: 'init'} → {type: 'ready', info}; {type: 'depth', data (RGBA), w, h} → {type: 'depth', data (Float32
// relative inverse depth, larger = nearer), w, h, ms}.
import * as ort from '../vendor/ort/ort.webgpu.min.mjs';

const MODEL = new URL('../vendor/models/depth_anything_v2_small_q8.onnx', import.meta.url).href;
ort.env.wasm.wasmPaths = new URL('../vendor/ort/', import.meta.url).href;
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
let model = null, cpu = null, gpu = null, use = null;
const info = { ep: null, cpuMs: null, gpuMs: null, diff: null, why: '', loadMs: null };

function tensor(rgba, w, h) {
  const t = new Float32Array(3 * w * h), n = w * h;
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) t[c * n + i] = (rgba[4 * i + c] / 255 - MEAN[c]) / STD[c];
  return new ort.Tensor('float32', t, [1, 3, h, w]);
}
const run = async (s, rgba, w, h) => { const o = await s.run({ [s.inputNames[0]]: tensor(rgba, w, h) }); return o[s.outputNames[0]].data; };
const norm = (d) => { let lo = Infinity, hi = -Infinity; for (const v of d) { if (v < lo) lo = v; if (v > hi) hi = v; } const k = 1 / Math.max(1e-6, hi - lo); return Float32Array.from(d, (v) => (v - lo) * k); };

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      const t0 = performance.now();
      const res = await fetch(MODEL);
      if (!res.ok) throw new Error('找不到 AI 深度模型（請先執行 npm run setup）');
      model = new Uint8Array(await res.arrayBuffer());
      cpu = await ort.InferenceSession.create(model, { executionProviders: ['wasm'] });
      if (self.navigator && self.navigator.gpu && m.gpu !== false) {
        try { gpu = await ort.InferenceSession.create(model, { executionProviders: ['webgpu'] }); } catch (err) { info.why = /adapter/i.test(err.message) ? '這部裝置的瀏覽器沒有提供顯示卡（WebGPU）' : '顯示卡不能載入模型'; info.error = String(err.message).slice(0, 200); gpu = null; }
      } else info.why = '這個瀏覽器沒有 WebGPU';
      info.loadMs = Math.round(performance.now() - t0);
      self.postMessage({ type: 'ready', id: m.id, info: { ...info } });
    } else if (m.type === 'depth') {
      const rgba = new Uint8ClampedArray(m.data), t0 = performance.now();
      let out;
      if (!use) { // first frame: decide (both warmed up once, then timed)
        if (gpu) {
          try {
            await run(cpu, rgba, m.w, m.h); await run(gpu, rgba, m.w, m.h);
            let t = performance.now(); const a = await run(cpu, rgba, m.w, m.h); info.cpuMs = Math.round(performance.now() - t);
            t = performance.now(); const b = await run(gpu, rgba, m.w, m.h); info.gpuMs = Math.round(performance.now() - t);
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
