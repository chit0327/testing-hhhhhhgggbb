// Processing worker: fits the reference grade, bakes LUTs, builds region masks from the CONFIRMED
// LightingIntent it receives, and renders frames. The page never renders pixels itself, so the dialog
// preview, the result preview and the export all come out of this one code path.
import { fitGrade, bakeLUT } from './engine/grade.js';
import { deserializeIntent, effectiveRegions, hashIntent, MASK_W } from './engine/intent.js';
import { regionMasks } from './engine/regionmasks.js';
import { evField, lightnessGrid } from './engine/masks.js';
import { renderFrame, FX_KEYS } from './engine/render.js';
import { fitPairLUT } from './engine/pairfit.js';

let model = null;
const pairs = new Map();       // pairId -> LUT distilled from an AI-graded frame (look.source === 'ai'); a few candidates
const luts = new Map();      // key: look settings + preserve flag -> LUT (bounded cache)
const jobs = new Map();      // jobId -> {intent, regions, hash, look, temporal}

const fxOf = (look) => (look && FX_KEYS.some((k) => look[k] !== undefined) ? Object.fromEntries(FX_KEYS.map((k) => [k, look[k]])) : null);

function lutFor(intent, look) {
  const full = { ...(look || {}), preserveLuminance: intent ? intent.mode === 'preserve' : false };
  for (const k of FX_KEYS) delete full[k]; // spatial effects are not part of the LUT: changing them needs no re-bake
  const key = JSON.stringify(full);
  if (!luts.has(key)) {
    if (luts.size > 24) luts.delete(luts.keys().next().value);
    const pair = full.source === 'ai' ? pairs.get(full.pairId) : null;
    luts.set(key, bakeLUT(pair ? { ...model, pair } : model, full, 33));
  }
  return luts.get(key);
}

function lookHash(look) {
  const s = JSON.stringify(look || {});
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h.toString(16).padStart(8, '0');
}

function prepare(intentJson) {
  const intent = intentJson ? deserializeIntent(intentJson) : null;
  return { intent, regions: intent ? effectiveRegions(intent) : [], hash: intent ? hashIntent(intent) : null };
}

self.onmessage = (e) => {
  const m = e.data;
  try {
    if (m.type === 'fit') {
      const t = performance.now();
      model = fitGrade(m.frames, m.reference);
      luts.clear();
      self.postMessage({ type: 'fitted', id: m.id, ms: performance.now() - t, src: model.src, ref: model.ref, palette: model.palette });
    } else if (m.type === 'fit-pair') {
      const t = performance.now();
      const r = fitPairLUT(new Uint8ClampedArray(m.src), new Uint8ClampedArray(m.ai), m.w, m.h);
      const accepted = r.shift.score >= 0.6; // a different composition must not replace a working LUT
      if (accepted) {
        pairs.set(m.pairId, r.lut);
        while (pairs.size > 6) pairs.delete(pairs.keys().next().value);
      }
      self.postMessage({ type: 'pair-fitted', id: m.id, ms: performance.now() - t, accepted, shift: r.shift, meanAbs: r.meanAbs, outlierShare: r.outlierShare });
    } else if (m.type === 'clear-pair') {
      pairs.clear();
      for (const k of [...luts.keys()]) if (k.includes('"source":"ai"')) luts.delete(k);
      self.postMessage({ type: 'pair-cleared', id: m.id });
    } else if (m.type === 'start-job') {
      jobs.set(m.jobId, { ...prepare(m.intent), look: m.look || null, temporal: { prev: {} } });
      const j = jobs.get(m.jobId);
      self.postMessage({ type: 'job-started', jobId: m.jobId, intentHash: j.hash, lookHash: lookHash(j.look), regions: j.regions.map((r) => ({ id: r.id, kind: r.kind, ev: r.ev })) });
    } else if (m.type === 'end-job') {
      jobs.delete(m.jobId);
    } else if (m.type === 'render') {
      if (!model && m.grade) throw new Error('grade model not fitted yet');
      const lk = m.jobId ? jobs.get(m.jobId)?.look : m.look;
      if (m.grade && lk && lk.source === 'ai' && !pairs.has(lk.pairId)) throw new Error('AI 調色圖的 LUT 與目前設定不一致，請重新上傳');
      // one-off renders (dialog preview) carry their own intent; job renders reuse the job's state
      const j = m.jobId ? jobs.get(m.jobId) : { ...prepare(m.intent), look: m.look || null, temporal: null };
      if (!j) throw new Error(`unknown job ${m.jobId}`);
      const src = new Uint8ClampedArray(m.rgba);
      let ev = null;
      if (j.regions.length) {
        const mh = j.intent.maskH, mw = MASK_W;
        const evidence = { seg: m.seg || null, L: lightnessGrid(src, m.w, m.h, mw, mh) };
        const masks = regionMasks(j.regions, evidence, mw, mh, j.intent.feather, j.temporal);
        ev = evField(j.regions, masks, mw, mh);
      }
      const out = renderFrame(src, m.w, m.h, m.grade ? lutFor(j.intent, j.look) : null, ev, undefined, m.grade ? fxOf(j.look) : null);
      self.postMessage({ type: 'rendered', id: m.id, jobId: m.jobId, index: m.index, rgba: out.buffer, w: m.w, h: m.h, intentHash: j.hash, localApplied: !!ev }, [out.buffer]);
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: m.id, jobId: m.jobId, message: String(err && err.message || err) });
  }
};
