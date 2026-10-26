// Processing worker: fits the reference grade, bakes LUTs, builds region masks from the CONFIRMED
// LightingIntent it receives, and renders frames. The page never renders pixels itself, so the dialog
// preview, the result preview and the export all come out of this one code path.
import { fitGrade, bakeLUT, sampleOklab } from './engine/grade.js';
import { fitCandidates } from './engine/autofit.js';
import { deserializeIntent, effectiveRegions, hashIntent, MASK_W } from './engine/intent.js';
import { regionMasks } from './engine/regionmasks.js';
import { evField, lightnessGrid } from './engine/masks.js';
import { renderFrame, FX_KEYS } from './engine/render.js';
import { fitPairLUT } from './engine/pairfit.js';
import { protectedPrim } from './engine/primary.js';

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

const skinRegion = (mh) => ({ id: '__skin', kind: 'ai-skin', enabled: true, status: 'confirmed', edits: { w: MASK_W, h: mh, add: null, erase: null } });
/** Skin protection for this frame: the tone-only LUT and the (temporally smoothed) AI skin mask, or null. */
function skinFor(intent, look, src, w, h, seg, temporal) {
  const k = look && look.prim && look.prim.skin;
  if (!k || !seg) return null;
  const mh = Math.round(MASK_W * h / w);
  if (!seg.skin || seg.skin.length !== MASK_W * mh) return null;
  const evidence = { seg, L: lightnessGrid(src, w, h, MASK_W, mh) };
  const mask = regionMasks([skinRegion(mh)], evidence, MASK_W, mh, 0.006, temporal)['__skin'];
  return { lut: lutFor(intent, { ...look, prim: protectedPrim(look.prim) }), mask: { data: mask, w: MASK_W, h: mh }, k };
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
      // pro-stage candidates (自然 / 忠實 / 濃烈) from the same keyframes and the reference's central 70%
      const unscale = (a) => { const o = new Float64Array(a.length); for (let i = 0; i < a.length; i++) o[i] = i % 3 ? a[i] / 2 : a[i]; return o; };
      const parts = m.frames.map((f, i) => sampleOklab(f.data, f.width, f.height, 2000, { seed: 21 + i }));
      const srcLab = new Float64Array(parts.reduce((n, q) => n + q.length, 0));
      let off = 0; for (const q of parts) { srcLab.set(q, off); off += q.length; }
      const refLab = sampleOklab(m.reference.data, m.reference.width, m.reference.height, 8000, { centre: 0.7, seed: 7 });
      const t2 = performance.now();
      const f0 = m.frames[Math.floor(m.frames.length / 2)], k = Math.min(1, 320 / f0.width); // texture-guard thumbnail
      const tw = Math.max(16, Math.round(f0.width * k)), th = Math.max(16, Math.round(f0.height * k)), td = new Uint8ClampedArray(tw * th * 4);
      for (let y = 0; y < th; y++) for (let x = 0; x < tw; x++) {
        const s = (Math.min(f0.height - 1, Math.floor(y / k)) * f0.width + Math.min(f0.width - 1, Math.floor(x / k))) * 4, o = (y * tw + x) * 4;
        td[o] = f0.data[s]; td[o + 1] = f0.data[s + 1]; td[o + 2] = f0.data[s + 2]; td[o + 3] = 255;
      }
      const candidates = fitCandidates(unscale(srcLab), unscale(refLab), { thumb: { data: td, width: tw, height: th } })
        .map((c) => ({ mode: c.report.mode, label: c.report.label, prim: c.prim, residual: c.report.residual, norm: c.report.norm }));
      self.postMessage({ type: 'fitted', id: m.id, ms: performance.now() - t, autofitMs: performance.now() - t2, src: model.src, ref: model.ref, palette: model.palette, candidates });
    } else if (m.type === 'bake') { // the look as a LUT (for .cube export); spatial effects are not part of it
      const intent = m.intent ? deserializeIntent(m.intent) : null, lut = lutFor(intent, m.look);
      const data = lut.data.slice();
      self.postMessage({ type: 'baked', id: m.id, size: lut.size, data: data.buffer }, [data.buffer]);
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
      const skin = m.grade ? skinFor(j.intent, j.look, src, m.w, m.h, m.seg, j.temporal) : null;
      const out = renderFrame(src, m.w, m.h, m.grade ? lutFor(j.intent, j.look) : null, ev, undefined, m.grade ? fxOf(j.look) : null, skin);
      self.postMessage({ type: 'rendered', id: m.id, jobId: m.jobId, index: m.index, rgba: out.buffer, w: m.w, h: m.h, intentHash: j.hash, localApplied: !!ev, skinProtected: !!skin }, [out.buffer]);
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: m.id, jobId: m.jobId, message: String(err && err.message || err) });
  }
};
