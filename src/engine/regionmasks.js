// Turn a region definition + per-frame evidence into the mask the renderer uses.
// evidence: { seg: {person, skin, background} Float32 grids (mw×mh) or null, L: lightness grid }.
import { lumaKey, applyEdits, feather, ema, guidedFilter } from './masks.js';
import { KINDS } from './intent.js';

export function rawMask(region, evidence, mw, mh) {
  const n = mw * mh;
  switch (region.kind) {
    case 'ai-person': case 'ai-skin': case 'ai-background': case 'ai-hair': case 'ai-clothes': {
      const key = region.kind.slice(3);
      if (!evidence.seg || !evidence.seg[key]) throw new Error('AI segmentation is not available for this frame');
      return Float32Array.from(evidence.seg[key]);
    }
    case 'luma-highlights': case 'luma-shadows': return lumaKey(evidence.L, region.kind);
    case 'manual': return new Float32Array(n);
    case 'global': return new Float32Array(n).fill(1);
    default: throw new Error(`unknown region kind ${region.kind}`);
  }
}

/**
 * Masks for the given (effective) regions on one frame. `state` carries temporal smoothing between
 * frames: {prev: {regionId: Float32Array}}; pass the same object for consecutive frames of one job.
 * Tracked kinds are recomputed from the frame and smoothed over time; brush edits are applied on top in
 * frame coordinates (they do not follow motion – documented limitation).
 */
export function regionMasks(regions, evidence, mw, mh, featherFrac, state = null, alpha = 0.5) {
  const out = {};
  for (const r of regions) {
    let m = rawMask(r, evidence, mw, mh);
    if (state && KINDS[r.kind].tracked) {
      m = ema(state.prev[r.id], m, alpha);
      state.prev[r.id] = m;
    }
    if (KINDS[r.kind].source === 'ai' && evidence.L) {
      // snap the soft AI mask to image edges, then a soft threshold: interior -> 1, residual spill -> 0
      m = guidedFilter(m, evidence.L, mw, mh);
      for (let i = 0; i < m.length; i++) { const t = Math.min(1, Math.max(0, (m[i] - 0.2) / 0.6)); m[i] = t * t * (3 - 2 * t); }
    }
    m = applyEdits(m, r.edits);
    out[r.id] = feather(m, mw, mh, featherFrac * mw);
  }
  return out;
}
