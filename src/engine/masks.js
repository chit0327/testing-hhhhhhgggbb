// Region masks on a small grid (MASK_W × maskH, values 0..1) and the per-pixel exposure field built
// from confirmed regions. Everything here is a pure function so the worker, the page and the tests agree.
import { S2L } from './color.js';

export function resize(src, sw, sh, dw, dh) { // bilinear
  const out = new Float32Array(dw * dh);
  for (let y = 0; y < dh; y++) {
    const fy = Math.min(sh - 1, Math.max(0, (y + 0.5) * sh / dh - 0.5)), y0 = Math.floor(fy), y1 = Math.min(sh - 1, y0 + 1), ty = fy - y0;
    for (let x = 0; x < dw; x++) {
      const fx = Math.min(sw - 1, Math.max(0, (x + 0.5) * sw / dw - 0.5)), x0 = Math.floor(fx), x1 = Math.min(sw - 1, x0 + 1), tx = fx - x0;
      const a = src[y0 * sw + x0] + (src[y0 * sw + x1] - src[y0 * sw + x0]) * tx;
      const b = src[y1 * sw + x0] + (src[y1 * sw + x1] - src[y1 * sw + x0]) * tx;
      out[y * dw + x] = a + (b - a) * ty;
    }
  }
  return out;
}

/** OKLab-like perceptual lightness (cube root of linear luminance) averaged on the mask grid. */
export function lightnessGrid(rgba, w, h, mw, mh) {
  const sum = new Float32Array(mw * mh), cnt = new Float32Array(mw * mh);
  for (let y = 0; y < h; y++) {
    const gy = Math.min(mh - 1, Math.floor(y * mh / h));
    for (let x = 0; x < w; x++) {
      const p = (y * w + x) * 4, gi = gy * mw + Math.min(mw - 1, Math.floor(x * mw / w));
      sum[gi] += Math.cbrt(0.2126 * S2L[rgba[p]] + 0.7152 * S2L[rgba[p + 1]] + 0.0722 * S2L[rgba[p + 2]]);
      cnt[gi]++;
    }
  }
  for (let i = 0; i < sum.length; i++) sum[i] /= Math.max(1, cnt[i]);
  return sum;
}

const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

/** Luminance qualifier (like a colourist's luma key). Fixed thresholds -> stable from frame to frame. */
export function lumaKey(L, kind) {
  const out = new Float32Array(L.length);
  for (let i = 0; i < L.length; i++) out[i] = kind === 'luma-highlights' ? smooth(0.62, 0.8, L[i]) : 1 - smooth(0.28, 0.45, L[i]);
  return out;
}

/** User brush corrections: 'add' raises the mask, 'erase' lowers it. */
export function applyEdits(mask, edits) {
  if (!edits) return mask;
  const out = Float32Array.from(mask);
  if (edits.add) for (let i = 0; i < out.length; i++) out[i] = Math.max(out[i], edits.add[i] / 255);
  if (edits.erase) for (let i = 0; i < out.length; i++) out[i] = Math.min(out[i], 1 - edits.erase[i] / 255);
  return out;
}

function boxBlur1(src, w, h, r, horizontal) {
  const out = new Float32Array(src.length);
  const n = horizontal ? w : h, m = horizontal ? h : w;
  for (let j = 0; j < m; j++) {
    let acc = 0;
    const at = (i) => src[horizontal ? j * w + Math.min(n - 1, Math.max(0, i)) : Math.min(n - 1, Math.max(0, i)) * w + j];
    for (let i = -r; i <= r; i++) acc += at(i);
    for (let i = 0; i < n; i++) {
      out[horizontal ? j * w + i : i * w + j] = acc / (2 * r + 1);
      acc += at(i + r + 1) - at(i - r);
    }
  }
  return out;
}

/** Soft edges: three separable box blurs ≈ Gaussian. radius in grid pixels. */
export function feather(mask, w, h, radius) {
  const r = Math.max(0, Math.round(radius));
  if (!r) return mask;
  let m = mask;
  for (let k = 0; k < 3; k++) m = boxBlur1(boxBlur1(m, w, h, r, true), w, h, r, false);
  return m;
}

/**
 * Edge-aware refinement (guided filter, He et al. 2010): the soft AI mask is re-fitted locally as a
 * linear function of the frame's lightness, so its edge snaps to the real object edge instead of
 * spreading into the background (which showed as a bright halo around brightened people).
 */
export function guidedFilter(p, I, w, h, r = 4, eps = 2e-3) {
  const n = p.length, Ip = new Float32Array(n), II = new Float32Array(n);
  for (let i = 0; i < n; i++) { Ip[i] = I[i] * p[i]; II[i] = I[i] * I[i]; }
  const box = (x) => boxBlur1(boxBlur1(x, w, h, r, true), w, h, r, false);
  const mI = box(I), mp = box(p), mIp = box(Ip), mII = box(II);
  const a = new Float32Array(n), b = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const v = mII[i] - mI[i] * mI[i], c = mIp[i] - mI[i] * mp[i];
    a[i] = c / (v + eps); b[i] = mp[i] - a[i] * mI[i];
  }
  const ma = box(a), mb = box(b), q = new Float32Array(n);
  for (let i = 0; i < n; i++) q[i] = Math.min(1, Math.max(0, ma[i] * I[i] + mb[i]));
  return q;
}

/** Temporal smoothing of a per-frame mask (exponential moving average). */
export function ema(prev, cur, alpha) {
  if (!prev) return cur;
  const out = new Float32Array(cur.length);
  for (let i = 0; i < cur.length; i++) out[i] = prev[i] + alpha * (cur[i] - prev[i]);
  return out;
}

/**
 * Exposure field (stops) from effective regions. masks: {regionId: Float32Array(w*h)} already
 * edited/feathered. Overlap rule: stops add, then clamp to ±limit. Returns null if nothing applies.
 */
export function evField(regions, masks, w, h, limit = 1.5) {
  if (!regions.length) return null;
  const ev = new Float32Array(w * h);
  for (const r of regions) {
    const m = masks[r.id];
    if (!m) throw new Error(`missing mask for confirmed region ${r.id}`);
    for (let i = 0; i < ev.length; i++) ev[i] += r.ev * m[i];
  }
  for (let i = 0; i < ev.length; i++) ev[i] = Math.max(-limit, Math.min(limit, ev[i]));
  return { data: ev, w, h };
}
