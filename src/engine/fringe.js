// 色散 (RGB 分離, lateral chromatic aberration) only where the picture is out of focus: red pushed outwards and blue
// inwards along the line from the frame centre, as a real lens does towards its edges, so contrast edges in the
// blurred background and soft hair get red / cyan fringes while the sharp subject stays clean.
//
// Where is "out of focus": per cell of the ~320 px effects grid (render.js gridDims), fine detail (perceptual lightness
// minus the mean of its 4 neighbours a short step away: 1 px at 640 px wide, 2 px at 1280 — the same scale at any size)
// is compared with medium detail (the cell means minus their wider mean). In focus, both are there; out of focus, fine
// detail is gone while the medium-scale shapes remain. The frame's noise floor (its quietest cells: sensor noise and
// compression grain look like fine detail everywhere) is taken off first, and the ratio is measured against the
// frame's own sharpest parts. Flat areas have nothing to judge: they take the decision of the nearby detail. A colour
// shift only shows on edges, so smooth in-focus skin stays unchanged anyway.
//
// fx.fringe 0..1 (strength: the shift at a frame corner, up to 0.4 % of the long side), fx.fringeFocus 0..1 (1 = only
// out of focus, 0 = everywhere like a plain lens). The GPU version (engine/gpufx.js) builds the same grid with its DOWN
// pass (mode 4) and uses fringeWeights and the same per-pixel shift. In place on RGBA.
import { gridDims, boxBlur } from './render.js';
import { S2L } from './color.js';

const clamp01 = (v) => Math.min(1, Math.max(0, +v || 0));
const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

export function fringePlan(fx) {
  const amt = clamp01(fx && fx.fringe);
  return amt ? { amt, focus: clamp01(fx.fringeFocus ?? 1) } : null;
}
export const fringeMaxShift = (w, h) => 0.004 * Math.max(w, h); // px at strength 1, frame corner (5 px at 1280)

export const fringeStep = (w, h) => Math.max(1, Math.round(Math.max(w, h) / 640));
const percL = (img, p) => Math.cbrt(0.2126 * S2L[img[p]] + 0.7152 * S2L[img[p + 1]] + 0.0722 * S2L[img[p + 2]]);
/** per grid cell: [mean perceptual lightness, mean fine detail] — what the GPU DOWN pass mode 4 gives */
export function focusGrid(img, w, h) {
  const { f, sw, sh } = gridDims(w, h), k = fringeStep(w, h), g = new Float32Array(sw * sh * 2), cnt = new Float32Array(sw * sh);
  const L = new Float32Array(w * h); for (let i = 0; i < w * h; i++) L[i] = percL(img, 4 * i);
  for (let y = 0; y < h; y++) {
    const yu = Math.max(0, y - k) * w, yd = Math.min(h - 1, y + k) * w, row = y * w;
    for (let x = 0; x < w; x++) {
      const v = L[row + x], m = (L[row + Math.max(0, x - k)] + L[row + Math.min(w - 1, x + k)] + L[yu + x] + L[yd + x]) / 4, q = Math.floor(y / f) * sw + Math.floor(x / f);
      g[2 * q] += v; g[2 * q + 1] += Math.abs(v - m); cnt[q]++;
    }
  }
  for (let q = 0; q < sw * sh; q++) { g[2 * q] /= cnt[q] || 1; g[2 * q + 1] /= cnt[q] || 1; }
  return { g, f, sw, sh };
}

const pct = (arr, p) => { const a = Array.from(arr).sort((x, y) => x - y); return a.length ? a[Math.min(a.length - 1, Math.floor(a.length * p))] : 0; };
/** per grid cell: the shift weight 0..1 (strength × out-of-focus share), from the focus grid */
export function fringeWeights(plan, fg) {
  const { g, sw, sh } = fg, n = sw * sh, three = () => new Float32Array(n * 3);
  const Lc = three(), F = three();
  for (let q = 0; q < n; q++) { Lc[3 * q] = Lc[3 * q + 1] = Lc[3 * q + 2] = g[2 * q]; F[3 * q] = F[3 * q + 1] = F[3 * q + 2] = g[2 * q + 1]; }
  const B = Lc.slice(); boxBlur(B, sw, sh, 2);
  const M = three(); for (let i = 0; i < n * 3; i++) M[i] = Math.abs(Lc[i] - B[i]);
  boxBlur(M, sw, sh, 3); boxBlur(F, sw, sh, 3);
  const noise = pct(Array.from({ length: n }, (_, q) => g[2 * q + 1]), 0.1);
  const R = new Float32Array(n), conf = new Float32Array(n);
  for (let q = 0; q < n; q++) { R[q] = Math.max(0, F[3 * q] - noise) / (M[3 * q] + 0.004); conf[q] = smooth(0.002, 0.01, M[3 * q]); }
  const r90 = Math.max(1e-3, pct(R.filter((_, q) => conf[q] > 0.5), 0.9) || pct(R, 0.9));
  const D = three(), C = three();
  for (let q = 0; q < n; q++) { const d = 1 - smooth(0.2, 0.55, R[q] / r90); D[3 * q] = d * conf[q]; C[3 * q] = conf[q]; }
  for (let pass = 0; pass < 2; pass++) { boxBlur(D, sw, sh, 3); boxBlur(C, sw, sh, 3); } // flat areas take the nearby detail's decision
  const out = new Float32Array(n);
  for (let q = 0; q < n; q++) { const d = ((D[3 * q] + 0.01) / (C[3 * q] + 0.01)) ** 2; out[q] = plan.amt * ((1 - plan.focus) + plan.focus * d); }
  return out;
}

export function applyFringe(img, w, h, fx) {
  const plan = fringePlan(fx);
  if (!plan) return img;
  const fg = focusGrid(img, w, h), Wt = fringeWeights(plan, fg), { f, sw, sh } = fg;
  const src = img.slice(), cx = (w - 1) / 2, cy = (h - 1) / 2, half = Math.hypot(cx, cy), ms = fringeMaxShift(w, h);
  const bil = (sx, sy, c) => {
    sx = Math.min(w - 1, Math.max(0, sx)); sy = Math.min(h - 1, Math.max(0, sy));
    const x0 = Math.floor(sx), y0 = Math.floor(sy), x1 = Math.min(w - 1, x0 + 1), y1 = Math.min(h - 1, y0 + 1), tx = sx - x0, ty = sy - y0;
    const a = src[(y0 * w + x0) * 4 + c] + (src[(y0 * w + x1) * 4 + c] - src[(y0 * w + x0) * 4 + c]) * tx;
    const b = src[(y1 * w + x0) * 4 + c] + (src[(y1 * w + x1) * 4 + c] - src[(y1 * w + x0) * 4 + c]) * tx;
    return a + (b - a) * ty;
  };
  for (let y = 0; y < h; y++) {
    const gy = Math.min(sh - 1, Math.max(0, (y + 0.5) / f - 0.5)), y0 = Math.floor(gy), y1 = Math.min(sh - 1, y0 + 1), ty = gy - y0, dy = y - cy;
    for (let x = 0; x < w; x++) {
      const gx = Math.min(sw - 1, Math.max(0, (x + 0.5) / f - 0.5)), x0 = Math.floor(gx), x1 = Math.min(sw - 1, x0 + 1), tx = gx - x0;
      const wa = Wt[y0 * sw + x0] + (Wt[y0 * sw + x1] - Wt[y0 * sw + x0]) * tx, wb = Wt[y1 * sw + x0] + (Wt[y1 * sw + x1] - Wt[y1 * sw + x0]) * tx;
      const dx = x - cx, r = Math.hypot(dx, dy);
      if (r < 1) continue;
      const s = ms * (wa + (wb - wa) * ty) * (0.3 + 0.7 * r / half);
      if (s < 0.05) continue;
      const ux = dx / r * s, uy = dy / r * s, p = (y * w + x) * 4;
      img[p] = bil(x + ux, y + uy, 0); img[p + 2] = bil(x - ux, y - uy, 2);
    }
  }
  return img;
}
