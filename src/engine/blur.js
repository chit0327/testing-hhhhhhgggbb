// Background blur (背景模糊): the selection, its tracking through camera motion, and the three blurs.
// Pure functions shared by the page (painting, overlay), the worker (rendering) and the tests.
//
// Selection = AI background (per frame, from segmentation) and/or a painted layer (brush, eraser, lasso) drawn on
// one frame (the anchor). The painted layer follows the camera: every frame is registered to the anchor with an
// affine motion model (coarse-to-fine Lucas–Kanade on small luma images) and the layer is warped with it.
// Blurs use normalized convolution: blur(colour × mask) / blur(mask), so sharp (unselected) pixels never bleed into
// the blurred area — no halo of the subject around its outline.
import { S2L, l2sFast, srgbToOklab } from './color.js';
import { rollOff } from './render.js';

export const BLUR_TYPES = ['gauss', 'bokeh', 'motion'];
/** edge softness 0..1 → feather radius in mask-grid cells (384 wide): 0 ≈ a crisp edge, 1 ≈ 6 % of the frame width */
export const featherCells = (f = 0.04) => 0.5 + 24 * Math.min(1, Math.max(0, f ?? 0.04)); // default 0.04 ≈ 1.5 cells
export const BRUSH_SIZES = { s: 0.02, m: 0.05, l: 0.1 }; // brush radius as a share of the frame width
export const TRACK_W = 160; // tracking luma width

// ---------------------------------------------------------------- painted layer (w × h grid, values -1..1)
// +1 = added to the selection, −1 = erased from it (also erases the AI background there), 0 = untouched
/** a round brush stroke from (x0,y0) to (x1,y1) in grid pixels, radius r, value v (+1 brush, −1 eraser) */
export function paintStroke(layer, w, h, x0, y0, x1, y1, r, v) {
  const len = Math.hypot(x1 - x0, y1 - y0), steps = Math.max(1, Math.ceil(len / Math.max(0.5, r / 3)));
  for (let s = 0; s <= steps; s++) {
    const cx = x0 + (x1 - x0) * s / steps, cy = y0 + (y1 - y0) * s / steps;
    for (let y = Math.max(0, Math.floor(cy - r - 1)); y <= Math.min(h - 1, Math.ceil(cy + r + 1)); y++) {
      for (let x = Math.max(0, Math.floor(cx - r - 1)); x <= Math.min(w - 1, Math.ceil(cx + r + 1)); x++) {
        const d = Math.hypot(x + 0.5 - cx, y + 0.5 - cy), a = Math.min(1, Math.max(0, r + 0.5 - d)); // 1-px anti-aliased edge
        if (a > 0) { const i = y * w + x; layer[i] = v > 0 ? Math.max(layer[i], a * v) : Math.min(layer[i], a * v); }
      }
    }
  }
  return layer;
}
/** fill a polygon [[x,y],…] (grid pixels) with value v; even-odd rule, sampled at pixel centres */
export function fillPolygon(layer, w, h, pts, v) {
  if (pts.length < 3) return layer;
  for (let y = 0; y < h; y++) {
    const yc = y + 0.5, xs = [];
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      const [xi, yi] = pts[i], [xj, yj] = pts[j];
      if ((yi > yc) !== (yj > yc)) xs.push(xi + (yc - yi) * (xj - xi) / (yj - yi));
    }
    xs.sort((a, b) => a - b);
    for (let k = 0; k + 1 < xs.length; k += 2) {
      for (let x = Math.max(0, Math.ceil(xs[k] - 0.5)); x <= Math.min(w - 1, Math.floor(xs[k + 1] - 0.5)); x++) layer[y * w + x] = v;
    }
  }
  return layer;
}

/** the frame's OKLab colours averaged onto a gw × gh grid (3 floats per cell), lightly smoothed (sensor noise) */
export function labGrid(rgba, w, h, gw, gh) {
  const acc = new Float32Array(gw * gh * 3), cnt = new Float32Array(gw * gh);
  for (let y = 0; y < h; y++) {
    const gy = Math.min(gh - 1, Math.floor(y * gh / h));
    for (let x = 0; x < w; x++) {
      const p = (y * w + x) * 4, q = gy * gw + Math.min(gw - 1, Math.floor(x * gw / w));
      acc[3 * q] += rgba[p]; acc[3 * q + 1] += rgba[p + 1]; acc[3 * q + 2] += rgba[p + 2]; cnt[q]++;
    }
  }
  const lab = new Float32Array(gw * gh * 3), t = [0, 0, 0];
  for (let q = 0; q < gw * gh; q++) {
    const n = cnt[q] || 1; srgbToOklab(acc[3 * q] / n / 255, acc[3 * q + 1] / n / 255, acc[3 * q + 2] / n / 255, t);
    lab[3 * q] = t[0]; lab[3 * q + 1] = t[1]; lab[3 * q + 2] = t[2];
  }
  const out = new Float32Array(lab.length); // 3×3 mean
  for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) for (let c = 0; c < 3; c++) {
    let s = 0, n = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const yy = y + dy, xx = x + dx; if (yy >= 0 && yy < gh && xx >= 0 && xx < gw) { s += lab[(yy * gw + xx) * 3 + c]; n++; } }
    out[(y * gw + x) * 3 + c] = s / n;
  }
  return out;
}
/**
 * 魔術棒 (magic wand): the connected area around (sx, sy) whose colour is within `tol` of the clicked colour
 * (OKLab distance × 100, ≈ ΔE; lightness counts half, so a surface in light and shade stays one area). Compared with
 * the clicked colour, not with the neighbour, so it does not creep along a gradient into other surfaces.
 * Returns a Uint8Array (1 = selected) on the gw × gh grid.
 */
export function magicWand(lab, gw, gh, sx, sy, tol) {
  const out = new Uint8Array(gw * gh), x0 = Math.min(gw - 1, Math.max(0, Math.floor(sx))), y0 = Math.min(gh - 1, Math.max(0, Math.floor(sy)));
  const s = (y0 * gw + x0) * 3, L0 = lab[s], a0 = lab[s + 1], b0 = lab[s + 2], t2 = (tol / 100) ** 2;
  const near = (q) => { const dL = (lab[3 * q] - L0) * 0.5, da = lab[3 * q + 1] - a0, db = lab[3 * q + 2] - b0; return dL * dL + da * da + db * db <= t2; };
  const stack = [y0 * gw + x0]; out[stack[0]] = 1;
  while (stack.length) {
    const q = stack.pop(), x = q % gw, y = (q / gw) | 0;
    for (const n of [x > 0 ? q - 1 : -1, x < gw - 1 ? q + 1 : -1, y > 0 ? q - gw : -1, y < gh - 1 ? q + gw : -1]) {
      if (n >= 0 && !out[n] && near(n)) { out[n] = 1; stack.push(n); }
    }
  }
  return out;
}
/** inverse of an affine p (so a selection made on the current frame can be written into the anchor's layer) */
export function invertAffine(p) {
  const a = 1 + p[0], b = p[1], c = p[3], d = 1 + p[4], det = a * d - b * c;
  const ia = d / det, ib = -b / det, ic = -c / det, id = a / det;
  return [ia - 1, ib, -(ia * p[2] + ib * p[5]), ic, id - 1, -(ic * p[2] + id * p[5])];
}
/** write a selection made on the CURRENT frame (sel: 0/1 on the w × h grid) into the anchor's layer with value v */
export function stampSelection(layer, w, h, sel, pCurToAnchor, v) {
  const inv = invertAffine(pCurToAnchor), id = pCurToAnchor.every((x) => x === 0);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let on;
    if (id) on = sel[y * w + x];
    else {
      const [u, vv] = toN(x, y, w, h), [X, Y] = fromN(...warpPoint(inv, u, vv), w, h), xi = Math.round(X), yi = Math.round(Y);
      on = xi >= 0 && yi >= 0 && xi < w && yi < h && sel[yi * w + xi];
    }
    if (on) layer[y * w + x] = v;
  }
  return layer;
}

/**
 * The selection (0..1) on the grid: AI background (optional, per frame) plus the painted layer, then inverted
 * if asked. layer may be null (AI only), bg may be null (painted only).
 */
export function combineSelection(layer, bg, { ai = false, invert = false } = {}, n = (layer || bg).length) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let m = ai && bg ? bg[i] : 0;
    if (layer) { const v = layer[i]; m = v >= 0 ? Math.max(m, v) : Math.min(m, 1 + v); }
    out[i] = invert ? 1 - m : m;
  }
  return out;
}

// ---------------------------------------------------------------- tracking (camera motion)
/** small luma image (0..1, perceptual) for tracking: {w, h, data} */
export function lumaSmall(rgba, w, h, tw = TRACK_W) {
  const sw = Math.min(tw, w), sh = Math.max(2, Math.round(sw * h / w)), d = new Float32Array(sw * sh), cnt = new Float32Array(sw * sh);
  for (let y = 0; y < h; y++) {
    const yy = Math.min(sh - 1, Math.floor(y * sh / h));
    for (let x = 0; x < w; x++) {
      const p = (y * w + x) * 4, q = yy * sw + Math.min(sw - 1, Math.floor(x * sw / w));
      d[q] += Math.cbrt(0.2126 * S2L[rgba[p]] + 0.7152 * S2L[rgba[p + 1]] + 0.0722 * S2L[rgba[p + 2]]); cnt[q]++;
    }
  }
  for (let i = 0; i < d.length; i++) d[i] /= cnt[i] || 1;
  return { w: sw, h: sh, data: d };
}
function half(im) {
  const w = Math.max(2, im.w >> 1), h = Math.max(2, im.h >> 1), d = new Float32Array(w * h), s = im.data;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const x0 = Math.min(im.w - 1, 2 * x), x1 = Math.min(im.w - 1, 2 * x + 1), y0 = Math.min(im.h - 1, 2 * y), y1 = Math.min(im.h - 1, 2 * y + 1);
    d[y * w + x] = (s[y0 * im.w + x0] + s[y0 * im.w + x1] + s[y1 * im.w + x0] + s[y1 * im.w + x1]) / 4;
  }
  return { w, h, data: d };
}
/** pyramid (finest first), each level zero-mean (exposure changes between frames matter less) with gradients */
export function pyramid(im, levels = 3) {
  const out = [];
  let cur = im;
  for (let l = 0; l < levels; l++) {
    const { w, h, data } = cur;
    let mean = 0; for (let i = 0; i < data.length; i++) mean += data[i]; mean /= data.length;
    const z = new Float32Array(data.length), gx = new Float32Array(data.length), gy = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) z[i] = data[i] - mean;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x;
      gx[i] = (z[y * w + Math.min(w - 1, x + 1)] - z[y * w + Math.max(0, x - 1)]) / 2;
      gy[i] = (z[Math.min(h - 1, y + 1) * w + x] - z[Math.max(0, y - 1) * w + x]) / 2;
    }
    out.push({ w, h, z, gx, gy });
    cur = half(cur);
  }
  return out;
}
const bil = (d, w, h, x, y) => {
  const x0 = Math.floor(x), y0 = Math.floor(y), tx = x - x0, ty = y - y0, x1 = x0 + 1, y1 = y0 + 1;
  const a = d[y0 * w + x0], b = d[y0 * w + x1], c = d[y1 * w + x0], e = d[y1 * w + x1];
  return (a + (b - a) * tx) + ((c + (e - c) * tx) - (a + (b - a) * tx)) * ty;
};
export const IDENTITY = [0, 0, 0, 0, 0, 0];
/** apply affine p in normalized, centred coordinates: u = (x + ½ − w/2) / w, v = (y + ½ − h/2) / w */
export const warpPoint = (p, u, v) => [(1 + p[0]) * u + p[1] * v + p[2], p[3] * u + (1 + p[4]) * v + p[5]];
export const toN = (x, y, w, h) => [(x + 0.5 - w / 2) / w, (y + 0.5 - h / 2) / w]; // pixel centre → normalized
export const fromN = (u, v, w, h) => [u * w + w / 2 - 0.5, v * w + h / 2 - 0.5];   // normalized → pixel coordinate
/** composition: (p ∘ q)(x) = p(q(x)) */
export function compose(p, q) {
  const A = [[1 + p[0], p[1], p[2]], [p[3], 1 + p[4], p[5]]], B = [[1 + q[0], q[1], q[2]], [q[3], 1 + q[4], q[5]]];
  const m = (i, j) => A[i][0] * B[0][j] + A[i][1] * B[1][j] + (j === 2 ? A[i][2] : 0);
  return [m(0, 0) - 1, m(0, 1), m(0, 2), m(1, 0), m(1, 1) - 1, m(1, 2)];
}
function solveN(H, b) { // Gaussian elimination with partial pivoting; H n×n (array of rows), returns x or null
  const n = b.length, M = H.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let piv = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    if (Math.abs(M[piv][c]) < 1e-12) return null;
    [M[c], M[piv]] = [M[piv], M[c]];
    for (let r = c + 1; r < n; r++) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) { let s = M[r][n]; for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k]; x[r] = s / M[r][r]; }
  return x;
}
// residual statistics of a warp on one level: [MEDIAN |r| (moving subjects covering less than half the frame do not
// count: it measures the background, i.e. the camera), share of the current frame that lands inside the anchor]
function residual(A, C, p) {
  const rs = [];
  for (let y = 0; y < C.h; y++) for (let x = 0; x < C.w; x++) {
    const [u, v] = toN(x, y, C.w, C.h), [u2, v2] = warpPoint(p, u, v), [X, Y] = fromN(u2, v2, A.w, A.h);
    if (X < 0 || Y < 0 || X > A.w - 1.001 || Y > A.h - 1.001) continue;
    rs.push(Math.abs(bil(A.z, A.w, A.h, X, Y) - C.z[y * C.w + x]));
  }
  if (!rs.length) return [1, 0];
  rs.sort((a, b) => a - b);
  return [rs[rs.length >> 1], rs.length / (C.w * C.h)];
}
/**
 * Camera motion from the current frame to the anchor: a similarity (pan, zoom, roll) stored as affine p with anchor(u', v') ≈ current(u, v),
 * (u', v') = warpPoint(p, u, v). anchorPyr / curPyr from pyramid(). init: starting estimate (e.g. the
 * previous frame's). Returns {p, err (mean |Δluma| after alignment), overlap (0..1), ok}.
 */
export function estimateMotion(anchorPyr, curPyr, init = null) {
  if (init) return refineMotion(anchorPyr, curPyr, init);
  const a = refineMotion(anchorPyr, curPyr, IDENTITY); // small motion: start from no motion
  if (a.ok) return a;
  const b = refineMotion(anchorPyr, curPyr, null);      // large pan: start from a coarse translation search
  return b.err < 0.7 * a.err ? b : a;                   // a searched jump must be clearly better, not just different
}
function refineMotion(anchorPyr, curPyr, init) {
  const L = Math.min(anchorPyr.length, curPyr.length);
  let p = init ? [(init[0] + init[4]) / 2, (init[1] - init[3]) / 2, init[2], (init[3] - init[1]) / 2, (init[0] + init[4]) / 2, init[5]] : [...IDENTITY];
  if (!init) { // coarse exhaustive translation search on the smallest level
    const A = anchorPyr[L - 1], C = curPyr[L - 1], R = Math.round(A.w * 0.3);
    let best = [Infinity, 0, 0];
    for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
      const q = [0, 0, dx / A.w, 0, 0, dy / A.w], [e, ov] = residual(A, C, q);
      if (ov < 0.4) continue;
      const score = e + 0.02 * (1 - ov); if (score < best[0]) best = [score, dx, dy];
    }
    if (best[0] < Infinity) p = [0, 0, best[1] / A.w, 0, 0, best[2] / A.w];
  }
  // similarity model (pan, zoom, roll): affine p = [k, −r, tx, r, k, ty]. A free affine also fits shear, which
  // soaks up moving subjects (hands, people) on a still camera; the camera itself does not shear.
  for (let l = L - 1; l >= 0; l--) {
    const A = anchorPyr[l], C = curPyr[l];
    for (let it = 0; it < (l === 0 ? 8 : 12); it++) {
      // robust scale: median |residual| on a subsample
      const smp = [];
      for (let y = 0; y < C.h; y += 2) for (let x = 0; x < C.w; x += 2) {
        const [u, v] = toN(x, y, C.w, C.h), [X, Y] = fromN(...warpPoint(p, u, v), A.w, A.h);
        if (X < 0 || Y < 0 || X > A.w - 1.001 || Y > A.h - 1.001) continue;
        smp.push(Math.abs(bil(A.z, A.w, A.h, X, Y) - C.z[y * C.w + x]));
      }
      smp.sort((a, b) => a - b);
      const k = Math.max(0.004, 1.4826 * (smp.length ? smp[smp.length >> 1] : 0.05)) * 2;
      const H = Array.from({ length: 4 }, () => new Array(4).fill(0)), g = new Array(4).fill(0);
      for (let y = 0; y < C.h; y++) for (let x = 0; x < C.w; x++) {
        const [u, v] = toN(x, y, C.w, C.h), [X, Y] = fromN(...warpPoint(p, u, v), A.w, A.h);
        if (X < 0 || Y < 0 || X > A.w - 1.001 || Y > A.h - 1.001) continue;
        const r = bil(A.z, A.w, A.h, X, Y) - C.z[y * C.w + x], ar = Math.abs(r);
        if (ar > 3 * k) continue; // Tukey-like: clear outliers (moving subjects) do not vote at all
        const wgt = ar <= k ? 1 : k / ar;
        const Ix = bil(A.gx, A.w, A.h, X, Y) * A.w, Iy = bil(A.gy, A.w, A.h, X, Y) * A.w;
        const J = [Ix * u + Iy * v, -Ix * v + Iy * u, Ix, Iy];
        for (let a = 0; a < 4; a++) { g[a] += wgt * J[a] * r; for (let b = a; b < 4; b++) H[a][b] += wgt * J[a] * J[b]; }
      }
      for (let a = 0; a < 4; a++) for (let b = 0; b < a; b++) H[a][b] = H[b][a];
      const tr = (H[0][0] + H[1][1] + H[2][2] + H[3][3]) / 4 + 1e-9;
      for (let a = 0; a < 4; a++) H[a][a] += 1e-3 * tr;
      const d = solveN(H, g.map((x) => -x));
      if (!d) break;
      p = [p[0] + d[0], p[1] - d[1], p[2] + d[2], p[3] + d[1], p[4] + d[0], p[5] + d[3]];
      if (Math.hypot(d[2], d[3]) * A.w < 0.01 && Math.hypot(d[0], d[1]) < 1e-4) break;
    }
  }
  const [err, overlap] = residual(anchorPyr[0], curPyr[0], p);
  const ok = overlap > 0.35 && err < 0.03 && Math.hypot(p[0], p[1], p[3], p[4]) < 0.5;
  return { p, err, overlap, ok };
}

/** the painted layer (anchor coordinates) seen in the current frame: out(x) = layer(p(x)); outside: edge value */
export function warpLayer(layer, w, h, p) {
  if (p.every((v) => v === 0)) return layer;
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const [u, v] = toN(x, y, w, h), [u2, v2] = warpPoint(p, u, v), [X0, Y0] = fromN(u2, v2, w, h);
    const X = Math.min(w - 1.001, Math.max(0, X0)), Y = Math.min(h - 1.001, Math.max(0, Y0));
    out[y * w + x] = bil(layer, w, h, X, Y);
  }
  return out;
}

// ---------------------------------------------------------------- the blurs
const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
function boxPass(src, dst, w, h, r, horizontal, ch) { // ch channels interleaved, clamped edges
  const n = horizontal ? w : h, m = horizontal ? h : w, idx = (i, j) => (horizontal ? j * w + i : i * w + j) * ch;
  for (let j = 0; j < m; j++) for (let c = 0; c < ch; c++) {
    let acc = 0;
    for (let i = -r; i <= r; i++) acc += src[idx(Math.min(n - 1, Math.max(0, i)), j) + c];
    for (let i = 0; i < n; i++) {
      dst[idx(i, j) + c] = acc / (2 * r + 1);
      acc += src[idx(Math.min(n - 1, i + r + 1), j) + c] - src[idx(Math.max(0, i - r), j) + c];
    }
  }
}
function gaussBlur(buf, w, h, sigma) {
  const r = Math.max(1, Math.round(sigma * 0.9)), tmp = new Float32Array(buf.length); // 3 box passes of radius r ≈ Gaussian σ ≈ r
  for (let k = 0; k < 3; k++) { boxPass(buf, tmp, w, h, r, true, 4); boxPass(tmp, buf, w, h, r, false, 4); }
  return buf;
}
function discBlur(buf, w, h, R) { // round aperture: sum of row chords via per-row prefix sums
  const P = new Float32Array((w + 1) * h * 4), out = new Float32Array(buf.length), r = Math.max(1, Math.round(R));
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < 4; c++) P[(y * (w + 1) + x + 1) * 4 + c] = P[(y * (w + 1) + x) * 4 + c] + buf[(y * w + x) * 4 + c];
  const chord = []; for (let dy = -r; dy <= r; dy++) chord.push([dy, Math.floor(Math.sqrt(Math.max(0, (r + 0.5) ** 2 - dy * dy)))]);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (const [dy, c] of chord) {
      const yy = Math.min(h - 1, Math.max(0, y + dy)), a = Math.max(0, x - c), b = Math.min(w, x + c + 1), row = yy * (w + 1);
      s0 += P[(row + b) * 4] - P[(row + a) * 4]; s1 += P[(row + b) * 4 + 1] - P[(row + a) * 4 + 1];
      s2 += P[(row + b) * 4 + 2] - P[(row + a) * 4 + 2]; s3 += P[(row + b) * 4 + 3] - P[(row + a) * 4 + 3];
    }
    const o = (y * w + x) * 4; out[o] = s0; out[o + 1] = s1; out[o + 2] = s2; out[o + 3] = s3;
  }
  return out; // (unnormalized: the caller divides by the weight channel)
}
function lineBlur(buf, w, h, len, angleDeg) { // straight streak, centred, length len px, along angle (0° = →, 90° = ↑)
  const out = new Float32Array(buf.length), K = Math.max(2, Math.ceil(len)), t = angleDeg * Math.PI / 180, dx = Math.cos(t), dy = -Math.sin(t); // counter-clockwise degrees, screen y down
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (let k = 0; k <= K; k++) {
      const f = (k / K - 0.5) * len, X = Math.min(w - 1.001, Math.max(0, x + dx * f)), Y = Math.min(h - 1.001, Math.max(0, y + dy * f));
      const x0 = X | 0, y0 = Y | 0, tx = X - x0, ty = Y - y0, i00 = (y0 * w + x0) * 4, i01 = i00 + 4, i10 = i00 + w * 4, i11 = i10 + 4;
      const w00 = (1 - tx) * (1 - ty), w01 = tx * (1 - ty), w10 = (1 - tx) * ty, w11 = tx * ty;
      s0 += buf[i00] * w00 + buf[i01] * w01 + buf[i10] * w10 + buf[i11] * w11;
      s1 += buf[i00 + 1] * w00 + buf[i01 + 1] * w01 + buf[i10 + 1] * w10 + buf[i11 + 1] * w11;
      s2 += buf[i00 + 2] * w00 + buf[i01 + 2] * w01 + buf[i10 + 2] * w10 + buf[i11 + 2] * w11;
      s3 += buf[i00 + 3] * w00 + buf[i01 + 3] * w01 + buf[i10 + 3] * w10 + buf[i11 + 3] * w11;
    }
    const o = (y * w + x) * 4; out[o] = s0; out[o + 1] = s1; out[o + 2] = s2; out[o + 3] = s3;
  }
  return out;
}
/** blur size in working pixels for a frame whose long side is `side` (relative to the frame: preview = export) */
export const blurRadius = (type, amount, side) => (type === 'motion' ? 0.12 : type === 'bokeh' ? 0.035 : 0.03) * Math.min(1, Math.max(0, amount)) * side;

/**
 * Blur the selected part of an RGBA frame in place. mask: {data 0..1, w, h} (any grid, resampled bilinearly).
 * cfg: {type 'gauss'|'bokeh'|'motion', amount 0..1, angle (motion, degrees)}. Work happens in linear light on a
 * grid of ≤ ~480 px (blurred content has no fine detail), composited back at full resolution.
 */
export function applyBlur(img, w, h, mask, cfg) {
  const amount = Math.min(1, Math.max(0, cfg.amount || 0)), type = BLUR_TYPES.includes(cfg.type) ? cfg.type : 'gauss';
  if (!amount || !mask) return img;
  // working grid: ≤ ~480 px; for bokeh fine enough that the disc spans ≥ 4 grid pixels — a small disc on a coarse grid
  // shows the grid cells (discs of 1–2 cells are squares, and boosted highlights make them stand out as blocks)
  // (Gaussian and motion blur stay smooth on the coarse grid; bokeh grid is at most ~1920 px for speed)
  const coarse = Math.max(1, Math.round(Math.max(w, h) / 480));
  const f = type === 'bokeh' ? Math.max(1, Math.round(Math.max(w, h) / 1920), Math.min(coarse, Math.floor(blurRadius(type, amount, Math.max(w, h)) / 4))) : coarse;
  const sw = Math.ceil(w / f), sh = Math.ceil(h / f);
  const M = mask.data, mw = mask.w, mh = mask.h;
  const maskAt = (x, y) => bil(M, mw, mh, Math.min(mw - 1.001, Math.max(0, (x + 0.5) * mw / w - 0.5)), Math.min(mh - 1.001, Math.max(0, (y + 0.5) * mh / h - 0.5)));
  // working buffer: [r·m, g·m, b·m, m] in linear light (bokeh: highlights boosted so bright points open into discs)
  let buf = new Float32Array(sw * sh * 4); const cnt = new Float32Array(sw * sh);
  let any = false;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = (y * w + x) * 4, q = (Math.floor(y / f) * sw + Math.floor(x / f)), m = maskAt(x, y);
    if (m > 0.002) any = true;
    let r = S2L[img[p]], g = S2L[img[p + 1]], b = S2L[img[p + 2]];
    if (type === 'bokeh') { const k = 1 + 5 * smooth(0.5, 1, 0.2126 * r + 0.7152 * g + 0.0722 * b); r *= k; g *= k; b *= k; }
    buf[4 * q] += r * m; buf[4 * q + 1] += g * m; buf[4 * q + 2] += b * m; buf[4 * q + 3] += m; cnt[q]++;
  }
  if (!any) return img;
  for (let q = 0; q < sw * sh; q++) for (let c = 0; c < 4; c++) buf[4 * q + c] /= cnt[q] || 1;
  const R = blurRadius(type, amount, Math.max(sw, sh));
  buf = type === 'gauss' ? gaussBlur(buf, sw, sh, R) : type === 'bokeh' ? discBlur(buf, sw, sh, R) : lineBlur(buf, sw, sh, R, cfg.angle || 0);
  for (let q = 0; q < sw * sh; q++) { const m = buf[4 * q + 3]; if (m > 1e-6) for (let c = 0; c < 3; c++) buf[4 * q + c] /= m; }
  // composite: sharp ↔ blurred by the mask, in linear light
  const lin = [0, 0, 0];
  for (let y = 0; y < h; y++) {
    const gy = Math.min(sh - 1.001, Math.max(0, (y + 0.5) / f - 0.5)), y0 = gy | 0, ty = gy - y0;
    for (let x = 0; x < w; x++) {
      const m = maskAt(x, y);
      if (m <= 0.002) continue;
      const p = (y * w + x) * 4, gx = Math.min(sw - 1.001, Math.max(0, (x + 0.5) / f - 0.5)), x0 = gx | 0, tx = gx - x0;
      const i00 = (y0 * sw + x0) * 4, i01 = i00 + 4, i10 = i00 + sw * 4, i11 = i10 + 4;
      const w00 = (1 - tx) * (1 - ty), w01 = tx * (1 - ty), w10 = (1 - tx) * ty, w11 = tx * ty;
      // weight by the blurred mask too, so samples outside the selection (no data there) do not darken the edge
      const a00 = w00 * (buf[i00 + 3] > 1e-6), a01 = w01 * (buf[i01 + 3] > 1e-6), a10 = w10 * (buf[i10 + 3] > 1e-6), a11 = w11 * (buf[i11 + 3] > 1e-6), at = a00 + a01 + a10 + a11;
      if (at <= 0) continue;
      for (let c = 0; c < 3; c++) lin[c] = (buf[i00 + c] * a00 + buf[i01 + c] * a01 + buf[i10 + c] * a10 + buf[i11 + c] * a11) / at;
      if (type === 'bokeh') { const Y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]; if (Y > 0.8) { const s = rollOff(Y) / Y; lin[0] *= s; lin[1] *= s; lin[2] *= s; } }
      for (let c = 0; c < 3; c++) {
        const s = S2L[img[p + c]], v = s + m * (Math.min(1, lin[c]) - s);
        img[p + c] = l2sFast(v) * 255;
      }
    }
  }
  return img;
}
