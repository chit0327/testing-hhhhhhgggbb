// Reference-guided global grading, baked into a 3D LUT.
//
// Port of the research method "B1 register" (research/lib/grade.py), the method human blind tests
// ranked best among those that need no large segmentation model: colour transfer as non-rigid
// point-set registration (weighted Coherent Point Drift) between k-means colour clusters of the target
// frames and the reference, in OKLab. The outlier term lets reference colours with no counterpart in
// the target stay unmatched, and the smooth displacement field keeps the mapping free of posterisation.
// Research findings applied here:
//   * fit once per video on keyframes spread over the clip (per-frame refits flicker),
//   * analyse only the central 70% of the reference (a vignette there is not part of the colour grade),
//   * the grade is a pure function of colour -> one LUT, identical in preview and export.
import { srgbToOklab, oklabToSrgb } from './color.js';
import { compilePrim, isNeutralPrim } from './primary.js';

const SCALE = [1, 2, 2]; // weight chroma a little more than lightness

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Sample n pixels (as OKLab*SCALE triples) from RGBA data, optionally only from a central box. */
export function sampleOklab(rgba, w, h, n, { centre = 1, seed = 1 } = {}) {
  const rnd = mulberry32(seed);
  const x0 = Math.floor(w * (1 - centre) / 2), y0 = Math.floor(h * (1 - centre) / 2);
  const cw = Math.max(1, w - 2 * x0), ch = Math.max(1, h - 2 * y0);
  const out = new Float64Array(n * 3), t = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const x = x0 + Math.floor(rnd() * cw), y = y0 + Math.floor(rnd() * ch), p = (y * w + x) * 4;
    srgbToOklab(rgba[p] / 255, rgba[p + 1] / 255, rgba[p + 2] / 255, t);
    out[3 * i] = t[0] * SCALE[0]; out[3 * i + 1] = t[1] * SCALE[1]; out[3 * i + 2] = t[2] * SCALE[2];
  }
  return out;
}

/** k-means with several k-means++ restarts; keeps the lowest-inertia run (stabler fits). */
export function kmeans(pts, k, iters = 15, seed = 0, restarts = 4) {
  let best = null;
  for (let r = 0; r < restarts; r++) {
    const run = kmeansOnce(pts, k, iters, seed * 101 + r);
    if (!best || run.inertia < best.inertia) best = run;
  }
  return { centres: best.centres, weights: best.weights };
}

function kmeansOnce(pts, k, iters, seed) {
  const n = pts.length / 3, rnd = mulberry32(seed + 7);
  const c = new Float64Array(k * 3), d = new Float64Array(n).fill(Infinity);
  let idx = Math.floor(rnd() * n);
  for (let j = 0; j < k; j++) { // k-means++
    c[3 * j] = pts[3 * idx]; c[3 * j + 1] = pts[3 * idx + 1]; c[3 * j + 2] = pts[3 * idx + 2];
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const dx = pts[3 * i] - c[3 * j], dy = pts[3 * i + 1] - c[3 * j + 1], dz = pts[3 * i + 2] - c[3 * j + 2];
      d[i] = Math.min(d[i], dx * dx + dy * dy + dz * dz); sum += d[i];
    }
    let r = rnd() * sum; idx = 0;
    while (idx < n - 1 && (r -= d[idx]) > 0) idx++;
  }
  const lab = new Int32Array(n), cnt = new Float64Array(k);
  for (let it = 0; it < iters; it++) {
    const acc = new Float64Array(k * 3); cnt.fill(0);
    for (let i = 0; i < n; i++) {
      let best = 0, bd = Infinity;
      for (let j = 0; j < k; j++) {
        const dx = pts[3 * i] - c[3 * j], dy = pts[3 * i + 1] - c[3 * j + 1], dz = pts[3 * i + 2] - c[3 * j + 2];
        const dd = dx * dx + dy * dy + dz * dz;
        if (dd < bd) { bd = dd; best = j; }
      }
      lab[i] = best; cnt[best]++;
      acc[3 * best] += pts[3 * i]; acc[3 * best + 1] += pts[3 * i + 1]; acc[3 * best + 2] += pts[3 * i + 2];
    }
    for (let j = 0; j < k; j++) if (cnt[j]) for (let q = 0; q < 3; q++) c[3 * j + q] = acc[3 * j + q] / cnt[j];
  }
  const w = new Float64Array(k);
  for (let j = 0; j < k; j++) w[j] = cnt[j] / n;
  let inertia = 0;
  for (let i = 0; i < n; i++) {
    const j = lab[i], dx = pts[3 * i] - c[3 * j], dy = pts[3 * i + 1] - c[3 * j + 1], dz = pts[3 * i + 2] - c[3 * j + 2];
    inertia += dx * dx + dy * dy + dz * dz;
  }
  return { centres: c, weights: w, inertia };
}

function solve(A, B, n, m) { // Gaussian elimination with partial pivoting; A n×n, B n×m (row-major, in place)
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(A[r * n + col]) > Math.abs(A[piv * n + col])) piv = r;
    if (piv !== col) {
      for (let q = 0; q < n; q++) { const t = A[col * n + q]; A[col * n + q] = A[piv * n + q]; A[piv * n + q] = t; }
      for (let q = 0; q < m; q++) { const t = B[col * m + q]; B[col * m + q] = B[piv * m + q]; B[piv * m + q] = t; }
    }
    const p = A[col * n + col] || 1e-12;
    for (let r = col + 1; r < n; r++) {
      const f = A[r * n + col] / p;
      if (!f) continue;
      for (let q = col; q < n; q++) A[r * n + q] -= f * A[col * n + q];
      for (let q = 0; q < m; q++) B[r * m + q] -= f * B[col * m + q];
    }
  }
  for (let r = n - 1; r >= 0; r--) {
    for (let q = 0; q < m; q++) {
      let s = B[r * m + q];
      for (let c = r + 1; c < n; c++) s -= A[r * n + c] * B[c * m + q];
      B[r * m + q] = s / (A[r * n + r] || 1e-12);
    }
  }
  return B;
}

/** Weighted non-rigid Coherent Point Drift: moves Y (M×3, weights u) towards X (N×3, weights v). */
export function cpd(Y, u, X, v, { beta = 0.25, lam = 3, w = 0.3, iters = 60 } = {}) {
  const M = u.length, N = v.length, D = 3;
  const G = new Float64Array(M * M);
  for (let i = 0; i < M; i++) for (let j = 0; j < M; j++) {
    const dx = Y[3 * i] - Y[3 * j], dy = Y[3 * i + 1] - Y[3 * j + 1], dz = Y[3 * i + 2] - Y[3 * j + 2];
    G[i * M + j] = Math.exp(-(dx * dx + dy * dy + dz * dz) / (2 * beta * beta));
  }
  let W = new Float64Array(M * 3);
  const T = Float64Array.from(Y);
  let s2 = 0;
  for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) {
    const dx = X[3 * j] - Y[3 * i], dy = X[3 * j + 1] - Y[3 * i + 1], dz = X[3 * j + 2] - Y[3 * i + 2];
    s2 += (dx * dx + dy * dy + dz * dz) * u[i] * v[j];
  }
  s2 /= D;
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const P of [X, Y]) for (let i = 0; i < P.length / 3; i++) for (let q = 0; q < 3; q++) {
    lo[q] = Math.min(lo[q], P[3 * i + q]); hi[q] = Math.max(hi[q], P[3 * i + q]);
  }
  const vol = (hi[0] - lo[0] + 1e-3) * (hi[1] - lo[1] + 1e-3) * (hi[2] - lo[2] + 1e-3);
  const P = new Float64Array(M * N);
  for (let it = 0; it < iters; it++) {
    const c = Math.pow(2 * Math.PI * s2, D / 2) * w / ((1 - w) * vol);
    for (let j = 0; j < N; j++) {
      let col = 0;
      for (let i = 0; i < M; i++) {
        const dx = X[3 * j] - T[3 * i], dy = X[3 * j + 1] - T[3 * i + 1], dz = X[3 * j + 2] - T[3 * i + 2];
        const k = u[i] * Math.exp(-(dx * dx + dy * dy + dz * dz) / (2 * s2));
        P[i * N + j] = k; col += k;
      }
      const s = v[j] / (col + c);
      for (let i = 0; i < M; i++) P[i * N + j] *= s;
    }
    const P1 = new Float64Array(M), Pt1 = new Float64Array(N), PX = new Float64Array(M * 3);
    let Np = 0;
    for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) {
      const p = P[i * N + j];
      P1[i] += p; Pt1[j] += p; Np += p;
      PX[3 * i] += p * X[3 * j]; PX[3 * i + 1] += p * X[3 * j + 1]; PX[3 * i + 2] += p * X[3 * j + 2];
    }
    if (Np < 1e-9) { s2 *= 2; continue; }
    const A = new Float64Array(M * M), B = new Float64Array(M * 3);
    for (let i = 0; i < M; i++) {
      const pi = Math.max(P1[i], 1e-9);
      for (let j = 0; j < M; j++) A[i * M + j] = G[i * M + j];
      A[i * M + i] += lam * s2 / pi;
      for (let q = 0; q < 3; q++) B[3 * i + q] = PX[3 * i + q] / pi - Y[3 * i + q];
    }
    W = solve(A, B, M, 3);
    for (let i = 0; i < M; i++) for (let q = 0; q < 3; q++) {
      let s = 0;
      for (let j = 0; j < M; j++) s += G[i * M + j] * W[3 * j + q];
      T[3 * i + q] = Y[3 * i + q] + s;
    }
    let num = 0;
    for (let j = 0; j < N; j++) num += Pt1[j] * (X[3 * j] ** 2 + X[3 * j + 1] ** 2 + X[3 * j + 2] ** 2);
    for (let i = 0; i < M; i++) {
      num -= 2 * (PX[3 * i] * T[3 * i] + PX[3 * i + 1] * T[3 * i + 1] + PX[3 * i + 2] * T[3 * i + 2]);
      num += P1[i] * (T[3 * i] ** 2 + T[3 * i + 1] ** 2 + T[3 * i + 2] ** 2);
    }
    s2 = Math.max(num / (Np * D), 1e-5);
  }
  return W;
}

const QN = 33; // lightness quantiles kept for tone matching

export const TONE_WHITE = 0.93; // reference lightness at/above which pixels are blown (sky, windows): no tone information
export const MAX_LIFT = 0.2;      // safety cap: most the tone match may raise/lower any lightness (OKLab L); beyond is the user's exposure

function stats(samples, { white = 1.01 } = {}) { // samples: OKLab*SCALE triples -> unscaled lightness quantiles and a/b mean, std
  const n = samples.length / 3, all = new Float64Array(n);
  let ma = 0, mb = 0;
  for (let i = 0; i < n; i++) { all[i] = samples[3 * i] / SCALE[0]; ma += samples[3 * i + 1] / SCALE[1]; mb += samples[3 * i + 2] / SCALE[2]; }
  ma /= n; mb /= n;
  let va = 0, vb = 0;
  for (let i = 0; i < n; i++) { va += (samples[3 * i + 1] / SCALE[1] - ma) ** 2; vb += (samples[3 * i + 2] / SCALE[2] - mb) ** 2; }
  // tone quantiles leave out blown highlights unless they are nearly the whole image: a reference that is a
  // third white sky otherwise made the tone match turn a third of a street white
  let L = all.filter((v) => v < white);
  if (L.length < n * 0.2) L = all;
  L.sort();
  const q = [];
  const m = L.length;
  for (let k = 0; k < QN; k++) q.push(L[Math.min(m - 1, Math.round((0.01 + 0.98 * k / (QN - 1)) * (m - 1)))]);
  // look descriptors used by the parameter suggestion and the AI prompt (same definitions as the skill's metrics)
  let lm = 0; for (let i = 0; i < m; i++) lm += L[i]; lm /= m;
  let lv = 0; for (let i = 0; i < m; i++) lv += (L[i] - lm) ** 2;
  const sorted = Float64Array.from(all).sort(), hiT = sorted[Math.floor(0.8 * (n - 1))], loT = sorted[Math.floor(0.2 * (n - 1))];
  let cm = 0; const hi = [0, 0, 0], lo = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const A = samples[3 * i + 1] / SCALE[1], B = samples[3 * i + 2] / SCALE[2];
    cm += Math.hypot(A, B);
    if (all[i] > hiT) { hi[0] += A; hi[1] += B; hi[2]++; }
    if (all[i] < loT) { lo[0] += A; lo[1] += B; lo[2]++; }
  }
  return {
    Lq: q, ma, mb, sa: Math.sqrt(va / n) + 1e-4, sb: Math.sqrt(vb / n) + 1e-4,
    Lmean: lm, Lsd: Math.sqrt(lv / m), chroma: cm / n,
    hiA: hi[0] / Math.max(1, hi[2]), hiB: hi[1] / Math.max(1, hi[2]), loA: lo[0] / Math.max(1, lo[2]), loB: lo[1] / Math.max(1, lo[2]),
  };
}

/** Look statistics of an image (same definitions as the fit uses): for metrics shown to Claude / tests. */
export function measure(rgba, w, h, { centre = 1, white = 1.01, samples = 12000 } = {}) {
  return stats(sampleOklab(rgba, w, h, samples, { centre, seed: 5 }), { white });
}

/**
 * Fit the grading model. targetFrames / reference: {data: RGBA Uint8ClampedArray, width, height}.
 * Returns a plain object (transferable to/from the worker): the colour displacement field (natural,
 * content-tolerant transfer) plus target/reference statistics (direct statistical match).
 */
export function fitGrade(targetFrames, reference, { k = 48, beta = 0.25, lam = 3, w = 0.3, samples = 20000 } = {}) {
  const per = Math.ceil(samples / targetFrames.length);
  const parts = targetFrames.map((f, i) => sampleOklab(f.data, f.width, f.height, per, { seed: 11 + i }));
  const src = new Float64Array(parts.reduce((s, p) => s + p.length, 0));
  let o = 0;
  for (const p of parts) { src.set(p, o); o += p.length; }
  const ref = sampleOklab(reference.data, reference.width, reference.height, samples, { centre: 0.7, seed: 3 });
  const a = kmeans(src, k, 15, 0), b = kmeans(ref, k, 15, 1);
  const W = cpd(a.centres, a.weights, b.centres, b.weights, { beta, lam, w });
  const S = stats(src), R = stats(ref, { white: TONE_WHITE });
  const order = Array.from(b.weights.keys()).sort((i, j) => b.weights[j] - b.weights[i]).slice(0, 6);
  const palette = order.map((i) => ({ L: b.centres[3 * i] / SCALE[0], a: b.centres[3 * i + 1] / SCALE[1], b: b.centres[3 * i + 2] / SCALE[2], w: b.weights[i] }));
  return { centres: Array.from(a.centres), W: Array.from(W), beta, src: S, ref: R, tone: toneCurve(S.Lq, R.Lq), palette };
}

/**
 * Lightness mapping target -> reference from their quantiles, made safe: slopes limited to [0.5, 2]
 * and smoothed (raw quantile matching against a bimodal reference produced steep steps = banding).
 */
function toneCurve(qs, qr, n = 65) {
  const xs = [], ys = [];
  for (let i = 0; i < n; i++) { const x = i / (n - 1); xs.push(x); ys.push(interp(x, qs, qr)); }
  let d = ys.slice(1).map((y, i) => (y - ys[i]) * (n - 1));
  for (let pass = 0; pass < 3; pass++) d = d.map((v, i) => (d[Math.max(0, i - 1)] + 2 * v + d[Math.min(d.length - 1, i + 1)]) / 4);
  d = d.map((v) => Math.min(2, Math.max(0.5, v)));
  const out = [0];
  for (let i = 0; i < d.length; i++) out.push(out[i] + d[i] / (n - 1));
  const shift = ys.reduce((a, y, i) => a + y - out[i], 0) / n; // keep the overall level of the raw match
  // limit the change at every level: different content (sky vs street) must not relight the whole scene
  return { xs, ys: out.map((y, i) => Math.min(1, Math.max(0, xs[i] + Math.max(-MAX_LIFT, Math.min(MAX_LIFT, y + shift - xs[i]))))) };
}

/** Default look settings (all neutral except a balanced match). Kept in the confirmed state. */
export const DEFAULT_LOOK = {
  strength: 1,       // 參考色彩強度: 0 = original colours, 1 = full reference look
  match: 0.5,        // 接近參考程度: 0 = natural transfer only, 1 = reference tone curve + saturation level
  protect: true,     // 保護材質顏色: limit hue rotation of clearly coloured surfaces
  exposure: 0,       // stops, -2..2
  contrast: 0,       // -1..1
  highlights: 0,     // -1..1
  shadows: 0,        // -1..1
  saturation: 0,     // -1..1  (chroma × (1 + s))
  temperature: 0,    // -1..1  (cool .. warm)
  tint: 0,           // -1..1  (green .. magenta)
  black: 0,          // output lightness floor 0..0.4 (lifted / faded blacks)
  white: 1,          // output lightness ceiling 0.7..1
  shadowHue: 60, shadowAmt: 0,  // colour pushed into shadows: OKLab hue (degrees) and amount 0..1
  midHue: 60, midAmt: 0,        // ... midtones
  hiHue: 100, hiAmt: 0,         // ... highlights
  source: 'reference',          // 'reference' (natural transfer) | 'ai' (LUT distilled from an AI-graded frame)
  amount: 1,                    // 整體調色強度: blend of the whole look with the original colour (0 = original)
  glow: 0, glowRadius: 0.5, haze: 0, hazeRadius: 0.5, // 柔光, 朦朧 (spatial: applied by the renderer, not the LUT)
  vignette: 0, vignetteSize: 0.5, // 暗角: + darkens / − lightens the edges (spatial)
  prim: null,                   // pro stage (primary.js): wheels, curves, hue curves, skin protection; null = off
};
const primCache = new WeakMap(); // look.prim object -> compiled function (single-colour calls outside a bake)
export const TONE_COLOUR = 0.04; // OKLab chroma offset at amount 1

const interp = (x, xs, ys) => {
  if (x <= xs[0]) return ys[0] + (x - xs[0]);
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1] + (x - xs[xs.length - 1]);
  let i = 1;
  while (xs[i] < x) i++;
  const t = (x - xs[i - 1]) / Math.max(xs[i] - xs[i - 1], 1e-6);
  return ys[i - 1] + t * (ys[i] - ys[i - 1]);
};
const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

/** Basic adjustments on OKLab (in place on the 3-array). */
function adjust(lab, k) {
  let L = lab[0];
  if (k.exposure) { // via linear luminance, with a soft shoulder so highlights do not clip
    let Y = Math.max(L, 0) ** 3 * Math.pow(2, k.exposure);
    if (Y > 0.8) Y = 0.8 + 0.2 * (Y - 0.8) / (Y - 0.8 + 0.2);
    L = Math.cbrt(Y);
  }
  if (k.contrast) L += k.contrast * 0.15 * Math.sin(2 * Math.PI * (L - 0.5)); // monotone S-curve for |contrast| ≤ 1
  if (k.highlights) L += k.highlights * 0.12 * smooth(0.45, 1, L) * (1.15 - L);
  if (k.shadows) L += k.shadows * 0.12 * (1 - smooth(0, 0.55, L)) * (L + 0.1);
  lab[0] = Math.min(1, Math.max(0, L));
  let a = lab[1], b = lab[2];
  if (k.saturation) { a *= 1 + k.saturation; b *= 1 + k.saturation; }
  const wl = 0.4 + 0.6 * smooth(0.05, 0.6, lab[0]); // casts are weaker in deep shadows
  if (k.temperature) { b += k.temperature * 0.045 * wl; a += k.temperature * 0.012 * wl; }
  if (k.tint) a += k.tint * 0.04 * wl;
  if (k.shadowAmt || k.midAmt || k.hiAmt) { // three-way colour: partition of unity over lightness
    const ws = 1 - smooth(0.2, 0.5, lab[0]), wh = smooth(0.5, 0.8, lab[0]), wm = Math.max(0, 1 - ws - wh);
    for (const [w, hue, amt] of [[ws, k.shadowHue, k.shadowAmt], [wm, k.midHue, k.midAmt], [wh, k.hiHue, k.hiAmt]]) {
      if (!amt) continue;
      const t = (hue * Math.PI) / 180;
      a += w * amt * TONE_COLOUR * Math.cos(t); b += w * amt * TONE_COLOUR * Math.sin(t);
    }
  }
  if (k.black || k.white !== 1) lab[0] = k.black + lab[0] * (k.white - k.black); // levels: fade / no paper white
  lab[1] = a; lab[2] = b;
  return lab;
}

/**
 * Apply the model + look settings to one sRGB colour (0..1).
 *   natural transfer (displacement field) ── `match` pulls tone curve + saturation level to the
 *   reference ── blend by `strength` with the original ── hue protection ── preserve lightness
 *   (Lighting Intent "preserve") ── basic adjustments.
 */
export function gradeColour(model, r, g, b, look = {}, out = [0, 0, 0], _lab = [0, 0, 0], primFn) {
  const k = { ...DEFAULT_LOOK, ...look };
  if (primFn === undefined && k.prim) { if (!primCache.has(k.prim)) primCache.set(k.prim, compilePrim(k.prim)); primFn = primCache.get(k.prim); }
  const lab = srgbToOklab(r, g, b, _lab);
  const L0 = lab[0], a0 = lab[1], b0 = lab[2];
  let L = L0, a = a0, bb = b0;
  if (model && k.source === 'ai') {
    if (!model.pair) throw new Error('AI 調色圖的 LUT 尚未建立');
    const c = lutLookup(model.pair, r, g, b, _pairRGB), t = srgbToOklab(c[0], c[1], c[2], _pairLab);
    L = L0 + k.strength * (t[0] - L0); a = a0 + k.strength * (t[1] - a0); bb = b0 + k.strength * (t[2] - b0);
  } else if (model) {
    const p0 = L0 * SCALE[0], p1 = a0 * SCALE[1], p2 = b0 * SCALE[2];
    const C = model.centres, W = model.W, M = C.length / 3, s = 2 * model.beta * model.beta;
    let d0 = 0, d1 = 0, d2 = 0;
    for (let i = 0; i < M; i++) {
      const dx = p0 - C[3 * i], dy = p1 - C[3 * i + 1], dz = p2 - C[3 * i + 2];
      const kk = Math.exp(-(dx * dx + dy * dy + dz * dz) / s);
      d0 += kk * W[3 * i]; d1 += kk * W[3 * i + 1]; d2 += kk * W[3 * i + 2];
    }
    let Ln = L0 + d0 / SCALE[0], an = a0 + d1 / SCALE[1], bn = b0 + d2 / SCALE[2];
    if (k.match && model.src) {
      // "match" pulls TONE (smoothed, slope-limited lightness curve) and SATURATION LEVEL (chroma spread
      // ratio) towards the reference; colour direction stays with the natural transfer above, so raising it
      // never swings hues (earlier per-channel/cast matching tinted shadows purple with unlike content)
      const S = model.src, R = model.ref;
      const Ls = model.tone ? interp(L0, model.tone.xs, model.tone.ys) : interp(L0, S.Lq, R.Lq);
      const kc = Math.min(2, Math.max(0.5, Math.hypot(R.sa, R.sb) / Math.hypot(S.sa, S.sb)));
      Ln += k.match * (Ls - Ln);
      const ck = 1 + k.match * (kc - 1);
      an *= ck; bn *= ck;
    }
    L = L0 + k.strength * (Ln - L0); a = a0 + k.strength * (an - a0); bb = b0 + k.strength * (bn - b0);
    const c0 = Math.hypot(a0, b0);
    if (k.protect && c0 > 0.01) { // material protection: limited hue rotation for clearly coloured surfaces
      const h0 = Math.atan2(b0, a0), c1 = Math.hypot(a, bb);
      let dh = Math.atan2(bb, a) - h0;
      dh = Math.atan2(Math.sin(dh), Math.cos(dh));
      const lim = (40 * Math.PI) / 180, wgt = Math.min(1, (c0 - 0.01) / 0.03);
      const h = h0 + dh + (Math.max(-lim, Math.min(lim, dh)) - dh) * wgt;
      a = c1 * Math.cos(h); bb = c1 * Math.sin(h);
    }
  }
  if (k.preserveLuminance) L = L0;
  lab[0] = L; lab[1] = a; lab[2] = bb;
  adjust(lab, k);
  if (primFn) { // pro stage on the adjusted colour (display-referred sRGB, like a colour page's primaries)
    const t = oklabToSrgb(lab[0], lab[1], lab[2], _primRGB);
    primFn(t[0], t[1], t[2], _primRGB);
    srgbToOklab(_primRGB[0], _primRGB[1], _primRGB[2], lab);
  }
  if (k.amount !== 1) { // overall strength: mix the finished look with the original, in OKLab
    const t = Math.min(1, Math.max(0, k.amount));
    lab[0] = L0 + t * (lab[0] - L0); lab[1] = a0 + t * (lab[1] - a0); lab[2] = b0 + t * (lab[2] - b0);
  }
  return oklabToSrgb(lab[0], lab[1], lab[2], out);
}

const _pairRGB = [0, 0, 0], _pairLab = [0, 0, 0], _primRGB = [0, 0, 0];
/** Trilinear lookup in {size, data} (index ((r*N+g)*N+b)*3). */
export function lutLookup(lut, r, g, b, out = [0, 0, 0]) {
  const N = lut.size, D = lut.data, n1 = N - 1;
  const fr = Math.min(1, Math.max(0, r)) * n1, fg = Math.min(1, Math.max(0, g)) * n1, fb = Math.min(1, Math.max(0, b)) * n1;
  const r0 = Math.min(n1 - 1, fr | 0), g0 = Math.min(n1 - 1, fg | 0), b0 = Math.min(n1 - 1, fb | 0);
  const dr = fr - r0, dg = fg - g0, db = fb - b0, sR = N * N * 3, sG = N * 3;
  for (let c = 0; c < 3; c++) {
    const i = ((r0 * N + g0) * N + b0) * 3 + c;
    const c00 = D[i] + (D[i + 3] - D[i]) * db, c01 = D[i + sG] + (D[i + sG + 3] - D[i + sG]) * db;
    const c10 = D[i + sR] + (D[i + sR + 3] - D[i + sR]) * db, c11 = D[i + sR + sG] + (D[i + sR + sG + 3] - D[i + sR + sG]) * db;
    const c0 = c00 + (c01 - c00) * dg, c1 = c10 + (c11 - c10) * dg;
    out[c] = c0 + (c1 - c0) * dr;
  }
  return out;
}

/**
 * Shaper (1D pre-LUT), as used for log-to-display LUTs: where the grade changes fast along the grey axis
 * (a LOG restoration stretching a narrow input range), the 3D grid nodes are placed densely; elsewhere
 * sparsely. Returns, for every 8-bit input level, its position in the grid (0..1), or null when the grade is
 * even enough that a uniform grid is as good. Density = OKLab change per input level, floored at a quarter of
 * the mean so that no range is starved of nodes.
 */
export function computeShaper(model, look = {}, { floor = 0.25, smooth = 0 } = {}) {
  const k = { ...DEFAULT_LOOK, ...look }, primFn = isNeutralPrim(k.prim) ? null : compilePrim(k.prim), o = [0, 0, 0], t = [0, 0, 0], prev = [0, 0, 0], lab = [0, 0, 0];
  const d = new Float64Array(255);
  for (let v = 0; v < 256; v++) {
    gradeColour(model, v / 255, v / 255, v / 255, look, o, t, primFn);
    srgbToOklab(Math.min(1, Math.max(0, o[0])), Math.min(1, Math.max(0, o[1])), Math.min(1, Math.max(0, o[2])), lab);
    if (v) d[v - 1] = Math.hypot(lab[0] - prev[0], lab[1] - prev[1], lab[2] - prev[2]);
    prev[0] = lab[0]; prev[1] = lab[1]; prev[2] = lab[2];
  }
  let mean = 0; for (const x of d) mean += x; mean /= 255;
  if (!(mean > 0)) return null;
  if (smooth) { // box-smooth the density: node spacing then changes gradually (abrupt changes cost accuracy off the grey axis)
    const c = Float64Array.from(d);
    for (let i = 0; i < 255; i++) { let s = 0, n = 0; for (let j = Math.max(0, i - smooth); j <= Math.min(254, i + smooth); j++) { s += c[j]; n++; } d[i] = s / n; }
  }
  let max = 0; for (let i = 0; i < 255; i++) { d[i] = Math.max(d[i], mean * floor); if (d[i] > max) max = d[i]; }
  let floored = 0; for (const x of d) floored += x;
  if (max < 2.5 * (floored / 255)) return null; // even enough: keep the uniform grid
  const pos = new Float32Array(256);
  for (let v = 1; v < 256; v++) pos[v] = pos[v - 1] + d[v - 1] / floored;
  pos[255] = 1;
  return pos;
}

/**
 * Bake into an N³ LUT (Float32Array, index ((r*N+g)*N+b)*3), sRGB in -> sRGB out. With a shaper the grid
 * nodes sit at the shaper's input levels and the LUT carries `index` (per 8-bit input level, its fractional
 * grid coordinate 0..N−1), which the renderer uses instead of v / 255 × (N − 1).
 */
export function bakeLUT(model, look = {}, N = 33, { shaper = null, memo = null } = {}) {
  const lut = new Float32Array(N * N * N * 3), o = [0, 0, 0], t = [0, 0, 0];
  const k = { ...DEFAULT_LOOK, ...look };
  const primFn = isNeutralPrim(k.prim) ? null : compilePrim(k.prim, { memo }); // memo: see compilePrim
  const neutral = !model && !primFn && !k.preserveLuminance && ['exposure', 'contrast', 'highlights', 'shadows', 'saturation', 'temperature', 'tint', 'black', 'shadowAmt', 'midAmt', 'hiAmt'].every((key) => !k[key]) && k.white === 1;
  // input value of grid node i: inverse of the shaper (linear between its 256 levels)
  const node = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const u = i / (N - 1);
    if (!shaper) { node[i] = u; continue; }
    let v = 0; while (v < 254 && shaper[v + 1] < u) v++;
    const a = shaper[v], b = shaper[v + 1];
    node[i] = (v + (b > a ? Math.min(1, Math.max(0, (u - a) / (b - a))) : 0)) / 255;
  }
  for (let r = 0; r < N; r++) for (let g = 0; g < N; g++) for (let b = 0; b < N; b++) {
    if (neutral) { o[0] = node[r]; o[1] = node[g]; o[2] = node[b]; }
    else gradeColour(model, node[r], node[g], node[b], look, o, t, primFn);
    const i = ((r * N + g) * N + b) * 3;
    lut[i] = o[0]; lut[i + 1] = o[1]; lut[i + 2] = o[2];
  }
  const out = { size: N, data: lut };
  if (shaper) { out.index = new Float32Array(256); for (let v = 0; v < 256; v++) out.index[v] = shaper[v] * (N - 1); }
  return out;
}

export function identityLUT(N = 17) { return bakeLUT(null, {}, N); }

/**
 * The LUT the app renders with (preview, processing, evaluation tools): 41³, plus a shaper when the look
 * restores LOG / flat footage. Measured against exact per-pixel maths (research/eval_runs/2026-10-26,
 * test/unit/lut.test.js): on LOG footage the share of pixels off by more than 4 levels went 26.6 % (33³) →
 * 1.3 %; on a bold hue-curve grade 5.4 % → 0.3 %. Without a restoration the shaper did not help (it moves
 * nodes along the grey axis while the error sits in hue), so it is used only with one. Known counter-case: a
 * synthetic flat ramp with tiny colour offsets and a 3× chroma boost (2.9 % instead of 0.4 % off by > 4).
 * Smoothing the density or a higher floor removed that, but also removed the gain on real LOG footage.
 * shaperLook: the look whose grey-axis response places the nodes (region layers pass their base look, so
 * every layer shares the main LUT's grid).
 */
export const LUT_SIZE = 41;
export function bakeLook(model, look = {}, { shaperLook = look, size = LUT_SIZE, memo = null } = {}) {
  const p = shaperLook && shaperLook.prim, restores = !!(p && p.norm && p.norm.amt > 0.02);
  return bakeLUT(model, look, size, { shaper: restores ? computeShaper(model, shaperLook) : null, memo });
}

/** .cube text (Resolve / Adobe format: red changes fastest, values 0..1) of a uniform-grid LUT. */
export function toCube(lut, title = 'zhaoyang') {
  if (lut.index) throw new Error('a shaper LUT cannot be written as a plain .cube; bake it on a uniform grid');
  const N = lut.size, lines = [`TITLE "${title}"`, `LUT_3D_SIZE ${N}`];
  for (let b = 0; b < N; b++) for (let g = 0; g < N; g++) for (let r = 0; r < N; r++) {
    const i = ((r * N + g) * N + b) * 3;
    lines.push(`${lut.data[i].toFixed(6)} ${lut.data[i + 1].toFixed(6)} ${lut.data[i + 2].toFixed(6)}`);
  }
  return lines.join('\n') + '\n';
}
