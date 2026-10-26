// Deterministic estimation of the pro-stage parameters from a reference: the "colourist's first pass".
//   tone   — luma curve from lightness quantiles (blown reference highlights ignored, change capped:
//            a reference's sky must not relight a street),
//   colour — shadow / midtone / highlight casts → Lift / Gamma / Gain wheels, solved iteratively by
//            rendering sample pixels through the real pipeline and correcting the residual (Newton steps
//            with a measured 2×2 response per wheel),
//   hues   — per hue anchor: rotate towards the reference's nearby hue, match its saturation, and mute
//            colours the reference does not have (a warm wall under a teal reference turns grey-green),
//   skin   — protection on, stronger for the natural interpretation.
// Before all that, LOG / flat-profile footage is restored to a normal image (estimateNorm, the "CST node"),
// and after it the whole parameter set is refined so that the graded source's colour DISTRIBUTION matches
// the reference (refineToReference: sliced Wasserstein in OKLab, regularised towards the first pass).
// The three interpretations are that refined grade at different strengths. Same input → same output, always.
import { oklabToSrgb, srgbToOklab } from './color.js';
import { neutralPrim, compilePrim, sanitizePrim, curveFn, normTone, HUE_ANCHORS, CURVE_X } from './primary.js';
import { structureScore } from './metrics.js';

export const MODES = {
  natural: { label: '自然', tone: 0.6, cap: 0.1, color: 0.6, cast: 0.6, hue: 0.6, maxShift: 25, mute: -0.3, skin: 1, contrast: 0 },
  faithful: { label: '忠實', tone: 1, cap: 0.16, color: 1, cast: 1, hue: 1, maxShift: 35, mute: -0.5, skin: 0.6, contrast: 0 },
  bold: { label: '濃烈', tone: 1.2, cap: 0.2, color: 1.25, cast: 1.2, hue: 1.15, maxShift: 45, mute: -0.6, skin: 0.35, contrast: 0.04 },
};

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const angDiff = (a, b) => ((a - b + 540) % 360) - 180;

function quantiles(Ls, n = 33) {
  const s = Float64Array.from(Ls).sort(), out = [];
  for (let k = 0; k < n; k++) out.push(s[Math.min(s.length - 1, Math.round((0.01 + 0.98 * k / (n - 1)) * (s.length - 1)))]);
  return out;
}
const interp = (x, xs, ys) => {
  if (x <= xs[0]) return ys[0] + (x - xs[0]);
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1] + (x - xs[xs.length - 1]);
  let i = 1; while (xs[i] < x) i++;
  const t = (x - xs[i - 1]) / Math.max(xs[i] - xs[i - 1], 1e-6);
  return ys[i - 1] + t * (ys[i] - ys[i - 1]);
};

/** Chroma-weighted hue statistics within ±win° of hue h: {mass share, circular mean hue, mean chroma}. */
function hueWindow(lab, h, win) {
  let m = 0, sx = 0, sy = 0, sc = 0, total = 0;
  for (let i = 0; i < lab.length; i += 3) {
    const C = Math.hypot(lab[i + 1], lab[i + 2]);
    if (C < 0.02) continue;
    total += C;
    const hh = (Math.atan2(lab[i + 2], lab[i + 1]) * 180) / Math.PI;
    if (Math.abs(angDiff(hh, h)) > win) continue;
    m += C; sx += C * Math.cos((hh * Math.PI) / 180); sy += C * Math.sin((hh * Math.PI) / 180); sc += C * C;
  }
  return { share: total ? m / total : 0, hue: (Math.atan2(sy, sx) * 180) / Math.PI, chroma: m ? sc / m : 0 };
}

/**
 * LOG / flat-profile detection and restoration from the source alone (no reference involved): footage whose
 * lightness spans less than a normal image (q1..q99 under 0.6 OKLab L) is stretched around its median so that
 * q1 / q99 land near 0.22 / 0.88, and its chroma is raised towards a normal level (mean 0.03), both scaled by
 * how flat it is. Normal footage gets amt 0 (untouched). The creative fit then starts from a Rec.709-like
 * image instead of fighting the flat profile with capped curves.
 */
export function estimateNorm(srcLab) {
  const Ls = []; let C = 0;
  for (let i = 0; i < srcLab.length; i += 3) { Ls.push(srcLab[i]); C += Math.hypot(srcLab[i + 1], srcLab[i + 2]); }
  const n = Ls.length; C /= Math.max(1, n);
  const s = Float64Array.from(Ls).sort(), q = (t) => s[Math.min(n - 1, Math.round(t * (n - 1)))];
  const q1 = q(0.01), m = q(0.5), q99 = q(0.99), range = q99 - q1;
  const amt = clamp((0.6 - range) / 0.2, 0, 1);
  const Tm = clamp(m + (0.6 - m) * 0.7, 0.3, 0.7), HI = 0.995, LO = 0.05;
  const slope = (target, x, bound) => { // slope that sends source x to target through the tanh roll-off
    const span = Math.abs(bound - Tm), want = Math.abs(target - Tm) / span;
    return (Math.atanh(Math.min(want, 0.97)) * span) / Math.max(Math.abs(x - m), 1e-3);
  };
  const sl = clamp((slope(0.88, q99, HI) + slope(0.22, q1, LO)) / 2, 1, 3);
  // parameters are estimated even for normal footage (amt 0) so that the user can still turn the restoration up
  return { amt: amt > 0.02 ? amt : 0, m, Tm, s: sl, sat: clamp(0.03 / Math.max(C, 1e-3), 1, 4), range, chroma: C };
}
function applyNormLab(lab, nm) { // the restoration in OKLab, to analyse the source as the grade will see it
  const out = new Float64Array(lab.length), k = 1 + nm.amt * (nm.sat - 1);
  for (let i = 0; i < lab.length; i += 3) { out[i] = normTone(nm, lab[i]); out[i + 1] = lab[i + 1] * k; out[i + 2] = lab[i + 2] * k; }
  return out;
}

/**
 * Keep every luma-curve segment's slope in [0.45, 2] (no flattened, detail-losing or step-like, banding tone
 * ranges) and the curve inside 0..1; returns the repaired monotone knots.
 */
function safeLuma(ys) {
  const y = ys.slice();
  for (let i = 1; i < 5; i++) y[i] = clamp(y[i], y[i - 1] + 0.45 * 0.25, y[i - 1] + 2 * 0.25);
  if (y[4] > 1) { const k = (1 - y[0]) / (y[4] - y[0]); for (let i = 1; i < 5; i++) y[i] = y[0] + (y[i] - y[0]) * k; }
  const f = curveFn(y);
  return CURVE_X.map((x) => f(x));
}

/** srcLab, refLab: Float64Array of OKLab triples (reference already limited to its central area). */
export function fitPrim(srcLab0, refLab, modeName = 'natural', { iters = 6, sample = 2500, norm = estimateNorm(srcLab0) } = {}) {
  const md = MODES[modeName];
  const p = neutralPrim();
  // ---- restoration (LOG / flat footage) first; everything below analyses the restored source
  p.norm = { amt: norm.amt, m: norm.m, Tm: norm.Tm, s: norm.s, sat: norm.sat };
  const srcLab = norm.amt ? applyNormLab(srcLab0, norm) : srcLab0;
  // ---- tone
  const srcL = [], refL = [];
  for (let i = 0; i < srcLab.length; i += 3) srcL.push(srcLab[i]);
  for (let i = 0; i < refLab.length; i += 3) if (refLab[i] < 0.93) refL.push(refLab[i]);
  if (refL.length < refLab.length / 3 * 0.2) { refL.length = 0; for (let i = 0; i < refLab.length; i += 3) refL.push(refLab[i]); }
  const qs = quantiles(srcL), qr = quantiles(refL);
  p.luma = CURVE_X.map((x) => {
    const d = clamp(interp(x, qs, qr) - x, -md.cap, md.cap) * md.tone;
    const s = x === 0.25 ? -md.contrast : x === 0.75 ? md.contrast : 0;
    return clamp(x + d + s, 0, 1); // never push black below 0 or white above 1 (crushed / clipped detail)
  });
  p.luma = safeLuma(p.luma);
  // ---- saturation level
  const meanC = (lab) => { let s = 0, n = 0; for (let i = 0; i < lab.length; i += 3) { s += Math.hypot(lab[i + 1], lab[i + 2]); n++; } return s / Math.max(1, n); };
  const k = clamp(meanC(refLab) / Math.max(meanC(srcLab), 1e-3), 0.5, 2);
  p.sat = clamp(Math.log2(k) * md.color, -0.8, 0.8);
  // ---- hue curves
  for (let i = 0; i < 8; i++) {
    const s = hueWindow(srcLab, HUE_ANCHORS[i], 30), r = hueWindow(refLab, HUE_ANCHORS[i], 50);
    if (s.share < 0.02) continue;
    if (r.share < 0.25 * s.share) { p.hueSat[i] = md.mute; continue; } // colour the reference does not have
    p.hueHue[i] = clamp(angDiff(r.hue, s.hue) * md.hue, -md.maxShift, md.maxShift);
    p.hueSat[i] = clamp(Math.log2(Math.max(r.chroma, 1e-3) / Math.max(s.chroma, 1e-3)) * md.color - p.sat, -0.7, 0.7);
  }
  p.skin = md.skin;
  // ---- tonal casts → wheels (iterative, through the real pipeline)
  const n = Math.min(sample, srcLab.length / 3), stride = Math.max(1, Math.floor(srcLab.length / 3 / n));
  const rgb = [], band = [], Ls = [];
  for (let i = 0, j = 0; j < n; i += 3 * stride, j++) { rgb.push(oklabToSrgb(srcLab0[i], srcLab0[i + 1], srcLab0[i + 2])); Ls.push(srcLab[i]); }
  const sl = Float64Array.from(Ls).sort(), t1 = sl[Math.floor(n / 3)], t2 = sl[Math.floor(2 * n / 3)];
  for (const L of Ls) band.push(L < t1 ? 0 : L < t2 ? 1 : 2);
  const bandAB = (lab) => { const a = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]; for (let i = 0; i < lab.length / 3; i++) { const t = a[band[i]]; t[0] += lab[3 * i + 1]; t[1] += lab[3 * i + 2]; t[2]++; } return a.map((t) => [t[0] / (t[2] || 1), t[1] / (t[2] || 1)]); };
  const refBands = (() => { // reference bands by its own lightness terciles
    const rl = []; for (let i = 0; i < refLab.length; i += 3) rl.push(refLab[i]);
    const s2 = Float64Array.from(rl).sort(), r1 = s2[Math.floor(rl.length / 3)], r2 = s2[Math.floor(2 * rl.length / 3)];
    const a = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < refLab.length; i += 3) { const t = a[refLab[i] < r1 ? 0 : refLab[i] < r2 ? 1 : 2]; t[0] += refLab[i + 1]; t[1] += refLab[i + 2]; t[2]++; }
    return a.map((t) => [t[0] / (t[2] || 1), t[1] / (t[2] || 1)]);
  })();
  const out = [0, 0, 0], lab = [0, 0, 0];
  const render = (prim) => { // achieved OKLab of the sample pixels
    const f = compilePrim(prim), res = new Float64Array(n * 3);
    for (let j = 0; j < n; j++) {
      const c = f ? f(rgb[j][0], rgb[j][1], rgb[j][2], out) : rgb[j];
      toLab(c, lab); res[3 * j] = lab[0]; res[3 * j + 1] = lab[1]; res[3 * j + 2] = lab[2];
    }
    return res;
  };
  const base = bandAB(render(p));
  // target: the (already saturation-scaled) source cast moved towards the reference's band cast
  const target = base.map((b, i) => [b[0] + (refBands[i][0] - b[0]) * md.cast, b[1] + (refBands[i][1] - b[1]) * md.cast]);
  const wheels = ['lift', 'gamma', 'gain'], vec = [[0, 0], [0, 0], [0, 0]];
  const setWheels = () => wheels.forEach((w, i) => { const a = Math.hypot(vec[i][0], vec[i][1]); p[w] = { m: 0, h: ((Math.atan2(vec[i][1], vec[i][0]) * 180) / Math.PI + 360) % 360, a: clamp(a, 0, 1) }; });
  // damped Gauss-Seidel Newton: per wheel, measure its 2×2 response at the current point (the response is
  // non-linear: hue curves and muting act after the wheels), take a capped step, halve it until the total
  // residual over all three bands improves
  const err = (bands) => bands.reduce((s, b, i) => s + Math.hypot(target[i][0] - b[0], target[i][1] - b[1]), 0);
  let achieved = base, cur = err(base);
  for (let it = 0; it < iters; it++) {
    for (let i = 0; i < 3; i++) {
      const v0 = vec[i].slice(), h = 0.15;
      const probe = (dx, dy) => { vec[i] = [v0[0] + dx, v0[1] + dy]; setWheels(); const b = bandAB(render(p))[i]; return b; };
      const ba = probe(h, 0), bb = probe(0, h); vec[i] = v0; setWheels();
      const b0 = achieved[i], M = [[(ba[0] - b0[0]) / h, (bb[0] - b0[0]) / h], [(ba[1] - b0[1]) / h, (bb[1] - b0[1]) / h]];
      const det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
      if (Math.abs(det) < 1e-8) continue;
      const r = [target[i][0] - b0[0], target[i][1] - b0[1]];
      let dx = (M[1][1] * r[0] - M[0][1] * r[1]) / det, dy = (-M[1][0] * r[0] + M[0][0] * r[1]) / det;
      const len = Math.hypot(dx, dy); if (len > 0.35) { dx *= 0.35 / len; dy *= 0.35 / len; }
      for (let tries = 0; tries < 4; tries++, dx /= 2, dy /= 2) {
        vec[i] = [v0[0] + dx, v0[1] + dy];
        const a = Math.hypot(vec[i][0], vec[i][1]); if (a > 1) { vec[i][0] /= a; vec[i][1] /= a; }
        setWheels();
        const bands = bandAB(render(p)), e = err(bands);
        if (e < cur) { achieved = bands; cur = e; break; }
        vec[i] = v0; setWheels();
      }
    }
  }
  const resid = achieved.map((a, i) => Math.hypot(target[i][0] - a[0], target[i][1] - a[1]));
  return { prim: p, report: { mode: modeName, label: md.label, norm, satRatio: k, targetBands: target, achievedBands: achieved, residual: resid } };
}

function toLab(c, out) { // sRGB 0..1 -> OKLab (local copy keeps this module free of the 8-bit table)
  const f = (v) => (v <= 0.04045 ? v / 12.92 : Math.pow((Math.max(v, 0) + 0.055) / 1.055, 2.4));
  const r = f(c[0]), g = f(c[1]), b = f(c[2]);
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  out[0] = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s;
  out[1] = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s;
  out[2] = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s;
  return out;
}

// ---- distribution refinement ------------------------------------------------------------------------
const CHROMA_W = 2;
const TONE_PRIOR = { m: 0.06, luma: 0.025 }; // a/b weigh twice L in the distribution distance: colour is what the reference is for
function sliceDirs(k = 24) { // fixed pseudo-random directions + the three axes (deterministic)
  let s = 3; const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647), d = [];
  for (let i = 0; i < k; i++) { const u = rnd() * 2 - 1, t = rnd() * 2 * Math.PI, r = Math.sqrt(1 - u * u); d.push([u, r * Math.cos(t), r * Math.sin(t)]); }
  d.push([1, 0, 0], [0, 1, 0], [0, 0, 1]);
  return d;
}
function sortedProjections(lab, dirs, nq) { // per direction: nq quantiles of the projected colours
  const m = lab.length / 3, out = [], v = new Float64Array(m);
  for (const d of dirs) {
    for (let i = 0; i < m; i++) v[i] = d[0] * lab[3 * i] + CHROMA_W * (d[1] * lab[3 * i + 1] + d[2] * lab[3 * i + 2]);
    v.sort();
    const q = new Float64Array(nq); for (let j = 0; j < nq; j++) q[j] = v[Math.min(m - 1, Math.floor(((j + 0.5) / nq) * m))];
    out.push(q);
  }
  return out;
}
// refined parameters: [key, index ('x'/'y' = wheel colour in cartesian form), step, prior scale]. Tone
// (master levels, luma curve) has a tight prior: a reference's bright sky or sand must not relight the
// source (the "完全過曝" failure); its brightness was already matched, capped, by the first pass.
const REFINE_PARAMS = (() => {
  const ps = [];
  for (const k of ['lift', 'gamma', 'gain']) ps.push([k, 'x', 0.1, 0.5], [k, 'y', 0.1, 0.5], [k, 'm', 0.025, TONE_PRIOR.m]);
  for (let i = 1; i < 4; i++) ps.push(['luma', i, 0.01, TONE_PRIOR.luma]);
  ps.push(['sat', null, 0.1, 0.5]);
  for (let i = 0; i < 8; i++) ps.push(['hueHue', i, 8, 40], ['hueSat', i, 0.1, 0.6]);
  return ps;
})();
const getP = (p, [k, i]) => {
  if (i === 'x' || i === 'y') { const t = (p[k].h * Math.PI) / 180; return (i === 'x' ? Math.cos(t) : Math.sin(t)) * p[k].a; }
  return i === null ? p[k] : p[k][i];
};
const setP = (p, [k, i], v) => {
  if (i === 'x' || i === 'y') {
    const t = (p[k].h * Math.PI) / 180; let x = Math.cos(t) * p[k].a, y = Math.sin(t) * p[k].a;
    if (i === 'x') x = v; else y = v;
    p[k] = { m: p[k].m, h: (((Math.atan2(y, x) * 180) / Math.PI) + 360) % 360, a: Math.min(1, Math.hypot(x, y)) };
  } else if (i === null) p[k] = v; else p[k][i] = v;
};

/**
 * Iterative optimisation of the whole parameter set (wheels, luma curve, saturation, hue curves) so that the
 * graded source's colour distribution matches the reference's: sliced Wasserstein-1 distance in OKLab,
 * plus lambda × squared distance from the first pass (content differs between source and reference; the
 * prior keeps the grade from forcing every colour of the source onto the reference's palette). Coordinate
 * descent with fixed steps, deterministic. On the user's LOG → GRADE example this reached the same pixel
 * error as parameters fitted to the true graded frame.
 */
export function refineToReference(prim0, srcRgb, refLab, { lambda = 0.005, rounds = 6, nq = 64 } = {}) {
  const dirs = sliceDirs(), R = sortedProjections(refLab, dirs, nq);
  const m = srcRgb.length / 3, outLab = new Float64Array(m * 3), c = [0, 0, 0], l = [0, 0, 0];
  const dist = (prim) => {
    const f = compilePrim(prim);
    for (let i = 0; i < m; i++) {
      const r = f ? f(srcRgb[3 * i], srcRgb[3 * i + 1], srcRgb[3 * i + 2], c) : [srcRgb[3 * i], srcRgb[3 * i + 1], srcRgb[3 * i + 2]];
      srgbToOklab(clamp(r[0], 0, 1), clamp(r[1], 0, 1), clamp(r[2], 0, 1), l);
      outLab[3 * i] = l[0]; outLab[3 * i + 1] = l[1]; outLab[3 * i + 2] = l[2];
    }
    const S = sortedProjections(outLab, dirs, nq); let s = 0;
    for (let k = 0; k < dirs.length; k++) for (let j = 0; j < nq; j++) s += Math.abs(S[k][j] - R[k][j]);
    return s / (dirs.length * nq);
  };
  // the composite tone response (restoration + masters + luma curve) on neutral greys across the source's
  // own lightness range must stay within slopes 0.45..2.5: no flattened midtones (lost texture), no steps
  const Ls = []; for (let i = 0; i < m; i++) { srgbToOklab(srcRgb[3 * i], srcRgb[3 * i + 1], srcRgb[3 * i + 2], l); Ls.push(l[0]); }
  Ls.sort((a, b) => a - b);
  const lo = Ls[Math.floor(0.05 * (m - 1))], hi = Ls[Math.floor(0.95 * (m - 1))], greys = [];
  for (let i = 0; i <= 16; i++) greys.push(oklabToSrgb(lo + ((hi - lo) * i) / 16, 0, 0));
  const response = (prim) => { // output lightness of the grey ramp
    const f = compilePrim(prim), out = [];
    for (const g of greys) { const r = f ? f(g[0], g[1], g[2], c) : g; srgbToOklab(clamp(r[0], 0, 1), clamp(r[1], 0, 1), clamp(r[2], 0, 1), l); out.push(l[0]); }
    return out;
  };
  // slopes are measured relative to the restoration alone (a LOG restoration is itself steep)
  const base = response({ ...neutralPrim(), norm: prim0.norm });
  const toneOK = (prim) => {
    const r = response(prim);
    for (let i = 1; i <= 16; i++) {
      const sl = (r[i] - r[i - 1]) / Math.max(base[i] - base[i - 1], 1e-4);
      if (sl < 0.45 || sl > 2.5) return false;
    }
    return true;
  };
  const v0 = REFINE_PARAMS.map((d) => getP(prim0, d));
  const cost = (p) => dist(p) + lambda * REFINE_PARAMS.reduce((s, d, j) => s + ((getP(p, d) - v0[j]) / d[3]) ** 2, 0) / REFINE_PARAMS.length;
  let p = sanitizePrim(prim0), cur = cost(p);
  const first = cur;
  for (let r = 0; r < rounds; r++) {
    const sc = r < rounds / 2 ? 1 : 0.5;
    for (const d of REFINE_PARAMS) for (const sgn of [1, -1]) for (let t = 0; t < 6; t++) {
      const q = sanitizePrim(p); setP(q, d, getP(q, d) + sgn * d[2] * sc);
      const qq = sanitizePrim(q);
      if (d[0] === 'luma') { qq.luma = safeLuma(qq.luma); if (qq.luma.every((v, j) => Math.abs(v - p.luma[j]) < 1e-9)) break; }
      if ((d[0] === 'luma' || d[1] === 'm') && !toneOK(qq)) break;
      const cc = cost(qq);
      if (cc < cur - 1e-6) { p = qq; cur = cc; } else break;
    }
  }
  return { prim: p, before: first, after: cur };
}

/** Parameters part-way between two grades (restoration from b). */
export function blendPrim(a, b, t) {
  const q = sanitizePrim(b);
  for (const d of REFINE_PARAMS) setP(q, d, getP(a, d) + (getP(b, d) - getP(a, d)) * t);
  for (const k of ['lift', 'gamma', 'gain']) q[k].m = a[k].m + (b[k].m - a[k].m) * t;
  const out = sanitizePrim(q); out.luma = safeLuma(out.luma);
  return out;
}
function renderPrim(img, prim) { // RGBA through the pro stage (texture guard only; the app renders through the LUT)
  const f = compilePrim(prim), d = img.data, o = new Uint8ClampedArray(d.length), c = [0, 0, 0];
  for (let i = 0; i < d.length; i += 4) {
    const r = f ? f(d[i] / 255, d[i + 1] / 255, d[i + 2] / 255, c) : [d[i] / 255, d[i + 1] / 255, d[i + 2] / 255];
    o[i] = r[0] * 255 + 0.5; o[i + 1] = r[1] * 255 + 0.5; o[i + 2] = r[2] * 255 + 0.5; o[i + 3] = 255;
  }
  return o;
}

/** A grade at another strength: every refined parameter scaled about neutral (restoration kept as is). */
export function scalePrim(p, t) {
  const q = sanitizePrim(p), n = neutralPrim();
  for (const d of REFINE_PARAMS) { const v0 = getP(n, d); setP(q, d, v0 + (getP(p, d) - v0) * t); }
  for (const k of ['lift', 'gamma', 'gain']) q[k].m = n[k].m + (p[k].m - n[k].m) * t; // master too (setP kept it)
  const out = sanitizePrim(q); out.luma = safeLuma(out.luma);
  return out;
}

const STRENGTH = { natural: 0.65, faithful: 1, bold: 1.25 };
/**
 * All three interpretations, natural first: restoration → first pass (natural settings, the safest start)
 * → distribution refinement = 忠實; 自然 / 濃烈 are the same grade at 65 % / 125 % strength.
 */
export function fitCandidates(srcLab, refLab, { refine = true, thumb = null } = {}) {
  const norm = estimateNorm(srcLab);
  if (!refine) return ['natural', 'faithful', 'bold'].map((m) => fitPrim(srcLab, refLab, m, { norm }));
  const first = fitPrim(srcLab, refLab, 'natural', { norm });
  const pick = (lab, n) => { const k = lab.length / 3, st = Math.max(1, Math.floor(k / n)), o = []; for (let i = 0; i < k; i += st) o.push(lab[3 * i], lab[3 * i + 1], lab[3 * i + 2]); return Float64Array.from(o); };
  const srcPick = pick(srcLab, 2000), srcRgb = new Float64Array(srcPick.length);
  for (let i = 0; i < srcPick.length; i += 3) srcRgb.set(oklabToSrgb(srcPick[i], srcPick[i + 1], srcPick[i + 2]), i);
  const bright = []; // like the first pass: blown reference pixels (sky, sand) say nothing about the grade
  for (let i = 0; i < refLab.length; i += 3) if (refLab[i] < 0.93) bright.push(refLab[i], refLab[i + 1], refLab[i + 2]);
  const refUse = bright.length > refLab.length * 0.2 ? Float64Array.from(bright) : refLab;
  const ref = refineToReference(first.prim, srcRgb, pick(refUse, 3000));
  // texture guard: on a real frame, the refined grade must keep structure ≥ 0.95 (it can recolour, not
  // flatten); otherwise it is moved back towards the first pass until it does
  let refined = ref.prim, back = 0;
  if (thumb) {
    const restored = renderPrim(thumb, { ...neutralPrim(), norm: first.prim.norm }); // compared after restoration
    const st = (prim) => structureScore(restored, renderPrim(thumb, prim), thumb.width, thumb.height);
    if (st(scalePrim(refined, STRENGTH.bold)) < 0.95) {
      let a = 0, b = 1; // a: share of the refinement known to be fine
      for (let k = 0; k < 6; k++) { const t = (a + b) / 2; if (st(scalePrim(blendPrim(first.prim, ref.prim, t), STRENGTH.bold)) >= 0.95) a = t; else b = t; }
      refined = blendPrim(first.prim, ref.prim, a); back = 1 - a;
    }
  }
  return ['natural', 'faithful', 'bold'].map((mode) => {
    const prim = scalePrim(refined, STRENGTH[mode]);
    prim.skin = MODES[mode].skin;
    return { prim, report: { ...first.report, mode, label: MODES[mode].label, distance: { first: ref.before, refined: ref.after }, structureBack: back } };
  });
}
