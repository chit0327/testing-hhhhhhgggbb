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
import { neutralPrim, compilePrim, sanitizePrim, curveFn, normTone, applyBal, HUE_ANCHORS, CURVE_X } from './primary.js';
import { structureScore } from './metrics.js';

export const MODES = {
  look: { label: '風格', skin: 1 },
  lookStrong: { label: '風格・濃', skin: 1 },
  infer: { label: '推斷風格', skin: 1 },
  natural: { label: '照搬・自然', tone: 0.6, cap: 0.1, color: 0.6, cast: 0.6, hue: 0.6, maxShift: 25, mute: -0.3, skin: 1, contrast: 0 },
  faithful: { label: '照搬・忠實', tone: 1, cap: 0.16, color: 1, cast: 1, hue: 1, maxShift: 35, mute: -0.5, skin: 0.6, contrast: 0 },
  bold: { label: '照搬・濃烈', tone: 1.2, cap: 0.2, color: 1.25, cast: 1.2, hue: 1.15, maxShift: 45, mute: -0.6, skin: 0.35, contrast: 0.04 },
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
export function fitPrim(srcLab0, refLab, modeName = 'natural', { iters = 6, sample = 2500, norm = estimateNorm(srcLab0), keepAbsent = false, tonePrior = false } = {}) {
  const absent = []; // hue anchors the source has and the reference does not
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
  if (tonePrior) { // tone of the LOOK, not of the reference's content: typical photo -> reference, half strength
    const sr = Float64Array.from(refL).sort(), qr2 = QS33.map((q) => sr[Math.min(sr.length - 1, Math.round(q * (sr.length - 1)))]);
    p.luma = CURVE_X.map((x) => clamp(x + clamp(interp(x, TYPICAL.Lq, qr2) - x, -0.12, 0.12) * 0.5 * md.tone, 0, 1));
  } else {
  p.luma = CURVE_X.map((x) => {
      const d = clamp(interp(x, qs, qr) - x, -md.cap, md.cap) * md.tone;
      const s = x === 0.25 ? -md.contrast : x === 0.75 ? md.contrast : 0;
      return clamp(x + d + s, 0, 1); // never push black below 0 or white above 1 (crushed / clipped detail)
    });
  }
  p.luma = safeLuma(p.luma);
  // ---- saturation level
  const meanC = (lab) => { let s = 0, n = 0; for (let i = 0; i < lab.length; i += 3) { s += Math.hypot(lab[i + 1], lab[i + 2]); n++; } return s / Math.max(1, n); };
  const k = clamp(meanC(refLab) / Math.max(meanC(srcLab), 1e-3), 0.5, 2);
  p.sat = clamp(Math.log2(k) * md.color, -0.8, 0.8);
  // ---- hue curves
  for (let i = 0; i < 8; i++) {
    const s = hueWindow(srcLab, HUE_ANCHORS[i], 30), r = hueWindow(refLab, HUE_ANCHORS[i], 50);
    if (s.share < 0.02) continue;
    if (r.share < 0.25 * s.share) { p.hueSat[i] = keepAbsent ? md.mute * 0.3 : md.mute; absent.push(i); continue; } // colour the reference does not have
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
  return { prim: p, report: { mode: modeName, label: md.label, norm, absent, satRatio: k, targetBands: target, achievedBands: achieved, residual: resid } };
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

// ---- look inference against typical-photo statistics ("推斷風格") ------------------------------------
// The reference is read as look(typical photo): its tone curve and neutral-axis casts are estimated by
// comparing it with the average of 59 natural, ungraded photos (Deep Photo Style Transfer inputs, centre
// 70 %), never with the source, so the source's own content cannot leak in. Tone at half strength (the
// typical distribution is a weak prior: its quantiles vary by ±0.17), casts from the least colourful 5 %
// of each tonal band, no saturation or hue changes (both made it worse). Measured on 41 known-look cases
// (third-party film LUTs, reference and source of different content): ΔE 12.30 untouched → 10.59; the
// direct matching candidates score 12.7–16.1 there but win on strong looks and similar content.
const TYPICAL = {"Lq": [0.2073, 0.2614, 0.2949, 0.3221, 0.3484, 0.3727, 0.3961, 0.419, 0.4425, 0.4724, 0.4934, 0.5151, 0.5324, 0.5491, 0.5652, 0.5802, 0.5956, 0.6118, 0.629, 0.6464, 0.6628, 0.6802, 0.6977, 0.7156, 0.7335, 0.7463, 0.7611, 0.7786, 0.799, 0.8201, 0.8443, 0.8755, 0.9273], "chroma": 0.0365, "casts": [[0.0016, 0.0078], [-0.0009, 0.0053], [-0.0006, 0.0044]]};
const QS33 = Array.from({ length: 33 }, (_, k) => 0.01 + (0.98 * k) / 32);
function bandCastsLow(lab, frac) { // per lightness tercile (by rank): median a/b of its least colourful `frac`
  const n = lab.length / 3, idx = Array.from({ length: n }, (_, i) => i).sort((x, y) => lab[3 * x] - lab[3 * y]);
  const C = (i) => Math.hypot(lab[3 * i + 1], lab[3 * i + 2]);
  return [0, 1, 2].map((b) => {
    const part = idx.slice(Math.floor((b * n) / 3), Math.floor(((b + 1) * n) / 3)).sort((x, y) => C(x) - C(y));
    const k = part.slice(0, Math.max(1, Math.floor(part.length * frac)));
    const med = (o) => { const v = k.map((i) => lab[3 * i + o]).sort((p, q) => p - q); return v[Math.floor(v.length / 2)]; };
    return [med(1), med(2)];
  });
}
export function fitInferred(srcLab, refLab, { tone = 0.5, cast = 1, toneCap = 0.12, norm = estimateNorm(srcLab), deltas = null } = {}) {
  const p = neutralPrim();
  p.norm = { amt: norm.amt, m: norm.m, Tm: norm.Tm, s: norm.s, sat: norm.sat };
  const srcR = norm.amt ? applyNormLab(srcLab, norm) : srcLab;
  const rl = []; for (let i = 0; i < refLab.length; i += 3) rl.push(refLab[i]);
  const sr = Float64Array.from(rl).sort(), qr = QS33.map((q) => sr[Math.min(sr.length - 1, Math.round(q * (sr.length - 1)))]);
  p.luma = safeLuma(CURVE_X.map((x) => clamp(x + clamp(interp(x, TYPICAL.Lq, qr) - x, -toneCap, toneCap) * tone, 0, 1)));
  const delta = deltas || bandCastsLow(refLab, 0.05).map((c, b) => [(c[0] - TYPICAL.casts[b][0]) * cast, (c[1] - TYPICAL.casts[b][1]) * cast]); // deltas: per-band casts given directly
  // wheels: the source's near-neutral samples (least colourful 20 % per band) must take on these casts
  const n = Math.min(2500, srcR.length / 3), st = Math.max(1, Math.floor(srcR.length / 3 / n)), rgb = [], sl = [];
  for (let i = 0, j = 0; j < n; i += 3 * st, j++) { rgb.push(oklabToSrgb(srcLab[i], srcLab[i + 1], srcLab[i + 2])); sl.push(srcR[i], srcR[i + 1], srcR[i + 2]); }
  const order = Array.from({ length: n }, (_, i) => i).sort((x, y) => sl[3 * x] - sl[3 * y]), band = new Uint8Array(n), keep = new Uint8Array(n);
  const C = (i) => Math.hypot(sl[3 * i + 1], sl[3 * i + 2]);
  for (let b = 0; b < 3; b++) {
    const part = order.slice(Math.floor((b * n) / 3), Math.floor(((b + 1) * n) / 3));
    for (const i of part) band[i] = b;
    for (const i of part.sort((x, y) => C(x) - C(y)).slice(0, Math.floor(part.length * 0.2))) keep[i] = 1;
  }
  const lab = [0, 0, 0], out = [0, 0, 0];
  const bands = (prim) => {
    const f = compilePrim(prim), a = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let j = 0; j < n; j++) {
      if (!keep[j]) continue;
      const c = f ? f(rgb[j][0], rgb[j][1], rgb[j][2], out) : rgb[j];
      srgbToOklab(clamp(c[0], 0, 1), clamp(c[1], 0, 1), clamp(c[2], 0, 1), lab);
      const t = a[band[j]]; t[0] += lab[1]; t[1] += lab[2]; t[2]++;
    }
    return a.map((t) => [t[0] / (t[2] || 1), t[1] / (t[2] || 1)]);
  };
  const base = bands(p), target = base.map((b, i) => [b[0] + delta[i][0], b[1] + delta[i][1]]);
  solveWheels(p, bands, target, 5);
  p.skin = MODES.infer.skin;
  return { prim: p, report: { mode: 'infer', label: MODES.infer.label, norm, casts: delta } };
}

// ---- 風格 (stable look) ---------------------------------------------------------------------------------
// The reference is read RELATIVE TO A TYPICAL PHOTO, not relative to the footage: the same reference gives the same grade
// on any footage (the matching candidates copy the reference's colour distribution, i.e. its content: a pink room went
// orange, a blue mug teal, skin sallow — "too unstable"). Every estimate is weighted by how far the reference lies beyond
// the normal photo-to-photo variation (σ, measured on 59 natural photos): a James–Stein-like weight max(floor, 1 − c/m²).
//   tone: quantiles vs typical (σ per quantile);  casts: median a/b of each tonal band vs typical (all pixels);
//   colour character: per hue sector (8 hue-curve anchors) hue / chroma of the reference vs the typical photo's sector
//   after the base grade, each weighted by its own m².
// Measured (research/look_stable): known-look benchmark ΔE to the truth 11.11 (自然 11.64, none 11.91); grading set
// closeness to the reference 14.2 (自然 14.8); stability — spread of a colour chart graded by fits on 6 different
// footages — 1.9 (自然 9.8). Blind check with the user: preferred over 自然 5 : 3.
const LOOK_TYP = {
  LqSd: [0.1382, 0.1532, 0.1599, 0.1646, 0.166, 0.1671, 0.1673, 0.1663, 0.1659, 0.1805, 0.18, 0.1816, 0.1799, 0.1782, 0.1766, 0.1749, 0.1735, 0.172, 0.1709, 0.1699, 0.1684, 0.1638, 0.1616, 0.1622, 0.1609, 0.1565, 0.1501, 0.1424, 0.135, 0.1251, 0.1108, 0.0957, 0.0747],
  casts: [[-0.0002, 0.0083], [-0.0035, 0.0098], [-0.0044, 0.0121]], castSd: [[0.0232, 0.049], [0.0301, 0.0533], [0.022, 0.0467]], // all pixels per band
  sectors: [[39.9, 11.1, 0.0631, 0.0308, 0.486], [74.6, 13.3, 0.0614, 0.0342, 0.563], [96.9, 14.2, 0.0669, 0.0447, 0.612], [130.1, 8.6, 0.0566, 0.0432, 0.503],
    [204.1, 10.5, 0.0342, 0.0119, 0.585], [254.2, 9.6, 0.054, 0.027, 0.627], [289.2, 11.5, 0.034, 0.0213, 0.46], [343.5, 6.8, 0.0344, 0.0174, 0.372]], // hue, hueSd, chroma, chromaSd, L
};
const LOOK = { toneFloor: 0.25, toneCap: 0.3, castMax: 2.5, castFloor: 0.25, kHue: 0.5, kChroma: 0.5, capHue: 20, capSkin: 10, capChroma: 0.5, minShare: 0.04 };
export const LOOK_SCALE = { look: 2, lookStrong: 3 };
const jsw = (m2, c = 1, floor = 0) => Math.max(floor, 1 - c / Math.max(m2, 1e-9));
function lookSectors(lab, win = 30) { // per hue anchor: share of total chroma, chroma-weighted hue / chroma
  const acc = HUE_ANCHORS.map(() => [0, 0, 0, 0]); let tot = 0;
  for (let i = 0; i < lab.length; i += 3) {
    const C = Math.hypot(lab[i + 1], lab[i + 2]); if (C < 0.02) continue; tot += C;
    const h = (Math.atan2(lab[i + 2], lab[i + 1]) * 180) / Math.PI;
    HUE_ANCHORS.forEach((a, k) => { if (Math.abs(angDiff(h, a)) > win) return; const t = acc[k]; t[0] += C; t[1] += C * Math.cos((h * Math.PI) / 180); t[2] += C * Math.sin((h * Math.PI) / 180); t[3] += C * C; });
  }
  return acc.map((t) => ({ share: tot ? t[0] / tot : 0, hue: ((Math.atan2(t[2], t[1]) * 180) / Math.PI + 360) % 360, chroma: t[0] ? t[3] / t[0] : 0 }));
}
/** 風格 / 風格・濃: the stable look (see above); mode 'look' (×2) or 'lookStrong' (×3). */
export function fitLook(srcLab, refLab, mode = 'look', { norm = estimateNorm(srcLab) } = {}) {
  const n = refLab.length / 3, L = []; for (let i = 0; i < n; i++) L.push(refLab[3 * i]); L.sort((a, b) => a - b);
  const q = QS33.map((x) => L[Math.round(x * (n - 1))]);
  const mt = q.reduce((s, v, k) => s + ((v - TYPICAL.Lq[k]) / LOOK_TYP.LqSd[k]) ** 2, 0) / q.length;
  const deltas = bandCastsLow(refLab, 1).map((c, b) => {
    const d = [c[0] - LOOK_TYP.casts[b][0], c[1] - LOOK_TYP.casts[b][1]], m2 = ((d[0] / LOOK_TYP.castSd[b][0]) ** 2 + (d[1] / LOOK_TYP.castSd[b][1]) ** 2) / 2, w = LOOK.castMax * jsw(m2, 1, LOOK.castFloor);
    return [d[0] * w, d[1] * w];
  });
  const base = fitInferred(srcLab, refLab, { tone: jsw(mt, 1, LOOK.toneFloor), toneCap: LOOK.toneCap, deltas, norm });
  const p = base.prim, f = compilePrim(p), rs = lookSectors(refLab), o = [0, 0, 0], l = [0, 0, 0];
  rs.forEach((r, k) => {
    if (r.share < LOOK.minShare) return;
    const [th, thSd, tc, tcSd, tL] = LOOK_TYP.sectors[k];
    const s = oklabToSrgb(tL, tc * Math.cos((th * Math.PI) / 180), tc * Math.sin((th * Math.PI) / 180)), g = f ? f(clamp(s[0], 0, 1), clamp(s[1], 0, 1), clamp(s[2], 0, 1), o) : s;
    srgbToOklab(clamp(g[0], 0, 1), clamp(g[1], 0, 1), clamp(g[2], 0, 1), l); // the typical sector colour after the base grade
    const h0 = angDiff(r.hue, (Math.atan2(l[2], l[1]) * 180) / Math.PI), c0 = Math.log2(Math.max(r.chroma, 1e-3) / Math.max(Math.hypot(l[1], l[2]), 1e-3));
    const cap = k <= 1 ? LOOK.capSkin : LOOK.capHue; // red / orange carry skin
    p.hueHue[k] = clamp(LOOK.kHue * h0 * jsw((h0 / thSd) ** 2), -cap, cap);
    p.hueSat[k] = clamp(LOOK.kChroma * c0 * jsw((c0 / Math.log2(1 + tcSd / tc)) ** 2), -LOOK.capChroma, LOOK.capChroma);
  });
  const out = scalePrim(p, LOOK_SCALE[mode] || 2);
  out.norm = p.norm; out.skin = MODES[mode].skin;
  return { prim: out, report: { mode, label: MODES[mode].label, norm, weights: { tone: jsw(mt, 1, LOOK.toneFloor) } } };
}

/**
 * Grey balance after the grade: the source's near-neutral pixels (least colourful 20 % of each tonal band:
 * whites, greys, shadows) are given the reference's neutral tint (median a/b of its least colourful 5 % per
 * band) by re-solving the Lift / Gamma / Gain colour, the rest of the grade unchanged. Raters judged
 * whites first ("whites a little bluer / more orange", "highlights too green"): this is what they saw.
 */
export function neutralBalance(p, srcLab, refLab, strength = 1) {
  const norm = p.norm || { amt: 0 }, srcR = norm.amt ? applyNormLab(srcLab, norm) : srcLab;
  const tint = bandCastsLow(refLab, 0.05);
  const n = Math.min(2500, srcR.length / 3), st = Math.max(1, Math.floor(srcR.length / 3 / n)), rgb = [], sl = [];
  for (let i = 0, j = 0; j < n; i += 3 * st, j++) { rgb.push(oklabToSrgb(srcLab[i], srcLab[i + 1], srcLab[i + 2])); sl.push(srcR[i], srcR[i + 1], srcR[i + 2]); }
  const order = Array.from({ length: n }, (_, i) => i).sort((x, y) => sl[3 * x] - sl[3 * y]), band = new Uint8Array(n), keep = new Uint8Array(n);
  const C = (i) => Math.hypot(sl[3 * i + 1], sl[3 * i + 2]);
  for (let b = 0; b < 3; b++) {
    const part = order.slice(Math.floor((b * n) / 3), Math.floor(((b + 1) * n) / 3));
    for (const i of part) band[i] = b;
    for (const i of part.sort((x, y) => C(x) - C(y)).slice(0, Math.floor(part.length * 0.2))) keep[i] = 1;
  }
  const lab = [0, 0, 0], out = [0, 0, 0];
  const bands = (prim) => {
    const f = compilePrim(prim), a = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let j = 0; j < n; j++) {
      if (!keep[j]) continue;
      const c = f ? f(rgb[j][0], rgb[j][1], rgb[j][2], out) : rgb[j];
      srgbToOklab(clamp(c[0], 0, 1), clamp(c[1], 0, 1), clamp(c[2], 0, 1), lab);
      const t = a[band[j]]; t[0] += lab[1]; t[1] += lab[2]; t[2]++;
    }
    return a.map((t) => [t[0] / (t[2] || 1), t[1] / (t[2] || 1)]);
  };
  const now = bands(p), target = now.map((b, i) => [b[0] + (tint[i][0] - b[0]) * strength, b[1] + (tint[i][1] - b[1]) * strength]);
  const q = sanitizePrim(p);
  solveWheels(q, bands, target, 4);
  return { prim: q, tint, before: now, target };
}

/**
 * Correction: the shot's own colour cast, as white-balance gains (linear light, luminance kept). The light
 * colour is estimated from grey pixels by shades of grey (Minkowski p = 6 mean per channel), iterated:
 * correct with the current estimate, keep the pixels that are then near grey (chroma < 0.05; 0.08 on the
 * first pass, unclipped, not dark), re-estimate. Saturated content (orange blocks, a green forest) never
 * takes part. On the 8 test clips with a known added cast (warm, cool, green, magenta) this removed 95 % of
 * it at full strength (worst 78 %); "least colourful pixels" removed 47 % (under warm light blue objects
 * look grey), plain shades of grey 93 % but changed untouched clips by content. Confidence falls when
 * fewer than 10 % of the pixels are grey candidates, and when the light colour is implausible (below). strength: share of the cast removed (geometric);
 * gains capped at ±25 %.
 */
export function estimateBalance(srcLab, { strength = 0.75, p = 6 } = {}) {
  const n = srcLab.length / 3, lin = (v) => (v <= 0.04045 ? v / 12.92 : Math.pow((Math.max(v, 0) + 0.055) / 1.055, 2.4));
  const enc = (v) => (v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055);
  const x = new Float64Array(srcLab.length);
  for (let i = 0; i < n; i++) { const c = oklabToSrgb(srcLab[3 * i], srcLab[3 * i + 1], srcLab[3 * i + 2]); for (let k = 0; k < 3; k++) x[3 * i + k] = lin(Math.min(1, Math.max(0, c[k]))); }
  const none = { bal: { r: 1, g: 1, b: 1 }, confidence: 0, cast: [0, 0] };
  let e = [1, 1, 1], share = 0;
  const l = [0, 0, 0];
  for (let it = 0; it < 5; it++) {
    const th = it === 0 ? 0.08 : 0.05, m = [0, 0, 0]; let cnt = 0;
    for (let i = 0; i < n; i++) {
      const r = x[3 * i], g = x[3 * i + 1], b = x[3 * i + 2];
      if (Math.max(r, g, b) >= 0.98) continue;
      srgbToOklab(enc(Math.min(1, r / e[0])), enc(Math.min(1, g / e[1])), enc(Math.min(1, b / e[2])), l);
      if (l[0] < 0.2 || Math.hypot(l[1], l[2]) > th) continue;
      m[0] += Math.pow(r, p); m[1] += Math.pow(g, p); m[2] += Math.pow(b, p); cnt++;
    }
    share = cnt / n;
    if (share < 0.02) return it === 0 ? none : { ...none, confidence: 0 };
    const ne = m.map((v) => Math.pow(v / cnt, 1 / p)), Y = 0.2126 * ne[0] + 0.7152 * ne[1] + 0.0722 * ne[2];
    e = ne.map((v) => v / Y);
  }
  // plausibility: real light sources lie along the warm–cool axis (tungsten orange ↔ daylight blue, OKLab hue
  // ≈ 73° / 253°); a light colour far off it (strong green / magenta) or very strong is content, not light
  const mx = Math.max(...e), il = [0, 0, 0]; srgbToOklab(enc(e[0] / mx), enc(e[1] / mx), enc(e[2] / mx), il);
  const ux = Math.cos((73 * Math.PI) / 180), uy = Math.sin((73 * Math.PI) / 180);
  const off = Math.abs(-uy * il[1] + ux * il[2]), along = Math.abs(ux * il[1] + uy * il[2]);
  const plausible = (1 - clamp((off - 0.03) / 0.03, 0, 1)) * (1 - clamp((along - 0.15) / 0.1, 0, 1));
  const confidence = clamp(share / 0.1, 0, 1) * plausible, k = strength * confidence;
  const g = e.map((v) => clamp(Math.pow(1 / v, k), 0.8, 1.25));
  return { bal: { r: g[0], g: g[1], b: g[2] }, confidence, cast: [il[1], il[2]], greyShare: share };
}
/** OKLab samples through the balance (to analyse the source as the grade will see it). */
export function balanceLab(lab, bal) {
  const out = new Float64Array(lab.length), c = [0, 0, 0], l = [0, 0, 0];
  for (let i = 0; i < lab.length; i += 3) { const s = oklabToSrgb(lab[i], lab[i + 1], lab[i + 2]); applyBal(bal, Math.min(1, Math.max(0, s[0])), Math.min(1, Math.max(0, s[1])), Math.min(1, Math.max(0, s[2])), c); srgbToOklab(c[0], c[1], c[2], l); out[i] = l[0]; out[i + 1] = l[1]; out[i + 2] = l[2]; }
  return out;
}

/** The reference's white: mean a/b of its brightest near-neutral pixels (chroma < 0.06, not clipped), or null. */
export function whiteOf(lab, top = 0.05) {
  const n = lab.length / 3, idx = [];
  for (let i = 0; i < n; i++) if (lab[3 * i] < 0.99 && Math.hypot(lab[3 * i + 1], lab[3 * i + 2]) < 0.06) idx.push(i);
  if (idx.length < n * 0.01) return null;
  idx.sort((x, y) => lab[3 * y] - lab[3 * x]);
  const k = idx.slice(0, Math.max(1, Math.floor(n * top)));
  let a = 0, b = 0; for (const i of k) { a += lab[3 * i + 1]; b += lab[3 * i + 2]; }
  return [a / k.length, b / k.length];
}
/**
 * White point after the grade: the source's whites (its brightest near-neutral pixels, after restoration)
 * are pushed through the whole grade and the Gain colour alone is re-solved until they take on the
 * reference's white. The tonal-band fit cannot do this: a band's mean is set by its grass, walls and
 * clothes, and the hue curves act after the wheels, so whites drifted (round-1/2 notes: "highlight whites
 * too green", "whites a little more orange").
 */
export function whitePoint(p, srcLab, refLab, strength = 1) {
  const tw = whiteOf(refLab);
  const norm = p.norm || { amt: 0 }, srcR = norm.amt ? applyNormLab(srcLab, norm) : srcLab;
  const n = srcR.length / 3, idx = [];
  for (let i = 0; i < n; i++) if (srcR[3 * i] < 0.99 && Math.hypot(srcR[3 * i + 1], srcR[3 * i + 2]) < 0.06) idx.push(i);
  if (!tw || idx.length < n * 0.01) return { prim: p, skipped: true };
  idx.sort((x, y) => srcR[3 * y] - srcR[3 * x]);
  const sel = idx.slice(0, Math.max(1, Math.floor(n * 0.05))), st = Math.max(1, Math.floor(sel.length / 600));
  const rgb = []; for (let j = 0; j < sel.length; j += st) { const i = sel[j]; rgb.push(oklabToSrgb(srcLab[3 * i], srcLab[3 * i + 1], srcLab[3 * i + 2])); }
  const lab = [0, 0, 0], out = [0, 0, 0];
  const white = (prim) => {
    const f = compilePrim(prim); let a = 0, b = 0;
    for (const c0 of rgb) { const c = f ? f(c0[0], c0[1], c0[2], out) : c0; srgbToOklab(clamp(c[0], 0, 1), clamp(c[1], 0, 1), clamp(c[2], 0, 1), lab); a += lab[1]; b += lab[2]; }
    return [a / rgb.length, b / rgb.length];
  };
  const q = sanitizePrim(p), before = white(q), target = [before[0] + (tw[0] - before[0]) * strength, before[1] + (tw[1] - before[1]) * strength];
  const t0 = (q.gain.h * Math.PI) / 180; let v = [Math.cos(t0) * q.gain.a, Math.sin(t0) * q.gain.a];
  const setV = (x) => { const a = Math.hypot(x[0], x[1]); q.gain = { m: q.gain.m, h: ((Math.atan2(x[1], x[0]) * 180) / Math.PI + 360) % 360, a: clamp(a, 0, 1) }; };
  const dist = (w) => Math.hypot(target[0] - w[0], target[1] - w[1]);
  let cur = before, e = dist(cur);
  for (let it = 0; it < 8 && e > 0.002; it++) {
    const h = 0.1, probe = (dx, dy) => { setV([v[0] + dx, v[1] + dy]); return white(q); };
    const wa = probe(h, 0), wb = probe(0, h); setV(v);
    const M = [[(wa[0] - cur[0]) / h, (wb[0] - cur[0]) / h], [(wa[1] - cur[1]) / h, (wb[1] - cur[1]) / h]], det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    if (Math.abs(det) < 1e-8) break;
    const r = [target[0] - cur[0], target[1] - cur[1]];
    let dx = (M[1][1] * r[0] - M[0][1] * r[1]) / det, dy = (-M[1][0] * r[0] + M[0][0] * r[1]) / det, moved = false;
    const len = Math.hypot(dx, dy); if (len > 0.3) { dx *= 0.3 / len; dy *= 0.3 / len; }
    for (let tries = 0; tries < 4; tries++, dx /= 2, dy /= 2) {
      const nv = [v[0] + dx, v[1] + dy], a = Math.hypot(nv[0], nv[1]); if (a > 1) { nv[0] /= a; nv[1] /= a; }
      setV(nv); const w = white(q), d = dist(w);
      if (d < e) { v = nv; cur = w; e = d; moved = true; break; }
    }
    setV(v);
    if (!moved) break;
  }
  return { prim: q, before, target, after: cur };
}

/** Damped Gauss-Seidel Newton on the Lift / Gamma / Gain colour so that bands(p) reaches target. */
function solveWheels(p, bands, target, iters) {
  const wheels = ['lift', 'gamma', 'gain'], vec = wheels.map((w) => { const t = (p[w].h * Math.PI) / 180; return [Math.cos(t) * p[w].a, Math.sin(t) * p[w].a]; });
  const setW = () => wheels.forEach((w, i) => { const a = Math.hypot(vec[i][0], vec[i][1]); p[w] = { m: p[w].m, h: ((Math.atan2(vec[i][1], vec[i][0]) * 180) / Math.PI + 360) % 360, a: clamp(a, 0, 1) }; });
  const err = (bs) => bs.reduce((s, b, i) => s + Math.hypot(target[i][0] - b[0], target[i][1] - b[1]), 0);
  let ach = bands(p), cur = err(ach);
  for (let it = 0; it < iters; it++) for (let i = 0; i < 3; i++) {
    const v0 = vec[i].slice(), h = 0.15;
    const probe = (dx, dy) => { vec[i] = [v0[0] + dx, v0[1] + dy]; setW(); return bands(p)[i]; };
    const ba = probe(h, 0), bb = probe(0, h); vec[i] = v0; setW();
    const b0 = ach[i], M = [[(ba[0] - b0[0]) / h, (bb[0] - b0[0]) / h], [(ba[1] - b0[1]) / h, (bb[1] - b0[1]) / h]];
    const det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    if (Math.abs(det) < 1e-8) continue;
    const r = [target[i][0] - b0[0], target[i][1] - b0[1]];
    let dx = (M[1][1] * r[0] - M[0][1] * r[1]) / det, dy = (-M[1][0] * r[0] + M[0][0] * r[1]) / det;
    const len = Math.hypot(dx, dy); if (len > 0.35) { dx *= 0.35 / len; dy *= 0.35 / len; }
    for (let tries = 0; tries < 4; tries++, dx /= 2, dy /= 2) {
      vec[i] = [v0[0] + dx, v0[1] + dy];
      const a = Math.hypot(vec[i][0], vec[i][1]); if (a > 1) { vec[i][0] /= a; vec[i][1] /= a; }
      setW();
      const bs = bands(p), e = err(bs);
      if (e < cur) { ach = bs; cur = e; break; }
      vec[i] = v0; setW();
    }
  }
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
export function refineToReference(prim0, srcRgb, refLab, { lambda = 0.005, rounds = 6, nq = 64, freeze = [], freezeTone = false } = {}) {
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
      if ((d[0] === 'hueHue' || d[0] === 'hueSat') && freeze.includes(d[1])) break; // colours the reference does not have
      if (freezeTone && (d[0] === 'luma' || d[1] === 'm')) break; // tone comes from the look prior (v3)
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
const NEUTRAL = { natural: 0.8, faithful: 1, bold: 1 }; // grey balance towards the reference's neutral tint
const WHITE = { natural: 1, faithful: 1, bold: 1 }; // white point: the reference's white, fully
/**
 * Candidates: 風格 and 風格・濃 first (stable, see fitLook), then the three matching interpretations, natural first: restoration → first pass (natural settings, the safest start)
 * → distribution refinement = 忠實; 自然 / 濃烈 are the same grade at 65 % / 125 % strength.
 */
export function fitCandidates(srcLab, refLab, { refine = true, thumb = null, v2 = false, v3 = false, white = true, balance = false } = {}) {
  if (v3) v2 = true;
  // correction first: the grade is fitted on the balanced source and the balance is prepended to every candidate
  const bal = balance ? estimateBalance(srcLab) : null;
  if (bal) srcLab = balanceLab(srcLab, bal.bal);
  if (bal && thumb) { const d = new Uint8ClampedArray(thumb.data.length), c = [0, 0, 0]; for (let i = 0; i < d.length; i += 4) { applyBal(bal.bal, thumb.data[i] / 255, thumb.data[i + 1] / 255, thumb.data[i + 2] / 255, c); d[i] = c[0] * 255 + 0.5; d[i + 1] = c[1] * 255 + 0.5; d[i + 2] = c[2] * 255 + 0.5; d[i + 3] = 255; } thumb = { data: d, width: thumb.width, height: thumb.height }; }
  const withBal = (cands) => (bal ? cands.map((c) => ({ ...c, prim: { ...c.prim, bal: { ...bal.bal } }, report: { ...c.report, balance: { gains: bal.bal, confidence: bal.confidence, cast: bal.cast } } })) : cands);
  const norm = estimateNorm(srcLab);
  const looks = ['look', 'lookStrong'].map((m) => fitLook(srcLab, refLab, m, { norm })); // 風格 first: the stable default
  if (!refine) return withBal([...looks, ...['natural', 'faithful', 'bold'].map((m) => fitPrim(srcLab, refLab, m, { norm }))]);
  const first = fitPrim(srcLab, refLab, 'natural', { norm, keepAbsent: v2, tonePrior: v3 });
  const pick = (lab, n) => { const k = lab.length / 3, st = Math.max(1, Math.floor(k / n)), o = []; for (let i = 0; i < k; i += st) o.push(lab[3 * i], lab[3 * i + 1], lab[3 * i + 2]); return Float64Array.from(o); };
  const srcPick = pick(srcLab, 2000), srcRgb = new Float64Array(srcPick.length);
  for (let i = 0; i < srcPick.length; i += 3) srcRgb.set(oklabToSrgb(srcPick[i], srcPick[i + 1], srcPick[i + 2]), i);
  const bright = []; // like the first pass: blown reference pixels (sky, sand) say nothing about the grade
  for (let i = 0; i < refLab.length; i += 3) if (refLab[i] < 0.93) bright.push(refLab[i], refLab[i + 1], refLab[i + 2]);
  const refUse = bright.length > refLab.length * 0.2 ? Float64Array.from(bright) : refLab;
  const ref = refineToReference(first.prim, srcRgb, pick(refUse, 3000), { freeze: v2 ? first.report.absent : [], freezeTone: v3 });
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
  return withBal([...looks, ...['natural', 'faithful', 'bold'].map((mode) => {
    let prim = scalePrim(refined, STRENGTH[mode]);
    if (v2) prim = neutralBalance(prim, srcLab, refUse, NEUTRAL[mode]).prim;
    if (white) { prim = whitePoint(prim, srcLab, refLab, WHITE[mode]).prim; prim.whiteHold = 1; } // refLab: refUse drops the brightest pixels
    prim.skin = MODES[mode].skin;
    return { prim, report: { ...first.report, mode, label: MODES[mode].label, distance: { first: ref.before, refined: ref.after }, structureBack: back } };
  })]);
}

/**
 * 膚色偏向參考圖, automatic: a reference whose WHOLE picture carries one strong colour (blue hour: everything blue,
 * mean colour ≈ 0.09–0.13) and almost no warm tones means faces in that light take the colour too — skin protection
 * would leave a warm face pasted into a blue scene. Ordinary looks (film, teal / orange, a green forest: mean colour
 * ≤ 0.06, or with warm tones) keep natural skin. lab: OKLab triples (a, b unscaled). Returns 0..0.85.
 */
export function autoSkinFollow(lab) {
  let sa = 0, sb = 0, warm = 0, n = 0;
  for (let i = 0; i < lab.length; i += 3) {
    const L = lab[i], a = lab[i + 1], b = lab[i + 2], C = Math.hypot(a, b), h = ((Math.atan2(b, a) * 180) / Math.PI + 360) % 360;
    sa += a; sb += b; n++;
    if (L > 0.3 && L < 0.9 && C > 0.03 && h >= 20 && h <= 95) warm++; // skin-like warm colours (skin, wood, warm light)
  }
  if (!n) return 0;
  const sm = (lo, hi, x) => { const t = Math.min(1, Math.max(0, (x - lo) / (hi - lo))); return t * t * (3 - 2 * t); };
  return +(0.85 * sm(0.06, 0.1, Math.hypot(sa / n, sb / n)) * (1 - sm(0.05, 0.15, warm / n))).toFixed(3);
}
