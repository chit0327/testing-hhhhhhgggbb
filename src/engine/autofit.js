// Deterministic estimation of the pro-stage parameters from a reference: the "colourist's first pass".
//   tone   — luma curve from lightness quantiles (blown reference highlights ignored, change capped:
//            a reference's sky must not relight a street),
//   colour — shadow / midtone / highlight casts → Lift / Gamma / Gain wheels, solved iteratively by
//            rendering sample pixels through the real pipeline and correcting the residual (Newton steps
//            with a measured 2×2 response per wheel),
//   hues   — per hue anchor: rotate towards the reference's nearby hue, match its saturation, and mute
//            colours the reference does not have (a warm wall under a teal reference turns grey-green),
//   skin   — protection on, stronger for the natural interpretation.
// Three interpretations differ only in how far they go. Same input → same output, always.
import { oklabToSrgb } from './color.js';
import { neutralPrim, compilePrim, curveFn, HUE_ANCHORS, CURVE_X } from './primary.js';

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

/** srcLab, refLab: Float64Array of OKLab triples (reference already limited to its central area). */
export function fitPrim(srcLab, refLab, modeName = 'natural', { iters = 6, sample = 2500 } = {}) {
  const md = MODES[modeName];
  const p = neutralPrim();
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
  // keep every segment's slope in [0.45, 2]: no flattened (detail-losing) or step-like (banding) tone ranges
  for (let i = 1; i < 5; i++) p.luma[i] = clamp(p.luma[i], p.luma[i - 1] + 0.45 * 0.25, p.luma[i - 1] + 2 * 0.25);
  if (p.luma[4] > 1) { const k = (1 - p.luma[0]) / (p.luma[4] - p.luma[0]); for (let i = 1; i < 5; i++) p.luma[i] = p.luma[0] + (p.luma[i] - p.luma[0]) * k; }
  const lumaF = curveFn(p.luma); p.luma = CURVE_X.map((x) => lumaF(x)); // store the repaired (monotone) knots
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
    p.hueSat[i] = clamp((Math.log2(Math.max(r.chroma, 1e-3) / Math.max(s.chroma, 1e-3)) - Math.log2(k)) * md.color, -0.7, 0.7);
  }
  p.skin = md.skin;
  // ---- tonal casts → wheels (iterative, through the real pipeline)
  const n = Math.min(sample, srcLab.length / 3), stride = Math.max(1, Math.floor(srcLab.length / 3 / n));
  const rgb = [], band = [], Ls = [];
  for (let i = 0, j = 0; j < n; i += 3 * stride, j++) { rgb.push(oklabToSrgb(srcLab[i], srcLab[i + 1], srcLab[i + 2])); Ls.push(srcLab[i]); }
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
  return { prim: p, report: { mode: modeName, label: md.label, satRatio: k, targetBands: target, achievedBands: achieved, residual: resid } };
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

/** All three interpretations, natural first. */
export function fitCandidates(srcLab, refLab) {
  return ['natural', 'faithful', 'bold'].map((m) => fitPrim(srcLab, refLab, m));
}
