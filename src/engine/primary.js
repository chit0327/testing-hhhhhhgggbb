// Pro grading stage, modelled on a colour page: Lift / Gamma / Gain / Offset wheels, RGB + luma curves and
// Hue vs Hue / Hue vs Sat / Hue vs Lum curves. Every operation depends on the colour only, so the whole stage
// is baked into the same 3D LUT as the rest of the look (deterministic, identical in preview and export, no
// flicker). Skin protection is NOT done by colour: beige coats and brick walls share skin's hue, so the
// renderer blends in a tone-only LUT (protectedPrim) inside the AI skin mask instead. The parameters are plain data so that the auto-fit and
// Claude can set them, and the user can read and change them.
import { srgbToOklab, oklabToSrgb } from './color.js';

/** OKLab hue (degrees) of the eight hue-curve anchors and their names. */
export const HUE_ANCHORS = [30, 70, 100, 145, 195, 255, 300, 340];
export const HUE_NAMES = ['紅', '橙', '黃', '綠', '青', '藍', '紫', '洋紅'];
export const CURVE_X = [0, 0.25, 0.5, 0.75, 1];

/** Neutral parameters. Wheels: master (luminance) + colour as OKLab hue (deg) and amount 0..1. */
export function neutralPrim() {
  const wheel = () => ({ m: 0, h: 0, a: 0 });
  return {
    lift: wheel(), gamma: wheel(), gain: wheel(), offset: wheel(),
    luma: CURVE_X.slice(), red: CURVE_X.slice(), green: CURVE_X.slice(), blue: CURVE_X.slice(),
    sat: 0,                                  // global saturation, log2 (−1 = ×0.5, +1 = ×2)
    hueHue: new Array(8).fill(0),            // degrees, ±60
    hueSat: new Array(8).fill(0),            // log2 factor, ±1
    hueLum: new Array(8).fill(0),            // OKLab L, ±0.15
    skin: 0,                                 // 0..1: inside the AI skin mask, how much skin keeps its colour (renderer)
    norm: neutralNorm(),                     // first node: LOG / flat-profile restoration (see estimateNorm)
  };
}

/**
 * Restoration of LOG / flat-profile footage to a normal display image, the "CST node" before the creative
 * grade. amt 0..1 blends it in; m: source median lightness, Tm: where the median goes, s: midtone slope,
 * sat: chroma factor. The tone map is a two-sided tanh around the median, so contrast is restored in the
 * midtones while the ends roll off softly towards lo/hi (no clipping, a filmic S-curve).
 */
export const NORM_LO = 0.05, NORM_HI = 0.995;
export function neutralNorm() { return { amt: 0, m: 0.5, Tm: 0.5, s: 1, sat: 1 }; }
export function normTone(n, L) {
  const y = n.Tm + n.s * (L - n.m);
  const g = y >= n.Tm ? n.Tm + (NORM_HI - n.Tm) * Math.tanh((y - n.Tm) / (NORM_HI - n.Tm))
    : n.Tm - (n.Tm - NORM_LO) * Math.tanh((n.Tm - y) / (n.Tm - NORM_LO));
  return L + n.amt * (g - L);
}
const isNeutralNorm = (n) => !n || !n.amt;

const WHEEL_SCALE = { lift: 0.1, gamma: 0.25, gain: 0.15, offset: 0.08 }; // channel deviation at amount 1
const MASTER_SCALE = { lift: 0.25, gamma: 1, gain: 0.5, offset: 0.2 };

// luminance-neutral RGB direction of an OKLab hue (so colour on a wheel does not change brightness)
const dirCache = new Map();
export function hueDirection(h) {
  const k = Math.round(((h % 360) + 360) % 360);
  if (dirCache.has(k)) return dirCache.get(k);
  const t = (k * Math.PI) / 180, g = oklabToSrgb(0.6, 0, 0), c = oklabToSrgb(0.6, 0.08 * Math.cos(t), 0.08 * Math.sin(t));
  const d = [c[0] - g[0], c[1] - g[1], c[2] - g[2]], y = 0.2126 * d[0] + 0.7152 * d[1] + 0.0722 * d[2];
  for (let i = 0; i < 3; i++) d[i] -= y;
  const mx = Math.max(Math.abs(d[0]), Math.abs(d[1]), Math.abs(d[2])) || 1;
  const u = d.map((v) => v / mx);
  dirCache.set(k, u);
  return u;
}
function wheelRGB(w, name) {
  const u = hueDirection(w.h), s = (w.a || 0) * WHEEL_SCALE[name], m = (w.m || 0) * MASTER_SCALE[name];
  return [m + s * u[0], m + s * u[1], m + s * u[2]];
}

/** Monotone cubic (Fritsch–Carlson) through the 5 fixed knots; y values are made non-decreasing. */
export function curveFn(ys) {
  const x = CURVE_X, n = x.length, y = ys.slice();
  for (let i = 1; i < n; i++) y[i] = Math.max(y[i], y[i - 1]);
  const d = [], m = new Array(n);
  for (let i = 0; i < n - 1; i++) d.push((y[i + 1] - y[i]) / (x[i + 1] - x[i]));
  m[0] = d[0]; m[n - 1] = d[n - 2];
  for (let i = 1; i < n - 1; i++) m[i] = d[i - 1] * d[i] <= 0 ? 0 : (d[i - 1] + d[i]) / 2;
  for (let i = 0; i < n - 1; i++) {
    if (d[i] === 0) { m[i] = 0; m[i + 1] = 0; continue; }
    const a = m[i] / d[i], b = m[i + 1] / d[i], s = a * a + b * b;
    if (s > 9) { const t = 3 / Math.sqrt(s); m[i] = t * a * d[i]; m[i + 1] = t * b * d[i]; }
  }
  return (v) => {
    if (v <= 0) return y[0] + m[0] * v;
    if (v >= 1) return y[n - 1] + m[n - 1] * (v - 1);
    let i = Math.min(n - 2, Math.floor(v * (n - 1)));
    const hh = x[i + 1] - x[i], t = (v - x[i]) / hh, t2 = t * t, t3 = t2 * t;
    return (2 * t3 - 3 * t2 + 1) * y[i] + (t3 - 2 * t2 + t) * hh * m[i] + (-2 * t3 + 3 * t2) * y[i + 1] + (t3 - t2) * hh * m[i + 1];
  };
}
const isIdentityCurve = (ys) => ys.every((v, i) => Math.abs(v - CURVE_X[i]) < 1e-6);

/** Smooth periodic interpolation of the 8 anchor values at hue h (degrees). */
export function hueInterp(vals, h) {
  h = ((h % 360) + 360) % 360;
  const A = HUE_ANCHORS, n = A.length;
  for (let i = 0; i < n; i++) {
    const a0 = A[i], a1 = i === n - 1 ? A[0] + 360 : A[i + 1];
    const hh = h < a0 ? h + 360 : h;
    if (hh >= a0 && hh < a1) { const t = (hh - a0) / (a1 - a0), s = t * t * (3 - 2 * t); return vals[i] + (vals[(i + 1) % n] - vals[i]) * s; }
  }
  return vals[0];
}

const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
export function isNeutralPrim(p) {
  if (!p) return true;
  const w = (x) => !x.m && !x.a;
  return isNeutralNorm(p.norm) && w(p.lift) && w(p.gamma) && w(p.gain) && w(p.offset) && !p.sat
    && [p.luma, p.red, p.green, p.blue].every(isIdentityCurve)
    && [p.hueHue, p.hueSat, p.hueLum].every((v) => v.every((x) => !x));
}

/** Prepare closures for one parameter set (called once per LUT bake). */
export function compilePrim(p) {
  if (isNeutralPrim(p)) return null;
  const L = wheelRGB(p.lift, 'lift'), G = wheelRGB(p.gamma, 'gamma'), K = wheelRGB(p.gain, 'gain'), O = wheelRGB(p.offset, 'offset');
  const expo = G.map((g) => Math.pow(2, -g));
  const rgbCurves = [p.red, p.green, p.blue].map((c) => (isIdentityCurve(c) ? null : curveFn(c)));
  const luma = isIdentityCurve(p.luma) ? null : curveFn(p.luma);
  const hueOn = [p.hueHue, p.hueSat, p.hueLum].some((v) => v.some((x) => x));
  const satK = Math.pow(2, p.sat || 0);
  const lab = [0, 0, 0], rgb = [0, 0, 0];
  const norm = isNeutralNorm(p.norm) ? null : p.norm, normSat = norm ? 1 + norm.amt * (norm.sat - 1) : 1;
  /** sRGB 0..1 in -> sRGB 0..1 out (out array). */
  return function apply(r, g, b, out) {
    if (norm) { // restoration first, like a CST node ahead of the grade
      srgbToOklab(r, g, b, lab);
      oklabToSrgb(normTone(norm, lab[0]), lab[1] * normSat, lab[2] * normSat, rgb);
      for (let c = 0; c < 3; c++) rgb[c] = Math.min(1, Math.max(0, rgb[c]));
    } else { rgb[0] = r; rgb[1] = g; rgb[2] = b; }
    for (let c = 0; c < 3; c++) { // Lift / Gain, Gamma, Offset (per channel, display-referred like a colour page)
      let v = rgb[c] * (1 + K[c]) + L[c] * (1 - rgb[c]);
      v = v > 0 ? Math.pow(v, expo[c]) : v;
      v += O[c];
      if (rgbCurves[c]) v = rgbCurves[c](v);
      rgb[c] = v;
    }
    // display clip, as on a colour page: a channel pushed past white clips while the others move, so gain can tint white
    srgbToOklab(Math.min(1, Math.max(0, rgb[0])), Math.min(1, Math.max(0, rgb[1])), Math.min(1, Math.max(0, rgb[2])), lab);
    if (luma) lab[0] = luma(lab[0]);
    lab[1] *= satK; lab[2] *= satK;
    if (hueOn) {
      const C = Math.hypot(lab[1], lab[2]);
      if (C > 1e-5) {
        const h0 = (Math.atan2(lab[2], lab[1]) * 180) / Math.PI, wc = smooth(0.005, 0.03, C); // greys stay grey
        const h = h0 + wc * hueInterp(p.hueHue, h0), cs = C * Math.pow(2, wc * hueInterp(p.hueSat, h0));
        lab[0] += wc * hueInterp(p.hueLum, h0);
        lab[1] = cs * Math.cos((h * Math.PI) / 180); lab[2] = cs * Math.sin((h * Math.PI) / 180);
      }
    }
    return oklabToSrgb(Math.min(1, Math.max(0, lab[0])), lab[1], lab[2], out);
  };
}

/** The same grade without its colour changes (tone only, restoration kept): what protected skin receives. */
export function protectedPrim(p) {
  const q = sanitizePrim(p);
  for (const k of ['lift', 'gamma', 'gain', 'offset']) q[k] = { m: q[k].m, h: q[k].h, a: 0 };
  q.red = CURVE_X.slice(); q.green = CURVE_X.slice(); q.blue = CURVE_X.slice();
  q.sat = 0; q.hueHue = new Array(8).fill(0); q.hueSat = new Array(8).fill(0); q.hueLum = new Array(8).fill(0);
  return q;
}

/** Clamp / repair parameters from any source (UI, auto-fit, Claude). Unknown keys dropped. */
export function sanitizePrim(q) {
  const p = neutralPrim();
  if (!q || typeof q !== 'object') return p;
  const num = (v, lo, hi, d) => { const x = Number(v); return Number.isFinite(x) ? Math.min(hi, Math.max(lo, x)) : d; };
  for (const k of ['lift', 'gamma', 'gain', 'offset']) {
    const w = q[k] || {};
    p[k] = { m: num(w.m, -1, 1, 0), h: ((num(w.h, -1e4, 1e4, 0) % 360) + 360) % 360, a: num(w.a, 0, 1, 0) };
  }
  for (const k of ['luma', 'red', 'green', 'blue']) if (Array.isArray(q[k]) && q[k].length === 5) p[k] = q[k].map((v, i) => num(v, -0.2, 1.2, CURVE_X[i]));
  p.sat = num(q.sat, -1, 1, 0);
  const arr = (v, lo, hi) => (Array.isArray(v) && v.length === 8 ? v.map((x) => num(x, lo, hi, 0)) : new Array(8).fill(0));
  p.hueHue = arr(q.hueHue, -60, 60); p.hueSat = arr(q.hueSat, -1, 1); p.hueLum = arr(q.hueLum, -0.15, 0.15);
  p.skin = num(q.skin, 0, 1, 0);
  if (q.norm && typeof q.norm === 'object') {
    const n = q.norm;
    p.norm = { amt: num(n.amt, 0, 1, 0), m: num(n.m, 0.05, 0.95, 0.5), Tm: num(n.Tm, 0.2, 0.8, 0.5), s: num(n.s, 0.5, 3, 1), sat: num(n.sat, 0.5, 5, 1) };
  }
  return p;
}
