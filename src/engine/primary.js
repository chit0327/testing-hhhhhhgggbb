// Pro grading stage, modelled on a colour page: Lift / Gamma / Gain / Offset wheels, RGB + luma curves and
// Hue vs Hue / Hue vs Sat / Hue vs Lum curves. Every operation depends on the colour only, so the whole stage
// is baked into the same 3D LUT as the rest of the look (deterministic, identical in preview and export, no
// flicker). Skin protection is NOT done by colour: beige coats and brick walls share skin's hue, so the
// renderer blends in a tone-only LUT (protectedPrim) inside the AI skin mask instead. The parameters are plain data so that the auto-fit and
// Claude can set them, and the user can read and change them.
import { srgbToOklab, oklabToSrgb } from './color.js';
import { GMAP_N, gradientAt } from './gradmap.js';

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
    bal: neutralBal(),                       // correction node, first of all: white balance as linear-light channel gains
    norm: neutralNorm(),                     // first node: LOG / flat-profile restoration (see estimateNorm)
    // automatic local grading (see local.js): per AI region an OKLab adjustment applied after the grade.
    // The renderer bakes one LUT per region and blends them by the region masks.
    local: null,                             // {skin, hair, clothes, background: {L, a, b, sat}} or null
    localAmt: 1,                             // strength of the local grade (pro panel)
    skinFollow: 0,                           // 0..1 膚色偏向參考圖色調: 0 = protected natural skin, 1 = skin takes the grade's colour fully
    keep: 0,                                 // 0..1 share of the colour change removed (tone kept): region layers only
    whiteHold: 0,                            // 0..1: the region adjustment fades out on whites (they keep the white point)
    skinLine: 1,                             // skin protection: 1 = skin follows the grade within natural skin tones; 0 = keeps its original colour
    post: null,                              // {L, a, b, sat}: the adjustment of the layer being baked
    gmap: null,                              // 漸層映射 {a: [17], b: [17], amt 0..1}: colour moves towards the gradient's colour at its lightness (last step)
  };
}

/**
 * Restoration of LOG / flat-profile footage to a normal display image, the "CST node" before the creative
 * grade. amt 0..1 blends it in; m: source median lightness, Tm: where the median goes, s: midtone slope,
 * sat: chroma factor. The tone map is a two-sided tanh around the median, so contrast is restored in the
 * midtones while the ends roll off softly towards lo/hi (no clipping, a filmic S-curve).
 */
/**
 * Correction (balance) of the source before anything else: white balance as linear-light channel gains,
 * the colourist's first node, so the look is laid on a neutral image rather than on the shot's own cast.
 */
export function neutralBal() { return { r: 1, g: 1, b: 1 }; }
const isNeutralBal = (q) => !q || (q.r === 1 && q.g === 1 && q.b === 1);
const toLin = (v) => (v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4));
const toEnc = (v) => (v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055);
/** One pixel through the balance (sRGB 0..1 in / out, clipped). */
export function applyBal(q, r, g, b, out) {
  out[0] = Math.min(1, toEnc(toLin(Math.max(0, r)) * q.r)); out[1] = Math.min(1, toEnc(toLin(Math.max(0, g)) * q.g)); out[2] = Math.min(1, toEnc(toLin(Math.max(0, b)) * q.b));
  return out;
}
export const NORM_LO = 0.05, NORM_HI = 0.995;
export function neutralNorm() { return { amt: 0, m: 0.5, Tm: 0.5, s: 1, sat: 1 }; }
export function normTone(n, L) {
  const y = n.Tm + n.s * (L - n.m);
  const g = y >= n.Tm ? n.Tm + (NORM_HI - n.Tm) * Math.tanh((y - n.Tm) / (NORM_HI - n.Tm))
    : n.Tm - (n.Tm - NORM_LO) * Math.tanh((n.Tm - y) / (n.Tm - NORM_LO));
  return L + n.amt * (g - L);
}
const isNeutralNorm = (n) => !n || !n.amt;
const isNeutralPost = (q) => !q || (!q.L && !q.a && !q.b && !q.sat);
export const LOCAL_REGIONS = ['skin', 'hair', 'clothes', 'background'];
export const LOCAL_NAMES = { skin: '皮膚', hair: '頭髮', clothes: '衣服', background: '背景' };

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

// OKLab hue range of natural skin (vectorscope skin line ≈ 55–60°). 80° let skin turn yellow-green (sallow, 'a bit
// cyan') under green-tinted looks; 70° keeps the warm–olive range
export const SKIN_LINE = [(35 * Math.PI) / 180, (70 * Math.PI) / 180];
/**
 * A graded skin hue (radians) moved into the skin line: unchanged inside; a little outside (≤ 45°, e.g. an olive film
 * look) → the nearest end; far outside (a cyan / teal grade: its direction means nothing for skin) → the skin's own
 * original hue `own`, itself kept inside the line (the classic teal look keeps skin warm).
 */
export function skinHue(h, own = (SKIN_LINE[0] + SKIN_LINE[1]) / 2) {
  const d = (x, y) => Math.abs(Math.atan2(Math.sin(x - y), Math.cos(x - y))), inside = (x) => d(x, (SKIN_LINE[0] + SKIN_LINE[1]) / 2) <= (SKIN_LINE[1] - SKIN_LINE[0]) / 2;
  const nearest = (x) => (d(x, SKIN_LINE[0]) < d(x, SKIN_LINE[1]) ? SKIN_LINE[0] : SKIN_LINE[1]);
  if (inside(h)) return h;
  if (Math.min(d(h, SKIN_LINE[0]), d(h, SKIN_LINE[1])) <= Math.PI / 4) return nearest(h);
  return inside(own) ? own : nearest(own);
}
const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
export function isNeutralPrim(p) {
  if (!p) return true;
  const w = (x) => !x.m && !x.a;
  return isNeutralBal(p.bal) && isNeutralNorm(p.norm) && !p.keep && isNeutralPost(p.post) && w(p.lift) && w(p.gamma) && w(p.gain) && w(p.offset) && !p.sat
    && [p.luma, p.red, p.green, p.blue].every(isIdentityCurve)
    && [p.hueHue, p.hueSat, p.hueLum].every((v) => v.every((x) => !x)) && !(p.gmap && p.gmap.amt);
}

/**
 * Prepare closures for one parameter set (called once per LUT bake). opts.memo(prim) may return a Map shared
 * by the LUTs baked for one render: the main LUT and the region layers evaluate the same base grade at the
 * same grid nodes, so each plain (non-layer) grade is computed once per input colour. Same results.
 */
export function compilePrim(p, opts = {}) {
  if (isNeutralPrim(p)) return null;
  if (p.keep || !isNeutralPost(p.post)) { // region layer: the grade, part of its colour change removed, then the region's own adjustment
    // prot: the tone-only grade (identity when the grade changes colour only — skin then keeps its colour exactly)
    const base = { ...p, keep: 0, post: null }, full = compilePrim(base, opts);
    const prot = p.keep ? compilePrim(protectedPrim(base)) || ((r, g, b, o) => { o[0] = r; o[1] = g; o[2] = b; return o; }) : null;
    const k = p.keep || 0, line = p.skinLine !== 0, q = p.post || {}, hold = p.whiteHold || 0, satK = Math.pow(2, q.sat || 0), o1 = [0, 0, 0], o2 = [0, 0, 0], lb = [0, 0, 0], lp = [0, 0, 0];
    return function layer(r, g, b, out) {
      let own; // the skin's own hue (protected grade, before the skin line)
      const c1 = full ? full(r, g, b, o1) : [r, g, b];
      srgbToOklab(c1[0], c1[1], c1[2], lb);
      const gA = lb[1], gB = lb[2]; // the grade's own colour, before skin protection
      if (prot) {
        const c2 = prot(r, g, b, o2); srgbToOklab(c2[0], c2[1], c2[2], lp);
        if (line) { // skin follows the grade, pulled back only as far as natural skin tones (the vectorscope skin line)
          const Cp = Math.hypot(lp[1], lp[2]), Cg = Math.hypot(lb[1], lb[2]);
          const hp = Math.atan2(lp[2], lp[1]), hg = Cg > 0.01 ? Math.atan2(lb[2], lb[1]) : hp;
          own = hp;
          const h = skinHue(hg, hp), C = Math.min(1.25 * Cp, Math.max(0.7 * Cp, Cg));
          lp[0] = lb[0]; lp[1] = C * Math.cos(h); lp[2] = C * Math.sin(h);
        }
        for (let i = 0; i < 3; i++) lb[i] += (lp[i] - lb[i]) * k;
      }
      // whiteHold: whites (bright, near-neutral after the grade) keep the white point; the region tint fades out on them
      const wk = hold ? 1 - hold * smooth(0.7, 0.88, lb[0]) * (1 - smooth(0.03, 0.08, Math.hypot(lb[1], lb[2]))) : 1;
      let A = lb[1] * satK + (q.a || 0) * wk, B = lb[2] * satK + (q.b || 0) * wk;
      if (prot && line) {
        // skin: a partial keep (忠實 / 濃烈) and the region's own adjustment may move it, but never outside natural skin
        // tones, and never drained to grey (a grade pulling hard towards cyan, mixed 35 % back, lands near grey — which
        // reads as cyan next to the graded surroundings)
        const Cp = Math.hypot(lp[1], lp[2]), C0 = Math.hypot(A, B), C = Math.max(C0, 0.6 * Cp);
        const h = skinHue(C0 > 0.005 ? Math.atan2(B, A) : Math.atan2(lp[2], lp[1]), own);
        if (C > 0.005) { A = C * Math.cos(h); B = C * Math.sin(h); }
      }
      if (prot && p.skinFollow) { // 膚色偏向參考圖色調: skin moves towards the colour the grade gives it unprotected
        const f = p.skinFollow;
        A += (gA * satK + (q.a || 0) * wk - A) * f; B += (gB * satK + (q.b || 0) * wk - B) * f;
      }
      return oklabToSrgb(Math.min(1, Math.max(0, lb[0] + (q.L || 0))), A, B, out);
    };
  }
  if (opts.memo) { // plain grade, memoised per input colour (nested maps: number keys, no string building)
    const raw = compilePrim(p), memo = opts.memo({ ...p, keep: 0, post: null });
    return function memoised(r, g, b, out) {
      let mr = memo.get(r); if (!mr) { mr = new Map(); memo.set(r, mr); }
      let mg = mr.get(g); if (!mg) { mg = new Map(); mr.set(g, mg); }
      let v = mg.get(b);
      if (!v) { raw(r, g, b, out); v = [out[0], out[1], out[2]]; mg.set(b, v); }
      out[0] = v[0]; out[1] = v[1]; out[2] = v[2]; return out;
    };
  }
  const L = wheelRGB(p.lift, 'lift'), G = wheelRGB(p.gamma, 'gamma'), K = wheelRGB(p.gain, 'gain'), O = wheelRGB(p.offset, 'offset');
  const expo = G.map((g) => Math.pow(2, -g));
  const rgbCurves = [p.red, p.green, p.blue].map((c) => (isIdentityCurve(c) ? null : curveFn(c)));
  const luma = isIdentityCurve(p.luma) ? null : curveFn(p.luma);
  const hueOn = [p.hueHue, p.hueSat, p.hueLum].some((v) => v.some((x) => x));
  const satK = Math.pow(2, p.sat || 0);
  const lab = [0, 0, 0], rgb = [0, 0, 0];
  const norm = isNeutralNorm(p.norm) ? null : p.norm, normSat = norm ? 1 + norm.amt * (norm.sat - 1) : 1;
  /** sRGB 0..1 in -> sRGB 0..1 out (out array). */
  const bal = isNeutralBal(p.bal) ? null : p.bal, cb = [0, 0, 0];
  const gm = p.gmap && p.gmap.amt ? p.gmap : null, gab = [0, 0];
  return function apply(r, g, b, out) {
    if (bal) { applyBal(bal, r, g, b, cb); r = cb[0]; g = cb[1]; b = cb[2]; } // correction before restoration and grade
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
    if (gm) { // 漸層映射, colour mode: lightness kept, colour towards the gradient's colour at this lightness
      gradientAt(gm, lab[0], gab);
      lab[1] += gm.amt * (gab[0] - lab[1]); lab[2] += gm.amt * (gab[1] - lab[2]);
    }
    return oklabToSrgb(Math.min(1, Math.max(0, lab[0])), lab[1], lab[2], out);
  };
}

/** The same grade without its colour changes (tone only, restoration kept): what protected skin receives. */
export function protectedPrim(p) {
  const q = sanitizePrim({ ...p, keep: 0, post: null });
  for (const k of ['lift', 'gamma', 'gain', 'offset']) q[k] = { m: q[k].m, h: q[k].h, a: 0 };
  q.red = CURVE_X.slice(); q.green = CURVE_X.slice(); q.blue = CURVE_X.slice();
  q.gmap = null; // the gradient map is a colour change: protected skin does not receive it
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
  const adj = (v) => (v && typeof v === 'object' ? { L: num(v.L, -0.1, 0.1, 0), a: num(v.a, -0.06, 0.06, 0), b: num(v.b, -0.06, 0.06, 0), sat: num(v.sat, -0.6, 0.6, 0) } : null);
  if (q.local && typeof q.local === 'object') {
    p.local = {};
    for (const r of LOCAL_REGIONS) p.local[r] = adj(q.local[r]) || { L: 0, a: 0, b: 0, sat: 0 };
  }
  p.localAmt = num(q.localAmt, 0, 1.5, 1);
  p.bal = q.bal && typeof q.bal === 'object' ? { r: num(q.bal.r, 0.5, 2, 1), g: num(q.bal.g, 0.5, 2, 1), b: num(q.bal.b, 0.5, 2, 1) } : neutralBal();
  p.keep = num(q.keep, 0, 1, 0);
  p.whiteHold = num(q.whiteHold, 0, 1, 0);
  p.skinLine = q.skinLine === 0 ? 0 : 1;
  p.skinFollow = num(q.skinFollow, 0, 1, 0);
  p.post = adj(q.post);
  const gq = q.gmap;
  if (gq && typeof gq === 'object' && Array.isArray(gq.a) && Array.isArray(gq.b) && gq.a.length === GMAP_N && gq.b.length === GMAP_N)
    p.gmap = { a: gq.a.map((v) => num(v, -0.3, 0.3, 0)), b: gq.b.map((v) => num(v, -0.3, 0.3, 0)), amt: num(gq.amt, 0, 1, 0) };
  return p;
}
