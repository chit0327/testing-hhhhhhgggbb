// 依深度分層匹配: the background matched to the reference layer by layer — near background and far background — instead
// of one colour for the whole picture. A reference whose distance is light and misty while its foreground is darker
// got one average before; now the far background of the video is moved towards the reference's far background and the
// near background towards its near background.
//
// Layers: people are left out (AI person mask: they keep the global grade and the skin protection — matched by depth
// alone, a reference subject in a black jacket darkened the subject of the video: clothing is content, not grade). The
// rest is split by depth relative to that picture's own background (its 10th…90th depth percentile): near below the
// middle, far above, with a soft overlap. Depth: the AI depth model (depth.js), 0..1 with 1 = near.
//
// Per layer, measured in OKLab after the global grade, towards the reference's same layer: brightness (mean L, capped
// ±0.12), contrast (spread of L: a misty distance is flat; ×0.6…1.3), tint (a / b of the less colourful half — greys,
// whites — so that colourful content does not count; capped ±0.03) and colourfulness — how far colours stray from the
// layer's cast, matched as a curve over its percentiles (a near-monochrome reference distance pulls red bicycles and a
// purple sign towards the cast; the greys hardly move) and colour strength at each brightness (a hazy distance has
// nearly grey brights: ×0.3…1.2 per band). Layers smaller than
// 5 % of either picture are left alone. fit = raw (strength 1); the strength is applied when rendering.
import { srgbToOklab, oklabToSrgb, S2L, l2sFast } from './color.js';

export const LAYERS = ['near', 'far'];
const LB = [0.3, 0.45, 0.6, 0.75]; // lightness bands (edges) for colour strength by brightness
const lband = (L) => { let i = 0; while (i < LB.length && L >= LB[i]) i++; return i; };
const CQ = [0.25, 0.5, 0.75, 0.9, 0.97, 0.995]; // percentiles of the distance from the layer's cast, matched as a curve (outliers come down most)
export const LAYER_NAMES = { near: '近景', far: '遠景' };
const MIN_SHARE = 0.05;
const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

const gridAt = (g, gw, gh, fx, fy) => { // bilinear on a grid, (fx, fy) in 0..1
  const x = Math.min(gw - 1, Math.max(0, fx * gw - 0.5)), y = Math.min(gh - 1, Math.max(0, fy * gh - 0.5));
  const x0 = x | 0, y0 = y | 0, x1 = Math.min(gw - 1, x0 + 1), y1 = Math.min(gh - 1, y0 + 1), tx = x - x0, ty = y - y0;
  return (g[y0 * gw + x0] * (1 - tx) + g[y0 * gw + x1] * tx) * (1 - ty) + (g[y1 * gw + x0] * (1 - tx) + g[y1 * gw + x1] * tx) * ty;
};
/** dn: {data 0..1 (1 = near), w, h}; seg: {person: grid, mw, mh} | null → per-pixel sampler of [person, near, far] weights */
export function layerSampler(dn, seg) {
  const P = seg && seg.person ? seg : null;
  // the background's own depth range (10th…90th percentile of the depth where there is no person)
  const v = [];
  for (let y = 0; y < dn.h; y++) for (let x = 0; x < dn.w; x++) {
    const p = P ? gridAt(P.person, P.mw, P.mh, (x + 0.5) / dn.w, (y + 0.5) / dn.h) : 0;
    if (p < 0.3) v.push(dn.data[y * dn.w + x]);
  }
  v.sort((a, b) => a - b);
  const lo = v.length ? v[Math.floor(v.length * 0.1)] : 0, hi = v.length ? v[Math.floor(v.length * 0.9)] : 1, span = Math.max(0.05, hi - lo);
  return (fx, fy, out = [0, 0, 0]) => {
    const p = P ? Math.min(1, Math.max(0, gridAt(P.person, P.mw, P.mh, fx, fy))) : 0, t = (gridAt(dn.data, dn.w, dn.h, fx, fy) - lo) / span;
    const near = smooth(0.35, 0.65, t); // (t: 0 = far end of the background, 1 = near end)
    out[0] = p; out[1] = (1 - p) * near; out[2] = (1 - p) * (1 - near);
    return out;
  };
}

/**
 * The depth of the background only: where the AI person mask (person grid mw wide) covers the depth grid, the depth is
 * filled in from the surrounding background (layer by layer inwards). For a depth reused over a few frames (a phone):
 * a person who moved left their old outline marked "near" in the background, and 分層匹配 darkened it — a ghost of the
 * head. 分層／同類別匹配 leave people out anyway, so they only need the background's depth. depth {data, w, h} → a new one.
 */
export function backgroundDepth(depth, person, mw) {
  const { w, h } = depth, mh = person.length / mw, d = Float32Array.from(depth.data), hole = new Uint8Array(w * h);
  let left = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (gridAt(person, mw, mh, (x + 0.5) / w, (y + 0.5) / h) > 0.2) { hole[y * w + x] = 1; left++; }
  if (!left || left === w * h) return { data: d, w, h };
  for (let i = 0, grow = Array.from(hole); i < w * h; i++) if (!hole[i] && ((i % w > 0 && grow[i - 1]) || (i % w < w - 1 && grow[i + 1]) || (i >= w && grow[i - w]) || (i < w * (h - 1) && grow[i + w]))) { hole[i] = 1; left++; } // (one cell more: the depth model's soft edge)
  let todo = []; for (let i = 0; i < w * h; i++) if (hole[i]) todo.push(i);
  while (todo.length) {
    const fill = [], rest = [];
    for (const i of todo) {
      const x = i % w; let s = 0, n = 0;
      if (x > 0 && !hole[i - 1]) { s += d[i - 1]; n++; } if (x < w - 1 && !hole[i + 1]) { s += d[i + 1]; n++; }
      if (i >= w && !hole[i - w]) { s += d[i - w]; n++; } if (i < w * (h - 1) && !hole[i + w]) { s += d[i + w]; n++; }
      if (n) fill.push(i, s / n); else rest.push(i);
    }
    if (!fill.length) break;
    for (let k = 0; k < fill.length; k += 2) { d[fill[k]] = fill[k + 1]; hole[fill[k]] = 0; }
    todo = rest;
  }
  return { data: d, w, h };
}

/** per layer: share, mean L, spread of L, tint (a, b of the less colourful half), mean chroma. rgba w×h. */
export function layerStats(rgba, w, h, dn, seg, step = 2) { return statsBy(rgba, w, h, layerSampler(dn, seg), LAYERS, step); }
/**
 * the same per-layer statistics for any split of the picture: sampler(fx, fy, out) gives [person, layer 1, layer 2, …]
 * weights, keys names the layers (同類別匹配 uses sky / plants / rest, classes.js)
 */
export function statsBy(rgba, w, h, at, keys, step = 2) {
  const wts = new Array(keys.length + 1).fill(0), lab = [0, 0, 0], px = Object.fromEntries(keys.map((k) => [k, []]));
  let total = 0;
  for (let y = 0; y < h; y += step) for (let x = 0; x < w; x += step) {
    at((x + 0.5) / w, (y + 0.5) / h, wts); total++;
    let k = null; for (let i = 0; i < keys.length; i++) if (wts[i + 1] > 0.6) { k = keys[i]; break; } // (clear members only)
    if (!k) continue;
    const p = 4 * (y * w + x); srgbToOklab(rgba[p] / 255, rgba[p + 1] / 255, rgba[p + 2] / 255, lab);
    px[k].push(lab[0], lab[1], lab[2]);
  }
  const out = {};
  for (const k of keys) {
    const v = px[k], n = v.length / 3;
    if (!n) { out[k] = { share: 0 }; continue; }
    let L = 0, L2 = 0, C = 0, ma = 0, mb = 0; const Cs = new Float64Array(n), Ds = new Float64Array(n);
    for (let i = 0; i < n; i++) { L += v[3 * i]; L2 += v[3 * i] ** 2; Cs[i] = Math.hypot(v[3 * i + 1], v[3 * i + 2]); C += Cs[i]; ma += v[3 * i + 1]; mb += v[3 * i + 2]; }
    ma /= n; mb /= n;
    // colour strength by brightness (5 lightness bands): a hazy reference has nearly grey brights and keeps its colour in
    // the darker parts; the same mean colour strength hid that (0.052 vs 0.056) while the video's glass and sky stayed cyan
    const bs = [0, 0, 0, 0, 0], bn = [0, 0, 0, 0, 0];
    for (let i = 0; i < n; i++) { const j = lband(v[3 * i]); bs[j] += Cs[i]; bn[j]++; }
    const Cb = bs.map((x, j) => (bn[j] >= Math.max(20, n * 0.03) ? x / bn[j] : null));
    // colourfulness = how far each colour strays from the layer's own cast (a graded picture's greys share its tint:
    // raw chroma could not tell a teal-grey wall from a red bicycle)
    for (let i = 0; i < n; i++) Ds[i] = Math.hypot(v[3 * i + 1] - ma, v[3 * i + 2] - mb);
    const sorted = Float64Array.from(Cs).sort(), med = sorted[n >> 1], ds = Float64Array.from(Ds).sort(), Cq = CQ.map((q) => ds[Math.min(n - 1, Math.floor(n * q))]);
    let ta = 0, tb = 0, tn = 0; for (let i = 0; i < n; i++) if (Cs[i] <= med) { ta += v[3 * i + 1]; tb += v[3 * i + 2]; tn++; }
    L /= n;
    out[k] = { share: n / total, L, sd: Math.sqrt(Math.max(0, L2 / n - L * L)), tintA: ta / tn, tintB: tb / tn, C: C / n, Cq, ma, mb, Cb };
  }
  return out;
}

/** the adjustment of each layer, src (after the global grade) → ref; raw (strength 1): {near, far: {m, dL, k, da, db, cs, cr} | null} */
export function fitLayers(src, ref) { return fitBy(src, ref, LAYERS); }
export function fitBy(src, ref, keys, minShare = MIN_SHARE) {
  const cl = (v, c) => Math.max(-c, Math.min(c, v)), out = {};
  for (const k of keys) {
    const S = src[k], R = ref[k];
    if (!S || !R || !(S.share >= minShare && R.share >= minShare)) { out[k] = null; continue; }
    out[k] = {
      m: +S.L.toFixed(4), dL: +cl(R.L - S.L, 0.12).toFixed(4),
      k: +Math.max(0.6, Math.min(1.3, R.sd / Math.max(S.sd, 1e-3))).toFixed(3),
      da: +cl(R.tintA - S.tintA, 0.03).toFixed(4), db: +cl(R.tintB - S.tintB, 0.03).toFixed(4),
      // colourfulness as a curve over its percentiles (src → ref): one factor for all (the mean: ×0.96, then the 90th
      // percentile: ×0.80) left red bicycles and a purple sign vivid in front of a near-monochrome reference distance —
      // the greys hardly move, the vivid objects must come down most. Kept rising (no colour order swapped) and between
      // ×0.25 and ×1.2 of the source at each point (adding colour is riskier than taking it away).
      ma: +S.ma.toFixed(4), mb: +S.mb.toFixed(4),
      cb: S.Cb.map((v, j) => (v && R.Cb[j] ? +Math.max(0.3, Math.min(1.2, R.Cb[j] / Math.max(v, 1e-3))).toFixed(3) : null)), cs: S.Cq.map((v) => +v.toFixed(4)), cr: S.Cq.map((v, i) => { const r = R.Cq[i]; return +Math.max(v * 0.25, Math.min(v * 1.2, r)).toFixed(4); }).map((v, i, a) => (i && v < a[i - 1] ? a[i - 1] : v)),
    };
  }
  return out;
}

/** colour strength factor at lightness L: by band (reference ÷ video), interpolated between band centres; 1 where unknown */
function bandGain(f, L) {
  const cb = f.cb; if (!cb) return 1;
  const centres = [0.2, 0.375, 0.525, 0.675, 0.85], g = cb.map((v) => v ?? 1);
  if (L <= centres[0]) return g[0];
  for (let i = 1; i < centres.length; i++) if (L <= centres[i]) { const t = (L - centres[i - 1]) / (centres[i] - centres[i - 1]); return g[i - 1] + (g[i] - g[i - 1]) * t; }
  return g[g.length - 1];
}
/** the factor on the distance from the layer's cast at distance C: the percentile curve (cs → cr), straight from 0, beyond the last point at its ratio */
function chromaGain(f, C) {
  const cs = f.cs, cr = f.cr;
  if (!cs || C < 1e-5) return f.sat ?? 1;
  let x0 = 0, y0 = 0;
  for (let i = 0; i < cs.length; i++) {
    if (C <= cs[i]) { const t = cs[i] > x0 ? (C - x0) / (cs[i] - x0) : 1; return (y0 + (cr[i] - y0) * t) / C; }
    x0 = cs[i]; y0 = cr[i];
  }
  // beyond the reference's most colourful level: compressed further (a red bicycle in a grey-teal distance)
  return x0 > 1e-5 ? (y0 + (C - x0) * Math.min(1, y0 / x0) * 0.35) / C : 1;
}
/** in place on RGBA: each pixel moved by its layers' adjustments (weights from depth and the person mask), × amt;
 * bgSat −1…0.5: the background's colour strength by hand (−1 = grey), people untouched */
export function applyLayers(img, w, h, dn, seg, fit, amt = 1, bgSat = 0) {
  if (!((amt > 0 && fit && LAYERS.some((k) => fit[k])) || bgSat)) return img;
  return applyBy(img, w, h, layerSampler(dn, seg), LAYERS, fit, amt, bgSat);
}
/**
 * keepVivid (同類別匹配): colours clearly more vivid than most of their class (beyond its 90th percentile of distance from
 * the class's cast, rising from its 75th) are content — a green sign, a red bicycle — and keep their colour strength;
 * only the greyer bulk is matched, and no colour loses more than 30 % of its strength. Without it a pale sea-side reference's 「建築地面」 took a green forest sign to a third of its colour.
 */
// The change a layer makes to a colour (at full weight) depends on the colour only, and the layers' changes add up
// (each is weighted by where the pixel is). So each layer's change is worked out once on a 33³ grid of colours and read
// per pixel by trilinear interpolation, instead of the curve maths per pixel per layer: 640 → ~115 ms a 1280×720 frame
// here (this ran on every exported frame; a phone took much longer). Output vs the per-pixel version: see layermatch.test.js.
const DN = 33, IX = Float32Array.from({ length: 256 }, (_, v) => (v * (DN - 1)) / 255);
const deltaCache = new WeakMap(); // fit layer object → {plain, vivid} tables
function layerDelta(f, lab, keepVivid, out) { // what the per-pixel version added at t = 1
  out[0] = (f.m + (lab[0] - f.m) * f.k + f.dL) - lab[0];
  const ea = lab[1] - f.ma, eb = lab[2] - f.mb, D = Math.hypot(ea, eb);
  let g = chromaGain(f, D), g2 = bandGain(f, lab[0]); // (towards / away from the layer's cast; colour strength at this brightness)
  if (keepVivid && f.cs) { // (from the class's most colourful quarter up: kept; and never below ×0.7 — a big sign is a quarter of its class)
    const v = smooth(f.cs[2], Math.max(f.cs[2] + 0.01, f.cs[3] * 1.4), D); g += (1 - g) * v; g2 += (1 - g2) * v;
    g = Math.max(0.7, g); g2 = Math.max(0.7, g2);
  }
  const na = lab[1] + ea * (g - 1), nb = lab[2] + eb * (g - 1);
  out[1] = f.da + na * g2 - lab[1]; out[2] = f.db + nb * g2 - lab[2];
  return out;
}
function deltaTable(f, keepVivid) {
  let c = deltaCache.get(f); if (!c) deltaCache.set(f, c = {});
  const k = keepVivid ? 'vivid' : 'plain';
  if (c[k]) return c[k];
  const T = new Float32Array(DN * DN * DN * 3), lab = [0, 0, 0], d = [0, 0, 0];
  for (let r = 0; r < DN; r++) for (let g = 0; g < DN; g++) for (let b = 0; b < DN; b++) {
    srgbToOklab(r / (DN - 1), g / (DN - 1), b / (DN - 1), lab); layerDelta(f, lab, keepVivid, d);
    const q = 3 * ((r * DN + g) * DN + b); T[q] = d[0]; T[q + 1] = d[1]; T[q + 2] = d[2];
  }
  return (c[k] = T);
}
function addDelta(T, x, y, z, t, acc) { // trilinear in the table at grid coordinates (x, y, z) = (r, g, b), × t, into acc
  const x0 = Math.min(DN - 2, x | 0), y0 = Math.min(DN - 2, y | 0), z0 = Math.min(DN - 2, z | 0), fx = x - x0, fy = y - y0, fz = z - z0;
  const o = 3 * ((x0 * DN + y0) * DN + z0), sz = 3, sy = 3 * DN, sx = 3 * DN * DN;
  const w000 = (1 - fx) * (1 - fy) * (1 - fz) * t, w001 = (1 - fx) * (1 - fy) * fz * t, w010 = (1 - fx) * fy * (1 - fz) * t, w011 = (1 - fx) * fy * fz * t;
  const w100 = fx * (1 - fy) * (1 - fz) * t, w101 = fx * (1 - fy) * fz * t, w110 = fx * fy * (1 - fz) * t, w111 = fx * fy * fz * t;
  for (let c = 0; c < 3; c++) {
    const q = o + c;
    acc[c] += w000 * T[q] + w001 * T[q + sz] + w010 * T[q + sy] + w011 * T[q + sy + sz] + w100 * T[q + sx] + w101 * T[q + sx + sz] + w110 * T[q + sx + sy] + w111 * T[q + sx + sy + sz];
  }
}
const _lin = [0, 0, 0];
function toSrgb(L, a, b, out) { // oklabToSrgb with the table-based gamma (the same gamut handling: less colour, same lightness)
  const lin = (aa, bb) => {
    const l1 = L + 0.3963377774 * aa + 0.2158037573 * bb, m1 = L - 0.1055613458 * aa - 0.0638541728 * bb, s1 = L - 0.0894841775 * aa - 1.2914855480 * bb, l = l1 * l1 * l1, m = m1 * m1 * m1, s = s1 * s1 * s1; // (x·x·x: ** is a general power, slow)
    _lin[0] = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s; _lin[1] = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s; _lin[2] = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;
    return _lin[0] >= -1e-4 && _lin[0] <= 1.0001 && _lin[1] >= -1e-4 && _lin[1] <= 1.0001 && _lin[2] >= -1e-4 && _lin[2] <= 1.0001;
  };
  if (!lin(a, b)) { let lo = 0, hi = 1; for (let k = 0; k < 12; k++) { const mid = (lo + hi) / 2; if (lin(a * mid, b * mid)) lo = mid; else hi = mid; } lin(a * lo, b * lo); }
  out[0] = l2sFast(_lin[0]); out[1] = l2sFast(_lin[1]); out[2] = l2sFast(_lin[2]);
  return out;
}
export function applyBy(img, w, h, at, keys, fit, amt = 1, bgSat = 0, { keepVivid = false } = {}) {
  const A = keys.map((k) => fit && fit[k]), extra = 1 + Math.max(-1, Math.min(0.5, bgSat || 0)); // 背景飽和度: by hand, on top of the fit
  if (!((amt > 0 && A.some(Boolean)) || extra !== 1)) return img;
  const T = A.map((f) => (f && amt > 0 ? deltaTable(f, keepVivid) : null));
  const wts = new Array(keys.length + 1).fill(0), rgb = [0, 0, 0], acc = [0, 0, 0];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    at((x + 0.5) / w, (y + 0.5) / h, wts);
    if (wts[0] > 0.995) continue; // (a person)
    const p = 4 * (y * w + x), R = img[p], G = img[p + 1], B = img[p + 2], lr = S2L[R], lg = S2L[G], lb = S2L[B];
    const l_ = Math.cbrt(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb), m_ = Math.cbrt(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb), s_ = Math.cbrt(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb);
    acc[0] = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_; acc[1] = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_; acc[2] = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;
    for (let i = 0; i < T.length; i++) { const t = wts[i + 1] * amt; if (T[i] && t > 0) addDelta(T[i], IX[R], IX[G], IX[B], t, acc); }
    let L = acc[0], a = acc[1], b = acc[2];
    if (extra !== 1) { let bg = 0; for (let i = 1; i < wts.length; i++) bg += wts[i]; const k = 1 + (extra - 1) * Math.min(1, bg); a *= k; b *= k; } // (last: the fit pulls colours towards the layer's cast, it must not put back what was taken out by hand)
    L = Math.min(1, Math.max(0, L));
    toSrgb(L, a, b, rgb); // (out of the screen's colours: less colour, same lightness — clipping each channel changed the lightness)
    img[p] = rgb[0] * 255; img[p + 1] = rgb[1] * 255; img[p + 2] = rgb[2] * 255;
  }
  return img;
}
/** the per-pixel version (before the tables), kept for the tests: the same result within interpolation error */
export function applyByExact(img, w, h, at, keys, fit, amt = 1, bgSat = 0, { keepVivid = false } = {}) {
  const A = keys.map((k) => fit && fit[k]), extra = 1 + Math.max(-1, Math.min(0.5, bgSat || 0)); // 背景飽和度: by hand, on top of the fit
  if (!((amt > 0 && A.some(Boolean)) || extra !== 1)) return img;
  const wts = new Array(keys.length + 1).fill(0), lab = [0, 0, 0], rgb = [0, 0, 0];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    at((x + 0.5) / w, (y + 0.5) / h, wts);
    if (wts[0] > 0.995) continue; // (a person)
    const p = 4 * (y * w + x);
    srgbToOklab(img[p] / 255, img[p + 1] / 255, img[p + 2] / 255, lab);
    let L = lab[0], a = lab[1], b = lab[2];
    for (let i = 0; i < A.length; i++) {
      const f = A[i], t = wts[i + 1] * amt;
      if (!f || t <= 0) continue;
      L += t * ((f.m + (lab[0] - f.m) * f.k + f.dL) - lab[0]);
      const ea = lab[1] - f.ma, eb = lab[2] - f.mb, D = Math.hypot(ea, eb);
      let g = chromaGain(f, D), g2 = bandGain(f, lab[0]); // (towards / away from the layer's cast; colour strength at this brightness)
      if (keepVivid && f.cs) { // (from the class's most colourful quarter up: kept; and never below ×0.7 — a big sign is a quarter of its class)
        const v = smooth(f.cs[2], Math.max(f.cs[2] + 0.01, f.cs[3] * 1.4), D); g += (1 - g) * v; g2 += (1 - g2) * v;
        g = Math.max(0.7, g); g2 = Math.max(0.7, g2);
      }
      const na = lab[1] + ea * (g - 1), nb = lab[2] + eb * (g - 1);
      a += t * (f.da + na * g2 - lab[1]); b += t * (f.db + nb * g2 - lab[2]);
    }
    if (extra !== 1) { let bg = 0; for (let i = 1; i < wts.length; i++) bg += wts[i]; const k = 1 + (extra - 1) * Math.min(1, bg); a *= k; b *= k; } // (last: the fit pulls colours towards the layer's cast, it must not put back what was taken out by hand)
    L = Math.min(1, Math.max(0, L));
    oklabToSrgb(L, a, b, rgb);
    if (rgb[0] < 0 || rgb[1] < 0 || rgb[2] < 0 || rgb[0] > 1 || rgb[1] > 1 || rgb[2] > 1) { // out of the screen's colours: less colour, same lightness
      let lo = 0, hi = 1; // (clipping each channel changed the lightness: a warm tint on a near-white wall darkened / brightened it)
      for (let it = 0; it < 8; it++) { const m = (lo + hi) / 2; oklabToSrgb(L, a * m, b * m, rgb); if (rgb[0] < 0 || rgb[1] < 0 || rgb[2] < 0 || rgb[0] > 1 || rgb[1] > 1 || rgb[2] > 1) hi = m; else lo = m; }
      oklabToSrgb(L, a * lo, b * lo, rgb);
    }
    for (let c = 0; c < 3; c++) img[p + c] = Math.min(1, Math.max(0, rgb[c])) * 255;
  }
  return img;
}
