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
import { srgbToOklab, oklabToSrgb } from './color.js';

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

/** per layer: share, mean L, spread of L, tint (a, b of the less colourful half), mean chroma. rgba w×h. */
export function layerStats(rgba, w, h, dn, seg, step = 2) {
  const at = layerSampler(dn, seg), wts = [0, 0, 0], lab = [0, 0, 0], px = { near: [], far: [] };
  let total = 0;
  for (let y = 0; y < h; y += step) for (let x = 0; x < w; x += step) {
    at((x + 0.5) / w, (y + 0.5) / h, wts); total++;
    const k = wts[1] > 0.6 ? 'near' : wts[2] > 0.6 ? 'far' : null; // (clear members only)
    if (!k) continue;
    const p = 4 * (y * w + x); srgbToOklab(rgba[p] / 255, rgba[p + 1] / 255, rgba[p + 2] / 255, lab);
    px[k].push(lab[0], lab[1], lab[2]);
  }
  const out = {};
  for (const k of LAYERS) {
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
export function fitLayers(src, ref) {
  const cl = (v, c) => Math.max(-c, Math.min(c, v)), out = {};
  for (const k of LAYERS) {
    const S = src[k], R = ref[k];
    if (!S || !R || !(S.share >= MIN_SHARE && R.share >= MIN_SHARE)) { out[k] = null; continue; }
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
  const A = LAYERS.map((k) => fit && fit[k]), extra = 1 + Math.max(-1, Math.min(0.5, bgSat || 0)); // 背景飽和度: by hand, on top of the fit
  if (!((amt > 0 && A.some(Boolean)) || extra !== 1)) return img;
  const at = layerSampler(dn, seg), wts = [0, 0, 0], lab = [0, 0, 0], rgb = [0, 0, 0];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    at((x + 0.5) / w, (y + 0.5) / h, wts);
    if (wts[0] > 0.995) continue; // (a person)
    const p = 4 * (y * w + x);
    srgbToOklab(img[p] / 255, img[p + 1] / 255, img[p + 2] / 255, lab);
    let L = lab[0], a = lab[1], b = lab[2];
    for (let i = 0; i < 2; i++) {
      const f = A[i], t = wts[i + 1] * amt;
      if (!f || t <= 0) continue;
      L += t * ((f.m + (lab[0] - f.m) * f.k + f.dL) - lab[0]);
      const ea = lab[1] - f.ma, eb = lab[2] - f.mb, g = chromaGain(f, Math.hypot(ea, eb)); // (towards / away from the layer's cast)
      const g2 = bandGain(f, lab[0]), na = lab[1] + ea * (g - 1), nb = lab[2] + eb * (g - 1); // then colour strength at this brightness
      a += t * (f.da + na * g2 - lab[1]); b += t * (f.db + nb * g2 - lab[2]);
    }
    if (extra !== 1) { const k = 1 + (extra - 1) * (wts[1] + wts[2]); a *= k; b *= k; } // (last: the fit pulls colours towards the layer's cast, it must not put back what was taken out by hand)
    oklabToSrgb(Math.min(1, Math.max(0, L)), a, b, rgb);
    for (let c = 0; c < 3; c++) img[p + c] = Math.min(1, Math.max(0, rgb[c])) * 255;
  }
  return img;
}
