// Technical checks of a graded frame against its original (and optionally the reference): the kinds of faults
// a colourist rejects at a glance. Each is a plain measurement; which ones predict a viewer's verdict is
// measured against blind ratings (research/eval_runs/…/qc_*), not assumed.
import { srgbToOklab } from './color.js';
import { structureScore } from './metrics.js';

const hueOf = (a, b) => (Math.atan2(b, a) * 180) / Math.PI;
const dHue = (x, y) => Math.abs(((x - y + 540) % 360) - 180);

function labs(rgba, step) {
  const out = [], l = [0, 0, 0];
  for (let i = 0; i < rgba.length; i += 4 * step) { srgbToOklab(rgba[i] / 255, rgba[i + 1] / 255, rgba[i + 2] / 255, l); out.push(l[0], l[1], l[2]); }
  return Float64Array.from(out);
}
function neutralTints(lab) { // per lightness tercile: median a/b of its least colourful 10 %
  const n = lab.length / 3, idx = Array.from({ length: n }, (_, i) => i).sort((p, q) => lab[3 * p] - lab[3 * q]), C = (i) => Math.hypot(lab[3 * i + 1], lab[3 * i + 2]);
  return [0, 1, 2].map((b) => {
    const part = idx.slice(Math.floor((b * n) / 3), Math.floor(((b + 1) * n) / 3)).sort((p, q) => C(p) - C(q)).slice(0, Math.max(1, Math.floor(n / 30)));
    const med = (o) => { const v = part.map((i) => lab[3 * i + o]).sort((p, q) => p - q); return v[Math.floor(v.length / 2)]; };
    return [med(1), med(2)];
  });
}
const stats = (lab) => { let s = 0, ss = 0; const n = lab.length / 3; for (let i = 0; i < lab.length; i += 3) { s += lab[i]; ss += lab[i] * lab[i]; } const m = s / n; return { mean: m, sd: Math.sqrt(Math.max(0, ss / n - m * m)) }; };

/**
 * orig, graded: RGBA (same size w×h). ref (optional): {rgba, w, h}. Returns measurements:
 *   dL: mean lightness change (OKLab L); contrast: lightness spread ratio graded / original;
 *   clipHi / clipLo: added share of blown (any channel ≥ 254) / crushed (all ≤ 1) pixels;
 *   salientChroma: chroma of the original's most colourful 15 % after / before (< 1 = colours went dull);
 *   salientHueShift: chroma-weighted mean hue change (°) of the original's colourful pixels (C > 0.06);
 *   skinOff: share of the original's skin-coloured pixels pushed outside natural skin hues (25–85°);
 *   neutralCast: strongest colour cast on near-neutral pixels (OKLab ab length × 100);
 *   castVsRef: distance of those neutral tints from the reference's (× 100), if a reference is given;
 *   structure: texture kept (luminance-gradient correlation).
 */
export function qualityChecks(orig, graded, w, h, ref = null, { step = 2 } = {}) {
  const O = labs(orig, step), G = labs(graded, step), n = O.length / 3;
  const so = stats(O), sg = stats(G);
  let hiO = 0, hiG = 0, loO = 0, loG = 0;
  for (let i = 0; i < orig.length; i += 4 * step) {
    if (orig[i] >= 254 || orig[i + 1] >= 254 || orig[i + 2] >= 254) hiO++;
    if (graded[i] >= 254 || graded[i + 1] >= 254 || graded[i + 2] >= 254) hiG++;
    if (orig[i] <= 1 && orig[i + 1] <= 1 && orig[i + 2] <= 1) loO++;
    if (graded[i] <= 1 && graded[i + 1] <= 1 && graded[i + 2] <= 1) loG++;
  }
  const C = (L, i) => Math.hypot(L[3 * i + 1], L[3 * i + 2]);
  const order = Array.from({ length: n }, (_, i) => i).sort((p, q) => C(O, q) - C(O, p)), top = order.slice(0, Math.max(1, Math.floor(n * 0.15)));
  let cO = 0, cG = 0; for (const i of top) { cO += C(O, i); cG += C(G, i); }
  let hs = 0, hw = 0, skin = 0, skinOff = 0;
  for (let i = 0; i < n; i++) {
    const co = C(O, i);
    if (co > 0.06) { const wgt = Math.min(co, 0.2); hs += wgt * dHue(hueOf(O[3 * i + 1], O[3 * i + 2]), hueOf(G[3 * i + 1], G[3 * i + 2])); hw += wgt; }
    const ho = hueOf(O[3 * i + 1], O[3 * i + 2]);
    if (co > 0.03 && co < 0.12 && ho > 30 && ho < 80 && O[3 * i] > 0.45 && O[3 * i] < 0.85) {
      skin++; const hg = hueOf(G[3 * i + 1], G[3 * i + 2]); if (hg < 25 || hg > 85 || C(G, i) < 0.015) skinOff++;
    }
  }
  const tG = neutralTints(G), cast = Math.max(...tG.map((t) => Math.hypot(t[0], t[1]))) * 100;
  const out = {
    dL: sg.mean - so.mean, contrast: sg.sd / Math.max(so.sd, 1e-6), clipHi: (hiG - hiO) / n, clipLo: (loG - loO) / n,
    salientChroma: cG / Math.max(cO, 1e-6), salientHueShift: hw ? hs / hw : 0, skinOff: skin > n * 0.01 ? skinOff / skin : 0,
    neutralCast: cast, structure: structureScore(orig, graded, w, h),
  };
  if (ref) { const tR = neutralTints(labs(ref.rgba, Math.max(1, Math.round((ref.w * ref.h) / (w * h / step))))); out.castVsRef = (tG.reduce((s, t, i) => s + Math.hypot(t[0] - tR[i][0], t[1] - tR[i][1]), 0) / 3) * 100; }
  return out;
}
