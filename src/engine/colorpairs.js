// 點選對應顏色: the user taps a colour in the reference and the matching thing in the video (the reference's sky, then
// the video's sky); the video's colour is moved onto the reference's, and colours near it move with it, fading out with
// the distance in colour (OKLab) — a colour qualifier with a soft edge, as a colourist would key the sky and push it. Up
// to a core distance the pull is full (a textured wall or a cloudy sky spreads around the tapped colour: with a plain
// Gaussian only its centre arrived, the tapped spot moved 60 % of the way), then it falls off. Several pairs blend (their pulls are averaged where
// they overlap). Being a function of the colour alone it is baked into the grade's LUT (grade.js bakeLUT, after the
// rest of the grade): no cost per frame, and the .cube export carries it.
//
// colorPairs: {on, amt 0..1, range 0..1 (how far in colour the pull reaches), items: [{s0: [r, g, b] the video's
// colour BEFORE grading (sRGB 0..1), r: [L, a, b] the reference's colour (OKLab), …}]}. The video's colour after the
// rest of the grade (s) is computed when baking, so the pair stays right when other settings change.
import { srgbToOklab, oklabToSrgb } from './color.js';

export const MAX_PAIRS = 6;
const rng = (range) => Math.min(1, Math.max(0, range ?? 0.5));
export const pairSigma = (range) => 0.025 + 0.06 * rng(range); // the fall-off beyond the core
export const pairCore = (range) => 0.025 + 0.04 * rng(range);   // full pull up to this distance

/** colorPairs → plan or null */
export function pairsPlan(cp) {
  if (!cp || !cp.on || !Array.isArray(cp.items) || !cp.items.length) return null;
  const amt = Math.min(1, Math.max(0, cp.amt ?? 1));
  if (!amt) return null;
  const items = cp.items.filter((it) => it && Array.isArray(it.s0) && Array.isArray(it.r)).slice(0, MAX_PAIRS);
  return items.length ? { amt, sigma: pairSigma(cp.range), core: pairCore(cp.range), items } : null;
}

/**
 * the warp, given each pair's video colour after the rest of the grade (s: OKLab) → fn(lab) moving lab in place.
 * gradeFn(r, g, b) → [r, g, b] graded sRGB (the LUT's own colour function).
 */
export function pairWarp(plan, gradeFn) {
  if (!plan) return null;
  const lab = [0, 0, 0];
  const P = plan.items.map((it) => {
    const g = gradeFn(it.s0[0], it.s0[1], it.s0[2]);
    srgbToOklab(Math.min(1, Math.max(0, g[0])), Math.min(1, Math.max(0, g[1])), Math.min(1, Math.max(0, g[2])), lab);
    return { s: [lab[0], lab[1], lab[2]], d: [it.r[0] - lab[0], it.r[1] - lab[1], it.r[2] - lab[2]] };
  });
  const k = 1 / (2 * plan.sigma * plan.sigma), amt = plan.amt, core = plan.core ?? 0;
  return (c) => {
    let wsum = 0, dl = 0, da = 0, db = 0;
    for (const p of P) {
      const e = Math.max(0, Math.hypot(c[0] - p.s[0], c[1] - p.s[1], c[2] - p.s[2]) - core), w = Math.exp(-e * e * k);
      wsum += w; dl += w * p.d[0]; da += w * p.d[1]; db += w * p.d[2];
    }
    if (wsum < 1e-4) return c;
    const n = amt / Math.max(1, wsum); // (overlapping pairs: their pulls averaged, never added up)
    c[0] += dl * n; c[1] += da * n; c[2] += db * n;
    return c;
  };
}

/** sRGB o (0..1, in place) through the warp */
export function warpRGB(warp, o, lab = [0, 0, 0], rgb = [0, 0, 0]) {
  srgbToOklab(Math.min(1, Math.max(0, o[0])), Math.min(1, Math.max(0, o[1])), Math.min(1, Math.max(0, o[2])), lab);
  warp(lab);
  oklabToSrgb(Math.min(1, Math.max(0, lab[0])), lab[1], lab[2], rgb);
  o[0] = Math.min(1, Math.max(0, rgb[0])); o[1] = Math.min(1, Math.max(0, rgb[1])); o[2] = Math.min(1, Math.max(0, rgb[2]));
  return o;
}
