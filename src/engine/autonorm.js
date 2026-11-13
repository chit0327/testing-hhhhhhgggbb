// 自動校正 before a mood (氛圍): the moods are fixed recipes, so a dark or colour-cast clip used to stay dark / cast under
// them. This measures one frame and makes a gentle, partial correction that is applied before the mood:
//
//   明暗 — a power curve on lightness (OKLab L' = L^g, black and white stay put, nothing clips) moving the median
//          lightness part of the way to a typical photo's (darkening: half as far, at most g 1.3, and not in the shadows); applied as a per-pixel gain in linear light, so hue and
//          saturation are kept.
//   白平衡 — the colour of the near-neutral surfaces (low chroma, mid lightness: walls, roads, clothes, clouds) is taken
//          as the light's colour and part of it is removed with channel gains in linear light (as a camera's white
//          balance does). Only part: golden-hour light is meant to be warm, and an orange path is not a cast.
//
// The correction is a few numbers ({g, gains}) kept in the look (look.autoFix, measured on the representative frame;
// each photo of a batch is measured on its own), so the preview, the thumbnails and the export all use the same one.
import { srgbToOklab, oklabToSrgb, s2l, l2s } from './color.js';

export const NORM_TARGET_L = 0.56;  // median OKLab lightness aimed at (well-exposed everyday photos: ~0.5–0.65)
export const NORM_TONE = 0.6;       // how much of the way to it
export const NORM_CAST = 0.6;       // how much of the measured cast is removed
export const NORM_CAST_MAX = 0.03;  // a larger "cast" is mostly the scene's own colour: counted only up to this

/** Measure an RGBA frame → the correction {g, gains, med, cast} (null-ish corrections are still returned). */
export function measureNorm(rgba, w, h, { tone = NORM_TONE, cast = NORM_CAST } = {}) {
  const n = w * h, step = Math.max(1, Math.floor(n / 40000)), Ls = [], lab = [0, 0, 0];
  let sa = 0, sb = 0, sw = 0;
  for (let i = 0; i < n; i += step) {
    const p = 4 * i;
    srgbToOklab(rgba[p] / 255, rgba[p + 1] / 255, rgba[p + 2] / 255, lab);
    Ls.push(lab[0]);
    const C = Math.hypot(lab[1], lab[2]);
    if (lab[0] > 0.25 && lab[0] < 0.95 && C < 0.06) { const k = 1 / (C + 0.01); sa += k * lab[1]; sb += k * lab[2]; sw += k; }
  }
  Ls.sort((a, b) => a - b);
  const med = Math.min(0.97, Math.max(0.03, Ls[Ls.length >> 1]));
  // darkening is weaker and capped: a bright picture is usually meant to be bright (white table, high-key room)
  const to = med + (med > NORM_TARGET_L ? 0.5 : 1) * tone * (NORM_TARGET_L - med), g = Math.min(1.3, Math.log(to) / Math.log(med));
  let ca = sw ? sa / sw : 0, cb = sw ? sb / sw : 0;
  const cm = Math.hypot(ca, cb);
  if (cm > NORM_CAST_MAX) { ca *= NORM_CAST_MAX / cm; cb *= NORM_CAST_MAX / cm; }
  // the light's colour as linear RGB at a mid grey, and the gains that make it neutral (same luminance)
  const grey = oklabToSrgb(0.6, cast * ca, cast * cb, [0, 0, 0]).map((v) => s2l(Math.min(1, Math.max(0, v))));
  const Yg = 0.2126 * grey[0] + 0.7152 * grey[1] + 0.0722 * grey[2];
  const gains = grey.map((v) => Yg / Math.max(1e-4, v));
  return { g: +g.toFixed(4), gains: gains.map((v) => +v.toFixed(4)), med: +med.toFixed(3), cast: [+ca.toFixed(4), +cb.toFixed(4)] };
}

const sm = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

/** True when a correction changes (practically) nothing. */
export const normIsNeutral = (nm) => !nm || (Math.abs(nm.g - 1) < 0.005 && nm.gains.every((v) => Math.abs(v - 1) < 0.003));

/** Apply a correction to one sRGB colour (0..1) → out (sRGB 0..1). */
export function applyNorm(r, g, b, nm, out = [0, 0, 0]) {
  let lr = s2l(r) * nm.gains[0], lg = s2l(g) * nm.gains[1], lb = s2l(b) * nm.gains[2];
  if (nm.g !== 1) { // tone: L' = L^g via a gain on linear light (OKLab L ~ cube root of luminance)
    const Y = 0.2126 * lr + 0.7152 * lg + 0.0722 * lb;
    if (Y > 1e-6) {
      const L = Math.cbrt(Math.min(1, Y)), e = nm.g > 1 ? 1 + (nm.g - 1) * sm(0.2, 0.6, L) : nm.g; // (darkening leaves the shadows: no crushed blacks)
      const k = Math.pow(Math.pow(L, e) / L, 3);
      lr *= k; lg *= k; lb *= k;
    }
  }
  const mx = Math.max(lr, lg, lb);
  if (mx > 1) { const Y = 0.2126 * lr + 0.7152 * lg + 0.0722 * lb; const t = (mx - 1) / (mx - Math.min(Y, 0.999)); lr += (Y - lr) * t; lg += (Y - lg) * t; lb += (Y - lb) * t; } // (over white: towards grey, not a hue shift)
  out[0] = l2s(Math.min(1, Math.max(0, lr))); out[1] = l2s(Math.min(1, Math.max(0, lg))); out[2] = l2s(Math.min(1, Math.max(0, lb)));
  return out;
}
