// Colour-space helpers shared by the page, the worker and the Node tests.
// sRGB values are 0..1 floats unless a name says 8-bit. OKLab is the perceptual working space.

export const S2L = new Float32Array(256); // 8-bit sRGB -> linear
for (let i = 0; i < 256; i++) {
  const c = i / 255;
  S2L[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

export function s2l(c) {
  return c <= 0.04045 ? c / 12.92 : Math.pow((Math.max(c, 0) + 0.055) / 1.055, 2.4);
}

export function l2s(c) {
  return c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(Math.max(c, 0), 1 / 2.4) - 0.055;
}

// linear -> sRGB through a 4097-entry table with linear interpolation (fast path for rendering)
const L2S_N = 4096;
const L2S = new Float32Array(L2S_N + 1);
for (let i = 0; i <= L2S_N; i++) L2S[i] = l2s(i / L2S_N);
export function l2sFast(c) {
  if (c <= 0) return 0;
  if (c >= 1) return 1;
  const x = c * L2S_N, i = x | 0, f = x - i;
  return L2S[i] + (L2S[i + 1] - L2S[i]) * f;
}

export function srgbToOklab(r, g, b, out = [0, 0, 0]) {
  const lr = s2l(r), lg = s2l(g), lb = s2l(b);
  const l = Math.cbrt(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb);
  const m = Math.cbrt(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb);
  const s = Math.cbrt(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb);
  out[0] = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s;
  out[1] = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s;
  out[2] = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s;
  return out;
}

function oklabToLinear(L, a, b, out) {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3;
  out[0] = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
  out[1] = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
  out[2] = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;
  return out;
}

const _t = [0, 0, 0];
const inGamut = (v) => v[0] >= -1e-4 && v[0] <= 1.0001 && v[1] >= -1e-4 && v[1] <= 1.0001 && v[2] >= -1e-4 && v[2] <= 1.0001;

// OKLab -> sRGB; out-of-gamut colours lose chroma (hue and lightness kept) instead of channel clipping
export function oklabToSrgb(L, a, b, out = [0, 0, 0]) {
  oklabToLinear(L, a, b, _t);
  if (!inGamut(_t)) {
    let lo = 0, hi = 1;
    for (let k = 0; k < 12; k++) {
      const mid = (lo + hi) / 2;
      oklabToLinear(L, a * mid, b * mid, _t);
      if (inGamut(_t)) lo = mid; else hi = mid;
    }
    oklabToLinear(L, a * lo, b * lo, _t);
  }
  out[0] = l2s(Math.min(1, Math.max(0, _t[0])));
  out[1] = l2s(Math.min(1, Math.max(0, _t[1])));
  out[2] = l2s(Math.min(1, Math.max(0, _t[2])));
  return out;
}

export function luma8(r, g, b) { // relative luminance (linear) of 8-bit sRGB
  return 0.2126 * S2L[r] + 0.7152 * S2L[g] + 0.0722 * S2L[b];
}
