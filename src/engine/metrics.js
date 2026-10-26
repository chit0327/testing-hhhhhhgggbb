// Verification metrics computed on actual frames: CIEDE2000 colour differences, tonal-band colours,
// structure preservation and clipping. Used to rank candidates and to report what a grade really did.
import { S2L } from './color.js';

/** 8-bit sRGB -> CIE L*a*b* (D65). */
export function srgb8ToLab(r, g, b, out = [0, 0, 0]) {
  const R = S2L[r], G = S2L[g], B = S2L[b];
  const X = (0.4124564 * R + 0.3575761 * G + 0.1804375 * B) / 0.95047;
  const Y = 0.2126729 * R + 0.7151522 * G + 0.0721750 * B;
  const Z = (0.0193339 * R + 0.1191920 * G + 0.9503041 * B) / 1.08883;
  const f = (t) => (t > 216 / 24389 ? Math.cbrt(t) : (24389 / 27 * t + 16) / 116);
  const fx = f(X), fy = f(Y), fz = f(Z);
  out[0] = 116 * fy - 16; out[1] = 500 * (fx - fy); out[2] = 200 * (fy - fz);
  return out;
}

/** CIEDE2000 colour difference between two L*a*b* colours. */
export function deltaE2000(l1, l2) {
  const [L1, a1, b1] = l1, [L2, a2, b2] = l2, rad = Math.PI / 180;
  const C1 = Math.hypot(a1, b1), C2 = Math.hypot(a2, b2), Cm = (C1 + C2) / 2;
  const G = 0.5 * (1 - Math.sqrt(Cm ** 7 / (Cm ** 7 + 25 ** 7)));
  const a1p = (1 + G) * a1, a2p = (1 + G) * a2, C1p = Math.hypot(a1p, b1), C2p = Math.hypot(a2p, b2);
  const hp = (b, a) => (b === 0 && a === 0 ? 0 : ((Math.atan2(b, a) / rad) + 360) % 360);
  const h1p = hp(b1, a1p), h2p = hp(b2, a2p);
  const dLp = L2 - L1, dCp = C2p - C1p;
  let dhp = 0;
  if (C1p * C2p !== 0) { dhp = h2p - h1p; if (dhp > 180) dhp -= 360; else if (dhp < -180) dhp += 360; }
  const dHp = 2 * Math.sqrt(C1p * C2p) * Math.sin((dhp / 2) * rad);
  const Lpm = (L1 + L2) / 2, Cpm = (C1p + C2p) / 2;
  let hpm = h1p + h2p;
  if (C1p * C2p !== 0) hpm = Math.abs(h1p - h2p) > 180 ? (h1p + h2p < 360 ? (h1p + h2p + 360) / 2 : (h1p + h2p - 360) / 2) : (h1p + h2p) / 2;
  const T = 1 - 0.17 * Math.cos((hpm - 30) * rad) + 0.24 * Math.cos(2 * hpm * rad) + 0.32 * Math.cos((3 * hpm + 6) * rad) - 0.2 * Math.cos((4 * hpm - 63) * rad);
  const dTheta = 30 * Math.exp(-(((hpm - 275) / 25) ** 2));
  const Rc = 2 * Math.sqrt(Cpm ** 7 / (Cpm ** 7 + 25 ** 7));
  const Sl = 1 + (0.015 * (Lpm - 50) ** 2) / Math.sqrt(20 + (Lpm - 50) ** 2), Sc = 1 + 0.045 * Cpm, Sh = 1 + 0.015 * Cpm * T;
  const Rt = -Math.sin(2 * dTheta * rad) * Rc;
  return Math.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh));
}

/**
 * Mean L*a*b* of the shadows / midtones / highlights (lightness terciles by L* rank) of an RGBA image.
 * centre: analyse only the central part (a reference's border may be vignetted). skipWhite: ignore blown pixels.
 */
export function bandColours(rgba, w, h, { centre = 1, skipWhite = false, step = 3 } = {}) {
  const x0 = Math.floor(w * (1 - centre) / 2), y0 = Math.floor(h * (1 - centre) / 2), lab = [0, 0, 0], px = [];
  for (let y = y0; y < h - y0; y += step) for (let x = x0; x < w - x0; x += step) {
    const p = (y * w + x) * 4;
    if (skipWhite && rgba[p] > 245 && rgba[p + 1] > 245 && rgba[p + 2] > 245) continue;
    srgb8ToLab(rgba[p], rgba[p + 1], rgba[p + 2], lab);
    px.push(lab[0], lab[1], lab[2]);
  }
  const n = px.length / 3, Ls = [];
  for (let i = 0; i < n; i++) Ls.push(px[3 * i]);
  const sorted = Float64Array.from(Ls).sort(), t1 = sorted[Math.floor(n / 3)], t2 = sorted[Math.floor((2 * n) / 3)];
  const acc = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  for (let i = 0; i < n; i++) {
    const L = px[3 * i], k = L < t1 ? 0 : L < t2 ? 1 : 2;
    acc[k][0] += L; acc[k][1] += px[3 * i + 1]; acc[k][2] += px[3 * i + 2]; acc[k][3]++;
  }
  return acc.map((a) => [a[0] / (a[3] || 1), a[1] / (a[3] || 1), a[2] / (a[3] || 1)]);
}

/** Mean CIEDE2000 between the band colours of an image and of a reference (0 = same tonal colours). */
export function bandDeltaE(bandsA, bandsB) {
  return (deltaE2000(bandsA[0], bandsB[0]) + deltaE2000(bandsA[1], bandsB[1]) + deltaE2000(bandsA[2], bandsB[2])) / 3;
}

/**
 * Structure preservation: correlation of luminance-gradient magnitude between the original and the graded
 * frame (1 = every edge and texture where it was; a colour grade should stay ≳ 0.95).
 */
export function structureScore(a, b, w, h) {
  const lum = (d, i) => 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2];
  let n = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
  for (let y = 1; y < h - 1; y += 2) for (let x = 1; x < w - 1; x += 2) {
    const i = (y * w + x) * 4;
    const ga = Math.hypot(lum(a, i + 4) - lum(a, i - 4), lum(a, i + 4 * w) - lum(a, i - 4 * w));
    const gb = Math.hypot(lum(b, i + 4) - lum(b, i - 4), lum(b, i + 4 * w) - lum(b, i - 4 * w));
    n++; sa += ga; sb += gb; saa += ga * ga; sbb += gb * gb; sab += ga * gb;
  }
  const va = saa - sa * sa / n, vb = sbb - sb * sb / n;
  return (sab - sa * sb / n) / Math.sqrt(Math.max(va * vb, 1e-12));
}

/** Share of pixels with a channel at the top (≥ 250) or all channels at the bottom (≤ 3). */
export function clipping(rgba) {
  let hi = 0, lo = 0;
  const n = rgba.length / 4;
  for (let i = 0; i < rgba.length; i += 4) {
    if (Math.max(rgba[i], rgba[i + 1], rgba[i + 2]) >= 250) hi++;
    if (rgba[i] <= 3 && rgba[i + 1] <= 3 && rgba[i + 2] <= 3) lo++;
  }
  return { highlights: hi / n, shadows: lo / n };
}

/** Mean CIEDE2000 between original and graded pixels where mask (0..1, same w×h) is high, e.g. skin. */
export function maskedDeltaE(a, b, mask, w, h) {
  const la = [0, 0, 0], lb = [0, 0, 0];
  let s = 0, n = 0;
  for (let i = 0; i < w * h; i += 2) {
    if (mask[i] < 0.5) continue;
    srgb8ToLab(a[4 * i], a[4 * i + 1], a[4 * i + 2], la); srgb8ToLab(b[4 * i], b[4 * i + 1], b[4 * i + 2], lb);
    s += deltaE2000(la, lb); n++;
  }
  return n ? s / n : null;
}
