// Colour analysis of an image (reference or target frame), in OKLab: lightness / chroma / hue histograms,
// shadow / midtone / highlight zones (lightness terciles) with their mean colour and NEUTRAL tint (median a/b
// of the least colourful 5 % — the grey-axis colour that a grade adds, as opposed to object colours), and
// the hue families present (chroma-weighted share per 30° sector). Pure, deterministic; used by the
// evaluation tool and available to the UI.
import { srgbToOklab } from './color.js';

const r3 = (v) => Math.round(v * 1000) / 1000;
export const HUE_SECTORS = 12; // 30° each, starting at 0° (OKLab hue)

export function analyse(rgba, w, h, { centre = 1, step = 2, bins = 32 } = {}) {
  const x0 = Math.floor((w * (1 - centre)) / 2), y0 = Math.floor((h * (1 - centre)) / 2), px = [], l = [0, 0, 0];
  for (let y = y0; y < h - y0; y += step) for (let x = x0; x < w - x0; x += step) {
    const p = (y * w + x) * 4;
    srgbToOklab(rgba[p] / 255, rgba[p + 1] / 255, rgba[p + 2] / 255, l);
    px.push(l[0], l[1], l[2]);
  }
  const n = px.length / 3, Lh = new Array(bins).fill(0), Ch = new Array(bins).fill(0), hue = new Array(HUE_SECTORS).fill(0);
  let chromaSum = 0;
  for (let i = 0; i < n; i++) {
    const L = px[3 * i], a = px[3 * i + 1], b = px[3 * i + 2], C = Math.hypot(a, b);
    Lh[Math.min(bins - 1, Math.floor(L * bins))]++;
    Ch[Math.min(bins - 1, Math.floor((C / 0.25) * bins))]++;
    chromaSum += C;
    if (C > 0.02) hue[Math.floor(((((Math.atan2(b, a) * 180) / Math.PI) + 360) % 360) / (360 / HUE_SECTORS))] += C;
  }
  const hueTotal = hue.reduce((s, v) => s + v, 0) || 1;
  const order = Array.from({ length: n }, (_, i) => i).sort((p, q) => px[3 * p] - px[3 * q]);
  const zones = ['shadows', 'midtones', 'highlights'].map((name, z) => {
    const idx = order.slice(Math.floor((z * n) / 3), Math.floor(((z + 1) * n) / 3));
    let L = 0, a = 0, b = 0, C = 0;
    for (const i of idx) { L += px[3 * i]; a += px[3 * i + 1]; b += px[3 * i + 2]; C += Math.hypot(px[3 * i + 1], px[3 * i + 2]); }
    const k = idx.length || 1;
    const low = idx.slice().sort((p, q) => Math.hypot(px[3 * p + 1], px[3 * p + 2]) - Math.hypot(px[3 * q + 1], px[3 * q + 2])).slice(0, Math.max(1, Math.floor(idx.length * 0.05)));
    const med = (o) => { const v = low.map((i) => px[3 * i + o]).sort((p, q) => p - q); return v[Math.floor(v.length / 2)] ?? 0; };
    return { name, L: r3(L / k), a: r3(a / k), b: r3(b / k), chroma: r3(C / k), neutralTint: [r3(med(1)), r3(med(2))], range: [r3(px[3 * idx[0]] ?? 0), r3(px[3 * idx[idx.length - 1]] ?? 0)] };
  });
  return {
    pixels: n, meanChroma: r3(chromaSum / Math.max(1, n)),
    lightnessHistogram: Lh.map((v) => r3(v / n)), chromaHistogram: Ch.map((v) => r3(v / n)),
    hueShares: hue.map((v) => r3(v / hueTotal)), zones,
  };
}
