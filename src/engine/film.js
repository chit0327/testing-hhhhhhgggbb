// 膠片質感: film grain, halation and softness, applied after the colour grade and the other effects (in place on
// RGBA), plus measuring them from a reference picture so the 膠片模板 can set them automatically.
//
// Sizes are relative to the frame width (a 1280 px export and a 640 px preview look alike).
//  - 柔化 (filmSoft 0..1): a small blur mixed in (film is less sharp than digital).
//  - 光暈 (halation 0..1): light from bright parts spreads a little into the surroundings in red-orange — on film the
//    light passes the emulsion, reflects off the film base and exposes the red layer again around highlights.
//  - 膠片顆粒 (filmGrain 0..1, filmGrainSize 0..1): grain that changes every frame (seeded by the frame number, so a
//    given frame is the same in the preview and in every export), strongest in the mid-tones, mostly luminance.
import { S2L } from './color.js';

export const FILM_KEYS = ['filmGrain', 'filmGrainSize', 'halation', 'filmSoft'];
export const FILM_DEFAULT = { filmGrain: 0.25, filmGrainSize: 0.5, halation: 0.3, filmSoft: 0.25 }; // template without a reference
const clamp01 = (v) => Math.min(1, Math.max(0, +v || 0));
const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

function blurChannels(buf, w, h, ch, r) { // in place, separable box blur, `ch` interleaved channels, clamped edges
  if (r < 1) return;
  const tmp = new Float32Array(buf.length), n = 2 * r + 1;
  for (let y = 0; y < h; y++) for (let c = 0; c < ch; c++) {
    let acc = 0;
    for (let k = -r; k <= r; k++) acc += buf[(y * w + Math.min(w - 1, Math.max(0, k))) * ch + c];
    for (let x = 0; x < w; x++) {
      tmp[(y * w + x) * ch + c] = acc / n;
      acc += buf[(y * w + Math.min(w - 1, x + r + 1)) * ch + c] - buf[(y * w + Math.max(0, x - r)) * ch + c];
    }
  }
  for (let x = 0; x < w; x++) for (let c = 0; c < ch; c++) {
    let acc = 0;
    for (let k = -r; k <= r; k++) acc += tmp[(Math.min(h - 1, Math.max(0, k)) * w + x) * ch + c];
    for (let y = 0; y < h; y++) {
      buf[(y * w + x) * ch + c] = acc / n;
      acc += tmp[(Math.min(h - 1, y + r + 1) * w + x) * ch + c] - tmp[(Math.max(0, y - r) * w + x) * ch + c];
    }
  }
}

// integer hash → [0, 1)
function hash3(x, y, z) {
  let h = (x * 374761393 + y * 668265263 + z * 2147483647) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

export const hasFilm = (fx) => !!fx && (clamp01(fx.filmGrain) > 0 || clamp01(fx.halation) > 0 || clamp01(fx.filmSoft) > 0);

/** fx: {filmGrain, filmGrainSize, halation, filmSoft}; frame: the frame number (grain pattern). In place on RGBA. */
export function applyFilm(img, w, h, fx, frame = 0) {
  if (!hasFilm(fx)) return img;
  const soft = clamp01(fx.filmSoft), hal = clamp01(fx.halation), grain = clamp01(fx.filmGrain), size = clamp01(fx.filmGrainSize ?? 0.5);
  const s = w / 1280; // scale: sizes are set for a 1280 px wide frame
  const n = w * h;
  // 柔化: blend towards a small blur (linear light)
  if (soft > 0) {
    const lin = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { lin[3 * i] = S2L[img[4 * i]]; lin[3 * i + 1] = S2L[img[4 * i + 1]]; lin[3 * i + 2] = S2L[img[4 * i + 2]]; }
    const bl = lin.slice(), r = Math.max(1, Math.round((1 + 1.5 * soft) * s));
    blurChannels(bl, w, h, 3, r); blurChannels(bl, w, h, 3, r);
    const k = 0.85 * soft;
    for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) {
      const v = lin[3 * i + c] + (bl[3 * i + c] - lin[3 * i + c]) * k;
      img[4 * i + c] = toSrgb8(v);
    }
  }
  // 光暈: the bright parts' light, blurred on a reduced grid, added back in red-orange around them
  if (hal > 0) {
    const f = Math.max(1, Math.round(Math.max(w, h) / 640)), sw = Math.ceil(w / f), sh = Math.ceil(h / f);
    const g = new Float32Array(sw * sh), cnt = new Float32Array(sw * sh);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const p = 4 * (y * w + x), Y = 0.2126 * S2L[img[p]] + 0.7152 * S2L[img[p + 1]] + 0.0722 * S2L[img[p + 2]];
      const q = Math.floor(y / f) * sw + Math.floor(x / f);
      g[q] += smooth(0.45, 0.95, Y) * Y; cnt[q]++;
    }
    for (let q = 0; q < g.length; q++) g[q] /= cnt[q] || 1;
    const rr = Math.max(1, Math.round(0.006 * Math.max(sw, sh)));
    for (let pass = 0; pass < 3; pass++) blurChannels(g, sw, sh, 1, rr);
    const tint = [1, 0.3, 0.1], k = 1.6 * hal;
    for (let y = 0; y < h; y++) {
      const gy = Math.min(sh - 1, Math.max(0, (y + 0.5) / f - 0.5)), y0 = Math.floor(gy), y1 = Math.min(sh - 1, y0 + 1), ty = gy - y0;
      for (let x = 0; x < w; x++) {
        const gx = Math.min(sw - 1, Math.max(0, (x + 0.5) / f - 0.5)), x0 = Math.floor(gx), x1 = Math.min(sw - 1, x0 + 1), tx = gx - x0;
        const a = g[y0 * sw + x0] + (g[y0 * sw + x1] - g[y0 * sw + x0]) * tx, b = g[y1 * sw + x0] + (g[y1 * sw + x1] - g[y1 * sw + x0]) * tx;
        const glow = a + (b - a) * ty;
        if (glow < 1e-4) continue;
        const p = 4 * (y * w + x);
        for (let c = 0; c < 3; c++) { const v = S2L[img[p + c]], add = Math.min(1, k * glow * tint[c]); img[p + c] = toSrgb8(1 - (1 - v) * (1 - add)); } // screened: never clips
      }
    }
  }
  // 膠片顆粒: smooth noise on a grid of grain cells, seeded by the frame number; mid-tones get the most
  if (grain > 0) {
    const cell = Math.max(0.6, (0.7 + 1.8 * size) * s), gw = Math.ceil(w / cell) + 2, gh = Math.ceil(h / cell) + 2;
    const L = new Float32Array(gw * gh), C = new Float32Array(gw * gh * 2), z = (frame | 0) + 1;
    for (let q = 0; q < gw * gh; q++) {
      const gx = q % gw, gy = (q / gw) | 0;
      L[q] = (hash3(gx, gy, z) + hash3(gx + 7919, gy, z) + hash3(gx, gy + 104729, z) - 1.5) * 2; // ≈ Gaussian, sd ≈ 1
      C[2 * q] = hash3(gx + 31, gy + 17, z * 3) - 0.5; C[2 * q + 1] = hash3(gx + 57, gy + 91, z * 5) - 0.5;
    }
    const amp = 20 * grain; // in 8-bit steps at mid-grey
    for (let y = 0; y < h; y++) {
      const gy = y / cell, y0 = Math.floor(gy), ty = gy - y0;
      for (let x = 0; x < w; x++) {
        const gx = x / cell, x0 = Math.floor(gx), tx = gx - x0, q = y0 * gw + x0;
        const nL = (L[q] * (1 - tx) + L[q + 1] * tx) * (1 - ty) + (L[q + gw] * (1 - tx) + L[q + gw + 1] * tx) * ty;
        const p = 4 * (y * w + x), v = (img[p] * 0.2126 + img[p + 1] * 0.7152 + img[p + 2] * 0.0722) / 255;
        const m = amp * (0.35 + 2.6 * v * (1 - v)); // mid-tones most, deep shadows and highlights less
        const cr = C[2 * q] * 0.25, cb = C[2 * q + 1] * 0.25; // a little colour in the grain
        img[p] = clamp8(img[p] + m * (nL + cr));
        img[p + 1] = clamp8(img[p + 1] + m * nL);
        img[p + 2] = clamp8(img[p + 2] + m * (nL + cb));
      }
    }
  }
  return img;
}
const clamp8 = (v) => (v < 0 ? 0 : v > 255 ? 255 : v);
function toSrgb8(v) { v = v < 0 ? 0 : v > 1 ? 1 : v; return 255 * (v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055); }

/**
 * 膠片模板: grain, softness and halation set from a reference picture (ImageData-like {data, width, height}).
 * Real grain rarely survives in a reference (screenshots and web stills are scaled and compressed: film stills measure
 * as smooth as digital frames), but the film tone does: milky lifted blacks and a narrow brightness range. So the
 * template rates how film-like the reference looks from that tone (letterbox bars and pure black excluded) and sets the
 * three effects from it; grain clearly visible in the reference (a full-resolution scan) adds to the grain.
 * Returns {filmGrain, filmGrainSize, halation, filmSoft, measured}.
 */
export function measureFilm(ref) {
  const { data: d, width: w, height: h } = ref;
  // tone: OKLab-like lightness (cube root of linear luminance) percentiles, pure black (bars) left out
  const Ls = [];
  for (let i = 0; i < w * h; i += 7) {
    const r = d[4 * i], g = d[4 * i + 1], b = d[4 * i + 2];
    if (r < 8 && g < 8 && b < 8) continue;
    Ls.push(Math.cbrt(0.2126 * S2L[r] + 0.7152 * S2L[g] + 0.0722 * S2L[b]));
  }
  Ls.sort((a, b) => a - b);
  const q = (p) => (Ls.length ? Ls[Math.floor(p * (Ls.length - 1))] : 0);
  const black = q(0.01), range = q(0.99) - black, bright = q(0.99);
  // grain: high-pass (pixel minus its 3×3 mean) spread in the flattest blocks (sky, walls), at full resolution
  const Y = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) Y[i] = 0.2126 * d[4 * i] + 0.7152 * d[4 * i + 1] + 0.0722 * d[4 * i + 2];
  const B = 16, blocks = [];
  for (let by = 1; by + B < h - 1; by += B) for (let bx = 1; bx + B < w - 1; bx += B) {
    let s = 0, s2 = 0, k = 0, mean = 0;
    for (let y = by; y < by + B; y++) for (let x = bx; x < bx + B; x++) {
      const i = y * w + x;
      const m3 = (Y[i - w - 1] + Y[i - w] + Y[i - w + 1] + Y[i - 1] + Y[i] + Y[i + 1] + Y[i + w - 1] + Y[i + w] + Y[i + w + 1]) / 9;
      const dd = Y[i] - m3; s += dd; s2 += dd * dd; k++; mean += Y[i];
    }
    mean /= k;
    if (mean < 25 || mean > 235) continue; // clipped areas (and bars) show no grain
    blocks.push(Math.sqrt(Math.max(0, s2 / k - (s / k) ** 2)));
  }
  blocks.sort((a, b) => a - b);
  const sigma = blocks.length ? blocks[Math.floor(blocks.length * 0.2)] : 0;
  // film-likeness 0..1: lifted blacks (digital ≈ 0.06–0.16, these film stills ≈ 0.29) and a narrow range (≈ 0.8 → 0.56)
  const film = 0.6 * clamp01((black - 0.1) / 0.18) + 0.4 * clamp01((0.82 - range) / 0.24);
  const visibleGrain = clamp01((sigma - 1.5) / 6); // only grain well above digital noise / compression (σ ≤ ≈ 1.2)
  const r2 = (v) => Math.round(v * 100) / 100;
  return {
    filmGrain: r2(Math.min(1, 0.1 + 0.2 * film + 0.5 * visibleGrain)), // (subtle: film stills show fine grain)
    filmGrainSize: 0.5,
    halation: r2((bright > 0.8 ? 0.1 : 0.05) + 0.3 * film), // a little even without highlights: lights in the video may have them
    filmSoft: r2(0.1 + 0.25 * film),
    measured: { black: r2(black), range: r2(range), sigma: +sigma.toFixed(2), film: r2(film) },
  };
}
