// The one renderer used by the dialog preview, the adjustment preview and the export (via the worker).
// Order: the global look first (reference grade + basic adjustments, one LUT fitted on the untouched
// frames), then the Lighting Intent's local exposure in LINEAR light on the graded pixels. Applying the
// local exposure last makes it exact and predictable: +0.6 EV in a region is +0.6 EV in the output
// (earlier, applying it before the LUT let the reference's contrast curve flatten brightened people).
import { S2L, l2sFast, oklabToSrgb, srgbToOklab } from './color.js';

const KNEE = 0.8; // linear luminance above which brightening is rolled off instead of clipped

export function rollOff(y) { // rational shoulder for y > KNEE: slope 1 at the knee, asymptote 1, never clips
  if (y <= KNEE) return y;
  const x = y - KNEE, r = 1 - KNEE;
  return KNEE + r * x / (x + r);
}

// sRGB (0..1 float) -> linear through a fine table (LUT outputs are floats, not 8-bit)
const S2L_N = 4096;
const S2LF = new Float32Array(S2L_N + 1);
for (let i = 0; i <= S2L_N; i++) { const c = i / S2L_N; S2LF[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); }
function s2lFast(c) {
  if (c <= 0) return 0;
  if (c >= 1) return 1;
  const x = c * S2L_N, i = x | 0;
  return S2LF[i] + (S2LF[i + 1] - S2LF[i]) * (x - i);
}

/** Scale a linear RGB colour by k via its luminance (hue/saturation kept), rolling highlights off. */
function expose(lin, k) {
  const Y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
  if (Y <= 1e-6) return;
  const s = (k > 1 ? rollOff(Y * k) : Y * k) / Y;
  lin[0] *= s; lin[1] *= s; lin[2] *= s;
  const mx = Math.max(lin[0], lin[1], lin[2]);
  if (mx > 1) { // pull an over-range colour towards its luminance instead of clipping one channel
    const Yn = Y * s, t = (mx - 1) / Math.max(mx - Yn, 1e-6);
    for (let c = 0; c < 3; c++) lin[c] += (Yn - lin[c]) * t;
  }
}

const sameIndex = (a, b) => (!a && !b) || (a && b && a.length === b.length && a.every((v, i) => v === b[i]));

/**
 * src: RGBA Uint8ClampedArray (w×h). lut: {size, data} or null (identity). ev: {data, w, h} stops or null.
 * Returns a new RGBA Uint8ClampedArray.
 */
/**
 * layers (optional): one {lut, mask: {data, w, h} 0..1, k} or an array of them — inside each mask the colour
 * is mixed (by mask × k) towards that layer's LUT (same size as the main one): skin protection and the
 * automatic local grade (one LUT per AI region).
 */
export function renderFrame(src, w, h, lut, ev, out = new Uint8ClampedArray(src.length), fx = null, layers = null) {
  const N = lut ? lut.size : 0, L = lut ? lut.data : null, n1 = N - 1;
  // region layers must share the main LUT's grid (same size and shaper) because they reuse its lookup position
  const LY = L && layers ? (Array.isArray(layers) ? layers : [layers]).filter((l) => l && l.lut && l.lut.size === N && sameIndex(l.lut.index, lut.index)) : [], col2 = [0, 0, 0];
  const IX = lut && lut.index ? lut.index : null; // shaper: 8-bit level -> fractional grid coordinate
  let gain = null, gw = 0, gh = 0;
  if (ev) {
    gw = ev.w; gh = ev.h;
    gain = new Float32Array(ev.data.length);
    for (let i = 0; i < gain.length; i++) gain[i] = Math.pow(2, ev.data[i]);
  }
  const col = [0, 0, 0], lin = [0, 0, 0], sR = N * N * 3, sG = N * 3;
  // per-column / per-row positions, computed once per frame (the same formulas the per-pixel code used)
  let gx0 = null, gx1 = null, gfx = null;
  if (gain) {
    gx0 = new Int32Array(w); gx1 = new Int32Array(w); gfx = new Float64Array(w);
    for (let x = 0; x < w; x++) { let fx = Math.min(gw - 1, Math.max(0, (x + 0.5) * gw / w - 0.5)); const x0 = Math.floor(fx); gx0[x] = x0; gx1[x] = Math.min(gw - 1, x0 + 1); gfx[x] = fx - x0; }
  }
  const lcol = LY.map((ly) => { const sm = ly.mask, a = new Int32Array(w); for (let x = 0; x < w; x++) a[x] = Math.min(sm.w - 1, Math.max(0, Math.round((x + 0.5) * sm.w / w - 0.5))); return a; });
  const lrow = new Int32Array(LY.length);
  for (let y = 0; y < h; y++) {
    let fy = 0, y0 = 0, y1 = 0;
    if (gain) { fy = Math.min(gh - 1, Math.max(0, (y + 0.5) * gh / h - 0.5)); y0 = Math.floor(fy); y1 = Math.min(gh - 1, y0 + 1); fy -= y0; }
    for (let li = 0; li < LY.length; li++) { const sm = LY[li].mask; lrow[li] = Math.min(sm.h - 1, Math.max(0, Math.round((y + 0.5) * sm.h / h - 0.5))) * sm.w; }
    for (let x = 0; x < w; x++) {
      const p = (y * w + x) * 4;
      if (L) { // trilinear 3D LUT lookup (global look)
        const fr = IX ? IX[src[p]] : src[p] / 255 * n1, fg = IX ? IX[src[p + 1]] : src[p + 1] / 255 * n1, fb = IX ? IX[src[p + 2]] : src[p + 2] / 255 * n1;
        const r0 = Math.min(n1 - 1, fr | 0), g0 = Math.min(n1 - 1, fg | 0), b0 = Math.min(n1 - 1, fb | 0);
        const dr = fr - r0, dg = fg - g0, db = fb - b0, base = ((r0 * N + g0) * N + b0) * 3;
        for (let c = 0; c < 3; c++) {
          const i000 = base + c;
          const c00 = L[i000] + (L[i000 + 3] - L[i000]) * db;
          const c01 = L[i000 + sG] + (L[i000 + sG + 3] - L[i000 + sG]) * db;
          const c10 = L[i000 + sR] + (L[i000 + sR + 3] - L[i000 + sR]) * db;
          const c11 = L[i000 + sR + sG] + (L[i000 + sR + sG + 3] - L[i000 + sR + sG]) * db;
          const c0 = c00 + (c01 - c00) * dg, c1 = c10 + (c11 - c10) * dg;
          col[c] = c0 + (c1 - c0) * dr;
        }
        for (let li = 0; li < LY.length; li++) { // region layers: same lookup in the layer's LUT, mixed in by its mask
          const ly = LY[li], SL = ly.lut.data;
          const m = ly.mask.data[lrow[li] + lcol[li][x]] * ly.k;
          if (m > 0.004) {
            for (let c = 0; c < 3; c++) {
              const i000 = base + c;
              const c00 = SL[i000] + (SL[i000 + 3] - SL[i000]) * db, c01 = SL[i000 + sG] + (SL[i000 + sG + 3] - SL[i000 + sG]) * db;
              const c10 = SL[i000 + sR] + (SL[i000 + sR + 3] - SL[i000 + sR]) * db, c11 = SL[i000 + sR + sG] + (SL[i000 + sR + sG + 3] - SL[i000 + sR + sG]) * db;
              const c0 = c00 + (c01 - c00) * dg, c1 = c10 + (c11 - c10) * dg;
              col2[c] = c0 + (c1 - c0) * dr;
              col[c] += (col2[c] - col[c]) * m;
            }
          }
        }
      } else { col[0] = src[p] / 255; col[1] = src[p + 1] / 255; col[2] = src[p + 2] / 255; }
      if (gain) { // local exposure from the confirmed Lighting Intent
        const x0 = gx0[x], x1 = gx1[x], fx = gfx[x];
        const a0 = gain[y0 * gw + x0] + (gain[y0 * gw + x1] - gain[y0 * gw + x0]) * fx;
        const a1 = gain[y1 * gw + x0] + (gain[y1 * gw + x1] - gain[y1 * gw + x0]) * fx;
        const k = a0 + (a1 - a0) * fy;
        if (k !== 1) {
          lin[0] = L ? s2lFast(col[0]) : S2L[src[p]]; lin[1] = L ? s2lFast(col[1]) : S2L[src[p + 1]]; lin[2] = L ? s2lFast(col[2]) : S2L[src[p + 2]];
          expose(lin, k);
          col[0] = l2sFast(lin[0]); col[1] = l2sFast(lin[1]); col[2] = l2sFast(lin[2]);
        }
      }
      out[p] = col[0] * 255; out[p + 1] = col[1] * 255; out[p + 2] = col[2] * 255; out[p + 3] = 255;
    }
  }
  if (fx) applyEffects(out, w, h, fx);
  return out;
}

// ---- spatial effects: they depend on position / neighbours, so they cannot live in the LUT ------------
// Both work in linear light on the finished frame and are defined relative to the frame size, so the
// 640 px preview and the 1280 px export look the same.
export const FX_KEYS = ['glow', 'glowRadius', 'glowHue', 'glowTint', 'haze', 'hazeRadius', 'hazeSoft', 'hazeBloom', 'hazeVeil', 'vignette', 'vignetteSize', 'filmGrain', 'filmGrainSize', 'halation', 'filmSoft', 'hazeAir', 'hazeDepth']; // (膠片質感: film.js; hazeAir: 霧色取自參考圖, hazeDepth: 背景霧更濃)
const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

function boxBlur(buf, w, h, r) { // in place, 3 channels, sliding window (clamped edges)
  const tmp = new Float32Array(buf.length), n = 2 * r + 1;
  for (let y = 0; y < h; y++) for (let c = 0; c < 3; c++) {
    let acc = 0;
    for (let k = -r; k <= r; k++) acc += buf[(y * w + Math.min(w - 1, Math.max(0, k))) * 3 + c];
    for (let x = 0; x < w; x++) {
      tmp[(y * w + x) * 3 + c] = acc / n;
      acc += buf[(y * w + Math.min(w - 1, x + r + 1)) * 3 + c] - buf[(y * w + Math.max(0, x - r)) * 3 + c];
    }
  }
  for (let x = 0; x < w; x++) for (let c = 0; c < 3; c++) {
    let acc = 0;
    for (let k = -r; k <= r; k++) acc += tmp[(Math.min(h - 1, Math.max(0, k)) * w + x) * 3 + c];
    for (let y = 0; y < h; y++) {
      buf[(y * w + x) * 3 + c] = acc / n;
      acc += tmp[(Math.min(h - 1, y + r + 1) * w + x) * 3 + c] - tmp[(Math.max(0, y - r) * w + x) * 3 + c];
    }
  }
}

let vgCache = { key: '', gain: null }; // one gain map per frame size and setting (every frame of a job reuses it)
function vignetteGain(w, h, vig, size) {
  const key = `${w}x${h}:${vig}:${size}`;
  if (vgCache.key === key) return vgCache.gain;
  const g = new Float32Array(w * h), v0 = 0.75 - 0.5 * size, v1 = v0 + 0.6;
  for (let y = 0; y < h; y++) {
    const dy = (y + 0.5) / h * 2 - 1;
    for (let x = 0; x < w; x++) {
      const m = smooth(v0, v1, Math.hypot((x + 0.5) / w * 2 - 1, dy) / Math.SQRT2);
      g[y * w + x] = vig > 0 ? 1 - vig * 0.8 * m : 1 - vig * 0.6 * m;
    }
  }
  vgCache = { key, gain: g };
  return g;
}

// the frame (linear light, weighted per pixel by wfn(luminance)) averaged onto a ~320 px grid and blurred
function blurredGrid(img, w, h, wfn, radius) {
  const f = Math.max(1, Math.round(Math.max(w, h) / 320)), sw = Math.ceil(w / f), sh = Math.ceil(h / f);
  const small = new Float32Array(sw * sh * 3), cnt = new Float32Array(sw * sh);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = (y * w + x) * 4, r = S2L[img[p]], g = S2L[img[p + 1]], b = S2L[img[p + 2]];
    const k = wfn(0.2126 * r + 0.7152 * g + 0.0722 * b), q = (Math.floor(y / f) * sw + Math.floor(x / f));
    small[3 * q] += r * k; small[3 * q + 1] += g * k; small[3 * q + 2] += b * k; cnt[q]++;
  }
  for (let q = 0; q < sw * sh; q++) for (let c = 0; c < 3; c++) small[3 * q + c] /= cnt[q] || 1;
  const rad = Math.max(1, Math.round(radius * Math.max(sw, sh)));
  for (let pass = 0; pass < 3; pass++) boxBlur(small, sw, sh, rad); // ≈ Gaussian
  return { small, sw, sh, f };
}
function sampleGrid(G, x, y, out) { // bilinear, full-resolution pixel (x, y)
  const { small, sw, sh, f } = G;
  let gy = Math.min(sh - 1, Math.max(0, (y + 0.5) / f - 0.5)), gx = Math.min(sw - 1, Math.max(0, (x + 0.5) / f - 0.5));
  const y0 = Math.floor(gy), y1 = Math.min(sh - 1, y0 + 1), x0 = Math.floor(gx), x1 = Math.min(sw - 1, x0 + 1); gy -= y0; gx -= x0;
  for (let c = 0; c < 3; c++) {
    const a = small[(y0 * sw + x0) * 3 + c] + (small[(y0 * sw + x1) * 3 + c] - small[(y0 * sw + x0) * 3 + c]) * gx;
    const b = small[(y1 * sw + x0) * 3 + c] + (small[(y1 * sw + x1) * 3 + c] - small[(y1 * sw + x0) * 3 + c]) * gx;
    out[c] = a + (b - a) * gy;
  }
  return out;
}

/**
 * fx: {glow 0..1, glowRadius 0..1, haze 0..1, hazeRadius 0..1, vignette -1..1 (+ darkens edges, − lightens),
 * vignetteSize 0..1}. In place on RGBA.
 * 柔光 (glow): bright parts are blurred on a ~320 px grid and added back (a diffusion / bloom filter).
 * 朦朧 (haze), dreamy light like a mist filter in sunlight: detail softens a little; the lights bloom (their blurred
 * light is screened over darker surroundings — halation around a white dress against water, sunlit ground into
 * shade); and the scene's light, in its own colour, is scattered evenly as a veil, so shadows are lifted by warm light
 * in a sunlit scene rather than by grey fog. In an already bright scene the bloom is scaled down so it does not wash out.
 */
/** 光芒顏色: per-channel factors (linear light, luminance 1) for light tinted towards OKLab hue h, tint 0..1 */
export function glowTintRGB(hue, tint) {
  const t = Math.min(1, Math.max(0, tint));
  if (!t) return [1, 1, 1];
  const c = oklabToSrgb(0.75, 0.13 * t * Math.cos((hue * Math.PI) / 180), 0.13 * t * Math.sin((hue * Math.PI) / 180));
  const l = c.map((v) => S2L[Math.round(Math.min(1, Math.max(0, v)) * 255)]), Y = 0.2126 * l[0] + 0.7152 * l[1] + 0.0722 * l[2];
  return l.map((v) => v / Math.max(Y, 1e-4));
}
/** hazeMap: optional {data, w, h} grid of the person (0..1, smoothed AI mask) for 背景霧更濃 (fx.hazeDepth 0..1). */
export function applyEffects(img, w, h, fx, hazeMap = null) {
  const glow = Math.min(1, Math.max(0, fx.glow || 0)), vig = Math.min(1, Math.max(-1, fx.vignette || 0)), haze = Math.min(1, Math.max(0, fx.haze || 0));
  if (!glow && !vig && !haze) return img;
  const G = glow ? blurredGrid(img, w, h, (Y) => smooth(0.2, 0.85, Y), 0.008 + 0.05 * (fx.glowRadius ?? 0.5)) : null;
  const hr = fx.hazeRadius ?? 0.5;
  // the three parts of 朦朧, each 0..1 with 0.5 = the default mix (×2 at 1, off at 0): 柔化 / 發光 / 光幕
  const part = (v) => 2 * Math.min(1, Math.max(0, v ?? 0.5)), kSoft = 0.3 * part(fx.hazeSoft), kBloom = 1.8 * part(fx.hazeBloom), kVeil = part(fx.hazeVeil);
  const H = haze ? blurredGrid(img, w, h, () => 1, 0.004 + 0.02 * hr) : null; // soft copy (detail softening)
  const Bm = haze ? blurredGrid(img, w, h, (Y) => smooth(0.18, 0.75, Y), 0.02 + 0.07 * hr) : null; // bloom of the lights
  // veil: the scene's light (colour of its bright parts) scattered evenly over the frame, like sun flare in a lens:
  // dark areas are lifted by warm light in a sunlit scene, not by grey fog
  const veil = [0, 0, 0]; let bk = 1;
  if (Bm) {
    const n = Bm.sw * Bm.sh, acc = [0, 0, 0]; for (let q = 0; q < n; q++) for (let c = 0; c < 3; c++) acc[c] += Bm.small[3 * q + c] / n;
    // light colour, luminance-normalised; in an already bright scene (white walls, sky) the bloom and veil are scaled
    // down so lights still read as lights instead of the whole frame washing out
    const Y = 0.2126 * acc[0] + 0.7152 * acc[1] + 0.0722 * acc[2], lvl = 0.03 + 0.06 * Math.min(1, Y * 5);
    bk = Math.min(1, Math.sqrt(0.15 / Math.max(Y, 1e-4)));
    // 霧色取自參考圖: the veil takes the colour of the reference's air (luminance-normalised linear RGB), not the footage's
    // own light — warm sunlit footage under a cool misty reference got a warm veil
    const air = Array.isArray(fx.hazeAir) && fx.hazeAir.length === 3 ? fx.hazeAir : null;
    for (let c = 0; c < 3; c++) veil[c] = air ? lvl * air[c] : Y > 1e-4 ? lvl * acc[c] / Y : lvl;
  }
  // 光芒顏色 (colour wheel): the added light (柔光, 朦朧's bloom and veil) tinted towards an OKLab hue; the tint keeps
  // the light's brightness (channel factors normalised to luminance 1), tint 0 = the light's own colour
  const tm = glowTintRGB(fx.glowHue ?? 60, fx.glowTint || 0);
  const vg = vig ? vignetteGain(w, h, vig, fx.vignetteSize ?? 0.5) : null;
  // 背景霧更濃: haze thins on the person and thickens on the background (distant things hazier, like air between them and
  // the camera); the person mask is the AI one, smoothed over time like the other AI regions
  const depth = haze && hazeMap ? Math.min(1, Math.max(0, fx.hazeDepth || 0)) : 0;
  // per-column / per-row grid indices and weights, computed once per frame (a per-pixel lookup cost +40 % effects time)
  let pxi0, pxi1, ptx, pyi0, pyi1, pty;
  if (depth) {
    const M = hazeMap; pxi0 = new Int32Array(w); pxi1 = new Int32Array(w); ptx = new Float32Array(w); pyi0 = new Int32Array(h); pyi1 = new Int32Array(h); pty = new Float32Array(h);
    for (let x = 0; x < w; x++) { const g = Math.min(M.w - 1, Math.max(0, (x + 0.5) / w * M.w - 0.5)), g0 = Math.floor(g); pxi0[x] = g0; pxi1[x] = Math.min(M.w - 1, g0 + 1); ptx[x] = g - g0; }
    for (let y = 0; y < h; y++) { const g = Math.min(M.h - 1, Math.max(0, (y + 0.5) / h * M.h - 0.5)), g0 = Math.floor(g); pyi0[y] = g0 * M.w; pyi1[y] = Math.min(M.h - 1, g0 + 1) * M.w; pty[y] = g - g0; }
  }
  const lin = [0, 0, 0], bl = [0, 0, 0];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p = (y * w + x) * 4;
      lin[0] = S2L[img[p]]; lin[1] = S2L[img[p + 1]]; lin[2] = S2L[img[p + 2]];
      let hz = haze;
      if (depth) {
        const md = hazeMap.data, r0 = pyi0[y], r1 = pyi1[y], a0 = pxi0[x], a1 = pxi1[x], tx = ptx[x], ty = pty[y];
        const pr = (md[r0 + a0] * (1 - tx) + md[r0 + a1] * tx) * (1 - ty) + (md[r1 + a0] * (1 - tx) + md[r1 + a1] * tx) * ty;
        hz = Math.min(1, haze * (1 - 0.85 * depth * pr) * (1 + 0.5 * depth * (1 - pr)));
      }
      if (H) { // dreamy light: soften detail a little, let the lights bloom (screen), then the light-coloured veil
        sampleGrid(H, x, y, bl);
        for (let c = 0; c < 3; c++) lin[c] += Math.min(1, hz * kSoft) * (bl[c] - lin[c]);
        sampleGrid(Bm, x, y, bl);
        for (let c = 0; c < 3; c++) { const add = Math.min(1, hz * tm[c] * (kBloom * bk * Math.max(0, bl[c] - 0.6 * lin[c]) + kVeil * veil[c])); /* light spills into darker surroundings */ lin[c] = 1 - (1 - lin[c]) * (1 - add); }
      }
      if (G) { sampleGrid(G, x, y, bl); for (let c = 0; c < 3; c++) lin[c] += glow * 1.2 * tm[c] * bl[c]; }
      if (vg) { const g = vg[y * w + x]; lin[0] *= g; lin[1] *= g; lin[2] *= g; }
      const Y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
      if (Y > KNEE) { const s = rollOff(Y) / Y; lin[0] *= s; lin[1] *= s; lin[2] *= s; } // added light rolls off, never clips
      const mx = Math.max(lin[0], lin[1], lin[2]);
      if (mx > 1) { const Yn = Math.min(Y, 1), t = (mx - 1) / Math.max(mx - Yn, 1e-6); for (let c = 0; c < 3; c++) lin[c] += (Yn - lin[c]) * t; }
      img[p] = l2sFast(lin[0]) * 255; img[p + 1] = l2sFast(lin[1]) * 255; img[p + 2] = l2sFast(lin[2]) * 255;
    }
  }
  return img;
}

/**
 * 霧色取自參考圖: the colour of a reference's air — the mean colour of its brighter part (above its 60th percentile of
 * lightness), strongly coloured pixels (chroma > 0.09: objects, deep sky) left out — as luminance-normalised linear RGB
 * factors for the haze veil. img: ImageData-like (bars already removed). null when there is nothing to read.
 */
export function refAirRGB(img) {
  const d = img.data, Ls = [], ab = [], o = [0, 0, 0];
  for (let i = 0; i < d.length; i += 4 * 3) { srgbToOklab(d[i] / 255, d[i + 1] / 255, d[i + 2] / 255, o); Ls.push(o[0]); ab.push(o[1], o[2]); }
  if (!Ls.length) return null;
  const sorted = Ls.slice().sort((a, b) => a - b), cut = sorted[Math.floor(sorted.length * 0.6)];
  let sa = 0, sb = 0, n = 0;
  for (let k = 0; k < Ls.length; k++) { if (Ls[k] < cut) continue; const a = ab[2 * k], b = ab[2 * k + 1]; if (Math.hypot(a, b) > 0.09) continue; sa += a; sb += b; n++; }
  if (!n) return null;
  const c = oklabToSrgb(0.8, sa / n, sb / n), l = c.map((v) => S2L[Math.round(Math.min(1, Math.max(0, v)) * 255)]), Y = 0.2126 * l[0] + 0.7152 * l[1] + 0.0722 * l[2];
  return Y > 1e-4 ? l.map((v) => +(v / Y).toFixed(4)) : null;
}
