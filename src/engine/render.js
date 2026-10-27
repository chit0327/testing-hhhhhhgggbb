// The one renderer used by the dialog preview, the adjustment preview and the export (via the worker).
// Order: the global look first (reference grade + basic adjustments, one LUT fitted on the untouched
// frames), then the Lighting Intent's local exposure in LINEAR light on the graded pixels. Applying the
// local exposure last makes it exact and predictable: +0.6 EV in a region is +0.6 EV in the output
// (earlier, applying it before the LUT let the reference's contrast curve flatten brightened people).
import { S2L, l2sFast } from './color.js';

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
  const col = [0, 0, 0], lin = [0, 0, 0];
  for (let y = 0; y < h; y++) {
    let fy = 0, y0 = 0, y1 = 0;
    if (gain) { fy = Math.min(gh - 1, Math.max(0, (y + 0.5) * gh / h - 0.5)); y0 = Math.floor(fy); y1 = Math.min(gh - 1, y0 + 1); fy -= y0; }
    for (let x = 0; x < w; x++) {
      const p = (y * w + x) * 4;
      if (L) { // trilinear 3D LUT lookup (global look)
        const fr = IX ? IX[src[p]] : src[p] / 255 * n1, fg = IX ? IX[src[p + 1]] : src[p + 1] / 255 * n1, fb = IX ? IX[src[p + 2]] : src[p + 2] / 255 * n1;
        const r0 = Math.min(n1 - 1, fr | 0), g0 = Math.min(n1 - 1, fg | 0), b0 = Math.min(n1 - 1, fb | 0);
        const dr = fr - r0, dg = fg - g0, db = fb - b0, sR = N * N * 3, sG = N * 3;
        for (let c = 0; c < 3; c++) {
          const i000 = ((r0 * N + g0) * N + b0) * 3 + c;
          const c00 = L[i000] + (L[i000 + 3] - L[i000]) * db;
          const c01 = L[i000 + sG] + (L[i000 + sG + 3] - L[i000 + sG]) * db;
          const c10 = L[i000 + sR] + (L[i000 + sR + 3] - L[i000 + sR]) * db;
          const c11 = L[i000 + sR + sG] + (L[i000 + sR + sG + 3] - L[i000 + sR + sG]) * db;
          const c0 = c00 + (c01 - c00) * dg, c1 = c10 + (c11 - c10) * dg;
          col[c] = c0 + (c1 - c0) * dr;
        }
        for (let li = 0; li < LY.length; li++) { // region layers: same lookup in the layer's LUT, mixed in by its mask
          const ly = LY[li], sm = ly.mask, SL = ly.lut.data;
          const mx = Math.min(sm.w - 1, Math.max(0, Math.round((x + 0.5) * sm.w / w - 0.5))), my = Math.min(sm.h - 1, Math.max(0, Math.round((y + 0.5) * sm.h / h - 0.5)));
          const m = sm.data[my * sm.w + mx] * ly.k;
          if (m > 0.004) {
            for (let c = 0; c < 3; c++) {
              const i000 = ((r0 * N + g0) * N + b0) * 3 + c;
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
        let fx = Math.min(gw - 1, Math.max(0, (x + 0.5) * gw / w - 0.5));
        const x0 = Math.floor(fx), x1 = Math.min(gw - 1, x0 + 1); fx -= x0;
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
export const FX_KEYS = ['glow', 'glowRadius', 'vignette', 'vignetteSize'];
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

/**
 * fx: {glow 0..1, glowRadius 0..1, vignette -1..1 (+ darkens edges, − lightens), vignetteSize 0..1}. In place on RGBA.
 * 柔光 (glow): bright parts are blurred on a ~320 px grid and added back (a diffusion / bloom filter).
 */
export function applyEffects(img, w, h, fx) {
  const glow = Math.min(1, Math.max(0, fx.glow || 0)), vig = Math.min(1, Math.max(-1, fx.vignette || 0));
  if (!glow && !vig) return img;
  let small = null, sw = 0, sh = 0, f = 1;
  if (glow) {
    f = Math.max(1, Math.round(Math.max(w, h) / 320)); sw = Math.ceil(w / f); sh = Math.ceil(h / f);
    small = new Float32Array(sw * sh * 3);
    const cnt = new Float32Array(sw * sh);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const p = (y * w + x) * 4, r = S2L[img[p]], g = S2L[img[p + 1]], b = S2L[img[p + 2]];
      const k = smooth(0.2, 0.85, 0.2126 * r + 0.7152 * g + 0.0722 * b), q = (Math.floor(y / f) * sw + Math.floor(x / f));
      small[3 * q] += r * k; small[3 * q + 1] += g * k; small[3 * q + 2] += b * k; cnt[q]++;
    }
    for (let q = 0; q < sw * sh; q++) for (let c = 0; c < 3; c++) small[3 * q + c] /= cnt[q] || 1;
    const rad = Math.max(1, Math.round((0.008 + 0.05 * (fx.glowRadius ?? 0.5)) * Math.max(sw, sh)));
    for (let pass = 0; pass < 3; pass++) boxBlur(small, sw, sh, rad); // ≈ Gaussian
  }
  const vg = vig ? vignetteGain(w, h, vig, fx.vignetteSize ?? 0.5) : null;
  const lin = [0, 0, 0];
  for (let y = 0; y < h; y++) {
    let gy = 0, y0 = 0, y1 = 0;
    if (small) { gy = Math.min(sh - 1, Math.max(0, (y + 0.5) / f - 0.5)); y0 = Math.floor(gy); y1 = Math.min(sh - 1, y0 + 1); gy -= y0; }
    for (let x = 0; x < w; x++) {
      const p = (y * w + x) * 4;
      lin[0] = S2L[img[p]]; lin[1] = S2L[img[p + 1]]; lin[2] = S2L[img[p + 2]];
      if (small) {
        let gx = Math.min(sw - 1, Math.max(0, (x + 0.5) / f - 0.5));
        const x0 = Math.floor(gx), x1 = Math.min(sw - 1, x0 + 1); gx -= x0;
        for (let c = 0; c < 3; c++) {
          const a = small[(y0 * sw + x0) * 3 + c] + (small[(y0 * sw + x1) * 3 + c] - small[(y0 * sw + x0) * 3 + c]) * gx;
          const b = small[(y1 * sw + x0) * 3 + c] + (small[(y1 * sw + x1) * 3 + c] - small[(y1 * sw + x0) * 3 + c]) * gx;
          lin[c] += glow * 1.2 * (a + (b - a) * gy);
        }
      }
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
