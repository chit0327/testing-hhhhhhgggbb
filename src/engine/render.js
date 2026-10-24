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

/**
 * src: RGBA Uint8ClampedArray (w×h). lut: {size, data} or null (identity). ev: {data, w, h} stops or null.
 * Returns a new RGBA Uint8ClampedArray.
 */
export function renderFrame(src, w, h, lut, ev, out = new Uint8ClampedArray(src.length)) {
  const N = lut ? lut.size : 0, L = lut ? lut.data : null, n1 = N - 1;
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
        const fr = src[p] / 255 * n1, fg = src[p + 1] / 255 * n1, fb = src[p + 2] / 255 * n1;
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
  return out;
}
