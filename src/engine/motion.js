// Two "hand-held, fast" lens / camera looks (the Hong Kong market sequence of a music video, after the step-printed
// slow-shutter images of Wong Kar-wai and Christopher Doyle):
//
// 鏡頭邊緣模糊 (lensBlur): the middle of the picture stays sharp, the edges smear — outwards like a zoom (放射) or round
// the middle like an old Petzval lens (旋轉). Per pixel, the smear grows from the edge of the sharp area (lensBlurSize)
// to the frame corner; it is averaged in linear light along a short line (radial or tangential) of up to 6 % of the
// long side, so lights streak brightly as in a long exposure. Centre (lensBlurX, lensBlurY) 0..1.
//
// 抽格拖影 (stepTrail, stepHold): a slow shutter and step printing. Each frame is added to an echo of the frames before
// it (linear light; trail 0..1 → how long the smear lasts), and the picture shown is refreshed only every stepHold
// frames (1 = every frame: shutter drag alone; 2–4 = the jerky, held images of step printing). Needs consecutive frames
// (a state per video job / playback); a single still frame is left as it is.
import { S2L } from './color.js';

const l2s = (v) => 255 * (v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(Math.max(0, v), 1 / 2.4) - 0.055);
const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
const c01 = (v) => Math.min(1, Math.max(0, +v || 0));

export function lensPlan(fx) {
  const amt = c01(fx && fx.lensBlur);
  if (!amt) return null;
  return { amt, size: c01(fx.lensBlurSize ?? 0.4), swirl: fx.lensBlurType === 'swirl', x: Number.isFinite(fx.lensBlurX) ? c01(fx.lensBlurX) : 0.5, y: Number.isFinite(fx.lensBlurY) ? c01(fx.lensBlurY) : 0.5 };
}

/** 鏡頭邊緣模糊, in place on RGBA */
export function applyLensBlur(img, w, h, fx) {
  const P = lensPlan(fx);
  if (!P) return img;
  // the smear is sampled from a copy at half the size above ~700 px (a smeared area has no fine detail to lose:
  // and the smear computed on the copy's grid: 473 ms a frame at 1280 × 720 before), in linear light
  const f = Math.max(w, h) > 700 ? 2 : 1, sw = Math.ceil(w / f), sh = Math.ceil(h / f), lin = new Float32Array(sw * sh * 3), cnt = new Float32Array(sw * sh);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const q = ((y / f) | 0) * sw + ((x / f) | 0), p = 4 * (y * w + x); lin[3 * q] += S2L[img[p]]; lin[3 * q + 1] += S2L[img[p + 1]]; lin[3 * q + 2] += S2L[img[p + 2]]; cnt[q]++; }
  for (let q = 0; q < sw * sh; q++) { const k = 1 / (cnt[q] || 1); lin[3 * q] *= k; lin[3 * q + 1] *= k; lin[3 * q + 2] *= k; }
  const cx = P.x * (w - 1), cy = P.y * (h - 1), half = 0.5 * Math.hypot(w, h), maxL = 0.06 * Math.max(w, h) * P.amt;
  const r0 = 0.05 + 0.6 * P.size, r1 = r0 + 0.5; // (sharp up to r0 of the half diagonal, full smear by r1)
  const at = (X, Y, o) => { // bilinear on the copy, clamped (X, Y in full-size pixels)
    const x = Math.min(sw - 1, Math.max(0, (X + 0.5) / f - 0.5)), y = Math.min(sh - 1, Math.max(0, (Y + 0.5) / f - 0.5));
    const x0 = x | 0, y0 = y | 0, x1 = Math.min(sw - 1, x0 + 1), y1 = Math.min(sh - 1, y0 + 1), tx = x - x0, ty = y - y0;
    const a = 3 * (y0 * sw + x0), b = 3 * (y0 * sw + x1), c = 3 * (y1 * sw + x0), d = 3 * (y1 * sw + x1);
    for (let k = 0; k < 3; k++) o[k] += (lin[a + k] * (1 - tx) + lin[b + k] * tx) * (1 - ty) + (lin[c + k] * (1 - tx) + lin[d + k] * tx) * ty;
  };
  // the smear itself on the copy's grid (its own pixel centres), then brought back to full size and mixed in by how
  // smeared each place is — the work per pixel is done on a quarter of the pixels
  const sm = new Float32Array(sw * sh * 4), acc = [0, 0, 0]; // (r, g, b, mix)
  for (let gy = 0; gy < sh; gy++) for (let gx = 0; gx < sw; gx++) {
    const x = (gx + 0.5) * f - 0.5, y = (gy + 0.5) * f - 0.5, dx = x - cx, dy = y - cy, r = Math.hypot(dx, dy), t = smooth(r0, r1, r / half);
    const q = 4 * (gy * sw + gx), L = t * maxL;
    if (t < 0.01 || r < 1 || L < 0.75) continue;
    // direction: away from the centre (zoom) or round it (swirl); the swirl's arc is a short straight line here
    const ux = P.swirl ? -dy / r : dx / r, uy = P.swirl ? dx / r : dy / r, N = Math.min(16, Math.max(3, Math.ceil(L / (1.5 * f))));
    acc[0] = acc[1] = acc[2] = 0;
    for (let s = 0; s < N; s++) { const o = (s / (N - 1) - 0.5) * L; at(x + ux * o, y + uy * o, acc); }
    sm[q] = acc[0] / N; sm[q + 1] = acc[1] / N; sm[q + 2] = acc[2] / N; sm[q + 3] = Math.min(1, L / (2 * f)); // (a short smear: mixed with the sharp pixel)
  }
  for (let y = 0; y < h; y++) {
    const gy = Math.min(sh - 1, Math.max(0, (y + 0.5) / f - 0.5)), y0 = gy | 0, y1 = Math.min(sh - 1, y0 + 1), ty = gy - y0;
    for (let x = 0; x < w; x++) {
      const gx = Math.min(sw - 1, Math.max(0, (x + 0.5) / f - 0.5)), x0 = gx | 0, x1 = Math.min(sw - 1, x0 + 1), tx = gx - x0;
      const i00 = 4 * (y0 * sw + x0), i01 = 4 * (y0 * sw + x1), i10 = 4 * (y1 * sw + x0), i11 = 4 * (y1 * sw + x1);
      const w00 = (1 - tx) * (1 - ty), w01 = tx * (1 - ty), w10 = (1 - tx) * ty, w11 = tx * ty;
      const m = sm[i00 + 3] * w00 + sm[i01 + 3] * w01 + sm[i10 + 3] * w10 + sm[i11 + 3] * w11;
      if (m < 0.004) continue;
      const p = 4 * (y * w + x);
      for (let k = 0; k < 3; k++) {
        // (premultiplied by the mix, so a cell with no smear does not pull its neighbours' colour towards black)
        const v = (sm[i00 + k] * sm[i00 + 3] * w00 + sm[i01 + k] * sm[i01 + 3] * w01 + sm[i10 + k] * sm[i10 + 3] * w10 + sm[i11 + k] * sm[i11 + 3] * w11) / m;
        img[p + k] = l2s(S2L[img[p + k]] + (v - S2L[img[p + k]]) * m);
      }
    }
  }
  return img;
}

export const stepPlan = (fx) => {
  const trail = c01(fx && fx.stepTrail), hold = Math.max(1, Math.min(4, Math.round(+(fx && fx.stepHold) || 1)));
  return trail > 0 || hold > 1 ? { trail, hold } : null;
};
/**
 * 抽格拖影 for one frame of a sequence, in place on RGBA. state: {} kept by the caller per video job / playback; index:
 * the frame's number in the clip — a jump back or of more than 4 frames starts afresh. Returns true when applied.
 */
export function stepFrame(img, w, h, fx, state, index) {
  const P = stepPlan(fx);
  if (!P || !state) return false;
  const n = w * h, fresh = !state.acc || state.w !== w || state.h !== h || !(index > state.index && index - state.index <= 4);
  const a = 0.92 * P.trail; // (echo: how much of the smear so far stays each frame)
  if (fresh) { state.acc = new Float32Array(n * 3); state.held = null; state.w = w; state.h = h; state.count = 0; state.wsum = 0; }
  // a running average with the newer frames counting more (weights a^age, normalised): the first frame of a clip does
  // not linger brighter than the frames after it, and a still part of the picture stays exactly as it is
  const acc = state.acc;
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) acc[3 * i + k] = a * acc[3 * i + k] + S2L[img[4 * i + k]];
  state.wsum = a * state.wsum + 1;
  state.index = index;
  if (state.count % P.hold === 0 || !state.held) { // a new image is "printed"
    const held = state.held && state.held.length === n * 4 ? state.held : new Uint8ClampedArray(n * 4);
    const k = 1 / state.wsum;
    for (let i = 0; i < n; i++) { held[4 * i] = l2s(acc[3 * i] * k); held[4 * i + 1] = l2s(acc[3 * i + 1] * k); held[4 * i + 2] = l2s(acc[3 * i + 2] * k); held[4 * i + 3] = 255; }
    state.held = held;
  }
  state.count++;
  img.set(state.held);
  return true;
}
