// 對焦（景深）: a shallow depth of field from an AI depth map, like a wide-aperture lens. The blur of each point grows
// with its distance in depth from the focus — behind it and in front of it (the circle of confusion of a lens is
// proportional to the difference in inverse depth, which is what the depth model gives) — so the picture gets the
// gradual near-to-far softening a real lens has instead of "subject sharp / everything else equally blurred".
//
// Depth: Depth Anything V2 Small (src/depthworker.js) gives relative inverse depth (larger = nearer) on a small grid.
// It is put on a 0..1 scale per frame (1st..99th percentile; the worker smooths that range over the frames of a video).
// Focus: on the person (when the tapped point is on one: the median depth of the AI person mask, so the focus follows
// them as they move) or on the tapped point (median depth around it).
//
// Rendering, layered (the way compositing software does lens blur): the frame is cut into depth slices; each slice is
// blurred with a round aperture (a disc: bright points open into round bokeh) of its own radius, far to near, each laid
// "over" the ones behind it with its blurred coverage — so a blurred foreground spreads over a sharp subject and a sharp
// subject does not leak into the blurred background behind it. Large blurs run on a coarser grid (the disc spans at most
// ~6 cells), the result is brought back to full size and mixed with the original pixels where the visible blur is
// under ~1 px, so what is in focus keeps every pixel of detail.
import { S2L } from './color.js';

const clamp01 = (v) => Math.min(1, Math.max(0, +v || 0));
const SLICES = 10;

/** dof: {on, amount 0..1, x, y (0..1, the tapped point), person (focus on the person), rel (where on the person)} → plan or null */
export function dofPlan(dof) {
  if (!dof || !dof.on) return null;
  const amount = clamp01(dof.amount ?? 0.5);
  if (!amount || !Number.isFinite(dof.x) || !Number.isFinite(dof.y)) return null;
  return { amount, x: clamp01(dof.x), y: clamp01(dof.y), person: !!dof.person, rel: Number.isFinite(dof.rel) ? clamp01(dof.rel) : null };
}
/** blur radius in px per unit of (0..1) depth difference, at the frame's size */
export const dofScale = (plan, w, h) => plan.amount * 0.04 * Math.max(w, h);

/** the 1st and 99th percentile of a raw depth grid */
export function depthRange(depth) {
  const s = Float32Array.from(depth.data).sort(), n = s.length;
  return [s[Math.floor(n * 0.01)], s[Math.min(n - 1, Math.floor(n * 0.99))]];
}
/** raw depth grid → 0..1 (1 = near) with the given range */
export function normaliseDepth(depth, [lo, hi]) {
  const d = new Float32Array(depth.data.length), k = 1 / Math.max(1e-6, hi - lo);
  for (let i = 0; i < d.length; i++) d[i] = Math.min(1, Math.max(0, (depth.data[i] - lo) * k));
  return { data: d, w: depth.w, h: depth.h };
}

const median = (a) => { if (!a.length) return null; a.sort((x, y) => x - y); return a[a.length >> 1]; };
/**
 * the focus depth (0..1). On the person (seg {person: grid, mw, mh}): the median depth of the band of the person at the
 * tapped height (plan.rel: 0 = top of the person, 1 = bottom; ±8 % of the person's height) — a tap on the face keeps the
 * face in focus as the person moves (the whole body's median was nearer than the face: the lower body is closer to the
 * camera). Elsewhere: the median around the tapped point.
 */
export function focusDepth(dn, plan, seg = null) {
  const { data, w, h } = dn;
  if (plan.person && seg && seg.person) {
    const { person, mw, mh } = seg, at = (x, y) => person[Math.min(mh - 1, Math.floor((y + 0.5) / h * mh)) * mw + Math.min(mw - 1, Math.floor((x + 0.5) / w * mw))];
    let top = -1, bot = -1;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (at(x, y) >= 0.6) { if (top < 0) top = y; bot = y; break; }
    if (top >= 0) {
      const rel = Number.isFinite(plan.rel) ? plan.rel : 0.15, c = top + rel * (bot - top), band = Math.max(1, 0.08 * (bot - top)), v = [];
      for (let y = Math.max(0, Math.floor(c - band)); y <= Math.min(h - 1, Math.ceil(c + band)); y++) for (let x = 0; x < w; x++) if (at(x, y) >= 0.6) v.push(data[y * w + x]);
      if (v.length >= 4) return median(v);
    }
  }
  const cx = plan.x * w - 0.5, cy = plan.y * h - 0.5, r = Math.max(1, Math.round(Math.min(w, h) * 0.02)), v = [];
  for (let y = Math.round(cy) - r; y <= Math.round(cy) + r; y++) for (let x = Math.round(cx) - r; x <= Math.round(cx) + r; x++) {
    if (x >= 0 && y >= 0 && x < w && y < h) v.push(data[y * w + x]);
  }
  return median(v) ?? 0.5;
}

/** where on the person a tap at (x, y) is: 0 = the top of the person, 1 = the bottom (seg as focusDepth's); null if not on one */
export function personRel(x, y, seg) {
  if (!seg || !seg.person) return null;
  const { person, mw, mh } = seg, gx = Math.min(mw - 1, Math.floor(x * mw)), gy = Math.min(mh - 1, Math.floor(y * mh));
  if (!(person[gy * mw + gx] >= 0.5)) return null;
  let top = -1, bot = -1;
  for (let r = 0; r < mh; r++) for (let c = 0; c < mw; c++) if (person[r * mw + c] >= 0.6) { if (top < 0) top = r; bot = r; break; }
  return top < 0 || bot <= top ? 0.15 : Math.min(1, Math.max(0, (gy - top) / (bot - top)));
}

/** round-aperture blur of a premultiplied [r, g, b, a] grid, radius R cells (row chords from prefix sums), normalised by area */
function disc(buf, w, h, R) {
  const r = Math.round(R);
  if (r < 1) return buf;
  const W1 = w + 1, P = new Float32Array(W1 * h * 4), out = new Float32Array(buf.length);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < 4; c++) P[(y * W1 + x + 1) * 4 + c] = P[(y * W1 + x) * 4 + c] + buf[(y * w + x) * 4 + c];
  const dys = [], cs = []; let n = 0;
  for (let dy = -r; dy <= r; dy++) { const c = Math.floor(Math.sqrt(Math.max(0, (r + 0.5) ** 2 - dy * dy))); dys.push(dy); cs.push(c); n += 2 * c + 1; }
  const inv = 1 / n;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (let k = 0; k < dys.length; k++) {
      const yy = Math.min(h - 1, Math.max(0, y + dys[k])), a = (yy * W1 + Math.max(0, x - cs[k])) * 4, b = (yy * W1 + Math.min(w, x + cs[k] + 1)) * 4;
      s0 += P[b] - P[a]; s1 += P[b + 1] - P[a + 1]; s2 += P[b + 2] - P[a + 2]; s3 += P[b + 3] - P[a + 3];
    }
    const o = (y * w + x) * 4; out[o] = s0 * inv; out[o + 1] = s1 * inv; out[o + 2] = s2 * inv; out[o + 3] = s3 * inv;
  }
  return out;
}

/**
 * In place on RGBA (w×h): dn = 0..1 depth grid {data, w, h} (1 = near), df = focus depth, K = blur px per unit depth.
 * Returns {maxRadius} (px) for the record.
 */
export function applyDof(img, w, h, dn, df, K) {
  const f0 = Math.max(1, Math.round(Math.max(w, h) / 640)), sw = Math.ceil(w / f0), sh = Math.ceil(h / f0), n = sw * sh;
  // the frame (linear light) and its depth on the base grid
  const lin = new Float32Array(n * 3), cnt = new Float32Array(n), dg = new Float32Array(n);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = (y * w + x) * 4, q = Math.floor(y / f0) * sw + Math.floor(x / f0);
    lin[3 * q] += S2L[img[p]]; lin[3 * q + 1] += S2L[img[p + 1]]; lin[3 * q + 2] += S2L[img[p + 2]]; cnt[q]++;
  }
  for (let q = 0; q < n; q++) { const k = 1 / (cnt[q] || 1); lin[3 * q] *= k; lin[3 * q + 1] *= k; lin[3 * q + 2] *= k; }
  const depthAt = (fx, fy) => { // bilinear on the depth grid, (fx, fy) in 0..1
    const gx = Math.min(dn.w - 1, Math.max(0, fx * dn.w - 0.5)), gy = Math.min(dn.h - 1, Math.max(0, fy * dn.h - 0.5));
    const x0 = gx | 0, y0 = gy | 0, x1 = Math.min(dn.w - 1, x0 + 1), y1 = Math.min(dn.h - 1, y0 + 1), tx = gx - x0, ty = gy - y0, d = dn.data;
    return (d[y0 * dn.w + x0] * (1 - tx) + d[y0 * dn.w + x1] * tx) * (1 - ty) + (d[y1 * dn.w + x0] * (1 - tx) + d[y1 * dn.w + x1] * tx) * ty;
  };
  for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) dg[y * sw + x] = depthAt((x + 0.5) / sw, (y + 0.5) / sh);
  // slices far → near, "over" composite (premultiplied); vis = the blur radius of what is visible (for the final mix)
  const acc = new Float32Array(n * 4), vis = new Float32Array(n);
  let maxR = 0;
  // slice centres one step apart with one exactly on the focus depth (in focus = no blur at all), covering 0..1
  const step = 1 / (SLICES - 1), centres = [];
  for (let k = -Math.ceil(df / step) - 1; df + k * step <= 1 + step; k++) if (df + k * step >= -step) centres.push(df + k * step);
  for (const c of centres) {
    const rPx = Math.abs(c - df) * K, rb = rPx / f0;
    const g = Math.max(1, Math.ceil(rb / 6)), gw = Math.ceil(sw / g), gh = Math.ceil(sh / g);
    const buf = new Float32Array(gw * gh * 4), gc = new Float32Array(gw * gh);
    let any = false;
    for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) {
      const q = y * sw + x, m = Math.max(0, 1 - Math.abs(dg[q] - c) * (SLICES - 1)), o = (Math.floor(y / g) * gw + Math.floor(x / g));
      gc[o]++;
      if (m <= 0) continue;
      any = true;
      buf[4 * o] += lin[3 * q] * m; buf[4 * o + 1] += lin[3 * q + 1] * m; buf[4 * o + 2] += lin[3 * q + 2] * m; buf[4 * o + 3] += m;
    }
    if (!any) continue;
    for (let o = 0; o < gw * gh; o++) { const k = 1 / (gc[o] || 1); for (let ch = 0; ch < 4; ch++) buf[4 * o + ch] *= k; }
    const bl = disc(buf, gw, gh, rb / g);
    if (rPx > maxR) maxR = rPx;
    for (let y = 0; y < sh; y++) {
      const gy = Math.min(gh - 1, Math.max(0, (y + 0.5) / g - 0.5)), y0 = gy | 0, y1 = Math.min(gh - 1, y0 + 1), ty = gy - y0;
      for (let x = 0; x < sw; x++) {
        const gx = Math.min(gw - 1, Math.max(0, (x + 0.5) / g - 0.5)), x0 = gx | 0, x1 = Math.min(gw - 1, x0 + 1), tx = gx - x0;
        const w00 = (1 - tx) * (1 - ty), w01 = tx * (1 - ty), w10 = (1 - tx) * ty, w11 = tx * ty;
        const i00 = (y0 * gw + x0) * 4, i01 = (y0 * gw + x1) * 4, i10 = (y1 * gw + x0) * 4, i11 = (y1 * gw + x1) * 4, q = y * sw + x;
        const a = bl[i00 + 3] * w00 + bl[i01 + 3] * w01 + bl[i10 + 3] * w10 + bl[i11 + 3] * w11;
        if (a <= 1e-5) continue;
        const keep = 1 - Math.min(1, a);
        for (let ch = 0; ch < 3; ch++) acc[4 * q + ch] = bl[i00 + ch] * w00 + bl[i01 + ch] * w01 + bl[i10 + ch] * w10 + bl[i11 + ch] * w11 + acc[4 * q + ch] * keep;
        acc[4 * q + 3] = a + acc[4 * q + 3] * keep;
        vis[q] = rPx * Math.min(1, a) + vis[q] * keep;
      }
    }
  }
  // back to full size: the blurred result where the visible blur is ≥ ~1 px, the original pixels where in focus
  for (let y = 0; y < h; y++) {
    const gy = Math.min(sh - 1, Math.max(0, (y + 0.5) / f0 - 0.5)), y0 = gy | 0, y1 = Math.min(sh - 1, y0 + 1), ty = gy - y0;
    for (let x = 0; x < w; x++) {
      const gx = Math.min(sw - 1, Math.max(0, (x + 0.5) / f0 - 0.5)), x0 = gx | 0, x1 = Math.min(sw - 1, x0 + 1), tx = gx - x0;
      const q00 = y0 * sw + x0, q01 = y0 * sw + x1, q10 = y1 * sw + x0, q11 = y1 * sw + x1;
      const w00 = (1 - tx) * (1 - ty), w01 = tx * (1 - ty), w10 = (1 - tx) * ty, w11 = tx * ty;
      const r = vis[q00] * w00 + vis[q01] * w01 + vis[q10] * w10 + vis[q11] * w11, k = Math.min(1, Math.max(0, (r - 0.5) / 1.5));
      if (k <= 0) continue;
      const p = (y * w + x) * 4;
      for (let ch = 0; ch < 3; ch++) {
        const v = (q) => acc[4 * q + ch] / Math.max(acc[4 * q + 3], 1e-6);
        const b = v(q00) * w00 + v(q01) * w01 + v(q10) * w10 + v(q11) * w11, o = S2L[img[p + ch]], m = Math.min(1, Math.max(0, o + (b - o) * k));
        img[p + ch] = 255 * (m <= 0.0031308 ? 12.92 * m : 1.055 * Math.pow(m, 1 / 2.4) - 0.055);
      }
    }
  }
  return { maxRadius: maxR };
}
