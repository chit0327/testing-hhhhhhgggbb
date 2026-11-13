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
import { rollOff, boxBlur } from './render.js';
import { guidedFilter } from './masks.js';
import { hash3 } from './film.js';

const clamp01 = (v) => Math.min(1, Math.max(0, +v || 0));
const SLICES = 10;

/**
 * dof: {on, amount 0..1, x, y (0..1, the tapped point), person (focus on the person), rel (where on the person), area,
 * glow, range, grain, mode, angle} → plan or null.
 * mode 'lens' (default): from the AI depth. mode 'tilt' (移軸／微縮): no depth model — a band across the frame through
 * (x, y) at angle (−1..1 = ±45°) stays sharp (its width from area) and the blur grows above and below it, like a
 * tilted lens plane (the "miniature" look). grain 0..1: 模糊處噪點 — the frame's own noise put back where it was blurred.
 */
export function dofPlan(dof) {
  if (!dof || !dof.on) return null;
  const amount = clamp01(dof.amount ?? 0.5), tilt = dof.mode === 'tilt';
  if (!amount || !Number.isFinite(dof.y) || (!tilt && !Number.isFinite(dof.x))) return null;
  const num = (v, d) => (Number.isFinite(v) ? clamp01(v) : d);
  return { amount, mode: tilt ? 'tilt' : 'lens', x: num(dof.x, 0.5), y: clamp01(dof.y), person: !tilt && !!dof.person, rel: Number.isFinite(dof.rel) ? clamp01(dof.rel) : null,
    glow: num(dof.glow, 0.5), range: num(dof.range, 0.5), grain: num(dof.grain, 0.5),
    angle: tilt && Number.isFinite(dof.angle) ? Math.min(1, Math.max(-1, dof.angle)) : 0,
    area: tilt ? num(dof.area, 0.3) : dof.person ? 1 : num(dof.area, 0.35) }; // 焦點範圍: a tap on the background keeps only its surroundings sharp
}
/**
 * 移軸: the depth grid a tilted focus plane gives (0..1, the focus at 0.5): distance from the line through (x, y) at the
 * plan's angle, in frame heights; inside the band (width 0.04 + 0.5·area) 0.5, outside it falling 1.5 per frame height.
 */
export function tiltDepth(plan, w, h) {
  const gw = 160, gh = Math.max(2, Math.round(160 * h / w)), d = new Float32Array(gw * gh), a = plan.angle * Math.PI / 4;
  const ca = Math.cos(a), sa = Math.sin(a), half = (0.04 + 0.5 * plan.area) / 2, ar = w / h;
  for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) {
    const dx = ((x + 0.5) / gw - plan.x) * ar, dy = (y + 0.5) / gh - plan.y, v = Math.abs(dy * ca + dx * sa); // (positive angle: the band rises to the right)
    d[y * gw + x] = 0.5 - 1.5 * Math.max(0, v - half);
  }
  return { data: d, w: gw, h: gh };
}
/** the frame's noise (σ in 8-bit code values of lightness), from each pixel minus the mean of its 4 neighbours: the
 *  25th percentile of that (flat areas: noise only; edges and texture sit higher) over 0.3186 (|N(0, 1)|'s 25th
 *  percentile) and √1.25 (the neighbours' mean carries noise too) */
export function noiseSigma(img, w, h) {
  const v = [], Y = (p) => 0.299 * img[p] + 0.587 * img[p + 1] + 0.114 * img[p + 2], st = Math.max(1, Math.round(Math.sqrt(w * h / 40000)));
  for (let y = 1; y < h - 1; y += st) for (let x = 1; x < w - 1; x += st) {
    const p = (y * w + x) * 4;
    v.push(Math.abs(Y(p) - (Y(p - 4) + Y(p + 4) + Y(p - 4 * w) + Y(p + 4 * w)) / 4));
  }
  v.sort((a, b) => a - b);
  return v.length ? v[v.length >> 2] / 0.3186 / Math.sqrt(1.25) : 0;
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
 * the people of a person mask as separate figures (connected parts of mask ≥ 0.5 on its grid, 8-neighbours); the one at
 * (fx, fy) 0..1 — or, when the point is not on anyone, the nearest one (its centre within a fifth of the frame) — as
 * {cells: Set, top, bot (grid rows), cx, cy (centre 0..1)} or null. A focus on "the person" used to mean every person in
 * the frame: with a big figure close to the camera, the tapped face further back went out of focus.
 */
export function figureAt(seg, fx, fy) {
  const { person, mw, mh } = seg, lab = new Int32Array(mw * mh).fill(-1), figs = [];
  for (let i = 0; i < mw * mh; i++) {
    if (lab[i] >= 0 || !(person[i] >= 0.5)) continue;
    const st = [i], cells = []; lab[i] = figs.length;
    while (st.length) {
      const q = st.pop(), x = q % mw, y = (q / mw) | 0; cells.push(q);
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const nx = x + dx, ny = y + dy; if (nx < 0 || ny < 0 || nx >= mw || ny >= mh) continue;
        const r = ny * mw + nx; if (lab[r] < 0 && person[r] >= 0.5) { lab[r] = figs.length; st.push(r); }
      }
    }
    let sx = 0, sy = 0, top = mh, bot = -1; for (const q of cells) { const y = (q / mw) | 0; sx += q % mw; sy += y; if (y < top) top = y; if (y > bot) bot = y; }
    figs.push({ cells: new Set(cells), top, bot, cx: (sx / cells.length + 0.5) / mw, cy: (sy / cells.length + 0.5) / mh, n: cells.length });
  }
  if (!figs.length) return null;
  const gx = Math.min(mw - 1, Math.max(0, Math.floor(fx * mw))), gy = Math.min(mh - 1, Math.max(0, Math.floor(fy * mh))), hit = lab[gy * mw + gx];
  if (hit >= 0 && figs[hit].n >= 4) return figs[hit];
  let best = null, bd = 0.2; // (not on anyone: the nearest figure, if close)
  for (const f of figs) { if (f.n < 4) continue; const d = Math.hypot(f.cx - fx, f.cy - fy); if (d < bd) { bd = d; best = f; } }
  return best;
}
/**
 * the focus depth (0..1). On a person (seg {person: grid, mw, mh}): the tapped figure only (figureAt; in a video the
 * caller passes where that figure was in the previous frame, so the focus follows them), the median depth of the band
 * of that figure at the tapped height (plan.rel: 0 = top of the figure, 1 = bottom; ±8 % of its height) — a tap on the
 * face keeps the face in focus (the whole body's median was nearer than the face: the lower body is closer). Elsewhere:
 * the median around the tapped point. Returns the depth; at.fig gets the figure used (for following it).
 */
export function focusDepth(dn, plan, seg = null, at = null) {
  const { data, w, h } = dn;
  if (plan.person && seg && seg.person) {
    const { mw, mh } = seg, ax = at && Number.isFinite(at.x) ? at.x : plan.x, ay = at && Number.isFinite(at.y) ? at.y : plan.y, fig = figureAt(seg, ax, ay);
    if (fig) {
      if (at) at.fig = fig;
      const rel = Number.isFinite(plan.rel) ? plan.rel : 0.15, c = fig.top + rel * (fig.bot - fig.top), band = Math.max(0.5, 0.08 * (fig.bot - fig.top)), v = [];
      for (let y = 0; y < h; y++) {
        const gy = Math.min(mh - 1, Math.floor((y + 0.5) / h * mh)); if (Math.abs(gy - c) > band + 0.5) continue;
        for (let x = 0; x < w; x++) { const gx = Math.min(mw - 1, Math.floor((x + 0.5) / w * mw)); if (fig.cells.has(gy * mw + gx)) v.push(data[y * w + x]); }
      }
      if (v.length >= 4) return median(v);
    }
  }
  const cx = plan.x * w - 0.5, cy = plan.y * h - 0.5, r = Math.max(1, Math.round(Math.min(w, h) * 0.02)), v = [];
  for (let y = Math.round(cy) - r; y <= Math.round(cy) + r; y++) for (let x = Math.round(cx) - r; x <= Math.round(cx) + r; x++) {
    if (x >= 0 && y >= 0 && x < w && y < h) v.push(data[y * w + x]);
  }
  return median(v) ?? 0.5;
}

/** where on the person a tap at (x, y) is: 0 = the top of THAT figure, 1 = its bottom (seg as focusDepth's); null if not on one */
export function personRel(x, y, seg) {
  if (!seg || !seg.person) return null;
  const { person, mw, mh } = seg, gx = Math.min(mw - 1, Math.floor(x * mw)), gy = Math.min(mh - 1, Math.floor(y * mh));
  if (!(person[gy * mw + gx] >= 0.5)) return null;
  const fig = figureAt(seg, x, y);
  return !fig || fig.bot <= fig.top ? 0.15 : Math.min(1, Math.max(0, (gy - fig.top) / (fig.bot - fig.top)));
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
 * bokeh {glow, range} 0..1: 光斑 — as in the background blur's bokeh (blur.js), bright points are boosted before the
 * round blur so they open into bright discs (a camera records lights far brighter than the screen shows; blurred
 * without that they turn into dull grey smudges), then rolled off so they never clip. glow 0.5 → ×16 (only points much brighter than
 * their surroundings count as lights, so a white wall is never brightened).
 * range: the brightness where a highlight starts to count as a light (0.8 … 0.2). The boost grows with the layer's blur,
 * so what is in focus is not brightened. Returns {maxRadius} (px) for the record.
 */
export function applyDof(img, w, h, dn, df, K, bokeh = { glow: 0.5, range: 0.5 }, spot = null, { grain = 0, frame = 0, refine = true } = {}) {
  // 模糊處噪點: a real lens blurs the light before it reaches the sensor, so the sensor's noise is on the blurred parts as
  // much as on the sharp ones; blurring the finished picture also wipes out its noise, and the blurred areas look
  // unnaturally clean (plastic, smooth bands) next to the grainy sharp subject. The frame's own noise level is measured
  // here and fresh noise of that level (at least 0.8 code values; ×2·grain) is added where the blur replaces the
  // pixels, changing every frame like sensor noise.
  const nAmp = grain > 0 ? 2 * grain * Math.min(6, Math.max(0.8, noiseSigma(img, w, h))) : 0;
  const gain = 30 * clamp01(bokeh.glow ?? 0.5), lo = 0.8 - 0.6 * clamp01(bokeh.range ?? 0.5);
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
  // the depth refined along the picture's own edges (guided filter, guide = lightness): the AI depth is ~3–4× coarser
  // than the picture and its edge along a slanted boundary steps cell by cell — those steps became a saw-tooth line
  // between a sharp subject and the blurred background (seen at a white table's edge against a dark floor). Fitted
  // locally as a linear function of the lightness, the depth edge follows the real edge smoothly. (Radius ~2 depth
  // cells; a wider one printed shadows on the table into the depth.)
  if (refine) {
    const I = new Float32Array(n);
    for (let q = 0; q < n; q++) I[q] = Math.cbrt(0.2126 * lin[3 * q] + 0.7152 * lin[3 * q + 1] + 0.0722 * lin[3 * q + 2]);
    dg.set(guidedFilter(dg, I, sw, sh, Math.max(1, Math.round(4 * sw / 640)), 1e-3));
  }
  // 焦點範圍 (spot {x, y, area}: a tap on the background): only the part of the picture around the tapped point stays
  // sharp — farther from it on screen counts as farther from the focus distance (up to 0.35 of the depth range), so a
  // wall or a row of buildings at the focus distance blurs away from the tapped corner instead of all staying sharp
  // (and making a blurred person in front stand out). area 1 = the whole focus plane sharp (a real lens).
  if (spot && spot.area < 0.99) {
    const r0 = 0.05 + 0.7 * spot.area, M = Math.max(sw, sh);
    for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) {
      const d = Math.hypot((x + 0.5 - spot.x * sw) / M, (y + 0.5 - spot.y * sh) / M), t = Math.min(1, Math.max(0, (d - r0) / 0.3)), e = 0.35 * t * t * (3 - 2 * t);
      if (!e) continue;
      const q = y * sw + x, off = dg[q] - df;
      if (Math.abs(off) < e) dg[q] = df + (off > 0 ? e : -e); // (at the focus distance: pushed back, as if farther)
    }
  }
  // 光斑: which cells are lights — bright AND much brighter than their surroundings (a lamp, a sparkle), not a large
  // bright surface (a white table, a sunlit wall stays as bright when blurred: boosting it washed it out)
  const pt = new Float32Array(n);
  if (gain) {
    const Yl = new Float32Array(n * 3);
    for (let q = 0; q < n; q++) Yl[3 * q] = Yl[3 * q + 1] = Yl[3 * q + 2] = 0.2126 * lin[3 * q] + 0.7152 * lin[3 * q + 1] + 0.0722 * lin[3 * q + 2];
    const M = Yl.slice(); boxBlur(M, sw, sh, Math.max(2, Math.round(6 / f0 * (sw / 640 * f0))));
    const sm = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
    for (let q = 0; q < n; q++) pt[q] = sm(lo, 1, Yl[3 * q]) * sm(0.45, 0.7, Yl[3 * q] - M[3 * q]); // (an edge of a large bright area stands out about half as much as a point)
  }
  // slices far → near, "over" composite (premultiplied); vis = the blur radius of what is visible (for the final mix)
  const acc = new Float32Array(n * 4), vis = new Float32Array(n);
  let maxR = 0;
  // slice centres one step apart with one exactly on the focus depth (in focus = no blur at all), covering 0..1
  // (covering the depth range actually present: 焦點範圍 can push points past 0 or 1 — beyond every layer they were
  // left out and kept their sharp pixels)
  let dMin = 0, dMax = 1; for (let q = 0; q < n; q++) { if (dg[q] < dMin) dMin = dg[q]; if (dg[q] > dMax) dMax = dg[q]; }
  const step = 1 / (SLICES - 1), centres = [];
  for (let k = Math.floor((dMin - df) / step) - 1; df + k * step <= dMax + step; k++) if (df + k * step >= dMin - step) centres.push(df + k * step);
  for (const c of centres) {
    const rPx = Math.abs(c - df) * K, rb = rPx / f0;
    const g = Math.max(1, Math.ceil(rb / 6)), gw = Math.ceil(sw / g), gh = Math.ceil(sh / g), gb = gain * Math.min(1, Math.max(0, (rPx - 1) / 3));
    const buf = new Float32Array(gw * gh * 4), gc = new Float32Array(gw * gh);
    let any = false;
    for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) {
      const q = y * sw + x, m = Math.max(0, 1 - Math.abs(dg[q] - c) * (SLICES - 1)), o = (Math.floor(y / g) * gw + Math.floor(x / g));
      gc[o]++;
      if (m <= 0) continue;
      any = true;
      const r = lin[3 * q], gg = lin[3 * q + 1], bb = lin[3 * q + 2];
      const kb = gb ? 1 + gb * pt[q] : 1;
      buf[4 * o] += r * m * kb; buf[4 * o + 1] += gg * m * kb; buf[4 * o + 2] += bb * m * kb; buf[4 * o + 3] += m;
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
      const p = (y * w + x) * 4, bl = [0, 0, 0];
      for (let ch = 0; ch < 3; ch++) {
        const v = (q) => acc[4 * q + ch] / Math.max(acc[4 * q + 3], 1e-6);
        bl[ch] = v(q00) * w00 + v(q01) * w01 + v(q10) * w10 + v(q11) * w11;
      }
      const Y = 0.2126 * bl[0] + 0.7152 * bl[1] + 0.0722 * bl[2]; // boosted lights: rolled off, never clipped
      if (Y > 0.8) { const sc = rollOff(Y) / Y; bl[0] *= sc; bl[1] *= sc; bl[2] *= sc; }
      const mx = Math.max(bl[0], bl[1], bl[2]); if (mx > 1) { const Yn = Math.min(1, Y > 0.8 ? rollOff(Y) : Y), t = (mx - 1) / Math.max(mx - Yn, 1e-6); for (let ch = 0; ch < 3; ch++) bl[ch] += (Yn - bl[ch]) * t; }
      // (noise: mostly in lightness, a little in colour, as a sensor's looks after the camera's processing; uniform
      // sums ≈ Gaussian with σ 1)
      const g = (z) => (hash3(x, y, frame * 4 + z) + hash3(x + 7919, y, frame * 4 + z) + hash3(x, y + 7919, frame * 4 + z) - 1.5) * 2;
      const nl = nAmp ? nAmp * k * g(0) : 0;
      for (let ch = 0; ch < 3; ch++) {
        const o = S2L[img[p + ch]], m = Math.min(1, Math.max(0, o + (bl[ch] - o) * k));
        img[p + ch] = 255 * (m <= 0.0031308 ? 12.92 * m : 1.055 * Math.pow(m, 1 / 2.4) - 0.055) + (nAmp ? nl + 0.35 * nAmp * k * g(ch + 1) : 0);
      }
    }
  }
  return { maxRadius: maxR };
}
