// 同類別匹配: the background split into what it shows — sky, plants, and everything else (buildings, walls, ground,
// roads, objects) — so each is matched to the same kind of thing in the reference (layermatch.js does the matching:
// brightness, contrast, tint, colourfulness per class). One average for the whole picture, or near / far by depth, still
// mixes a reference's pale sky with its warm walls; now the video's sky follows the reference's sky, its trees the
// reference's trees, its walls the reference's walls. People are left out (AI person mask), as in 依深度分層匹配.
//
// The classes come from the picture itself on a coarse grid (no extra AI model, the same for every frame of a video):
//   sky   — the farthest part of the picture by the AI depth (the depth model puts the sky at the far end), bright for
//           this picture and connected to the top edge through such cells; clouds inside it count. Without depth only a
//           smooth, bright, blue or near-white area at the top (brightness and smoothness alone took a white bedroom
//           wall and a brick wall for sky, and a grey-teal cloudy sky for plants);
//   plants — green hues (OKLab 85…178°, so a yellow-green or slightly teal grade still counts; teal-graded buildings do not), some colour and some
//           texture (leaves, grass); not sky;
//   rest  — the remaining background.
// Brightness and smoothness are measured against the picture's own range, so a graded reference (orange sky, teal
// leaves) is classified the same way as the ungraded video. Soft edges (a blur over the grid).
import { srgbToOklab } from './color.js';
import { boxBlur } from './render.js';

export const CLASSES = ['sky', 'plants', 'rest'];
export const CLASS_NAMES = { sky: '天空', plants: '植物', rest: '建築地面' };
const GW = 96;
const smooth = (a, b, x) => { const t = b > a ? Math.min(1, Math.max(0, (x - a) / (b - a))) : x >= b ? 1 : 0; return t * t * (3 - 2 * t); };
const gridAt = (g, gw, gh, fx, fy) => {
  const x = Math.min(gw - 1, Math.max(0, fx * gw - 0.5)), y = Math.min(gh - 1, Math.max(0, fy * gh - 0.5));
  const x0 = x | 0, y0 = y | 0, x1 = Math.min(gw - 1, x0 + 1), y1 = Math.min(gh - 1, y0 + 1), tx = x - x0, ty = y - y0;
  return (g[y0 * gw + x0] * (1 - tx) + g[y0 * gw + x1] * tx) * (1 - ty) + (g[y1 * gw + x0] * (1 - tx) + g[y1 * gw + x1] * tx) * ty;
};
const pct = (arr, q) => { const s = Float64Array.from(arr).sort(); return s.length ? s[Math.min(s.length - 1, Math.floor(s.length * q))] : 0; };

/**
 * rgba w×h → {gw, gh, sky, plants, rest, person} soft masks on a grid (0..1, sky + plants + rest + person = 1).
 * seg: {person: grid, mw, mh} | null; dn: AI depth {data 0..1 (1 = near), w, h} | null.
 */
export function classGrid(rgba, w, h, seg = null, dn = null) {
  const gw = Math.min(GW, w), gh = Math.max(1, Math.round(gw * h / w)), n = gw * gh;
  // per cell: mean OKLab, and fine detail (mean |L − right neighbour| + |L − lower neighbour| at ~2 px of a 640 frame)
  const L = new Float32Array(n), A = new Float32Array(n), B = new Float32Array(n), T = new Float32Array(n), cnt = new Float32Array(n);
  const step = Math.max(1, Math.round(Math.max(w, h) / 640)), lab = [0, 0, 0], lab2 = [0, 0, 0];
  const Lat = (x, y) => { const p = 4 * (y * w + x); srgbToOklab(rgba[p] / 255, rgba[p + 1] / 255, rgba[p + 2] / 255, lab2); return lab2[0]; };
  for (let y = 0; y < h - step; y += step) for (let x = 0; x < w - step; x += step) {
    const p = 4 * (y * w + x); srgbToOklab(rgba[p] / 255, rgba[p + 1] / 255, rgba[p + 2] / 255, lab);
    const q = Math.min(gh - 1, Math.floor(y * gh / h)) * gw + Math.min(gw - 1, Math.floor(x * gw / w));
    L[q] += lab[0]; A[q] += lab[1]; B[q] += lab[2]; T[q] += Math.abs(lab[0] - Lat(x + step, y)) + Math.abs(lab[0] - Lat(x, y + step)); cnt[q]++;
  }
  for (let q = 0; q < n; q++) { const k = 1 / (cnt[q] || 1); L[q] *= k; A[q] *= k; B[q] *= k; T[q] *= k; }
  const person = new Float32Array(n);
  if (seg && seg.person) for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) person[y * gw + x] = Math.min(1, Math.max(0, gridAt(seg.person, seg.mw, seg.mh, (x + 0.5) / gw, (y + 0.5) / gh)));
  const depth = dn ? (() => { const d = new Float32Array(n); for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) d[y * gw + x] = gridAt(dn.data, dn.w, dn.h, (x + 0.5) / gw, (y + 0.5) / gh); return d; })() : null;
  // the picture's own scale: brightness by percentile, detail by percentile (of the non-person cells)
  const bgL = [], bgT = [], bgD = [];
  for (let q = 0; q < n; q++) if (person[q] < 0.5) { bgL.push(L[q]); bgT.push(T[q]); if (depth) bgD.push(depth[q]); }
  const L50 = pct(bgL, 0.5), L90 = pct(bgL, 0.9), T30 = Math.max(0.002, pct(bgT, 0.3)), /* (a perfectly smooth picture: 0, and 0 / 0 made every class NaN) */ T70 = pct(bgT, 0.7), D02 = depth ? pct(bgD, 0.02) : 0, D60 = depth ? pct(bgD, 0.6) : 1;
  const hue = (q) => ((Math.atan2(B[q], A[q]) * 180) / Math.PI + 360) % 360, chroma = (q) => Math.hypot(A[q], B[q]);
  // plants: green…teal hue with some colour, some texture
  const plantScore = (q) => { const hh = hue(q), C = chroma(q); const inHue = smooth(80, 95, hh) * (1 - smooth(172, 185, hh)); /* (a teal-graded city reached 190°+: not plants) */ return inHue * smooth(0.025, 0.045, C) * smooth(T30 * 0.6, T30 * 1.2, T[q]) * (1 - smooth(0.88, 0.95, L[q])); };
  // sky candidates: bright for this picture, smooth, not plant-coloured, not a person, far (if depth)
  const smoothThr = Math.max(T30, 0.004);
  const cand = new Uint8Array(n);
  for (let q = 0; q < n; q++) {
    if (person[q] >= 0.5) continue;
    const bright = L[q] >= Math.max(0.4, Math.min(L90 - 0.08, L50 + 0.08)) || L[q] > 0.82;
    if (depth) { // the far end of this picture's depth (the sky is at infinity: well beyond everything else); with depth
      // as the main clue, only "not dark" (a deep blue sky is darker than sunlit walls: it failed "bright for this picture")
      cand[q] = L[q] >= Math.max(0.35, L50 - 0.1) && depth[q] <= D02 + 0.12 * Math.max(0.05, D60 - D02) ? 1 : 0;
    } else {
      const hh = hue(q), C = chroma(q), skyHue = (hh >= 200 && hh <= 290 && C > 0.01) || C < 0.02;
      cand[q] = bright && skyHue && T[q] <= Math.max(smoothThr * 1.6, Math.min(T70, 0.03)) && plantScore(q) < 0.5 ? 1 : 0;
    }
  }
  // connected to the top edge (4-neighbours), only in the upper 80 % of the frame
  const sky = new Float32Array(n), stack = [];
  for (let x = 0; x < gw; x++) if (cand[x]) { sky[x] = 1; stack.push(x); }
  while (stack.length) {
    const q = stack.pop(), x = q % gw, y = (q / gw) | 0;
    for (const [nx, ny] of [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]) {
      if (nx < 0 || ny < 0 || nx >= gw || ny >= gh || ny > gh * 0.8) continue;
      const r = ny * gw + nx; if (sky[r] || !cand[r]) continue;
      sky[r] = 1; stack.push(r);
    }
  }
  // holes inside the sky (a cloud edge, a bird): cells surrounded by sky on 3+ sides join it
  for (let pass = 0; pass < 2; pass++) for (let y = 1; y < gh - 1; y++) for (let x = 1; x < gw - 1; x++) {
    const q = y * gw + x; if (sky[q] || person[q] >= 0.5) continue;
    if (sky[q - 1] + sky[q + 1] + sky[q - gw] + sky[q + gw] >= 3 && plantScore(q) < 0.5) sky[q] = 1;
  }
  const plants = new Float32Array(n); for (let q = 0; q < n; q++) plants[q] = sky[q] ? 0 : plantScore(q);
  // soft edges: blurred (3 channels at once), then shared out so that every cell sums to 1 with the person
  const buf = new Float32Array(n * 3); for (let q = 0; q < n; q++) { buf[3 * q] = sky[q]; buf[3 * q + 1] = plants[q]; buf[3 * q + 2] = 0; }
  boxBlur(buf, gw, gh, 1);
  const out = { gw, gh, sky: new Float32Array(n), plants: new Float32Array(n), rest: new Float32Array(n), person };
  for (let q = 0; q < n; q++) {
    const bg = 1 - person[q], s = Math.min(1, buf[3 * q]), pl = Math.min(1 - s, buf[3 * q + 1]);
    out.sky[q] = bg * s; out.plants[q] = bg * pl; out.rest[q] = Math.max(0, bg * (1 - s - pl));
  }
  return out;
}

/** a class grid → per-pixel sampler of [person, sky, plants, rest] weights (for layermatch.js) */
export function classSampler(cg) {
  return (fx, fy, out = [0, 0, 0, 0]) => {
    out[0] = gridAt(cg.person, cg.gw, cg.gh, fx, fy); out[1] = gridAt(cg.sky, cg.gw, cg.gh, fx, fy);
    out[2] = gridAt(cg.plants, cg.gw, cg.gh, fx, fy); out[3] = gridAt(cg.rest, cg.gw, cg.gh, fx, fy);
    return out;
  };
}
/** share of each class (0..1 of the frame) */
export function classShares(cg) {
  const n = cg.gw * cg.gh, s = { sky: 0, plants: 0, rest: 0 };
  for (let q = 0; q < n; q++) { s.sky += cg.sky[q] / n; s.plants += cg.plants[q] / n; s.rest += cg.rest[q] / n; }
  return s;
}
