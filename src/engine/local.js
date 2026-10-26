// Automatic local grading: after the global grade, each AI region (skin, hair, clothes, background) gets a
// small OKLab adjustment fitted against the same region of the reference — what a colourist does with
// masks. Only what is "grade" rather than "content" is matched:
//   skin       — its colour (hue / chroma) towards the reference's skin, kept inside natural skin tones;
//   background — the TINT of its near-neutral pixels (greys, whites: the less colourful half) towards the
//                reference environment's tint;
//   hair, clothes — keep the global grade (no background tint): their colour is content. Matching them
//                was tried and turned a beige coat lavender under a reference with a navy jacket.
// Regions too small in either image are left alone. Deterministic; the renderer bakes one LUT per region
// and blends them by the region masks.
import { srgbToOklab } from './color.js';
import { compilePrim, sanitizePrim, LOCAL_REGIONS } from './primary.js';

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
export const LOCAL_STRENGTH = { natural: 0.6, faithful: 1, bold: 1.2 };
const MIN_SHARE = 0.015;            // region must cover ≥ 1.5 % of both images
const CAP = { skin: 0.035, other: 0.025, satSkin: 0.4 };
const SKIN_HUE = [25, 85], SKIN_C = [0.03, 0.12]; // OKLab: natural skin tones

/** Region weight (0..1) of every analysed pixel: masks from the AI grids (skin, hair, clothes, background). */
function regionOf(seg, x, y, w, h) {
  const mw = seg.w, mh = seg.h, i = Math.min(mh - 1, Math.floor((y * mh) / h)) * mw + Math.min(mw - 1, Math.floor((x * mw) / w));
  return { skin: seg.skin[i], hair: seg.hair[i], clothes: seg.clothes[i], background: seg.background[i] };
}

/**
 * Per-region statistics of an image (optionally through a colour transform): share, mean OKLab, and the
 * mean a/b of the less colourful half (the region's tint). seg: {w, h, skin, hair, clothes, background}.
 */
export function regionStats(rgba, w, h, seg, transform = null, step = 2) {
  const px = Object.fromEntries(LOCAL_REGIONS.map((r) => [r, []]));
  const lab = [0, 0, 0], c = [0, 0, 0];
  let total = 0;
  for (let y = 0; y < h; y += step) for (let x = 0; x < w; x += step) {
    total++;
    const m = regionOf(seg, x, y, w, h);
    let best = null; for (const r of LOCAL_REGIONS) if (m[r] > 0.6 && (!best || m[r] > m[best])) best = r;
    if (!best) continue; // edges: no clear region
    const p = (y * w + x) * 4;
    let r = rgba[p] / 255, g = rgba[p + 1] / 255, b = rgba[p + 2] / 255;
    if (transform) { const t = transform(r, g, b, c); r = t[0]; g = t[1]; b = t[2]; }
    srgbToOklab(clamp(r, 0, 1), clamp(g, 0, 1), clamp(b, 0, 1), lab);
    px[best].push(lab[0], lab[1], lab[2]);
  }
  const out = {};
  for (const r of LOCAL_REGIONS) {
    const v = px[r], n = v.length / 3;
    if (!n) { out[r] = { share: 0 }; continue; }
    let L = 0, a = 0, b = 0; const C = [];
    for (let i = 0; i < n; i++) { L += v[3 * i]; a += v[3 * i + 1]; b += v[3 * i + 2]; C.push(Math.hypot(v[3 * i + 1], v[3 * i + 2])); }
    const med = Float64Array.from(C).sort()[Math.floor(n / 2)];
    let ta = 0, tb = 0, tn = 0;
    for (let i = 0; i < n; i++) if (C[i] <= med) { ta += v[3 * i + 1]; tb += v[3 * i + 2]; tn++; }
    out[r] = { share: n / total, L: L / n, a: a / n, b: b / n, C: C.reduce((s, x) => s + x, 0) / n, tintA: ta / tn, tintB: tb / tn };
  }
  return out;
}

/** The layer parameters a region is rendered with (before its own adjustment). */
export function layerPrim(prim, region) {
  return { ...sanitizePrim(prim), local: null, post: null, keep: region === 'skin' ? prim.skin || 0 : 0 };
}

/**
 * Fit the per-region adjustments for one grade. src / ref: {rgba, w, h, seg}. Returns {local, report}:
 * local = {skin, hair, clothes, background: {L, a, b, sat}} to store in prim.local.
 */
export function fitLocal(prim, src, ref, mode = 'faithful') {
  const s = LOCAL_STRENGTH[mode] ?? 1;
  const R = regionStats(ref.rgba, ref.w, ref.h, ref.seg);
  const local = {}, report = {};
  for (const r of LOCAL_REGIONS) {
    const f = compilePrim(layerPrim(prim, r));
    const G = regionStats(src.rgba, src.w, src.h, src.seg, f)[r], Rr = R[r];
    local[r] = { L: 0, a: 0, b: 0, sat: 0 };
    if (r === 'hair' || r === 'clothes') { report[r] = { kept: true }; continue; }
    if (!(G.share >= MIN_SHARE && Rr.share >= MIN_SHARE)) { report[r] = { skipped: true, src: G.share, ref: Rr.share }; continue; }
    if (r === 'skin') {
      // reference skin colour, limited to natural skin hues / chroma
      const hue = clamp((Math.atan2(Rr.b, Rr.a) * 180) / Math.PI, SKIN_HUE[0], SKIN_HUE[1]), C = clamp(Math.hypot(Rr.a, Rr.b), SKIN_C[0], SKIN_C[1]);
      const ta = C * Math.cos((hue * Math.PI) / 180), tb = C * Math.sin((hue * Math.PI) / 180);
      let da = (ta - G.a) * s, db = (tb - G.b) * s;
      const d = Math.hypot(da, db); if (d > CAP.skin) { da *= CAP.skin / d; db *= CAP.skin / d; }
      local[r] = { L: 0, a: da, b: db, sat: 0 };
    } else {
      let da = (Rr.tintA - G.tintA) * s, db = (Rr.tintB - G.tintB) * s;
      const d = Math.hypot(da, db); if (d > CAP.other) { da *= CAP.other / d; db *= CAP.other / d; }
      local[r] = { L: 0, a: da, b: db, sat: 0 };
    }
    report[r] = { src: G, ref: Rr, adj: local[r] };
  }
  return { local, report };
}
