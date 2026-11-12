// Letterbox / pillarbox bars around a reference picture: black bars of a film still, or the dark grey background of a
// video player in a screenshot. Left in, they count as the reference's "blacks" and the fit then darkens the shadows
// of a milky, lifted-black look — the opposite of the reference. findBars returns the picture's own area.

/**
 * img: ImageData-like {data, width, height}. Returns {x, y, w, h} (pixels of img): the area inside flat dark bars on
 * any side. A bar line is nearly uniform (the 2nd–98th percentile spread of its luma ≤ 20: a player background may have a slight gradient) and dark (mean ≤ 70); bright flat
 * lines (an even sky) are never removed. Bars thinner than 1 % are ignored; at most 40 % per side.
 */
export function findBars(img) {
  const { data: d, width: W, height: H } = img;
  const lum = (x, y) => { const p = 4 * (y * W + x); return 0.2126 * d[p] + 0.7152 * d[p + 1] + 0.0722 * d[p + 2]; };
  const flatDark = (vals) => {
    const s = vals.slice().sort((a, b) => a - b), lo = s[Math.floor(s.length * 0.02)], hi = s[Math.floor(s.length * 0.98)];
    const mean = vals.reduce((a, v) => a + v, 0) / vals.length;
    return hi - lo <= 20 && mean <= 70;
  };
  const row = (y, x0, x1) => { const v = []; for (let x = x0; x < x1; x++) v.push(lum(x, y)); return v; };
  const col = (x, y0, y1) => { const v = []; for (let y = y0; y < y1; y++) v.push(lum(x, y)); return v; };
  let left = 0, right = 0, top = 0, bottom = 0;
  const maxX = Math.floor(W * 0.4), maxY = Math.floor(H * 0.4);
  while (left < maxX && flatDark(col(left, 0, H))) left++;
  while (right < maxX && flatDark(col(W - 1 - right, 0, H))) right++;
  while (top < maxY && flatDark(row(top, left, W - right))) top++;
  while (bottom < maxY && flatDark(row(H - 1 - bottom, left, W - right))) bottom++;
  if (left < W * 0.01) left = 0; if (right < W * 0.01) right = 0;
  if (top < H * 0.01) top = 0; if (bottom < H * 0.01) bottom = 0;
  return { x: left, y: top, w: W - left - right, h: H - top - bottom };
}

/**
 * Panels around a reference picture that are not part of the picture: a colour-palette strip (the swatches film-palette
 * posts put under a still: rows made of a few flat blocks, each row the same as the next), white gaps and a flat light
 * frame. Left in, the swatches counted as picture (a reference's near layer measured contrast ×1.30 and colour ×1.20
 * with them, ×0.60 / ×0.41 without). img as findBars; returns {x, y, w, h}: the area inside the panels (then findBars
 * finds dark bars inside it).
 * Rules — a block row: ≥ 3 wide (≥ 4 %) flat blocks of clearly different colour side by side, ≤ 60 sharp steps (a smooth
 * sky has no steps; lamp posts on a pale sky have the same colour on both sides); a palette strip:
 * block rows (with flat gaps and a few anti-aliased rows) over ≥ 3 % of the height, ≥ 80 % of them the same as the next
 * row inwards (mean difference ≤ 2.5 levels); a flat row: luma spread ≤ 8. From each edge, a run of such rows is
 * removed when it holds swatch rows over ≥ 3 % of the height, or when it is flat and light (mean ≥ 200), ≤ 8 % thick and
 * ends at a sharp change into the picture (an overcast sky fades in: kept). Columns: flat light frame only.
 */
export function findPanels(img) {
  const { data: d, width: W, height: H } = img;
  const lum = (i) => 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2];
  const rowStats = (y) => {
    const v = new Float32Array(W); for (let x = 0; x < W; x++) v[x] = lum(4 * (y * W + x));
    const s = Float32Array.from(v).sort(), spread = s[Math.floor(W * 0.98)] - s[Math.floor(W * 0.02)];
    let steps = 0, flatPairs = 0; for (let x = 1; x < W; x++) { const g = Math.abs(v[x] - v[x - 1]); if (g > 12) steps++; else if (g <= 2) flatPairs++; }
    // wide blocks of clearly different colour side by side (swatches) — not thin lines on one background (lamp posts
    // against a pale sky: the same colour on both sides of each step)
    const segs = []; let a = 0;
    for (let x = 1; x <= W; x++) if (x === W || Math.abs(v[x] - v[x - 1]) > 12) { if (x - a >= W * 0.04) { let m = 0; for (let k = a; k < x; k++) m += v[k]; segs.push(m / (x - a)); } a = x; }
    let blocks = 0; for (let i = 1; i < segs.length; i++) if (Math.abs(segs[i] - segs[i - 1]) > 10) blocks++;
    return { v, spread, steps, blocks, flat: flatPairs / (W - 1), mean: v.reduce((a, b) => a + b, 0) / W };
  };
  const rowDiff = (a, b) => { let s = 0; for (let x = 0; x < W; x++) s += Math.abs(a.v[x] - b.v[x]); return s / W; };
  const side = (from, dir) => { // rows from the edge inwards: how many to remove
    const max = Math.floor(H * 0.45);
    let n = 0, swatch = 0, same = 0, odd = 0;
    const rows = [];
    for (let y = from; n < max; n++, y += dir) {
      const r = rowStats(y), isFlat = r.spread <= 8;
      // block rows (≥ 3 sharp steps: a smooth sky has none; flat between them)
      const isBlock = !isFlat && r.blocks >= 2 && r.steps <= 60 && r.flat >= 0.8;
      if (!isFlat && !isBlock) { if (odd < 3 && rows.length) { odd++; rows.push({ isFlat: false, mean: r.mean }); continue; } break; } // (a few anti-aliased rows where a swatch meets a gap)
      if (isBlock) { swatch++; if (rowDiff(r, rowStats(y + dir)) <= 2.5) same++; }
      rows.push({ isFlat, mean: r.mean });
    }
    while (rows.length && !rows[rows.length - 1].isFlat && rows.length > 0 && odd > 0) { rows.pop(); n--; odd--; } // (stray rows at the end belong to the picture)
    // a palette strip: block rows over ≥ 3 % of the height, nearly all the same as the next row (a picture's rows are not)
    if (swatch >= H * 0.03 && same >= swatch * 0.8) return n;
    if (n && rows.every((r) => r.isFlat && r.mean >= 200) && n <= H * 0.08) { // a flat light frame: only with a sharp edge into the picture
      const edge = rowDiff(rowStats(from + (n - 1) * dir), rowStats(from + n * dir));
      if (edge >= 20) return n;
    }
    return 0;
  };
  const top = side(0, 1), bottom = side(H - 1, -1);
  // columns: a flat light frame only (the same test on the columns inside the kept rows)
  const colSide = (from, dir) => {
    const max = Math.floor(W * 0.08), y0 = top, y1 = H - bottom;
    const col = (x) => { const v = []; for (let y = y0; y < y1; y++) v.push(lum(4 * (y * W + x))); return v; };
    const flatLight = (v) => { const s = v.slice().sort((a, b) => a - b); return s[Math.floor(v.length * 0.98)] - s[Math.floor(v.length * 0.02)] <= 8 && v.reduce((a, b) => a + b, 0) / v.length >= 200; };
    let n = 0; while (n < max && flatLight(col(from + n * dir))) n++;
    if (!n) return 0;
    const a = col(from + (n - 1) * dir), b = col(from + n * dir); let s = 0; for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
    return s / a.length >= 20 ? n : 0;
  };
  const left = colSide(0, 1), right = colSide(W - 1, -1);
  return { x: left, y: top, w: W - left - right, h: H - top - bottom };
}

/** the picture's own area: panels (palette strips, light frames) first, then dark bars inside what is left */
export function findPicture(img) {
  const p = findPanels(img);
  if (p.w === img.width && p.h === img.height) return findBars(img);
  const { data: d, width: W } = img, sub = new Uint8ClampedArray(p.w * p.h * 4);
  for (let y = 0; y < p.h; y++) sub.set(d.subarray(4 * ((p.y + y) * W + p.x), 4 * ((p.y + y) * W + p.x + p.w)), 4 * y * p.w);
  const b = findBars({ data: sub, width: p.w, height: p.h });
  return { x: p.x + b.x, y: p.y + b.y, w: b.w, h: b.h, panels: true };
}
