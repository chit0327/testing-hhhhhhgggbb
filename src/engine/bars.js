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
