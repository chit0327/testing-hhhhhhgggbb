// Alternative export path: our own RGB -> YUV (BT.709, limited "TV" range) with explicit colour tags.
// The app currently hands RGBA frames to VideoEncoder; test/e2e/codec_roundtrip.mjs encodes a known
// pattern both ways and verified both keep colour (mean 8-bit error < 1 when decoded by Chromium and
// by ffmpeg). Kept for players that mishandle the full-range tags Chromium writes for RGBA input.
export const MP4_COLOR = { primaries: 'bt709', transfer: 'bt709', matrix: 'bt709', fullRange: false };

export function rgbaToI420(rgba, w, h, out = new Uint8Array(w * h * 3 / 2)) {
  const Y = out.subarray(0, w * h), U = out.subarray(w * h, w * h * 5 / 4), V = out.subarray(w * h * 5 / 4);
  // BT.709 on gamma-encoded R'G'B' (what every player inverts), limited range
  for (let i = 0, p = 0; i < w * h; i++, p += 4) Y[i] = 16 + (0.2126 * rgba[p] + 0.7152 * rgba[p + 1] + 0.0722 * rgba[p + 2]) * 219 / 255 + 0.5;
  const cw = w >> 1;
  for (let y = 0; y < h; y += 2) for (let x = 0; x < w; x += 2) {
    let r = 0, g = 0, b = 0;
    for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) { const p = ((y + dy) * w + x + dx) * 4; r += rgba[p]; g += rgba[p + 1]; b += rgba[p + 2]; }
    r /= 4; g /= 4; b /= 4;
    const yy = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    const ci = (y >> 1) * cw + (x >> 1);
    U[ci] = 128 + (b - yy) / 1.8556 * 224 / 255 + 0.5;
    V[ci] = 128 + (r - yy) / 1.5748 * 224 / 255 + 0.5;
  }
  return out;
}
