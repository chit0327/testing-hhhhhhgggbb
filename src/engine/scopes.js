// Video scopes, as on a colour page: luma waveform, RGB parade and vectorscope (Rec.709 Y'CbCr, the
// convention of broadcast scopes, with the usual skin-tone line). Pure functions: the accumulation is
// testable, drawing only maps counts to pixels. Input is 8-bit sRGB RGBA (display-referred, as exported).

const KR = 0.2126, KB = 0.0722, KG = 1 - KR - KB;

/** Y', Cb, Cr (Rec.709, full range: Y' 0..1, Cb/Cr −0.5..0.5) of 8-bit R'G'B'. */
export function ycbcr(r, g, b, out = [0, 0, 0]) {
  const R = r / 255, G = g / 255, B = b / 255, Y = KR * R + KG * G + KB * B;
  out[0] = Y; out[1] = (B - Y) / (2 * (1 - KB)); out[2] = (R - Y) / (2 * (1 - KR));
  return out;
}

/**
 * Waveform: counts[ch][y * cols + x] for column bins (x) and level bins (y, 0 = black at the bottom).
 * mode 'luma' -> one channel (Y'); 'parade' -> three channels R, G, B.
 */
export function waveformData(rgba, w, h, { mode = 'luma', cols = 256, levels = 128, step = 1 } = {}) {
  const nch = mode === 'parade' ? 3 : 1, counts = Array.from({ length: nch }, () => new Uint32Array(cols * levels));
  for (let y = 0; y < h; y += step) for (let x = 0; x < w; x += step) {
    const p = (y * w + x) * 4, cx = Math.min(cols - 1, Math.floor((x * cols) / w));
    if (nch === 1) {
      const Y = KR * rgba[p] + KG * rgba[p + 1] + KB * rgba[p + 2];
      counts[0][Math.min(levels - 1, Math.floor((Y / 255) * levels)) * cols + cx]++;
    } else for (let c = 0; c < 3; c++) counts[c][Math.min(levels - 1, Math.floor((rgba[p + c] / 255) * levels)) * cols + cx]++;
  }
  return { counts, cols, levels, mode };
}

/** Vectorscope: counts[(v) * size + u], u = Cb, v = −Cr (red up-left, as on a scope), full scale = ±0.5. */
export function vectorscopeData(rgba, w, h, { size = 256, step = 1, gain = 1 } = {}) {
  const counts = new Uint32Array(size * size), o = [0, 0, 0], half = size / 2;
  for (let y = 0; y < h; y += step) for (let x = 0; x < w; x += step) {
    const p = (y * w + x) * 4;
    ycbcr(rgba[p], rgba[p + 1], rgba[p + 2], o);
    const u = Math.round(half + o[1] * 2 * gain * (half - 1)), v = Math.round(half - o[2] * 2 * gain * (half - 1));
    if (u >= 0 && u < size && v >= 0 && v < size) counts[v * size + u]++;
  }
  return { counts, size, gain };
}

/** Where a colour lands on the vectorscope (unit circle = full scale): used for the targets and skin line. */
export function vectorOf(r, g, b) { const o = ycbcr(r, g, b); return [o[1] * 2, -o[2] * 2]; }
// skin-tone line: the hue of typical skin (all skin tones fall near one line; only saturation differs)
export const SKIN_ANGLE = (() => { const v = vectorOf(224, 172, 140); return Math.atan2(v[1], v[0]); })();

/** Share of pixels at the clip points (any channel ≥ 254 / all ≤ 1): the scope's warning numbers. */
export function clipShares(rgba) {
  let hi = 0, lo = 0; const n = rgba.length / 4;
  for (let i = 0; i < rgba.length; i += 4) {
    if (rgba[i] >= 254 || rgba[i + 1] >= 254 || rgba[i + 2] >= 254) hi++;
    if (rgba[i] <= 1 && rgba[i + 1] <= 1 && rgba[i + 2] <= 1) lo++;
  }
  return { hi: hi / n, lo: lo / n };
}

// ---- drawing (canvas 2D) -----------------------------------------------------------------------------
const TRACE = { luma: [150, 235, 160], parade: [[255, 90, 80], [90, 230, 110], [90, 150, 255]] };
function plot(img, W, x, y, rgb, a) { // additive, capped
  const i = (y * W + x) * 4;
  img[i] = Math.min(255, img[i] + rgb[0] * a); img[i + 1] = Math.min(255, img[i + 1] + rgb[1] * a); img[i + 2] = Math.min(255, img[i + 2] + rgb[2] * a); img[i + 3] = 255;
}
const level = (c, norm) => (c ? Math.min(1, 0.15 + Math.log1p(c) / norm) : 0);

/** Draw a waveform / parade into ctx (canvas size W×H), with 0/25/50/75/100 % graticule. */
export function drawWaveform(ctx, data, W, H) {
  const { counts, cols, levels, mode } = data, nch = counts.length, im = ctx.createImageData(W, H), d = im.data;
  for (let i = 3; i < d.length; i += 4) d[i] = 255;
  const panelW = Math.floor(W / nch);
  for (let c = 0; c < nch; c++) {
    let max = 1; for (const v of counts[c]) if (v > max) max = v;
    const norm = Math.log1p(max), col = nch === 1 ? TRACE.luma : TRACE.parade[c];
    for (let ly = 0; ly < levels; ly++) for (let cx = 0; cx < cols; cx++) {
      const v = counts[c][ly * cols + cx]; if (!v) continue;
      const a = level(v, norm), x0 = c * panelW + Math.floor((cx * panelW) / cols), x1 = c * panelW + Math.floor(((cx + 1) * panelW) / cols);
      const y0 = H - 1 - Math.floor(((ly + 1) * (H - 1)) / levels), y1 = H - 1 - Math.floor((ly * (H - 1)) / levels);
      for (let y = y0; y <= y1; y++) for (let x = x0; x < Math.max(x1, x0 + 1); x++) plot(d, W, x, y, col, a);
    }
  }
  ctx.putImageData(im, 0, 0);
  ctx.strokeStyle = 'rgba(255,255,255,0.18)'; ctx.fillStyle = 'rgba(255,255,255,0.55)'; ctx.font = '10px ui-monospace, monospace'; ctx.lineWidth = 1;
  for (const pct of [0, 25, 50, 75, 100]) {
    const y = Math.round(H - 1 - (pct / 100) * (H - 1)) + 0.5;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    ctx.fillText(String(pct), 3, Math.min(H - 3, Math.max(10, y - 3)));
  }
  for (let c = 1; c < nch; c++) { ctx.beginPath(); ctx.moveTo(c * panelW + 0.5, 0); ctx.lineTo(c * panelW + 0.5, H); ctx.stroke(); }
}

/** Draw a vectorscope into a square canvas (size S), with colour targets (75 %) and the skin-tone line. */
export function drawVectorscope(ctx, data, S) {
  const { counts, size, gain } = data, im = ctx.createImageData(S, S), d = im.data;
  for (let i = 3; i < d.length; i += 4) d[i] = 255;
  let max = 1; for (const v of counts) if (v > max) max = v;
  const norm = Math.log1p(max);
  for (let v = 0; v < size; v++) for (let u = 0; u < size; u++) {
    const c = counts[v * size + u]; if (!c) continue;
    const x = Math.floor((u * S) / size), y = Math.floor((v * S) / size);
    plot(d, S, Math.min(S - 1, x), Math.min(S - 1, y), [210, 240, 215], level(c, norm));
  }
  ctx.putImageData(im, 0, 0);
  const cx = S / 2, R = S / 2 - 1;
  ctx.strokeStyle = 'rgba(255,255,255,0.2)'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(cx, cx, R, 0, Math.PI * 2); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, S); ctx.moveTo(0, cx); ctx.lineTo(S, cx); ctx.stroke();
  // 75 % colour bars targets
  ctx.font = '10px ui-monospace, monospace'; ctx.fillStyle = 'rgba(255,255,255,0.6)';
  for (const [n, rgb] of [['R', [191, 0, 0]], ['Yl', [191, 191, 0]], ['G', [0, 191, 0]], ['Cy', [0, 191, 191]], ['B', [0, 0, 191]], ['Mg', [191, 0, 191]]]) {
    const v = vectorOf(...rgb), x = cx + v[0] * gain * R, y = cx + v[1] * gain * R;
    ctx.strokeRect(x - 5, y - 5, 10, 10); ctx.fillText(n, x + 7, y + 3);
  }
  ctx.strokeStyle = 'rgba(255,200,150,0.55)'; // skin-tone line
  ctx.beginPath(); ctx.moveTo(cx, cx); ctx.lineTo(cx + Math.cos(SKIN_ANGLE) * R, cx + Math.sin(SKIN_ANGLE) * R); ctx.stroke();
}
