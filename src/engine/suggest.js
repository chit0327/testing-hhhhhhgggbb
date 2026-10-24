// Turn the measured look of the reference into (a) starting values for the adjustment sliders and (b) a
// prompt for an external image AI. Both come from the same statistics the grade fit already computes
// (grade.js stats(): reference without blown highlights, central 70%).
//
// The suggestion is the parameter route of the reference-grade skill: levels (black/white) map the video's
// black and white points towards the reference's, contrast fixes the remaining spread, saturation follows
// the chroma ratio, three-way colour moves the highlight and shadow casts. It replaces the natural transfer
// (strength 0), so every change is visible on a slider and can be undone there.
import { DEFAULT_LOOK, TONE_COLOUR } from './grade.js';

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const q = (s, p) => s.Lq[Math.round(p * (s.Lq.length - 1))];

/** src, ref: stats objects (model.src, model.ref). Returns a full look. */
export function suggestLook(src, ref) {
  const vb = q(src, 0), vw = q(src, 1), rb = q(ref, 0), rw = q(ref, 1); // ≈ p1 / p99
  const slope = clamp((rw - rb) / Math.max(vw - vb, 0.05), 0.6, 1.2);
  const black = clamp(rb - slope * vb, 0, 0.35);
  const white = clamp(black + slope, 0.75, 1);
  const s = white - black; // achieved slope
  // adjust(): contrast slope at mid-grey ≈ 1 + 0.94·c; levels already scale the spread by s
  const contrast = clamp((ref.Lsd / Math.max(src.Lsd * s, 1e-3) - 1) / 0.94, -0.6, 0.6);
  // brighten midtones only modestly: a bright reference is often bright because of its content (sky)
  const predMid = black + src.Lmean * s;
  const exposure = clamp(3 * Math.log2(Math.max(ref.Lmean, 1e-3) / Math.max(predMid, 1e-3)), -0.5, 0.4);
  const saturation = clamp(ref.chroma / Math.max(src.chroma, 1e-3) - 1, -0.6, 1.2);
  const k = 1 + saturation;
  const tone = (ra, rb_, sa, sb) => { // colour to add so the scaled source cast lands on the reference cast
    const da = ra - sa * k, db = rb_ - sb * k, m = Math.hypot(da, db);
    if (m < 0.006) return [0, 0];
    return [((Math.atan2(db, da) * 180) / Math.PI + 360) % 360, clamp(m / TONE_COLOUR, 0, 1)];
  };
  const [hiHue, hiAmt] = tone(ref.hiA, ref.hiB, src.hiA, src.hiB);
  const [shadowHue, shadowAmt] = tone(ref.loA, ref.loB, src.loA, src.loB);
  const r2 = (v) => Math.round(v * 100) / 100;
  return {
    ...DEFAULT_LOOK, strength: 0, match: 0,
    exposure: r2(exposure), contrast: r2(contrast), black: r2(black), white: r2(white), saturation: r2(saturation),
    hiHue: Math.round(hiHue), hiAmt: r2(hiAmt), shadowHue: Math.round(shadowHue), shadowAmt: r2(shadowAmt),
  };
}

/** OKLab hue (degrees) -> colour name. */
export function hueName(h) {
  const names = [[20, '粉紅'], [50, '紅'], [80, '橙'], [105, '黃'], [130, '黃綠'], [165, '綠'], [210, '青綠'], [255, '藍'], [300, '紫'], [340, '洋紅'], [361, '粉紅']];
  return names.find(([lim]) => ((h % 360) + 360) % 360 < lim)[1];
}

function cast(a, b) {
  const m = Math.hypot(a, b);
  if (m < 0.008) return '中性';
  return `${m > 0.03 ? '明顯' : '略'}偏${hueName((Math.atan2(b, a) * 180) / Math.PI)}`;
}

/** Prompt for an image AI: re-grade one frame only, in the reference's look. */
export function promptFor(ref, palette = []) {
  const contrast = ref.Lsd < 0.1 ? '低對比、柔和' : ref.Lsd < 0.16 ? '中等對比' : '高對比';
  const blacks = q(ref, 0) > 0.3 ? '暗部明顯提亮、帶霧感，不要純黑' : q(ref, 0) > 0.2 ? '暗部略為提亮' : '保留深黑的暗部';
  const whites = q(ref, 1) < 0.88 ? '亮部柔和、不要死白' : '亮部明亮';
  const sat = ref.chroma < 0.03 ? '低飽和' : ref.chroma < 0.06 ? '中等飽和、偏粉彩' : '鮮豔高飽和';
  const cols = palette.filter((p) => Math.hypot(p.a, p.b) > 0.02).slice(0, 4)
    .map((p) => `${hueName((Math.atan2(p.b, p.a) * 180) / Math.PI)}色`).filter((v, i, a) => a.indexOf(v) === i);
  return [
    '請只調整這張影格的顏色和明暗，不要改變構圖、人物、臉部、物件、文字或光線方向，不要磨皮、不要加光暈或濾鏡效果。',
    '請參考我附上的參考圖的色調和氛圍。',
    `具體要求：${contrast}；${blacks}；${whites}；${sat}；亮部${cast(ref.hiA, ref.hiB)}，暗部${cast(ref.loA, ref.loB)}${cols.length ? `；主要色彩傾向：${cols.join('、')}` : ''}。`,
    '輸出與原圖相同尺寸、相同構圖的 PNG。',
  ].join('\n');
}
