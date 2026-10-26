// Claude looks at the reference, the original frame and the current result, and sets the adjustment
// sliders — the automated version of the reference-grade skill's "Claude parameters" mode. It runs on the
// viewer's own Claude account through the artifact `sample` capability (images declared), so no API key is
// involved; outside a claude.ai viewer the feature is simply absent.
//
// Claude only chooses slider values. The pixels still come from the worker's LUT, so every change stays
// visible on the sliders and can be undone there.
import { hueName } from './engine/suggest.js';
import { canvas2d } from './pipeline.js';
import { sanitizePrim, neutralPrim } from './engine/primary.js';

/** Slider ranges in look units (what the adjustment panel can show). */
export const RANGES = {
  exposure: [-2, 2], contrast: [-1, 1], highlights: [-1, 1], shadows: [-1, 1],
  saturation: [-1, 1], temperature: [-1, 1], tint: [-1, 1],
  black: [0, 0.4], white: [0.7, 1],
  shadowHue: [0, 359], shadowAmt: [0, 1], midHue: [0, 359], midAmt: [0, 1], hiHue: [0, 359], hiAmt: [0, 1],
};

/** The sample function when this view can send images to Claude, else null. */
export async function claudeAvailable() {
  try {
    if (!window.claude || !window.claude.use) return null;
    const sample = await window.claude.use('sample');
    if (!sample) return null;
    const lim = await sample.limits().catch(() => null);
    return lim && lim.images && lim.images.maxCount >= 3 ? sample : null;
  } catch { return null; }
}

/** Keep only known keys, as finite numbers inside the slider ranges (hues wrap). */
export function sanitizeParams(params, base) {
  const out = { ...base };
  if (!params || typeof params !== 'object') return out;
  for (const [k, [lo, hi]] of Object.entries(RANGES)) {
    const v = Number(params[k]);
    if (!Number.isFinite(v)) continue;
    out[k] = k.endsWith('Hue') ? Math.round(((v % 360) + 360) % 360) : Math.round(Math.min(hi, Math.max(lo, v)) * 100) / 100;
  }
  if (params.prim && typeof params.prim === 'object') out.prim = sanitizePrim({ ...(base.prim || neutralPrim()), ...params.prim });
  return out;
}

const fmt = (s) => {
  const q = (p) => s.Lq[Math.round(p * (s.Lq.length - 1))];
  const cast = (a, b) => { const m = Math.hypot(a, b); return m < 0.008 ? '中性' : `${hueName((Math.atan2(b, a) * 180) / Math.PI)} ${m.toFixed(3)}`; };
  return `平均亮度 ${s.Lmean.toFixed(3)}，對比(亮度標準差) ${s.Lsd.toFixed(3)}，黑位 ${q(0).toFixed(3)}，白位 ${q(1).toFixed(3)}，平均彩度 ${s.chroma.toFixed(3)}，亮部偏色 ${cast(s.hiA, s.hiB)}，暗部偏色 ${cast(s.loA, s.loB)}`;
};

export function buildPrompt({ round, maxRounds, params, metrics, instruction = '' }) {
  const current = Object.fromEntries(Object.keys(RANGES).map((k) => [k, params[k]]));
  current.prim = params.prim || neutralPrim();
  return `你是專業調色師。任務：調整影片的色彩與光暗，讓它呈現參考圖的氛圍（色調、明暗、柔和度、飽和度、亮部與暗部的偏色）。${instruction ? `\n使用者的要求（最優先）：「${instruction.slice(0, 300)}」——在目前設定的基礎上照這個要求修改，其他部分盡量不動。` : ''}
附上三張圖：
1. 參考圖（要模仿的風格）
2. 影片原始影格
3. 目前設定套用後的結果（第 ${round} 輪，最多 ${maxRounds} 輪）

量化數據（OKLab，亮度 0–1；彩度 0.05 左右已算鮮明）：
- 參考圖：${fmt(metrics.ref)}
- 原始影格：${fmt(metrics.src)}
- 目前結果：${fmt(metrics.cur)}
注意：參考圖的數字包含它的內容（例如大片天空會拉高亮度、鮮豔服裝會拉高彩度）。請分辨哪些是風格、哪些只是內容，不要為了追數字讓影片過曝或變色失真；膚色要自然。

可調參數（目前值）：${JSON.stringify(current)}
參數說明：
- exposure：曝光檔數 -2..2（線性光，亮部柔和壓縮）
- contrast：對比 -1..1；highlights / shadows：高光、陰影 -1..1
- saturation：飽和度 -1..1（彩度 ×(1+值)）
- temperature：色溫 -1..1（正值偏暖）；tint：色調 -1..1（正值偏洋紅）
- black：黑位提亮 0..0.4（輸出亮度下限，霧面感）；white：白位 0.7..1（輸出亮度上限）
- shadowHue/midHue/hiHue：陰影、中間調、亮部要加入的顏色，OKLab 色相角 0..359（約：30 紅、70 橙、100 黃、130 黃綠、160 綠、200 青綠、250 藍、300 紫、340 粉紅）
- shadowAmt/midAmt/hiAmt：上面顏色的份量 0..1（1 約等於彩度 0.04）
- prim：專業調色（與達文西調色頁相同的概念，優先用這組調整色調）：
  - lift / gamma / gain / offset：色輪，各為 {"m": 亮度 -1..1, "h": 顏色的 OKLab 色相角 0..359, "a": 顏色份量 0..1}，分別作用於暗部、中間調、亮部、整體
  - luma / red / green / blue：曲線，5 個輸出值，對應輸入 0、0.25、0.5、0.75、1（不改變時為 [0,0.25,0.5,0.75,1]）
  - sat：整體飽和度 log2，-1..1
  - hueHue / hueSat / hueLum：色相曲線，各 8 個值，依序對應 紅30°、橙70°、黃100°、綠145°、青195°、藍255°、紫300°、洋紅340°；hueHue 為色相偏移度數 -60..60，hueSat 為飽和度 log2 -1..1，hueLum 為亮度 -0.15..0.15
  - skin：膚色保護 0..1（AI 皮膚遮罩內保留原膚色，只跟隨明暗）

請比較圖 3 和圖 1，決定下一組參數（每次只做必要的修改）。如果圖 3 已經很接近參考圖的氛圍，done 設為 true。
只回覆一個 JSON：{"params": {上面全部參數，含 prim}, "done": true 或 false, "note": "一句繁體中文，說明這次調整了什麼、為什麼"}`;
}

/** ImageData or ImageBitmap -> JPEG Blob (long side ≤ maxSide). */
export async function toJpeg(src, maxSide = 1024) {
  const w0 = src.width, h0 = src.height, s = Math.min(1, maxSide / Math.max(w0, h0));
  const w = Math.round(w0 * s), h = Math.round(h0 * s);
  const c = canvas2d(w, h), ctx = c.getContext('2d');
  if (src instanceof ImageData) {
    const tmp = canvas2d(w0, h0); tmp.getContext('2d').putImageData(src, 0, 0);
    ctx.drawImage(tmp, 0, 0, w, h);
  } else ctx.drawImage(src, 0, 0, w, h);
  return c.convertToBlob ? c.convertToBlob({ type: 'image/jpeg', quality: 0.9 }) : new Promise((res) => c.toBlob(res, 'image/jpeg', 0.9));
}

/**
 * Iterate: render the current look, show Claude reference + original + result, take its parameters.
 * render(look) -> {image: ImageData, stats}; returns {look, notes, rounds}.
 */
export async function claudeGrade({ sample, refImage, frameImage, metrics, look0, render, onStatus = () => {}, signal, maxRounds = 3, instruction = '' }) {
  const ref = await toJpeg(refImage), frame = await toJpeg(frameImage);
  let look = { ...look0 };
  const notes = [];
  for (let round = 1; round <= maxRounds; round++) {
    onStatus(`Claude 正在比較參考圖和目前結果（第 ${round}/${maxRounds} 輪）…`);
    const cur = await render(look);
    const prompt = buildPrompt({ round, maxRounds, params: look, metrics: { ...metrics, cur: cur.stats }, instruction });
    const r = await sample.json(prompt, { images: [ref, frame, await toJpeg(cur.image)], modelTier: 'default', signal });
    look = sanitizeParams(r && r.params, look);
    if (r && typeof r.note === 'string') notes.push(r.note.slice(0, 200));
    if (r && r.done === true) return { look, notes, rounds: round };
  }
  return { look, notes, rounds: maxRounds };
}
