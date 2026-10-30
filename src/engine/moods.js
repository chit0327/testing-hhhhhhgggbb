// Mood presets (氛圍): ready-made looks for when there is no reference image. Each is a plain look in the same
// parameter space as the sliders (exposure … vignette, three-way colour), plus a few hue-curve moves where a
// mood needs them (teal greens, mint, olive film greens). Choosing one fills the sliders, so it can be tuned.
// OKLab hues used below: red ≈ 30°, orange ≈ 60°, yellow ≈ 100°, green ≈ 145°, cyan ≈ 195°, blue ≈ 255°,
// magenta ≈ 330° (the hue-curve anchors are primary.js HUE_ANCHORS: 30 70 100 145 195 255 300 340).
import { DEFAULT_LOOK } from './grade.js';
import { neutralPrim, HUE_ANCHORS } from './primary.js';

const ANCHOR = { red: 30, orange: 70, yellow: 100, green: 145, cyan: 195, blue: 255, purple: 300, magenta: 340 };
/** hue curves: {green: [hue shift °, saturation log2, lightness]} → a pro-stage parameter set (or null) */
function hues(spec) {
  if (!spec) return null;
  const p = neutralPrim();
  for (const [name, [shift, sat, lum = 0]] of Object.entries(spec)) {
    const i = HUE_ANCHORS.indexOf(ANCHOR[name]);
    p.hueHue[i] = shift; p.hueSat[i] = sat; p.hueLum[i] = lum;
  }
  return p;
}

export const MOOD_GROUPS = ['日常、清新', '電影感', '懷舊、膠片', '時間、光線', '情緒'];

export const MOODS = [
  { id: 'jp_fresh', group: 0, name: '日系清新', desc: '明亮、低對比、暗部微提亮，帶淡淡青藍',
    look: { exposure: 0.25, contrast: -0.25, highlights: -0.2, shadows: 0.35, saturation: -0.15, temperature: -0.25, tint: -0.04, black: 0.06, shadowHue: 205, shadowAmt: 0.25, hiHue: 200, hiAmt: 0.12 },
    hue: { green: [14, -0.3], orange: [0, -0.1] } },
  { id: 'creamy', group: 0, name: '奶油暖白', desc: '整體偏暖，白色像奶油，柔和',
    look: { exposure: 0.12, contrast: -0.15, highlights: -0.4, shadows: 0.2, saturation: -0.1, temperature: 0.18, black: 0.04, white: 0.95, hiHue: 90, hiAmt: 0.28, shadowHue: 60, shadowAmt: 0.08, glow: 0.1 } },
  { id: 'clear', group: 0, name: '晴朗通透', desc: '天空更藍、綠色更鮮，對比清楚',
    look: { contrast: 0.2, highlights: -0.1, shadows: 0.1, saturation: 0.15, temperature: -0.05 },
    hue: { blue: [0, 0.3], cyan: [0, 0.2], green: [0, 0.2] } },
  { id: 'teal_orange', group: 1, name: '青橙電影', desc: '暗部和背景偏青、膚色偏橙',
    look: { contrast: 0.25, highlights: -0.1, temperature: -0.05, shadowHue: 205, shadowAmt: 0.9, midHue: 200, midAmt: 0.2, hiHue: 60, hiAmt: 0.45, vignette: 0.25 },
    hue: { green: [45, -0.3], yellow: [-10, -0.2], blue: [-20, 0], orange: [0, 0.25] } },
  // 電影柔光: a Black Pro-Mist filter in late-afternoon sun (tuned on the user's example, a film still): bright parts
  // glow in a tight halo hugging their edges, blacks stay deep (no veil), whites held below white, low saturation,
  // warm highlights (example: darkest 1 % at OKLab L 0.18, brightest 1 % 0.81, chroma 0.032)
  { id: 'promist', group: 1, name: '電影柔光', desc: '午後陽光，亮部柔柔地發光、暗部保持深沉，顏色淡雅，像加了柔光濾鏡的電影畫面',
    look: { exposure: 0.1, highlights: -0.1, saturation: -0.35, temperature: 0.15, white: 0.95, hiHue: 80, hiAmt: 0.2, haze: 0.45, hazeRadius: 0.3, hazeSoft: 0.3, hazeBloom: 0.6, hazeVeil: 0 } },
  { id: 'cool_city', group: 1, name: '冷調都市', desc: '低飽和、偏青灰，帶點疏離感',
    look: { contrast: 0.15, saturation: -0.35, temperature: -0.35, tint: -0.08, black: 0.03, shadowHue: 200, shadowAmt: 0.4, vignette: 0.2 } },
  { id: 'pastel', group: 1, name: '粉彩復古', desc: '粉色、薄荷綠、淡黃，柔和夢幻',
    look: { exposure: 0.1, contrast: -0.25, highlights: -0.15, saturation: -0.05, black: 0.08, white: 0.96, shadowHue: 340, shadowAmt: 0.4, midHue: 350, midAmt: 0.15, hiHue: 95, hiAmt: 0.3 },
    hue: { green: [35, -0.2], red: [-10, -0.15], blue: [-15, -0.2] } },
  { id: 'warm_film', group: 2, name: '暖調膠片', desc: '暖黃、柔和對比，暗部不死黑',
    look: { exposure: -0.05, contrast: 0.05, saturation: -0.05, temperature: 0.3, black: 0.07, white: 0.95, shadowHue: 175, shadowAmt: 0.3, hiHue: 70, hiAmt: 0.35, glow: 0.1 },
    hue: { green: [-18, -0.35], blue: [-10, -0.2] } },
  { id: 'faded', group: 2, name: '褪色懷舊', desc: '顏色變淡、黑位變灰，像舊照片',
    look: { contrast: -0.25, saturation: -0.45, temperature: 0.2, black: 0.15, white: 0.9, hiHue: 90, hiAmt: 0.25, shadowHue: 60, shadowAmt: 0.1, vignette: 0.3 } },
  { id: 'bw', group: 2, name: '黑白經典', desc: '去色，保留豐富的明暗層次',
    look: { contrast: 0.25, saturation: -1, vignette: 0.15 } },
  { id: 'golden', group: 3, name: '黃金時刻', desc: '日落前的金橙色光線，暖、溫柔',
    look: { contrast: 0.05, highlights: -0.35, saturation: 0.1, temperature: 0.5, tint: 0.06, white: 0.96, hiHue: 65, hiAmt: 0.5, shadowHue: 40, shadowAmt: 0.2, glow: 0.12 } },
  // 藍調時刻: everything in one saturated azure / royal blue, shadows included (lifted, blue blacks), soft contrast,
  // lamps pale rather than orange (tuned on a blue-hour city photo: chroma 0.09 shadows, 0.12–0.14 mids, 0.11
  // brights, hue 250–260°, median lightness 0.59, no warm colour left)
  { id: 'blue_hour', group: 3, name: '藍調時刻', desc: '整體沉浸在飽和的藍色裏，暗部也是藍，燈光偏白不偏橙',
    look: { exposure: -0.3, contrast: -0.1, highlights: -0.3, shadows: 0.15, saturation: 0.1, temperature: -0.9, black: 0.07, white: 0.95, shadowHue: 262, shadowAmt: 1, midHue: 252, midAmt: 1, hiHue: 236, hiAmt: 0.8 },
    hue: { blue: [0, 1], cyan: [30, 1], purple: [-30, 0.6], green: [45, 0, -0.04], yellow: [0, -0.6, 0.05], orange: [0, -0.5, 0.03], red: [0, -0.3] } },
  { id: 'overcast', group: 3, name: '陰天柔和', desc: '低對比、偏冷灰，安靜帶點憂鬱',
    look: { exposure: -0.15, contrast: -0.4, highlights: -0.4, saturation: -0.45, temperature: -0.25, black: 0.06, white: 0.94, shadowHue: 210, shadowAmt: 0.25, midHue: 220, midAmt: 0.1 } },
  { id: 'memory', group: 4, name: '溫暖回憶', desc: '暖色、柔光、輕微暗角，像回憶片段',
    look: { exposure: -0.1, contrast: -0.25, highlights: -0.3, saturation: -0.25, temperature: 0.45, tint: 0.05, black: 0.1, white: 0.94, shadowHue: 35, shadowAmt: 0.3, hiHue: 70, hiAmt: 0.35, glow: 0.22, glowRadius: 0.6, vignette: 0.4 } },
  { id: 'thriller', group: 4, name: '冷峻懸疑', desc: '暗、偏綠青、高對比',
    look: { exposure: -0.35, contrast: 0.4, highlights: -0.2, saturation: -0.3, temperature: -0.2, tint: -0.3, shadowHue: 165, shadowAmt: 0.6, midHue: 170, midAmt: 0.15, vignette: 0.5 } },
  // 夢幻朦朧: tuned on a backlit portrait in sunset light (the user's example): soft pastel colours, warm peach light,
  // lights blooming, shadows lifted by warm light (chroma ≈ 0.036, darkest 1 % at OKLab L ≈ 0.3, hue 40–55°)
  { id: 'dreamy', group: 4, name: '夢幻朦朧', desc: '暖暖的逆光感，亮部向外發光，暗部被暖光提亮，顏色淡雅柔和',
    look: { exposure: 0.1, contrast: -0.2, highlights: -0.25, saturation: -0.25, temperature: 0.2, tint: 0.1, black: 0.04, shadowHue: 30, shadowAmt: 0.3, hiHue: 75, hiAmt: 0.2, haze: 0.6, hazeRadius: 0.6 } },
  { id: 'vivid', group: 4, name: '高飽和活力', desc: '顏色鮮艷、對比強',
    look: { contrast: 0.3, highlights: -0.1, shadows: 0.1, saturation: 0.45 },
    hue: { orange: [0, 0.1], red: [0, 0.1] } },
];

/** The full look for a mood (no reference: strength / match off, the mood's sliders and hue curves). */
export function moodLook(mood, keep = {}) {
  return { ...DEFAULT_LOOK, strength: 0, match: 0, source: 'mood', mood: mood.id, ...mood.look, prim: hues(mood.hue), ...keep };
}
