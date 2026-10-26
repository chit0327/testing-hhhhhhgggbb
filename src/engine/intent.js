// LightingIntent: the structured, serialisable state that carries the user's confirmed lighting choices
// from the dialog to the worker, the renderer and the export. The UI edits a DRAFT copy; only
// confirmIntent() produces the CONFIRMED copy that processing reads.

export const MAX_EV = 1.0;     // strength 100% = ±1 stop (EV); the default 60% = ±0.6 EV
export const EV_LIMIT = 1.5;   // overlapping regions add in stops, clamped to ±1.5 EV
export const MASK_W = 384;     // mask / edit resolution (width); height follows the video aspect

export const MODES = {
  auto: '自動匹配參考圖的光暗（建議）',
  brighten: '變亮',
  darken: '變暗',
  mixed: '不同區域分別變亮或變暗',
  preserve: '保留原始光暗，只匹配參考色彩',
};
export const SELECTIONS = { ai: 'AI 辨識後由我確認', manual: '手動選區', global: '整體光暗調整' };

// region kinds and how their per-frame mask is produced
export const KINDS = {
  'ai-person': { label: '人物', source: 'ai', tracked: true },
  'ai-skin': { label: '臉部與皮膚', source: 'ai', tracked: true },
  'ai-background': { label: '背景（人物以外）', source: 'ai', tracked: true },
  'luma-highlights': { label: '亮部（亮度選區）', source: 'luma', tracked: true },
  'luma-shadows': { label: '暗部（亮度選區）', source: 'luma', tracked: true },
  manual: { label: '手動選區', source: 'manual', tracked: false },
  global: { label: '整個畫面', source: 'global', tracked: true },
  // internal (automatic local grade only, never offered in the dialog)
  'ai-hair': { label: '頭髮', source: 'ai', tracked: true, internal: true },
  'ai-clothes': { label: '衣服', source: 'ai', tracked: true, internal: true },
};

let _id = 0;
export function newRegion(kind, maskH, extra = {}) {
  return {
    id: `r${Date.now().toString(36)}${(_id++).toString(36)}`,
    kind,
    label: KINDS[kind].label,
    enabled: kind === 'ai-person',          // a sensible first suggestion; user may change it
    status: kind === 'global' ? 'confirmed' : 'proposed',
    direction: 1,                           // +1 brighter, -1 darker, 0 unchanged
    strength: 0.6,
    edits: { w: MASK_W, h: maskH, add: null, erase: null }, // user brush corrections (Uint8 0..255)
    ...extra,
  };
}

export function createIntent(maskH = 216) {
  return { version: 2, mode: 'auto', selection: 'ai', strength: 0.6, feather: 0.006, regions: [], maskH };
}

export function cloneIntent(it) {
  return {
    ...it,
    regions: it.regions.map((r) => ({
      ...r,
      edits: { ...r.edits, add: r.edits.add ? new Uint8Array(r.edits.add) : null, erase: r.edits.erase ? new Uint8Array(r.edits.erase) : null },
    })),
  };
}

/** Any change to a region's mask (brush edit) makes it 'proposed' again: it must be re-confirmed. */
export function markEdited(region) { if (region.kind !== 'global') region.status = 'proposed'; }

/**
 * Regions that may actually change pixels: enabled, CONFIRMED, non-zero direction, and the mode allows
 * local adjustment. Direction is forced by the mode for 'brighten'/'darken'; strength by the global
 * slider unless the mode is 'mixed' (per-region strength).
 */
export function effectiveRegions(it) {
  if (!it || it.mode === 'preserve' || it.mode === 'auto') return []; // auto: lighting comes from the global look
  const sel = it.selection;
  return it.regions
    .filter((r) => r.enabled && r.status === 'confirmed')
    .filter((r) => (sel === 'global' ? r.kind === 'global' : sel === 'manual' ? r.kind === 'manual' : r.kind !== 'global'))
    .map((r) => {
      const direction = it.mode === 'brighten' ? 1 : it.mode === 'darken' ? -1 : r.direction;
      const strength = it.mode === 'mixed' ? r.strength : it.strength;
      return { ...r, direction, strength, ev: direction * strength * MAX_EV };
    })
    .filter((r) => r.ev !== 0);
}

/** Validation used by the dialog's confirm button. Returns {ok, reason}. */
export function canConfirm(it) {
  if (it.mode === 'preserve' || it.mode === 'auto') return { ok: true };
  const enabled = it.regions.filter((r) => r.enabled && (it.selection === 'global' ? r.kind === 'global' : it.selection === 'manual' ? r.kind === 'manual' : r.kind !== 'global'));
  if (!enabled.length) return { ok: false, reason: '請至少選擇一個要調整的區域。' };
  const unconfirmed = enabled.filter((r) => r.status !== 'confirmed');
  if (unconfirmed.length) return { ok: false, reason: `還有 ${unconfirmed.length} 個已選區域的遮罩尚未確認。` };
  if (!effectiveRegions(it).length) return { ok: false, reason: '已選區域的調整方向都是「不變」。' };
  return { ok: true };
}

/** Freeze the draft into the confirmed state. Throws if not confirmable. */
export function confirmIntent(draft) {
  const v = canConfirm(draft);
  if (!v.ok) throw new Error(v.reason);
  return { ...cloneIntent(draft), confirmedAt: Date.now() };
}

// ---- persistence ---------------------------------------------------------------------------------
// chunked: spreading a whole mask into String.fromCharCode overflows the stack in WebKit (≈65k argument limit)
function bytesToBinary(u8) {
  let s = '';
  for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  return s;
}
const b64 = (u8) => (u8 ? (typeof Buffer !== 'undefined' ? Buffer.from(u8).toString('base64') : btoa(bytesToBinary(u8))) : null);
const unb64 = (s) => (s ? (typeof Buffer !== 'undefined' ? new Uint8Array(Buffer.from(s, 'base64')) : Uint8Array.from(atob(s), (c) => c.charCodeAt(0))) : null);

export function serializeIntent(it) {
  return JSON.stringify({ ...it, regions: it.regions.map((r) => ({ ...r, edits: { ...r.edits, add: b64(r.edits.add), erase: b64(r.edits.erase) } })) });
}

export function deserializeIntent(s) {
  const o = JSON.parse(s);
  return { ...o, regions: o.regions.map((r) => ({ ...r, edits: { ...r.edits, add: unb64(r.edits.add), erase: unb64(r.edits.erase) } })) };
}

/** Stable FNV-1a hash of the serialised intent (logged by UI, worker and export to prove they agree). */
export function hashIntent(it) {
  const s = serializeIntent({ ...it, confirmedAt: undefined });
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h.toString(16).padStart(8, '0');
}
