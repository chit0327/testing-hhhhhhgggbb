// Gemini image editing with the USER'S OWN API key ("bring your own key"). The browser calls Google
// directly: the key stays in this browser (memory, or localStorage only if the user asks) and is sent only
// to generativelanguage.googleapis.com in a header, never in a URL. Nothing here runs unless the user
// enters a key and agrees to send the frame and the reference to Google.
//
// Model ids for image output change often, so none is hard-coded as the only choice: the page lists the
// models this key can use and keeps the image-capable ones.
const BASE = 'https://generativelanguage.googleapis.com/v1beta';
const KEY_STORE = 'zhaoyang.geminiKey';

export const keyStore = {
  load() { try { return localStorage.getItem(KEY_STORE) || ''; } catch { return ''; } },
  save(k) { try { localStorage.setItem(KEY_STORE, k); } catch { /* storage unavailable: memory only */ } },
  clear() { try { localStorage.removeItem(KEY_STORE); } catch { /* nothing stored */ } },
};

export class GeminiError extends Error {
  constructor(message, code) { super(message); this.code = code; }
}

async function call(path, key, init = {}) {
  let res;
  try {
    res = await fetch(`${BASE}/${path}`, { ...init, headers: { 'x-goog-api-key': key, 'Content-Type': 'application/json', ...(init.headers || {}) } });
  } catch (err) {
    if (err && err.name === 'AbortError') throw new GeminiError('已停止', 'cancelled');
    throw new GeminiError('無法連線到 Google Gemini（網路被阻擋或離線；在 claude.ai 網頁裏不能連線外部服務）', 'network');
  }
  let body = null;
  try { body = await res.json(); } catch { /* not JSON */ }
  if (!res.ok) {
    const msg = body && body.error && body.error.message || res.statusText;
    const reason = JSON.stringify(body && body.error && body.error.details || '');
    if (res.status === 400 && /API_KEY_INVALID|API key not valid/i.test(msg + reason)) throw new GeminiError('API 金鑰無效，請檢查是否貼錯', 'bad_key');
    if (res.status === 403) throw new GeminiError(`這個金鑰沒有使用權限：${msg}`, 'forbidden');
    if (res.status === 429) throw new GeminiError('Gemini 用量已達上限（免費額度或速率限制），請稍後再試', 'quota');
    throw new GeminiError(`Gemini 回報錯誤（${res.status}）：${msg}`, res.status === 400 ? 'bad_request' : 'upstream');
  }
  return body;
}

/** Image-output models this key can use, best guess first. */
export async function listImageModels(key) {
  const out = [];
  let token = '';
  for (let page = 0; page < 5; page++) {
    const body = await call(`models?pageSize=200${token ? `&pageToken=${encodeURIComponent(token)}` : ''}`, key);
    for (const m of body.models || []) {
      const id = String(m.name || '').replace(/^models\//, '');
      if (/image/i.test(id) && (m.supportedGenerationMethods || []).includes('generateContent') && !/imagen/i.test(id)) out.push(id);
    }
    token = body.nextPageToken;
    if (!token) break;
  }
  const rank = (id) => (/preview|exp/i.test(id) ? 1 : 0) + (/banana/i.test(id) ? -2 : 0) + (/flash/i.test(id) ? -1 : 0);
  return out.sort((a, b) => rank(a) - rank(b) || b.localeCompare(a));
}

async function blobToBase64(blob) {
  const u8 = new Uint8Array(await blob.arrayBuffer());
  let bin = '';
  for (let i = 0; i < u8.length; i += 0x8000) bin += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  return btoa(bin);
}

const RATIOS = ['1:1', '2:3', '3:2', '3:4', '4:3', '4:5', '5:4', '9:16', '16:9', '21:9'];
export function nearestRatio(w, h) {
  const r = Math.log(w / h);
  let best = '1:1', bd = Infinity;
  for (const s of RATIOS) { const [a, b] = s.split(':').map(Number), d = Math.abs(Math.log(a / b) - r); if (d < bd) { bd = d; best = s; } }
  return best;
}

/**
 * Ask the model to re-grade `frame` in the style of `reference`. Resolves with the returned image Blob.
 * The first image is the frame to edit, the second the style reference (the prompt says so).
 */
export async function geminiRegrade({ key, model, prompt, frame, reference, aspectRatio, signal }) {
  const parts = [
    { text: prompt },
    { inline_data: { mime_type: frame.type || 'image/jpeg', data: await blobToBase64(frame) } },
    { inline_data: { mime_type: reference.type || 'image/jpeg', data: await blobToBase64(reference) } },
  ];
  const body = (withRatio) => JSON.stringify({
    contents: [{ role: 'user', parts }],
    generationConfig: { responseModalities: ['TEXT', 'IMAGE'], ...(withRatio && aspectRatio ? { imageConfig: { aspectRatio } } : {}) },
  });
  let res;
  try {
    res = await call(`models/${encodeURIComponent(model)}:generateContent`, key, { method: 'POST', body: body(true), signal });
  } catch (err) { // older models reject imageConfig: try once without it
    if (err.code !== 'bad_request' || !/imageConfig|aspect/i.test(err.message)) throw err;
    res = await call(`models/${encodeURIComponent(model)}:generateContent`, key, { method: 'POST', body: body(false), signal });
  }
  const cand = (res.candidates || [])[0];
  const outParts = (cand && cand.content && cand.content.parts) || [];
  for (const p of outParts) {
    const d = p.inlineData || p.inline_data;
    if (d && /^image\//.test(d.mimeType || d.mime_type || '')) {
      const bin = atob(d.data), u8 = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
      return new Blob([u8], { type: d.mimeType || d.mime_type });
    }
  }
  const text = outParts.map((p) => p.text).filter(Boolean).join(' ').slice(0, 160);
  const why = cand && cand.finishReason && cand.finishReason !== 'STOP' ? `（${cand.finishReason}）` : '';
  throw new GeminiError(`Gemini 沒有回傳圖片${why}${text ? `：${text}` : ''}`, 'no_image');
}
