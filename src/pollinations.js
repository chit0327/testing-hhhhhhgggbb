// Pollinations image editing with the USER'S OWN key (an `sk_` secret key from enter.pollinations.ai).
// Same contract as gemini.js: the key stays in this browser and goes only to gen.pollinations.ai in the
// Authorization header. Generation spends the account's Pollen (Quest Pollen first, then paid Pollen).
// API: GET /image/models (public catalogue), POST /v1/images/edits (OpenAI-compatible, multipart).
import { GeminiError as ProviderError } from './gemini.js';

const BASE = 'https://gen.pollinations.ai';
const KEY_STORE = 'zhaoyang.pollinationsKey';

export const pollinationsKeyStore = {
  load() { try { return localStorage.getItem(KEY_STORE) || ''; } catch { return ''; } },
  save(k) { try { localStorage.setItem(KEY_STORE, k); } catch { /* memory only */ } },
  clear() { try { localStorage.removeItem(KEY_STORE); } catch { /* nothing stored */ } },
};

async function call(path, key, init = {}) {
  let res;
  try {
    res = await fetch(`${BASE}${path}`, { ...init, headers: { ...(key ? { Authorization: `Bearer ${key}` } : {}), ...(init.headers || {}) } });
  } catch (err) {
    if (err && err.name === 'AbortError') throw new ProviderError('已停止', 'cancelled');
    throw new ProviderError('無法連線到 Pollinations（網路被阻擋或離線；在 claude.ai 網頁裏不能連線外部服務）', 'network');
  }
  let body = null;
  try { body = await res.json(); } catch { /* not JSON */ }
  if (!res.ok) {
    const e = body && (body.error || body);
    const msg = (e && (e.message || e.error)) || res.statusText;
    if (res.status === 401) throw new ProviderError('Pollinations 金鑰無效或缺少（請使用 sk_ 開頭的金鑰）', 'bad_key');
    if (res.status === 402) throw new ProviderError(`Pollen 點數不足（帳戶或這把金鑰的預算已用完）：${String(msg).slice(0, 160)}`, 'quota');
    if (res.status === 429) {
      const retry = res.headers && res.headers.get && res.headers.get('Retry-After');
      throw new ProviderError(`Pollinations 請求太頻繁${retry ? `，請約 ${retry} 秒後再試` : '，請稍後再試'}`, 'quota');
    }
    if (res.status === 403) throw new ProviderError(`這把金鑰沒有使用這個模型的權限：${String(msg).slice(0, 160)}`, 'forbidden');
    throw new ProviderError(`Pollinations 回報錯誤（${res.status}）：${String(msg).slice(0, 200)}`, res.status === 400 ? 'bad_request' : 'upstream');
  }
  return body;
}

/** Image models that take an image as input (editing), best guess first. With a key, the list honours its permissions. */
export async function listEditModels(key) {
  const list = await call('/image/models?source=official', key);
  const has = (arr, v) => Array.isArray(arr) && arr.includes(v);
  const out = (Array.isArray(list) ? list : [])
    .filter((m) => has(m.input_modalities, 'image') && (!m.output_modalities || has(m.output_modalities, 'image')) && !has(m.output_modalities, 'video'))
    .map((m) => ({ id: m.name, refs: Number(m.max_reference_images) || 1 }));
  const rank = (m) => (/kontext|edit/i.test(m.id) ? -2 : 0) + (m.refs >= 2 ? -1 : 0);
  return out.sort((a, b) => rank(a) - rank(b));
}

/**
 * Re-grade `frame` in the style of `reference`. refs: how many images the model accepts (the reference is
 * sent as a second image only when it can take two). Resolves with the returned image Blob.
 */
export async function pollinationsRegrade({ key, model, refs = 1, prompt, frame, reference, width, height, signal }) {
  const fd = new FormData();
  fd.append('image', frame, 'frame.jpg');
  if (refs >= 2) fd.append('image', reference, 'reference.jpg');
  fd.append('prompt', refs >= 2 ? prompt : prompt.replace(/第二張圖是參考圖。?/, ''));
  fd.append('model', model);
  if (width && height) fd.append('size', `${Math.round(width / 16) * 16}x${Math.round(height / 16) * 16}`);
  fd.append('response_format', 'b64_json');
  const body = await call('/v1/images/edits', key, { method: 'POST', body: fd, signal });
  const d = body && Array.isArray(body.data) && body.data[0];
  if (d && d.b64_json) {
    const bin = atob(d.b64_json), u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return new Blob([u8], { type: d.media_type || 'image/png' });
  }
  if (d && d.url) {
    const r = await fetch(d.url, { signal }).catch(() => null);
    if (r && r.ok) return r.blob();
  }
  throw new ProviderError('Pollinations 沒有回傳圖片', 'no_image');
}
