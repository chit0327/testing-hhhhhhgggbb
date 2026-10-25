// Page controller: asset loading, the Lighting Intent dialog (draft -> confirmed), processing and export.
// Pixels are only ever produced by the worker (src/worker.js) from the serialised CONFIRMED intent.
import {
  createIntent, newRegion, cloneIntent, confirmIntent, canConfirm, effectiveRegions, markEdited,
  serializeIntent, hashIntent, MODES, KINDS, MASK_W,
} from './engine/intent.js';
import { rawMask } from './engine/regionmasks.js';
import { applyEdits, feather, lightnessGrid } from './engine/masks.js';
import { EngineClient, probeVideo, grabFrame, fitSize, loadImage, imageDataOf, processVideo, seek, frameTime, canvas2d } from './pipeline.js';
import { initSegmenter, segment } from './segment.js';
import { DEFAULT_LOOK, measure } from './engine/grade.js';
import { claudeAvailable, claudeGrade, toJpeg } from './claudegrade.js';
import { keyStore, listImageModels, geminiRegrade, nearestRatio } from './gemini.js';
import { pollinationsKeyStore, listEditModels, pollinationsRegrade } from './pollinations.js';
import { suggestLook, promptFor } from './engine/suggest.js';
import { oklabToSrgb } from './engine/color.js';

const $ = (id) => document.getElementById(id);
const engine = new EngineClient();

const PHASE_TEXT = {
  idle: '請先上傳參考圖片和目標影片', loading: '素材載入中…', analysing: '正在分析參考圖片與影片…',
  draft: '草稿設定：請在光暗設定視窗中確認', unset: '尚未確認光暗設定（按「重新設定光暗」開始）',
  confirmed: '已確認設定，準備處理…', processing: '正在處理影片…', done: '已完成處理',
  exporting: '匯出中…', exported: '匯出完成', error: '發生錯誤',
};

const S = {
  ref: { status: 'empty', id: 0 },
  video: { status: 'empty', id: 0 },
  promptedPair: null,
  grade: { pair: null, ready: null },
  rep: null,                       // representative frame {t, img, seg, L}
  intent: { draft: null, confirmed: null },
  look: { draft: { ...DEFAULT_LOOK }, applied: null, auto: true }, // draft (sliders) vs applied (last processing); auto: follow the reference until the user edits
  phase: 'idle',
  job: { token: 0 },
  output: null,
  ai: { status: 'idle', error: null },
  aiPair: null,                    // {id, t, meanAbs, outlierShare, score}: LUT distilled from an AI-graded frame
  aiFrameT: null,                  // time of the frame offered for AI grading
  viewing: null,
  log: [],                         // event log (used by the end-to-end test)
};
window.__app = { S, hashIntent, engine };
const log = (ev, data = {}) => { S.log.push({ ev, t: Date.now(), ...data }); };

function setPhase(p, detail = '') {
  S.phase = p;
  const el = $('status');
  el.dataset.state = p;
  el.textContent = PHASE_TEXT[p] + (detail ? `：${detail}` : '');
}

const pairKey = () => (S.ref.status === 'ready' && S.video.status === 'ready' ? `${S.ref.id}|${S.video.id}` : null);
const maskH = () => Math.round(MASK_W * S.video.meta.height / S.video.meta.width);

// ------------------------------------------------------------------ assets
$('refInput').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const id = ++S.ref.id;
  S.ref = { status: 'loading', id };
  $('refInfo').textContent = '載入中…';
  try {
    if (!file.type.startsWith('image/')) throw new Error(`不支援的檔案類型（${file.type || '未知'}），請選擇圖片`);
    const bmp = await loadImage(file);
    if (id !== S.ref.id) return;
    S.ref = { status: 'ready', id, file, bitmap: bmp, imageData: imageDataOf(bmp, 512) };
    const img = $('refPreview');
    img.src = URL.createObjectURL(file); img.hidden = false;
    $('refInfo').textContent = `${file.name}　${bmp.width}×${bmp.height}`;
    log('ref-loaded', { name: file.name });
  } catch (err) {
    S.ref = { status: 'error', id };
    $('refInfo').textContent = `無法載入圖片：${err.message}`;
  }
  assetsChanged('ref');
});

$('videoInput').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const id = ++S.video.id;
  S.video = { status: 'loading', id };
  $('videoInfo').textContent = '載入中…';
  try {
    if (!file.type.startsWith('video/') && !/\.(mp4|webm|mov|m4v)$/i.test(file.name)) throw new Error(`不支援的檔案類型（${file.type || '未知'}），請選擇影片`);
    const meta = await probeVideo(file);
    if (id !== S.video.id) return;
    S.video = { status: 'ready', id, file, meta };
    const v = $('videoPreview');
    v.src = meta.url; v.hidden = false;
    $('videoInfo').textContent = `${file.name}　${meta.width}×${meta.height}　${meta.duration.toFixed(2)} 秒　${meta.fps} fps${meta.fpsMeasured ? '' : '（無法測得，假設值）'}　約 ${meta.frameCount} 格`
      + (meta.duration > 60 ? '　⚠ 第一版建議 60 秒以內' : '');
    log('video-loaded', { name: file.name, fps: meta.fps, frames: meta.frameCount });
  } catch (err) {
    S.video = { status: 'error', id };
    $('videoInfo').textContent = `無法載入影片：${err.message}（瀏覽器可能不支援此編碼）`;
  }
  assetsChanged('video');
});

async function assetsChanged(which) {
  const key = pairKey();
  if (!key) { setPhase(S.ref.status === 'loading' || S.video.status === 'loading' ? 'loading' : 'idle'); return; }
  // new pair: old results no longer valid; a new video also invalidates masks (geometry/content changed)
  S.output = null; $('resultCard').hidden = true; $('adjustCard').hidden = true;
  if (which === 'video') { S.intent.confirmed = null; S.rep = null; clearAiPair(); }
  updateIntentCard();
  $('openIntent').disabled = false;
  if (S.grade.pair !== key) fitGrade(key);
  if (S.promptedPair !== key) { // auto-open exactly once per newly loaded pair
    S.promptedPair = key;
    log('dialog-auto-open', { pair: key });
    openDialog();
  }
}

function fitGrade(key) {
  S.grade.pair = key;
  setPhase('analysing');
  S.grade.ready = (async () => {
    const m = S.video.meta, { w, h } = fitSize(m.width, m.height, 320);
    const frames = [];
    for (const f of [0.1, 0.25, 0.4, 0.55, 0.7, 0.85]) { // frame-centred times -> deterministic frames
      const img = await grabFrame(m.video, frameTime(m, Math.round(f * (m.frameCount - 1))), w, h);
      frames.push({ data: img.data, width: w, height: h });
    }
    const ref = S.ref.imageData;
    const r = await engine.fit(frames, { data: ref.data, width: ref.width, height: ref.height });
    S.grade.stats = { src: r.src, ref: r.ref, palette: r.palette };
    $('aiPrompt').value = promptFor(r.ref, r.palette);
    if (S.look.auto) applyAutoLook();
    log('grade-fitted', { ms: Math.round(r.ms) });
    return key;
  })();
  S.grade.ready.then(() => { if (S.phase === 'analysing') setPhase(S.intent.confirmed ? 'confirmed' : 'draft'); })
    .catch((err) => setPhase('error', `參考分析失敗：${err.message}`));
  return S.grade.ready;
}

// ------------------------------------------------------------------ dialog
const dlg = $('intentDialog');

function defaultRegions(mh) {
  return ['ai-person', 'ai-skin', 'ai-background', 'luma-highlights', 'luma-shadows'].map((k) => newRegion(k, mh));
}

async function openDialog() {
  const mh = maskH();
  if (S.intent.confirmed) S.intent.draft = cloneIntent(S.intent.confirmed);
  else { S.intent.draft = createIntent(mh); S.intent.draft.regions = defaultRegions(mh); }
  S.viewing = S.intent.draft.regions.find((r) => r.kind === 'ai-person')?.id || null;
  if (!dlg.open) dlg.showModal();
  if (!S.intent.confirmed) setPhase('draft');
  syncDialogControls();
  await loadRepFrame(S.rep ? S.rep.t : frameTime(S.video.meta, Math.floor(S.video.meta.frameCount / 2)));
}

async function loadRepFrame(t) {
  const m = S.video.meta, { w, h } = fitSize(m.width, m.height, 640);
  const img = await grabFrame(m.video, t, w, h);
  S.rep = { t, img, seg: null, L: lightnessGrid(img.data, w, h, MASK_W, maskH()) };
  for (const c of [$('maskCanvas'), $('previewCanvas')]) { c.width = w; c.height = h; }
  $('frameTime').value = Math.round(t / m.duration * 1000);
  drawOverlay(); schedulePreview(); validate(); // confirm needs the frame; auto mode needs nothing else
  await runAI();
}

async function runAI() {
  if (!S.rep) return;
  $('aiStatus').textContent = 'AI 辨識中…';
  S.ai.status = 'loading';
  try {
    await initSegmenter();
    const c = canvas2d(256, maskH());
    c.getContext('2d').drawImage(await createImageBitmap(S.rep.img), 0, 0, c.width, c.height);
    S.rep.seg = segment(c, MASK_W, maskH());
    S.ai.status = 'ready';
    const cover = S.rep.seg.person.reduce((a, b) => a + b, 0) / S.rep.seg.person.length;
    $('aiStatus').textContent = `AI 已提出候選區域（人物約佔 ${Math.round(cover * 100)}%）`;
    log('ai-proposed', { personCover: cover });
  } catch (err) {
    S.ai = { status: 'error', error: err.message };
    $('aiStatus').textContent = `AI 辨識無法使用（${err.message}），請改用手動選區或亮度選區`;
    log('ai-failed', { message: err.message });
  }
  renderRegionList(); drawOverlay(); schedulePreview();
}

function visibleRegions() {
  const d = S.intent.draft;
  return d.regions.filter((r) => (d.selection === 'global' ? r.kind === 'global' : d.selection === 'manual' ? r.kind === 'manual' : !['global', 'manual'].includes(r.kind)));
}

function syncDialogControls() {
  const d = S.intent.draft;
  dlg.querySelector(`input[name=mode][value=${d.mode}]`).checked = true;
  dlg.querySelector(`input[name=selection][value=${d.selection}]`).checked = true;
  $('strength').value = Math.round(d.strength * 100);
  $('strengthVal').textContent = `${Math.round(d.strength * 100)}%`;
  const preserve = d.mode === 'preserve' || d.mode === 'auto'; // no local regions in either
  $('selectionSet').disabled = preserve; $('regionSet').disabled = preserve;
  $('strengthSet').disabled = preserve || d.mode === 'mixed';
  $('addManual').hidden = d.selection !== 'manual';
  $('brushBar').hidden = preserve || d.selection === 'global';
  $('autoNote').hidden = d.mode !== 'auto';
  renderRegionList(); validate();
}

function renderRegionList() {
  const d = S.intent.draft, list = $('regionList');
  if (!d) return;
  list.innerHTML = '';
  const regs = visibleRegions();
  if (!regs.length) list.innerHTML = `<p class="info">${d.selection === 'manual' ? '按「新增手動選區」，再於左邊影格上畫出範圍。' : '沒有可用的區域。'}</p>`;
  for (const r of regs) {
    const aiMissing = KINDS[r.kind].source === 'ai' && S.ai.status !== 'ready';
    const div = document.createElement('div');
    div.className = 'region' + (S.viewing === r.id ? ' viewing' : '');
    div.dataset.kind = r.kind;
    div.innerHTML = `<div class="line">
        <label><input type="checkbox" class="use" ${r.enabled ? 'checked' : ''} ${aiMissing ? 'disabled' : ''}> ${r.label}</label>
        <span class="chip ${r.status}">${r.status === 'confirmed' ? '已確認' : '候選（未確認）'}</span>
        <button type="button" class="view">查看</button>
        ${r.kind !== 'global' ? `<button type="button" class="confirm" ${r.status === 'confirmed' || aiMissing ? 'disabled' : ''}>確認遮罩</button>` : ''}
      </div>
      ${d.mode === 'mixed' ? `<div class="line">
        <select class="dir"><option value="1" ${r.direction === 1 ? 'selected' : ''}>變亮</option><option value="-1" ${r.direction === -1 ? 'selected' : ''}>變暗</option><option value="0" ${r.direction === 0 ? 'selected' : ''}>不變</option></select>
        <input type="range" class="rs" min="0" max="100" value="${Math.round(r.strength * 100)}"> <span>${Math.round(r.strength * 100)}%</span></div>` : ''}
      ${aiMissing ? '<p class="info">AI 尚未提供此區域</p>' : ''}`;
    div.querySelector('.use').onchange = (e) => { r.enabled = e.target.checked; if (r.enabled) S.viewing = r.id; afterEdit(); };
    div.querySelector('.view').onclick = () => { S.viewing = r.id; renderRegionList(); drawOverlay(); };
    const cb = div.querySelector('.confirm');
    if (cb) cb.onclick = () => { r.status = 'confirmed'; r.enabled = true; log('mask-confirmed', { kind: r.kind }); afterEdit(); };
    const dir = div.querySelector('.dir');
    if (dir) dir.onchange = (e) => { r.direction = +e.target.value; afterEdit(); };
    const rs = div.querySelector('.rs');
    if (rs) rs.oninput = (e) => { r.strength = e.target.value / 100; e.target.nextElementSibling.textContent = `${e.target.value}%`; validate(); schedulePreview(); };
    list.appendChild(div);
  }
}

function afterEdit() { renderRegionList(); drawOverlay(); validate(); schedulePreview(); }

function validate() {
  const v = canConfirm(S.intent.draft);
  $('confirmIntent').disabled = !v.ok || !S.rep;
  $('confirmReason').textContent = v.ok ? '' : v.reason;
  return v.ok;
}

function regionMaskForView(r) {
  const mh = maskH();
  let m;
  try { m = rawMask(r, { seg: S.rep.seg, L: S.rep.L }, MASK_W, mh); } catch { m = new Float32Array(MASK_W * mh); }
  return feather(applyEdits(m, r.edits), MASK_W, mh, S.intent.draft.feather * MASK_W);
}

function drawOverlay() {
  if (!S.rep) return;
  const c = $('maskCanvas'), ctx = c.getContext('2d');
  ctx.putImageData(S.rep.img, 0, 0);
  const r = S.intent.draft?.regions.find((x) => x.id === S.viewing);
  $('maskLabel').textContent = r ? `　目前查看：${r.label}（${r.status === 'confirmed' ? '已確認' : '候選'}）` : '';
  if (!r || S.intent.draft.mode === 'preserve') return;
  const m = regionMaskForView(r), mh = maskH();
  const ov = new ImageData(MASK_W, mh);
  const col = (S.intent.draft.mode === 'darken' || (S.intent.draft.mode === 'mixed' && r.direction < 0)) ? [40, 120, 255] : [255, 170, 0];
  for (let i = 0; i < m.length; i++) { ov.data[4 * i] = col[0]; ov.data[4 * i + 1] = col[1]; ov.data[4 * i + 2] = col[2]; ov.data[4 * i + 3] = m[i] * 150; }
  const tmp = canvas2d(MASK_W, mh);
  tmp.getContext('2d').putImageData(ov, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(tmp, 0, 0, c.width, c.height);
}

let previewTimer = null, previewSeq = 0;
function schedulePreview() {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(async () => {
    if (!S.rep || !S.intent.draft) return;
    const seq = ++previewSeq;
    try {
      await S.grade.ready;
      const r = await engine.render(S.rep.img, { intent: serializeIntent(S.intent.draft), seg: S.rep.seg, grade: true, look: S.look.draft });
      if (seq !== previewSeq) return; // a newer preview superseded this one
      $('previewCanvas').getContext('2d').putImageData(r.imageData, 0, 0);
      log('preview-rendered', { localApplied: r.localApplied, intentHash: r.intentHash });
    } catch (err) { $('confirmReason').textContent = `預覽失敗：${err.message}`; }
  }, 120);
}

// dialog controls
dlg.addEventListener('change', (e) => {
  const d = S.intent.draft;
  if (e.target.name === 'mode') { d.mode = e.target.value; syncDialogControls(); drawOverlay(); schedulePreview(); }
  if (e.target.name === 'selection') {
    d.selection = e.target.value;
    if (d.selection === 'global' && !d.regions.some((r) => r.kind === 'global')) d.regions.push(newRegion('global', maskH(), { enabled: true }));
    const first = visibleRegions()[0];
    S.viewing = first ? first.id : null;
    syncDialogControls(); drawOverlay(); schedulePreview();
  }
});
$('strength').addEventListener('input', (e) => { S.intent.draft.strength = e.target.value / 100; $('strengthVal').textContent = `${e.target.value}%`; schedulePreview(); });
$('frameTime').addEventListener('change', (e) => loadRepFrame(frameTime(S.video.meta, Math.round(e.target.value / 1000 * (S.video.meta.frameCount - 1)))));
$('addManual').onclick = () => {
  const r = newRegion('manual', maskH(), { enabled: true });
  r.label = `手動選區 ${S.intent.draft.regions.filter((x) => x.kind === 'manual').length + 1}`;
  S.intent.draft.regions.push(r); S.viewing = r.id; afterEdit();
};

// brush editing on the mask canvas (edits the region currently viewed)
let painting = false;
function paint(e) {
  // no brushing where there are no regions to edit (on a tablet a scroll gesture would otherwise paint)
  if (!S.intent.draft || ['auto', 'preserve'].includes(S.intent.draft.mode) || $('brushBar').hidden) return;
  const r = S.intent.draft.regions.find((x) => x.id === S.viewing);
  if (!r || r.kind === 'global') return;
  const c = $('maskCanvas'), b = c.getBoundingClientRect(), mh = maskH();
  const gx = (e.clientX - b.left) / b.width * MASK_W, gy = (e.clientY - b.top) / b.height * mh;
  const rad = +$('brushSize').value / 2, mode = dlg.querySelector('input[name=brush]:checked').value;
  const n = MASK_W * mh;
  r.edits.add = r.edits.add || new Uint8Array(n); r.edits.erase = r.edits.erase || new Uint8Array(n);
  const [on, off] = mode === 'add' ? [r.edits.add, r.edits.erase] : [r.edits.erase, r.edits.add];
  for (let y = Math.max(0, Math.floor(gy - rad)); y <= Math.min(mh - 1, gy + rad); y++)
    for (let x = Math.max(0, Math.floor(gx - rad)); x <= Math.min(MASK_W - 1, gx + rad); x++)
      if ((x - gx) ** 2 + (y - gy) ** 2 <= rad * rad) { on[y * MASK_W + x] = 255; off[y * MASK_W + x] = 0; }
  if (r.status === 'confirmed') { markEdited(r); renderRegionList(); validate(); }
  r.enabled = true;
  drawOverlay();
}
$('maskCanvas').addEventListener('pointerdown', (e) => { painting = true; $('maskCanvas').setPointerCapture(e.pointerId); paint(e); });
$('maskCanvas').addEventListener('pointermove', (e) => { if (painting) paint(e); });
$('maskCanvas').addEventListener('pointerup', () => { if (painting) { painting = false; log('brush-edit', { region: S.viewing }); renderRegionList(); validate(); schedulePreview(); } });

$('cancelIntent').onclick = () => { // discard the draft; the confirmed intent (if any) is untouched
  S.intent.draft = null;
  dlg.close();
  log('dialog-cancelled');
  if (!S.intent.confirmed) setPhase('unset');
  else setPhase(S.output ? 'done' : 'confirmed');
};
dlg.addEventListener('cancel', (e) => { e.preventDefault(); $('cancelIntent').click(); });

$('confirmIntent').onclick = async () => {
  if (!validate()) return;
  const withClaude = S.intent.draft.mode === 'auto' && S.claude && $('useClaude').checked;
  S.intent.confirmed = confirmIntent(S.intent.draft);
  S.intent.draft = null;
  dlg.close();
  log('intent-confirmed', { hash: hashIntent(S.intent.confirmed) });
  setPhase('confirmed');
  updateIntentCard();
  showAdjustCard();
  if (withClaude) await runClaude();
  startProcessing();
};
$('openIntent').onclick = () => openDialog();
$('editIntent').onclick = () => openDialog();

function describeLook(k) {
  const parts = k.source === 'ai' ? [`強度 ${Math.round(k.strength * 100)}%`] : [`接近參考 ${Math.round(k.match * 100)}%`, `強度 ${Math.round(k.strength * 100)}%`];
  for (const [key, name] of [['exposure', '曝光'], ['contrast', '對比'], ['highlights', '高光'], ['shadows', '陰影'], ['saturation', '飽和'], ['temperature', '色溫'], ['tint', '色調']]) {
    if (k[key]) parts.push(`${name} ${k[key] > 0 ? '+' : ''}${key === 'exposure' ? k[key].toFixed(2) : Math.round(k[key] * 100)}`);
  }
  if (k.black) parts.push(`黑位 +${Math.round(k.black * 100)}`);
  if (k.white !== 1) parts.push(`白位 ${Math.round(k.white * 100)}`);
  for (const [key, name] of [['shadow', '陰影色'], ['mid', '中間調色'], ['hi', '亮部色']]) {
    if (k[`${key}Amt`]) parts.push(`${name} ${Math.round(k[`${key}Hue`])}° ${Math.round(k[`${key}Amt`] * 100)}`);
  }
  if (k.source === 'ai') parts.unshift('AI 調色圖');
  if (k.amount !== undefined && k.amount !== 1) parts.unshift(`整體強度 ${Math.round(k.amount * 100)}%`);
  if (!k.protect) parts.push('不保護材質顏色');
  return parts.join('、');
}

function updateIntentCard() {
  const c = S.intent.confirmed;
  if (!c) { $('intentSummary').textContent = pairKey() ? '尚未確認光暗設定。' : '素材載入後會自動開啟設定視窗。'; $('intentHash').textContent = ''; return; }
  const regs = effectiveRegions(c);
  $('intentSummary').textContent = `已確認：${MODES[c.mode]}` + (regs.length ? `；${regs.map((r) => `${r.label} ${r.ev > 0 ? '+' : ''}${r.ev.toFixed(2)} 檔`).join('、')}` : c.mode === 'preserve' ? '' : '；沒有局部調整');
  $('intentHash').textContent = `設定雜湊 ${hashIntent(c)}`;
}

// ------------------------------------------------------------------ processing / results / export
// ------------------------------------------------------------------ colour & tone adjustments
const ADJ = { // control id -> [look key, slider scale, display]
  amount: ['amount', 100, (v) => `${v}%`],
  match: ['match', 100, (v) => `${v}%`], strength: ['strength', 100, (v) => `${v}%`],
  exposure: ['exposure', 100, (v) => `${v > 0 ? '+' : ''}${(v / 100).toFixed(2)} 檔`], contrast: ['contrast', 100, (v) => `${v > 0 ? '+' : ''}${v}`],
  highlights: ['highlights', 100, (v) => `${v > 0 ? '+' : ''}${v}`], shadows: ['shadows', 100, (v) => `${v > 0 ? '+' : ''}${v}`],
  saturation: ['saturation', 100, (v) => `${v > 0 ? '+' : ''}${v}`], temperature: ['temperature', 100, (v) => `${v > 0 ? '+' : ''}${v}`],
  tint: ['tint', 100, (v) => `${v > 0 ? '+' : ''}${v}`],
  black: ['black', 100, (v) => `${v}`], white: ['white', 100, (v) => `${v}`],
  shadowHue: ['shadowHue', 1, null], midHue: ['midHue', 1, null], hiHue: ['hiHue', 1, null],
  shadowAmt: ['shadowAmt', 100, (v) => `${v}`], midAmt: ['midAmt', 100, (v) => `${v}`], hiAmt: ['hiAmt', 100, (v) => `${v}`],
};
// hue sliders show the colour they add (OKLab hue at a mid lightness, so the gradient matches the effect)
{
  const stops = [];
  for (let h = 0; h <= 360; h += 30) { const c = oklabToSrgb(0.72, 0.12 * Math.cos(h * Math.PI / 180), 0.12 * Math.sin(h * Math.PI / 180)); stops.push(`rgb(${c.map((v) => Math.round(v * 255)).join(',')}) ${(h / 360 * 100).toFixed(1)}%`); }
  for (const el of document.querySelectorAll('input.hue')) el.style.setProperty('--hue-gradient', `linear-gradient(90deg, ${stops.join(', ')})`);
}
function syncAdjustControls() {
  for (const [id, [key, scale, fmt]] of Object.entries(ADJ)) {
    const v = Math.round(S.look.draft[key] * scale);
    $(`adj-${id}`).value = v; if (fmt) $(`adj-${id}-v`).textContent = fmt(v);
  }
  $('adj-protect').checked = S.look.draft.protect;
  const ai = S.look.draft.source === 'ai';
  for (const el of document.querySelectorAll('input[name=source]')) el.checked = el.value === S.look.draft.source;
  $('adj-match').disabled = ai; $('adj-protect').disabled = ai;
  updateAdjState();
}
function updateAdjState() {
  const dirty = !S.look.applied || JSON.stringify(S.look.applied) !== JSON.stringify(S.look.draft);
  $('adjState').textContent = !S.intent.confirmed ? '確認光暗設定後即可處理影片。'
    : dirty ? '調整尚未套用到影片：按「套用並處理影片」產生新的結果。' : '影片已使用目前的調整。';
  $('adjApply').disabled = !S.intent.confirmed || S.phase === 'processing';
}
for (const [id, [key, scale, fmt]] of Object.entries(ADJ)) {
  $(`adj-${id}`).addEventListener('input', (e) => { S.look.auto = false; S.look.draft[key] = e.target.value / scale; if (fmt) $(`adj-${id}-v`).textContent = fmt(+e.target.value); updateAdjState(); scheduleAdjPreview(); });
  $(`adj-${id}`).addEventListener('dblclick', () => { S.look.auto = false; S.look.draft[key] = DEFAULT_LOOK[key]; syncAdjustControls(); scheduleAdjPreview(); });
}
$('adj-protect').addEventListener('change', (e) => { S.look.auto = false; S.look.draft.protect = e.target.checked; updateAdjState(); scheduleAdjPreview(); });
$('adjReset').onclick = () => { S.look.auto = false; S.look.draft = { ...DEFAULT_LOOK }; syncAdjustControls(); scheduleAdjPreview(); };
for (const el of document.querySelectorAll('input[name=source]')) {
  el.addEventListener('change', () => {
    if (el.value === 'ai' && !S.aiPair) return;
    S.look.auto = false;
    S.look.draft = el.value === 'ai' ? { ...S.look.draft, source: 'ai', pairId: S.aiPair.id, strength: 1 } : { ...S.look.draft, source: 'reference', pairId: undefined };
    log('source-changed', { source: el.value });
    syncAdjustControls(); scheduleAdjPreview();
  });
}
function applyAutoLook() {
  const st = S.grade.stats;
  if (!st) return;
  S.look.draft = { ...suggestLook(st.src, st.ref), source: 'reference', amount: S.look.draft.amount ?? 1 };
  S.look.auto = true;
  $('suggestInfo').textContent = `已依參考圖自動匹配：${describeLook(S.look.draft)}。可以再用滑桿微調。`;
  log('suggested', { look: { ...S.look.draft } });
  if (!$('adjustCard').hidden) { syncAdjustControls(); scheduleAdjPreview(); }
}
$('adjSuggest').onclick = async () => { await S.grade.ready; applyAutoLook(); syncAdjustControls(); scheduleAdjPreview(); };

// ---- Claude looks at reference + frame + result and sets the sliders (artifact `sample` capability)
let claudeCtl = null;
claudeAvailable().then((sample) => {
  S.claude = sample;
  $('claudeOpt').hidden = !sample; $('claudeGrade').hidden = !sample;
  log('claude-availability', { available: !!sample });
});
const CLAUDE_ERR = {
  not_granted: '你沒有允許這個網頁使用 Claude，已停用這個功能。', sampling_disabled: '這個帳號不能使用 Claude。',
  rate_limited: 'Claude 使用量已達上限，請稍後再試。', refused: 'Claude 拒絕了這次要求。', invalid_json: 'Claude 的回覆格式不正確，請再試一次。',
  images_unavailable: '這個檢視不能傳送圖片給 Claude。', session_expired: '請重新登入 claude.ai。',
};
async function runClaude() {
  if (!S.claude || !S.rep || S.claudeBusy) return false;
  await S.grade.ready;
  S.claudeBusy = true; claudeCtl = new AbortController();
  $('claudeStop').hidden = false; $('claudeGrade').disabled = true; $('adjApply').disabled = true;
  const info = $('claudeInfo');
  const intent = S.intent.confirmed ? serializeIntent(S.intent.confirmed) : null;
  const { width: w, height: h } = S.rep.img;
  try {
    const res = await claudeGrade({
      sample: S.claude, refImage: S.ref.bitmap, frameImage: S.rep.img, signal: claudeCtl.signal,
      metrics: { ref: S.grade.stats.ref, src: measure(S.rep.img.data, w, h) }, look0: { ...S.look.draft },
      render: async (look) => {
        const r = await engine.render(S.rep.img, { intent, seg: S.rep.seg, grade: true, look });
        return { image: r.imageData, stats: measure(r.imageData.data, w, h) };
      },
      onStatus: (t) => { info.textContent = t; },
    });
    S.look.draft = res.look; S.look.auto = false;
    info.textContent = `Claude 調整了 ${res.rounds} 輪：${res.notes.join(' ')}`;
    log('claude-graded', { rounds: res.rounds, look: { ...res.look } });
    syncAdjustControls(); scheduleAdjPreview();
    return true;
  } catch (e) {
    const code = e && e.code;
    if (['not_granted', 'sampling_disabled', 'not_declared', 'capability_disabled', 'capability_removed', 'images_unavailable'].includes(code)) {
      S.claude = null; $('claudeOpt').hidden = true; $('claudeGrade').hidden = true;
    }
    info.textContent = code === 'cancelled' ? '已停止，保留原本的設定。' : `Claude 自動調色未完成：${CLAUDE_ERR[code] || (e && e.message) || '連線問題，請稍後再試。'}`;
    log('claude-failed', { code });
    return false;
  } finally {
    S.claudeBusy = false; $('claudeStop').hidden = true; $('claudeGrade').disabled = false; updateAdjState();
  }
}
$('claudeGrade').onclick = () => runClaude();
$('claudeStop').onclick = () => claudeCtl && claudeCtl.abort();

// ---- AI-graded frame: offer one frame + prompt, take the AI's version back, distil a LUT in the worker
async function saveBlob(blob, filename) {
  const downloads = window.claude && window.claude.use ? await window.claude.use('downloads') : null;
  if (downloads) return downloads.save({ filename, data: blob });
  const url = URL.createObjectURL(blob), a = document.createElement('a');
  a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}
function clearAiPair() {
  S.aiPair = null; S.aiFrameT = null;
  engine.clearPair().catch(() => {});
  $('sourceAi').disabled = true; $('aiResult').textContent = ''; $('geminiCands').innerHTML = '';
  if (S.look.draft.source === 'ai') S.look.draft = { ...S.look.draft, source: 'reference', pairId: undefined };
}
$('aiDownload').onclick = async () => {
  if (!S.rep) { $('aiResult').textContent = '請先開啟光暗設定視窗，選好代表影格。'; return; }
  const m = S.video.meta, { w, h } = fitSize(m.width, m.height, 1280);
  const img = await grabFrame(m.video, S.rep.t, w, h);
  const c = document.createElement('canvas'); c.width = w; c.height = h; c.getContext('2d').putImageData(img, 0, 0);
  const blob = await new Promise((res) => c.toBlob(res, 'image/png'));
  S.aiFrameT = S.rep.t;
  try {
    await saveBlob(blob, `${S.video.file.name.replace(/\.[^.]+$/, '')}_frame_${S.rep.t.toFixed(2)}s.png`);
    log('ai-frame-downloaded', { t: S.rep.t });
  } catch (err) { if (!(err && err.code === 'declined')) $('aiResult').textContent = `下載失敗：${err.message || err.code}`; }
};
$('aiCopy').onclick = async () => {
  try { await navigator.clipboard.writeText($('aiPrompt').value); $('aiCopy').textContent = '已複製'; }
  catch { $('aiPrompt').select(); $('aiCopy').textContent = '請按 Ctrl+C 複製'; }
  setTimeout(() => { $('aiCopy').textContent = '複製提示詞'; }, 2000);
};
/**
 * Distil a LUT from an AI-graded version of the frame at time t. crop: centre-crop an image whose aspect
 * differs a little (generators may pick their own size) instead of refusing it.
 * Returns the worker's verdict plus the pair id; throws with a user-facing message.
 */
async function fitAiImage(bmp, t, { crop = false } = {}) {
  const m = S.video.meta, target = m.width / m.height, ar = bmp.width / bmp.height;
  let sx = 0, sy = 0, sw = bmp.width, sh = bmp.height;
  if (Math.abs(ar / target - 1) > 0.03) {
    if (!crop || Math.abs(ar / target - 1) > 0.2) throw new Error(`圖片比例（${bmp.width}×${bmp.height}）和影片不同，AI 可能裁切或改變了構圖。請使用與影格相同比例的圖片。`);
    if (ar > target) { sw = Math.round(bmp.height * target); sx = Math.round((bmp.width - sw) / 2); } else { sh = Math.round(bmp.width / target); sy = Math.round((bmp.height - sh) / 2); }
  }
  const { w, h } = fitSize(m.width, m.height, 320);
  const src = await grabFrame(m.video, t, w, h);
  const cv = canvas2d(w, h), cx = cv.getContext('2d', { willReadFrequently: true });
  cx.imageSmoothingQuality = 'high'; cx.drawImage(bmp, sx, sy, sw, sh, 0, 0, w, h);
  const id = `p${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
  const r = await engine.fitPair(src, cx.getImageData(0, 0, w, h), id);
  log('ai-pair-fitted', { id, t, score: r.shift.score, meanAbs: r.meanAbs, outlierShare: r.outlierShare, ms: Math.round(r.ms) });
  if (!r.accepted) throw new Error(`這張圖和代表影格的構圖對不上（相似度 ${r.shift.score.toFixed(2)}）`);
  return { id, t, score: r.shift.score, meanAbs: r.meanAbs, outlierShare: r.outlierShare };
}
function pairWarnings(p) {
  const warn = [];
  if (p.score < 0.8) warn.push('構圖有些差異');
  if (p.meanAbs > 12 || p.outlierShare > 0.15) warn.push('AI 可能改了光線或內容，只能部分重現');
  return warn;
}
function useAiPair(p) {
  S.aiPair = p;
  $('sourceAi').disabled = false;
  S.look.auto = false;
  S.look.draft = { ...DEFAULT_LOOK, source: 'ai', pairId: p.id, amount: S.look.draft.amount ?? 1 }; // the AI image is the target: neutral sliders, the user's overall strength kept
  syncAdjustControls(); scheduleAdjPreview();
}
$('aiInput').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file) return;
  const out = $('aiResult');
  try {
    if (!S.video.meta) throw new Error('請先上傳目標影片');
    const t = S.aiFrameT ?? (S.rep ? S.rep.t : null);
    if (t === null) throw new Error('請先開啟光暗設定視窗，再下載代表影格');
    out.textContent = '分析 AI 調色圖中…';
    const p = await fitAiImage(await loadImage(file), t);
    const warn = pairWarnings(p);
    useAiPair(p);
    out.textContent = `已建立 AI 調色設定：構圖相似度 ${p.score.toFixed(2)}，重現誤差約 ${p.meanAbs.toFixed(1)}/255，`
      + `${(p.outlierShare * 100).toFixed(1)}% 的像素無法以調色重現${warn.length ? `（注意：${warn.join('；')}）` : '（AI 調色可以準確重現）'}。`
      + '調色來源已切換為「AI 調色圖」，滑桿已歸零，可以再微調。';
  } catch (err) { out.textContent = `無法使用這張圖：${err.message.replace(/）$/, '）。請確認上傳的是步驟 1 下載的那一格調色後的版本')}`; }
});

// ---- Image AI with the user's own key (Gemini or Pollinations): generate candidates, check, distil, rank,
// pick the closest. Element ids keep the gemini* prefix; the provider decides key storage, models and calls.
const inClaudeViewer = !!(window.claude && window.claude.use); // claude.ai pages cannot reach other services
const PROVIDERS = {
  gemini: {
    name: 'Google Gemini', store: keyStore, placeholder: '貼上 Gemini API 金鑰',
    info: '金鑰只保存在這個瀏覽器，直接送到 Google。生成時會把代表影格和參考圖傳給 Google；圖像模型可能沒有免費額度，費用和資料使用方式依 Google 的條款。',
    list: async (key) => (await listImageModels(key)).map((id) => ({ id, refs: 2 })),
    run: (o) => geminiRegrade({ ...o, aspectRatio: nearestRatio(o.width, o.height) }),
  },
  pollinations: {
    name: 'Pollinations', store: pollinationsKeyStore, placeholder: '貼上 Pollinations 的 sk_ 金鑰（enter.pollinations.ai）',
    info: '金鑰只保存在這個瀏覽器，直接送到 Pollinations。每次生成會扣帳戶的 Pollen 點數（免費點數來自官方任務，不是無限免費），影格和參考圖會傳給 Pollinations 和它使用的模型供應商。',
    list: listEditModels,
    run: (o) => pollinationsRegrade(o),
  },
};
const provider = () => PROVIDERS[$('genProvider').value] || PROVIDERS.gemini;
let geminiCtl = null;
function syncProvider() {
  const p = provider();
  $('geminiKey').value = p.store.load(); $('geminiKey').placeholder = p.placeholder;
  $('geminiRemember').checked = !!$('geminiKey').value;
  $('genProviderInfo').textContent = p.info; $('genProviderName').textContent = p.name;
  $('geminiModel').innerHTML = '<option value="">（先連線）</option>';
  if (!inClaudeViewer) $('geminiInfo').textContent = '';
}
syncProvider();
$('genProvider').addEventListener('change', () => { syncProvider(); geminiState(); });
function geminiState() {
  const ok = $('geminiKey').value.trim() && $('geminiModel').value && $('geminiConsent').checked && !S.geminiBusy;
  $('geminiRun').disabled = !ok;
}
if (inClaudeViewer) {
  for (const id of ['genProvider', 'geminiKey', 'geminiConnect', 'geminiRemember', 'geminiModel', 'geminiCount', 'geminiConsent']) $(id).disabled = true;
  $('geminiInfo').textContent = '目前在 claude.ai 網頁裏開啟，不能連線外部的圖像 AI。請在本機（npm start）或自己的網址開啟網站。';
}
$('geminiRemember').addEventListener('change', (e) => { if (e.target.checked) provider().store.save($('geminiKey').value.trim()); else provider().store.clear(); });
$('geminiKey').addEventListener('input', () => { if ($('geminiRemember').checked) provider().store.save($('geminiKey').value.trim()); geminiState(); });
for (const id of ['geminiModel', 'geminiConsent']) $(id).addEventListener('change', geminiState);
$('geminiConnect').onclick = async () => {
  const key = $('geminiKey').value.trim();
  if (!key) { $('geminiInfo').textContent = '請先貼上 API 金鑰。'; return; }
  $('geminiInfo').textContent = '連線中…';
  try {
    const models = await provider().list(key);
    const sel = $('geminiModel');
    sel.innerHTML = '';
    for (const m of models) { const o = document.createElement('option'); o.value = o.textContent = m.id; o.dataset.refs = m.refs; sel.appendChild(o); }
    if (!models.length) { const o = document.createElement('option'); o.value = ''; o.textContent = '（這個金鑰沒有可用的圖像模型）'; sel.appendChild(o); }
    $('geminiInfo').textContent = models.length ? `已連線，可用的修圖模型 ${models.length} 個。` : '已連線，但這個金鑰沒有可以修圖的模型（可能要開通或付費）。';
    log('gemini-models', { provider: $('genProvider').value, models: models.map((m) => m.id) });
  } catch (err) { $('geminiInfo').textContent = `連線失敗：${err.message}`; }
  geminiState();
};
function lookDistance(st, ref) { // the same look descriptors the suggestion uses
  const q0 = (x) => x.Lq[0];
  return Math.abs(q0(st) - q0(ref)) + Math.abs(st.Lsd - ref.Lsd) + Math.abs(st.chroma - ref.chroma)
    + Math.hypot(st.hiA - ref.hiA, st.hiB - ref.hiB) + Math.hypot(st.loA - ref.loA, st.loB - ref.loB);
}
function renderCandidates(cands, chosenId) {
  const box = $('geminiCands');
  box.innerHTML = '';
  for (const c of cands) {
    const d = document.createElement('div');
    d.className = `cand${c.id && c.id === chosenId ? ' chosen' : ''}${c.ok ? '' : ' bad'}`;
    if (c.url) { const img = document.createElement('img'); img.src = c.url; img.alt = `AI 候選 ${c.n}`; d.appendChild(img); }
    const p = document.createElement('p');
    p.textContent = c.ok ? `候選 ${c.n}：構圖相似度 ${c.score.toFixed(2)}，重現誤差 ${c.meanAbs.toFixed(1)}/255，與參考圖差距 ${c.dist.toFixed(3)}${c.warn.length ? `（${c.warn.join('；')}）` : ''}`
      : `候選 ${c.n}：不採用，${c.reason}`;
    d.appendChild(p);
    if (c.ok) {
      const b = document.createElement('button'); b.type = 'button';
      b.textContent = c.id === chosenId ? '使用中' : '改用這個'; b.disabled = c.id === chosenId;
      b.onclick = () => { useAiPair(c.pair); renderCandidates(cands, c.id); log('gemini-chosen', { id: c.id, manual: true }); };
      d.appendChild(b);
    }
    box.appendChild(d);
  }
}
$('geminiStop').onclick = () => geminiCtl && geminiCtl.abort();
$('geminiRun').onclick = async () => {
  if (!S.rep || !S.video.meta) { $('geminiInfo').textContent = '請先開啟光暗設定視窗，選好代表影格。'; return; }
  await S.grade.ready;
  const key = $('geminiKey').value.trim(), model = $('geminiModel').value, count = +$('geminiCount').value;
  const info = $('geminiInfo'), m = S.video.meta, t = S.rep.t;
  S.geminiBusy = true; geminiState(); geminiCtl = new AbortController(); $('geminiStop').hidden = false;
  const cands = [];
  try {
    const { w, h } = fitSize(m.width, m.height, 1024);
    const frame = await toJpeg(await grabFrame(m.video, t, w, h), 1024), reference = await toJpeg(S.ref.bitmap, 1024);
    const refs = +($('geminiModel').selectedOptions[0] && $('geminiModel').selectedOptions[0].dataset.refs) || 1, p = provider();
    // a model that takes one image only gets the frame; the style then comes from the prompt's description alone
    const prompt = refs >= 2 ? `第一張圖是要調色的影片影格，第二張圖是參考圖。請輸出第一張圖調色後的版本。\n${$('aiPrompt').value}`
      : `請輸出這張影片影格調色後的版本。\n${$('aiPrompt').value.replace(/請參考我附上的參考圖的色調和氛圍。?\n?/, '')}`;
    const intent = S.intent.confirmed ? serializeIntent(S.intent.confirmed) : null;
    for (let n = 1; n <= count; n++) {
      info.textContent = `${p.name} 生成候選 ${n}/${count} 中…`;
      const c = { n, ok: false };
      cands.push(c);
      try {
        const blob = await p.run({ key, model, refs, prompt, frame, reference, width: w, height: h, signal: geminiCtl.signal });
        c.url = URL.createObjectURL(blob);
        info.textContent = `檢查候選 ${n}/${count}…`;
        c.pair = await fitAiImage(await loadImage(blob), t, { crop: true });
        const r = await engine.render(S.rep.img, { intent, seg: S.rep.seg, grade: true, look: { ...DEFAULT_LOOK, source: 'ai', pairId: c.pair.id } });
        const d = r.imageData.data;
        let blown = 0; for (let i = 0; i < d.length; i += 4) if (Math.max(d[i], d[i + 1], d[i + 2]) >= 250) blown++;
        c.blown = blown / (d.length / 4);
        Object.assign(c, { id: c.pair.id, score: c.pair.score, meanAbs: c.pair.meanAbs, warn: pairWarnings(c.pair), dist: lookDistance(measure(d, r.imageData.width, r.imageData.height), S.grade.stats.ref) });
        if (c.blown > 0.03) c.reason = `死白比例 ${(c.blown * 100).toFixed(1)}%，會過曝`; else c.ok = true;
      } catch (err) {
        c.reason = err.message;
        if (['bad_key', 'quota', 'network', 'cancelled', 'forbidden'].includes(err.code)) break; // no point trying again
      }
      renderCandidates(cands, null);
    }
    const good = cands.filter((c) => c.ok).sort((a, b) => a.dist - b.dist);
    log('gemini-candidates', { provider: $('genProvider').value, model, results: cands.map((c) => ({ n: c.n, ok: c.ok, reason: c.reason || null, dist: c.dist ?? null })) });
    if (good.length) {
      useAiPair(good[0].pair);
      renderCandidates(cands, good[0].id);
      info.textContent = `完成：${good.length}/${cands.length} 個候選可用，已自動選用最接近參考圖的候選 ${good[0].n}（調色來源已切換為「AI 調色圖」，滑桿已歸零）。按「套用並處理影片」產生影片。`;
    } else info.textContent = `沒有可用的候選：${cands.map((c) => `候選 ${c.n} ${c.reason}`).join('；')}`;
  } catch (err) { info.textContent = `${provider().name} 生成失敗：${err.message}`; }
  finally { S.geminiBusy = false; $('geminiStop').hidden = true; geminiState(); }
};
$('adjApply').onclick = () => startProcessing();
$('reprocess').onclick = () => startProcessing();
$('adjWipe').addEventListener('input', () => drawAdj());

let adjTimer = null, adjSeq = 0, adjRendered = null;
function scheduleAdjPreview() {
  clearTimeout(adjTimer);
  adjTimer = setTimeout(async () => {
    if (!S.rep || $('adjustCard').hidden) return;
    const seq = ++adjSeq;
    try {
      await S.grade.ready;
      const intent = S.intent.confirmed ? serializeIntent(S.intent.confirmed) : null;
      const r = await engine.render(S.rep.img, { intent, seg: S.rep.seg, grade: true, look: S.look.draft });
      if (seq !== adjSeq) return;
      adjRendered = r.imageData;
      drawAdj();
      log('adjust-preview', { look: { ...S.look.draft } });
    } catch (err) { $('adjState').textContent = `預覽失敗：${err.message}`; }
  }, 80);
}
function drawAdj() {
  if (!S.rep || !adjRendered) return;
  const c = $('adjCanvas'), w = S.rep.img.width, h = S.rep.img.height, x = Math.round($('adjWipe').value / 100 * w);
  c.width = w; c.height = h;
  const ctx = c.getContext('2d');
  ctx.putImageData(S.rep.img, 0, 0);
  ctx.putImageData(adjRendered, 0, 0, x, 0, w - x, h);
  ctx.fillStyle = '#fff'; ctx.fillRect(x - 1, 0, 2, h);
}
function showAdjustCard() { $('adjustCard').hidden = false; syncAdjustControls(); scheduleAdjPreview(); }

async function startProcessing() {
  const intent = S.intent.confirmed;
  if (!intent) return;
  const token = ++S.job.token;
  await S.grade.ready;
  S.look.applied = { ...S.look.draft };
  setPhase('processing');
  updateAdjState();
  $('resultCard').hidden = false; $('progressWrap').hidden = false;
  for (const id of ['exportBtn', 'reprocess', 'editIntent', 'syncPlay', 'cmpTime']) $(id).disabled = true;
  const regs = effectiveRegions(intent);
  const needsSeg = regs.some((r) => KINDS[r.kind].source === 'ai');
  if (needsSeg) await initSegmenter();
  try {
    const res = await processVideo({
      engine, meta: S.video.meta, intentJson: serializeIntent(intent), needsSeg, segmentFn: segment, maskH: intent.maskH,
      look: S.look.applied, isCancelled: () => token !== S.job.token, keepFrames: window.__keepFrames || [],
      onProgress: ({ done, total, elapsed }) => {
        $('progressBar').style.width = `${(done / total) * 100}%`;
        $('progressText').textContent = `${done} / ${total} 格　${elapsed.toFixed(1)} 秒`;
      },
    });
    if (token !== S.job.token) return; // a newer job started; drop this stale result
    if (S.output) URL.revokeObjectURL(S.output.url);
    S.output = { ...res, url: URL.createObjectURL(res.blob), confirmedHash: hashIntent(intent) };
    $('origVideo').src = S.video.meta.url;
    $('outVideo').src = S.output.url;
    $('resultInfo').textContent = `${res.frames} 格，${res.fps} fps，${res.width}×${res.height}，${res.codec}，${(res.blob.size / 1e6).toFixed(1)} MB。`
      + `處理引擎回報的光暗設定雜湊 ${res.intentHash}${res.intentHash === S.output.confirmedHash ? '（與確認的設定一致）' : '（⚠ 不一致）'}，色彩設定雜湊 ${res.lookHash}`
    + `（${describeLook(S.look.applied)}）`
    + `；局部調整區域：${res.jobRegions.length ? res.jobRegions.map((r) => `${KINDS[r.kind].label} ${r.ev > 0 ? '+' : ''}${r.ev.toFixed(2)} 檔`).join('、') : '無'}。`;
    log('processing-done', { frames: res.frames, hash: res.intentHash, size: res.blob.size });
    setPhase('done');
    updateAdjState();
    for (const id of ['exportBtn', 'reprocess', 'editIntent', 'syncPlay', 'cmpTime']) $(id).disabled = false;
    $('wipe').hidden = false;
    drawWipe();
  } catch (err) {
    if (token !== S.job.token) return;
    log('processing-failed', { message: err.message });
    setPhase('error', err.message);
    for (const id of ['reprocess', 'editIntent']) $(id).disabled = false;
  }
}

$('syncPlay').onclick = () => {
  const a = $('origVideo'), b = $('outVideo');
  b.currentTime = a.currentTime;
  Promise.all([a.play(), b.play()]).catch(() => {});
};
$('origVideo').addEventListener('timeupdate', () => {
  const a = $('origVideo'), b = $('outVideo');
  if (!a.paused && !b.paused && Math.abs(a.currentTime - b.currentTime) > 0.15) b.currentTime = a.currentTime;
});

async function drawWipe() {
  if (!S.output) return;
  const a = $('origVideo'), b = $('outVideo');
  const t = $('cmpTime').value / 1000 * Math.min(a.duration || S.video.meta.duration, b.duration || S.video.meta.duration);
  await Promise.all([seek(a, Math.max(0, t - 1e-3)), seek(b, Math.max(0, t - 1e-3))]);
  const c = $('wipeCanvas'), w = S.output.width, h = S.output.height, x = Math.round($('wipePos').value / 100 * w);
  c.width = w; c.height = h;
  const ctx = c.getContext('2d');
  ctx.drawImage(a, 0, 0, w, h);
  ctx.save(); ctx.beginPath(); ctx.rect(x, 0, w - x, h); ctx.clip(); ctx.drawImage(b, 0, 0, w, h); ctx.restore();
  ctx.fillStyle = '#fff'; ctx.fillRect(x - 1, 0, 2, h);
}
$('cmpTime').addEventListener('input', drawWipe);
$('wipePos').addEventListener('input', drawWipe);

$('exportBtn').onclick = async () => {
  if (!S.output) return;
  setPhase('exporting');
  const name = `${S.video.file.name.replace(/\.[^.]+$/, '')}_graded_${S.output.intentHash}.mp4`;
  try {
    // inside a claude.ai artifact plain download links are blocked; the platform's save dialog is used instead
    const downloads = window.claude && window.claude.use ? await window.claude.use('downloads') : null;
    if (downloads) {
      await downloads.save({ filename: name, data: S.output.blob });
    } else {
      const a = document.createElement('a');
      a.href = S.output.url; a.download = name;
      document.body.appendChild(a); a.click(); a.remove();
    }
    log('exported', { name, size: S.output.blob.size });
    setPhase('exported', name);
  } catch (err) {
    setPhase(err && err.code === 'declined' ? 'done' : 'error', err && err.code === 'declined' ? '已取消下載' : `匯出失敗：${err.message || err.code}`);
  }
};

setPhase('idle');
