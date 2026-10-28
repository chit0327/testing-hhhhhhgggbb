// Page controller: asset loading, the Lighting Intent dialog (draft -> confirmed), processing and export.
// Pixels are only ever produced by the worker (src/worker.js) from the serialised CONFIRMED intent.
import {
  createIntent, newRegion, cloneIntent, confirmIntent, canConfirm, effectiveRegions, markEdited,
  serializeIntent, hashIntent, MODES, KINDS, MASK_W,
} from './engine/intent.js';
import { rawMask } from './engine/regionmasks.js';
import { applyEdits, feather, lightnessGrid } from './engine/masks.js';
import { EngineClient, probeVideo, probeImages, PHOTO_MAX, grabFrame, fitSize, loadImage, imageDataOf, processVideo, processPhoto, seek, frameTime, canvas2d } from './pipeline.js';
import { initSegmenter, segment } from './segment.js';
import { DEFAULT_LOOK, measure, toCube } from './engine/grade.js';
import { waveformData, vectorscopeData, drawWaveform, drawVectorscope, clipShares } from './engine/scopes.js';
import { neutralPrim, isNeutralPrim, sanitizePrim, compilePrim, curveFn, HUE_ANCHORS, HUE_NAMES, CURVE_X, LOCAL_NAMES } from './engine/primary.js';
import { bandColours, bandDeltaE, structureScore, clipping, maskedDeltaE } from './engine/metrics.js';
import { claudeAvailable, claudeGrade, toJpeg } from './claudegrade.js';
import { keyStore, listImageModels, geminiRegrade, nearestRatio } from './gemini.js';
import { pollinationsKeyStore, listEditModels, pollinationsRegrade } from './pollinations.js';
import { suggestLook, promptFor } from './engine/suggest.js';
import { zipStore } from './zip.js';
import { oklabToSrgb, srgbToOklab } from './engine/color.js';

const $ = (id) => document.getElementById(id);
const engine = new EngineClient();

const PHASE_TEXT = {
  idle: '請先上傳參考圖片和要調色的影片或照片', loading: '素材載入中…', analysing: '正在分析參考圖片與影片…',
  draft: '草稿設定：請在光暗設定視窗中確認', unset: '尚未確認光暗設定（按「重新設定光暗」開始）',
  confirmed: '已確認設定，準備處理…', processing: '正在處理…', done: '已完成處理',
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
const AUTOFIT_V2 = new URLSearchParams(location.search).has('v2'); // grey balance + absent colours kept (under evaluation)
const AUTOFIT_V3 = new URLSearchParams(location.search).has('v3'); // v2 + tone from the look prior, not the reference content (under evaluation)
const AUTOFIT_WHITE = !new URLSearchParams(location.search).has('nowhite'); // white point matched to the reference's white (default since round-3 blind rating; ?nowhite = before)
const SKIN_KEEP = new URLSearchParams(location.search).has('skinkeep'); // skin protection keeps the original colour (before the skin-line change)
const AUTOFIT_BALANCE = new URLSearchParams(location.search).has('balance'); // correction (white balance of the shot) before the grade (under evaluation)
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

const isImageFile = (f) => f.type.startsWith('image/') || /\.(jpe?g|png|webp|heic|avif)$/i.test(f.name);
$('videoInput').addEventListener('change', async (e) => {
  const files = [...e.target.files], file = files[0];
  if (!file) return;
  const id = ++S.video.id;
  S.video = { status: 'loading', id };
  $('videoInfo').textContent = '載入中…';
  const photo = files.every(isImageFile);
  try {
    if (files.length > 1 && !photo) throw Object.assign(new Error('一次只能處理一條影片；選多個檔案時只支援照片'), { plain: true });
    if (files.length > PHOTO_MAX) throw Object.assign(new Error(`一次最多 ${PHOTO_MAX} 張照片（選了 ${files.length} 張），請分批處理`), { plain: true });
    if (!photo && !file.type.startsWith('video/') && !/\.(mp4|webm|mov|m4v)$/i.test(file.name)) throw new Error(`不支援的檔案類型（${file.type || '未知'}），請選擇影片或照片`);
    const meta = photo ? await probeImages(files) : await probeVideo(file);
    if (id !== S.video.id) return;
    S.video = { status: 'ready', id, file, files, meta };
    showTargetPreview(meta);
    if (photo) {
      const big = meta.video.sizes.some(([w, h]) => Math.max(w, h) > 4096) ? '　（匯出時長邊縮為 4096）' : '';
      $('videoInfo').textContent = files.length > 1 ? `${files.length} 張照片（第一張 ${file.name}　${meta.width}×${meta.height}）${big}` : `${file.name}　照片　${meta.width}×${meta.height}${big}`;
      log('photo-loaded', { name: file.name, count: files.length, w: meta.width, h: meta.height });
    } else {
      $('videoInfo').textContent = `${file.name}　${meta.width}×${meta.height}　${meta.duration.toFixed(2)} 秒　${meta.fps} fps${meta.fpsMeasured ? '' : '（無法測得，假設值）'}　約 ${meta.frameCount} 格`
        + (meta.duration > 60 ? '　⚠ 第一版建議 60 秒以內' : '');
      log('video-loaded', { name: file.name, fps: meta.fps, frames: meta.frameCount });
    }
  } catch (err) {
    S.video = { status: 'error', id };
    showTargetPreview(null);
    $('videoInfo').textContent = err.plain ? err.message : photo ? `無法載入照片：${err.message}（瀏覽器可能不支援此格式）` : `無法載入影片：${err.message}（瀏覽器可能不支援此編碼）`;
  }
  assetsChanged('video');
});

async function assetsChanged(which) {
  const key = pairKey();
  updateStart();
  if (!key) { setPhase(S.ref.status === 'loading' || S.video.status === 'loading' ? 'loading' : 'idle'); return; }
  // new pair: old results no longer valid; a new video also invalidates masks (geometry/content changed)
  S.output = null; setAdjustVisible(false); resetResultViews();
  if (which === 'video') { S.intent.confirmed = null; S.rep = null; clearAiPair(); }
  updateIntentCard();
  $('openIntent').disabled = false;
  if (S.grade.pair !== key) fitGrade(key); // analysis starts while the user is still on the upload page
  if (S.page === 'work') promptOnce();
}
function promptOnce() { // auto-open the Lighting Intent dialog exactly once per newly loaded pair
  const key = pairKey();
  if (!key || S.promptedPair === key) return;
  S.promptedPair = key;
  log('dialog-auto-open', { pair: key });
  openDialog();
}

function fitGrade(key) {
  S.grade.pair = key;
  setPhase('analysing');
  S.grade.ready = (async () => {
    const m = S.video.meta, { w, h } = fitSize(m.width, m.height, 320);
    const frames = [];
    if (m.kind === 'image') { // photos: up to six spread over the set (one look for all), each at its own aspect
      for (const i of [...new Set([0.1, 0.25, 0.4, 0.55, 0.7, 0.85].map((f) => Math.round(f * (m.count - 1))))]) {
        const img = await m.video.frameAt(i, 320);
        frames.push({ data: img.data, width: img.width, height: img.height });
      }
    } else for (const f of [0.1, 0.25, 0.4, 0.55, 0.7, 0.85]) { // frame-centred times -> deterministic frames
      const img = await grabFrame(m.video, frameTime(m, Math.round(f * (m.frameCount - 1))), w, h);
      frames.push({ data: img.data, width: w, height: h });
    }
    const ref = S.ref.imageData;
    const r = await engine.fit(frames, { data: ref.data, width: ref.width, height: ref.height }, { v2: AUTOFIT_V2, v3: AUTOFIT_V3, white: AUTOFIT_WHITE, balance: AUTOFIT_BALANCE });
    S.grade.stats = { src: r.src, ref: r.ref, palette: r.palette };
    S.grade.cands = (r.candidates || []).map((c) => (SKIN_KEEP ? { ...c, prim: { ...c.prim, skinLine: 0 } } : c));
    S.grade.refBands = bandColours(S.ref.imageData.data, S.ref.imageData.width, S.ref.imageData.height, { centre: 0.7, skipWhite: true });
    log('autofit', { ms: Math.round(r.autofitMs || 0), modes: S.grade.cands.map((c) => c.mode) });
    $('aiPrompt').value = promptFor(r.ref, r.palette);
    if (S.look.auto) applyAutoLook();
    renderCands();
    log('grade-fitted', { ms: Math.round(r.ms) });
    setTimeout(fitLocalGrade, 0);
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

// LOG / flat footage is restored before segmentation (the model misses people in flat, grey frames).
function currentNorm() {
  const p = S.look.draft && S.look.draft.prim, c = S.grade.cands && S.grade.cands[0];
  const n = (p && p.norm) || (c && c.prim && c.prim.norm);
  return n && n.amt > 0.02 ? n : null;
}
function restoreCanvas(c, norm = currentNorm()) {
  if (!norm) return c;
  const f = compilePrim({ ...neutralPrim(), norm }), ctx = c.getContext('2d'), id = ctx.getImageData(0, 0, c.width, c.height), d = id.data, o = [0, 0, 0];
  for (let i = 0; i < d.length; i += 4) { f(d[i] / 255, d[i + 1] / 255, d[i + 2] / 255, o); d[i] = o[0] * 255 + 0.5; d[i + 1] = o[1] * 255 + 0.5; d[i + 2] = o[2] * 255 + 0.5; }
  ctx.putImageData(id, 0, 0);
  return c;
}
const segmentRestored = (c, mw, mh) => segment(restoreCanvas(c), mw, mh);

async function runSegmentation() {
  await initSegmenter();
  const c = canvas2d(256, maskH());
  c.getContext('2d').drawImage(await createImageBitmap(S.rep.img), 0, 0, c.width, c.height);
  const norm = currentNorm();
  S.rep.seg = segment(restoreCanvas(c, norm), MASK_W, maskH());
  S.rep.segNorm = norm ? JSON.stringify(norm) : null;
}

async function runAI() {
  if (!S.rep) return;
  $('aiStatus').textContent = 'AI 辨識中…';
  S.ai.status = 'loading';
  try {
    await runSegmentation();
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
  fitLocalGrade();
}

// ---- automatic local grading: per-region adjustments (skin, background) fitted on the representative
// frame against the reference; needs both segmentations, so it runs when the later of the two is ready
async function referenceSeg() {
  const ref = S.ref.imageData;
  if (S.ref.seg && S.ref.segFor === ref) return S.ref.seg;
  await initSegmenter();
  const mh = Math.round(MASK_W * ref.height / ref.width), c = canvas2d(256, Math.round(256 * ref.height / ref.width));
  c.getContext('2d').drawImage(await createImageBitmap(ref), 0, 0, c.width, c.height);
  S.ref.seg = { w: MASK_W, h: mh, ...segment(c, MASK_W, mh) }; S.ref.segFor = ref;
  return S.ref.seg;
}
// one fit per (frame, restoration, candidates); callers share the running promise
async function fitLocalGrade() {
  try {
    if (!S.rep || S.ai.status !== 'ready' || !S.ref || !S.ref.imageData) return;
    await S.grade.ready;
    const cands = S.grade.cands || [];
    if (!cands.length) return;
    const norm = currentNorm(), nk = norm ? JSON.stringify(norm) : null;
    if (S.rep.segNorm !== nk) { await runSegmentation(); drawOverlay(); schedulePreview(); } // re-segment the restored frame first
    const key = `${S.rep.t}|${nk}|${cands.map((c) => JSON.stringify(c.prim.luma)).join()}`;
    if (S.grade.localJob && S.grade.localJob.key === key) return S.grade.localJob.promise;
    const job = { key, promise: null };
    S.grade.localJob = job;
    job.promise = (async () => {
    const refSeg = await referenceSeg(), seg = { w: MASK_W, h: maskH(), ...S.rep.seg };
    const r = await engine.fitLocal(S.rep.img, seg, S.ref.imageData, refSeg, cands.map((c) => ({ mode: c.mode, prim: { ...c.prim, local: null } })));
    for (const res of r.results) { const c = cands.find((x) => x.mode === res.mode); if (c) { c.prim = { ...c.prim, local: res.local }; c.localReport = res.report; } }
    log('local-fitted', { ms: Math.round(r.ms), results: r.results.map((x) => ({ mode: x.mode, local: x.local })) });
    if (S.look.auto && S.candMode) applyAutoLook(S.candMode);
    renderCands();
    })();
    return await job.promise;
  } catch (err) { log('local-failed', { message: err.message }); }
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
  $('editIntent').disabled = false;
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
  if (k.prim && !isNeutralPrim(k.prim)) parts.push(`專業調色${S.candMode && S.look.auto ? `（${{ infer: '推斷風格', natural: '自然', faithful: '忠實', bold: '濃烈' }[S.candMode]}）` : ''}${k.prim.skin ? `、膚色保護 ${Math.round(k.prim.skin * 100)}` : ''}${k.prim.local && (k.prim.localAmt ?? 1) > 0 ? `、局部調色 ${Math.round((k.prim.localAmt ?? 1) * 100)}` : ''}${k.prim.norm && k.prim.norm.amt > 0.02 ? `、LOG 還原 ${Math.round(k.prim.norm.amt * 100)}` : ''}`);
  if (k.glow) parts.push(`柔光 ${Math.round(k.glow * 100)}`);
  if (k.vignette) parts.push(`暗角 ${k.vignette > 0 ? '+' : ''}${Math.round(k.vignette * 100)}`);
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
  glow: ['glow', 100, (v) => `${v}`], glowRadius: ['glowRadius', 100, (v) => `${v}`],
  vignette: ['vignette', 100, (v) => `${v > 0 ? '+' : ''}${v}`], vignetteSize: ['vignetteSize', 100, (v) => `${v}`],
};
// settings that belong to the user's finishing, not to a colour source: kept when the source replaces the look
const KEEP = ['amount', 'glow', 'glowRadius', 'vignette', 'vignetteSize'];
const kept = () => Object.fromEntries(KEEP.map((k) => [k, S.look.draft[k] ?? DEFAULT_LOOK[k]]));
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
  syncPro();
  updateAdjState();
}
function updateAdjState() {
  const dirty = !S.look.applied || JSON.stringify(S.look.applied) !== JSON.stringify(S.look.draft);
  $('adjState').textContent = !S.intent.confirmed ? '確認光暗設定後即可處理影片。'
    : dirty ? `調整尚未套用：按「${isPhoto() ? '套用並處理照片' : '套用並處理影片'}」產生新的結果。` : `${isPhoto() ? '照片' : '影片'}已使用目前的調整。`;
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
const candLook = (c) => ({ ...DEFAULT_LOOK, strength: 0, match: 0, source: 'reference', prim: sanitizePrim(c.prim), ...kept() });
function applyAutoLook(mode = 'natural') {
  const st = S.grade.stats;
  if (!st) return;
  const c = (S.grade.cands || []).find((x) => x.mode === mode);
  S.look.draft = c ? candLook(c) : { ...suggestLook(st.src, st.ref), source: 'reference', ...kept() };
  S.candMode = c ? mode : null;
  S.look.auto = true;
  $('suggestInfo').textContent = `已依參考圖自動匹配：${describeLook(S.look.draft)}。可以再用滑桿微調。`;
  log('suggested', { look: { ...S.look.draft } });
  if (!$('adjustCard').hidden) { syncAdjustControls(); scheduleAdjPreview(); }
}
$('adjSuggest').onclick = async () => { await S.grade.ready; applyAutoLook(S.candMode || 'natural'); syncAdjustControls(); scheduleAdjPreview(); renderCands(); };

// ---- automatic candidates: render each on the representative frame and measure it
function skinMaskFor(img) { // AI skin grid resampled to the frame (for the skin ΔE report), or null
  const seg = S.rep && S.rep.seg;
  if (!seg || !seg.skin) return null;
  const mw = MASK_W, mh = Math.round(seg.skin.length / mw), w = img.width, h = img.height, m = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) m[y * w + x] = seg.skin[Math.min(mh - 1, Math.floor(y * mh / h)) * mw + Math.min(mw - 1, Math.floor(x * mw / w))];
  return m;
}
async function renderCands() {
  const box = $('candBox');
  if (!S.rep || !(S.grade.cands || []).length || $('adjustCard').hidden) return;
  await S.grade.ready;
  const run = S.candRun = (S.candRun || 0) + 1; // a newer run (e.g. after the local fit) replaces this one
  const intent = S.intent.confirmed ? serializeIntent(S.intent.confirmed) : null;
  const img = S.rep.img, skinM = skinMaskFor(img);
  const results = [], cards = [];
  for (const c of S.grade.cands) {
    const look = candLook(c);
    const r = await engine.render(img, { intent, seg: S.rep.seg, grade: true, look });
    if (run !== S.candRun) return;
    const d = r.imageData.data, cl = clipping(d);
    const m = {
      mode: c.mode, de: bandDeltaE(bandColours(d, img.width, img.height), S.grade.refBands),
      structure: structureScore(img.data, d, img.width, img.height), clip: cl.highlights + cl.shadows,
      skin: skinM ? maskedDeltaE(img.data, d, skinM, img.width, img.height) : null,
    };
    results.push(m);
    const card = document.createElement('div');
    card.className = `acand${S.candMode === c.mode && S.look.draft.prim ? ' chosen' : ''}`;
    card.dataset.mode = c.mode;
    const cv = document.createElement('canvas'); cv.width = img.width; cv.height = img.height; cv.getContext('2d').putImageData(r.imageData, 0, 0);
    const p = document.createElement('p');
    p.textContent = `${c.label}${c.mode === 'infer' ? '（推斷參考圖的調色方式，不照搬它的顏色；內容和影片差很遠時較可靠）' : ''}：與參考圖色差 ΔE ${m.de.toFixed(1)}｜結構保留 ${m.structure.toFixed(3)}${m.skin !== null ? `｜膚色變化 ΔE ${m.skin.toFixed(1)}` : ''}${m.clip > 0.01 ? `｜裁切 ${(m.clip * 100).toFixed(1)}%` : ''}`;
    card.append(cv, p);
    card.onclick = () => { applyAutoLook(c.mode); log('candidate-chosen', { mode: c.mode }); syncAdjustControls(); scheduleAdjPreview(); for (const el of box.children) el.classList.toggle('chosen', el.dataset.mode === c.mode); };
    cards.push(card);
  }
  box.replaceChildren(...cards);
  const ref0 = bandDeltaE(bandColours(img.data, img.width, img.height), S.grade.refBands);
  const nm = S.grade.cands && S.grade.cands[0] && S.grade.cands[0].norm;
  const restore = nm && nm.amt > 0.02 ? `偵測到 LOG／平面素材（亮度範圍 ${nm.range.toFixed(2)}，正常約 0.6 以上），已先自動還原 ${Math.round(nm.amt * 100)}%，再匹配參考圖。` : '';
  const lr = S.grade.cands[0] && S.grade.cands[0].localReport;
  const localTxt = lr ? `局部調色：${['skin', 'background'].filter((k) => lr[k] && lr[k].adj && (lr[k].adj.a || lr[k].adj.b)).map((k) => LOCAL_NAMES[k]).join('、') || '參考圖或影片缺少可對應的區域，未使用'}（頭髮、衣服保持整體調色）。` : '';
  $('candInfo').textContent = `${restore}${localTxt}原片與參考圖色差 ΔE ${ref0.toFixed(1)}。ΔE 越小越接近參考圖的明暗色調（「推斷風格」不追求接近參考圖的顏色分佈，ΔE 通常較大）；結構保留 1.000 代表構圖與紋理完全未變；膚色變化是和原片相比（LOG 還原和局部調色會刻意把膚色調向參考圖的自然膚色，所以不是越小越好，請以畫面判斷）。`;
  S.candResults = results;
  log('candidates-rendered', { results, original: ref0 });
}

// ---- professional controls (wheels, curves, hue curves), bound to look.prim
const PRO = []; // {el, get(prim), set(prim, v)}
function proPrim() { if (!S.look.draft.prim) S.look.draft.prim = neutralPrim(); return S.look.draft.prim; }
function proSlider(parent, { min, max, value, label, get, set }) {
  const wrap = document.createElement('label'); wrap.className = 'pslider';
  if (label) { const sp = document.createElement('span'); sp.className = 'lbl'; sp.textContent = label; wrap.appendChild(sp); }
  const el = document.createElement('input'); el.type = 'range'; el.min = min; el.max = max; el.value = value; wrap.appendChild(el);
  el.addEventListener('input', () => { S.look.auto = false; set(proPrim(), +el.value); updateAdjState(); scheduleAdjPreview(); });
  parent.appendChild(wrap);
  PRO.push({ el, get });
  return el;
}
function buildPro() {
  let root = $('proLocal');
  const h4 = (t) => { const e = document.createElement('h4'); e.textContent = t; root.appendChild(e); };
  h4('自動步驟：LOG／平面素材還原（自動偵測，0 = 不還原）與局部調色');
  const nrow = document.createElement('div'); nrow.className = 'hue-row';
  nrow.appendChild(document.createElement('span'));
  proSlider(nrow, { min: 0, max: 100, value: 0, label: '還原強度', get: (p) => (p.norm ? p.norm.amt : 0) * 100, set: (p, v) => { p.norm = { ...(p.norm || { m: 0.5, Tm: 0.5, s: 1, sat: 1 }), amt: v / 100 }; } });
  proSlider(nrow, { min: 0, max: 100, value: 100, label: '自動局部調色（皮膚、背景）', get: (p) => (p.localAmt ?? 1) * 100, set: (p, v) => { p.localAmt = v / 100; } });
  root.appendChild(nrow);
  root = $('proWheels');
  h4('色輪（Lift 暗部／Gamma 中間調／Gain 亮部／Offset 整體）');
  for (const [k, name] of [['lift', 'Lift'], ['gamma', 'Gamma'], ['gain', 'Gain'], ['offset', 'Offset']]) {
    const row = document.createElement('div'); row.className = 'wheel-row';
    const t = document.createElement('span'); t.textContent = name; row.appendChild(t);
    proSlider(row, { min: -100, max: 100, value: 0, label: '亮度', get: (p) => p[k].m * 100, set: (p, v) => { p[k].m = v / 100; } });
    const hue = proSlider(row, { min: 0, max: 359, value: 0, label: '顏色', get: (p) => p[k].h, set: (p, v) => { p[k].h = v; } });
    hue.classList.add('hue');
    proSlider(row, { min: 0, max: 100, value: 0, label: '份量', get: (p) => p[k].a * 100, set: (p, v) => { p[k].a = v / 100; } });
    root.appendChild(row);
  }
  root = $('proCurves');
  h4('曲線（0%／25%／50%／75%／100% 的輸出）');
  for (const [k, name] of [['luma', '亮度'], ['red', '紅'], ['green', '綠'], ['blue', '藍']]) {
    const row = document.createElement('div'); row.className = 'curve-row';
    const t = document.createElement('span'); t.textContent = name; row.appendChild(t);
    CURVE_X.forEach((x, i) => proSlider(row, { min: -5, max: 105, value: x * 100, get: (p) => p[k][i] * 100, set: (p, v) => { p[k][i] = v / 100; } }));
    root.appendChild(row);
  }
  h4('色相曲線（各顏色的色相偏移／飽和度／亮度）');
  HUE_ANCHORS.forEach((h, i) => {
    const row = document.createElement('div'); row.className = 'hue-row';
    const t = document.createElement('span'); t.textContent = HUE_NAMES[i]; row.appendChild(t);
    proSlider(row, { min: -60, max: 60, value: 0, label: '色相', get: (p) => p.hueHue[i], set: (p, v) => { p.hueHue[i] = v; } });
    proSlider(row, { min: -100, max: 100, value: 0, label: '飽和', get: (p) => p.hueSat[i] * 100, set: (p, v) => { p.hueSat[i] = v / 100; } });
    proSlider(row, { min: -15, max: 15, value: 0, label: '亮度', get: (p) => p.hueLum[i] * 100, set: (p, v) => { p.hueLum[i] = v / 100; } });
    root.appendChild(row);
  });
  root = $('proSat');
  h4('專業調色');
  const row = document.createElement('div'); row.className = 'hue-row';
  row.appendChild(document.createElement('span'));
  proSlider(row, { min: -100, max: 100, value: 0, label: '整體飽和度', get: (p) => p.sat * 100, set: (p, v) => { p.sat = v / 100; } });
  root.appendChild(row);
  root = $('proLocal');
  h4('膚色保護');
  const srow = document.createElement('div'); srow.className = 'hue-row';
  srow.appendChild(document.createElement('span'));
  proSlider(srow, { min: 0, max: 100, value: 0, label: '膚色保護（AI 皮膚遮罩）', get: (p) => p.skin * 100, set: (p, v) => { p.skin = v / 100; } });
  root.appendChild(srow);
  const stops = [];
  for (let hh = 0; hh <= 360; hh += 30) { const c = oklabToSrgb(0.72, 0.12 * Math.cos(hh * Math.PI / 180), 0.12 * Math.sin(hh * Math.PI / 180)); stops.push(`rgb(${c.map((v) => Math.round(v * 255)).join(',')}) ${(hh / 360 * 100).toFixed(1)}%`); }
  for (const el of document.querySelectorAll('.pro input.hue')) el.style.setProperty('--hue-gradient', `linear-gradient(90deg, ${stops.join(', ')})`);
}
buildPro();
function syncPro() {
  const p = S.look.draft.prim || neutralPrim();
  for (const c of PRO) c.el.value = Math.round(c.get(p));
  drawCurve();
}
$('exportCube').onclick = async () => {
  try {
    await S.grade.ready;
    const lut = await engine.bake(S.look.draft, S.intent.confirmed ? serializeIntent(S.intent.confirmed) : null), N = lut.size;
    const name = `${(S.video.file ? S.video.file.name : 'look').replace(/\.[^.]+$/, '')}_look.cube`;
    await saveBlob(new Blob([toCube(lut)], { type: 'text/plain' }), name);
    log('cube-exported', { size: N, name });
    $('cubeInfo').textContent = `已匯出 ${name}（${N}³）。自動局部調色、局部曝光、膚色保護、柔光與暗角取決於畫面位置，不在 LUT 內。`;
  } catch (err) { if (!(err && err.code === 'declined')) $('cubeInfo').textContent = `匯出失敗：${err.message || err.code}`; }
};

// ---- Claude looks at reference + frame + result and sets the sliders (artifact `sample` capability)
let claudeCtl = null;
claudeAvailable().then((sample) => {
  S.claude = sample;
  $('claudeOpt').hidden = !sample; $('claudeGrade').hidden = !sample; $('tweakRow').hidden = !sample;
  log('claude-availability', { available: !!sample });
});
const CLAUDE_ERR = {
  not_granted: '你沒有允許這個網頁使用 Claude，已停用這個功能。', sampling_disabled: '這個帳號不能使用 Claude。',
  rate_limited: 'Claude 使用量已達上限，請稍後再試。', refused: 'Claude 拒絕了這次要求。', invalid_json: 'Claude 的回覆格式不正確，請再試一次。',
  images_unavailable: '這個檢視不能傳送圖片給 Claude。', session_expired: '請重新登入 claude.ai。',
};
async function runClaude(instruction = '') {
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
      onStatus: (t) => { info.textContent = t; }, instruction, maxRounds: instruction ? 1 : 3,
    });
    S.look.draft = res.look; S.look.auto = false;
    info.textContent = `Claude 調整了 ${res.rounds} 輪：${res.notes.join(' ')}`;
    log('claude-graded', { rounds: res.rounds, look: { ...res.look } });
    syncAdjustControls(); scheduleAdjPreview();
    return true;
  } catch (e) {
    const code = e && e.code;
    if (['not_granted', 'sampling_disabled', 'not_declared', 'capability_disabled', 'capability_removed', 'images_unavailable'].includes(code)) {
      S.claude = null; $('claudeOpt').hidden = true; $('claudeGrade').hidden = true; $('tweakRow').hidden = true;
    }
    info.textContent = code === 'cancelled' ? '已停止，保留原本的設定。' : `Claude 自動調色未完成：${CLAUDE_ERR[code] || (e && e.message) || '連線問題，請稍後再試。'}`;
    log('claude-failed', { code });
    return false;
  } finally {
    S.claudeBusy = false; $('claudeStop').hidden = true; $('claudeGrade').disabled = false; updateAdjState();
  }
}
$('claudeGrade').onclick = () => runClaude();
$('tweakBtn').onclick = () => { const t = $('tweakText').value.trim(); if (t) runClaude(t); };
$('tweakText').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('tweakBtn').click(); });
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
  S.look.draft = { ...DEFAULT_LOOK, source: 'ai', pairId: p.id, ...kept() }; // the AI image is the target: neutral sliders, the user's strength and effects kept
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

// ---- live preview, After Effects style: a chosen preview resolution (完整 = the export size, 1/2, 1/4) and
// 快速預覽 — while a control moves, renders use a 17³ LUT at half that resolution (the 41³ + shaper bake is
// ≈ 600 ms, the 17³ ≈ 50 ms); 250 ms after the last change the full-quality render replaces it. Renders run
// back to back on the latest settings (no debounce), so the picture follows a drag.
const PV_SIDE = { full: 1280, half: 640, quarter: 320 };
const PV = { q: 'full', fast: true, busy: false, pending: null, finalTimer: null, frames: new Map() };
let adjSeq = 0, adjRendered = null;
S.preview = null; // {src, out, w, h, draft, ms} — the frame on the canvas (tests read it)
try { const st = JSON.parse(localStorage.getItem('zhaoyang-preview') || 'null'); if (st && PV_SIDE[st.q]) PV.q = st.q; if (st && typeof st.fast === 'boolean') PV.fast = st.fast; } catch {}
async function previewFrame(side) { // the representative frame at a preview resolution (cached per frame and size)
  const m = S.video.meta, key = `${S.video.id}|${S.rep.t}|${side}`;
  if (!PV.frames.has(key)) {
    if (PV.frames.size > 6) PV.frames.delete(PV.frames.keys().next().value);
    const { w, h } = fitSize(m.width, m.height, side);
    PV.frames.set(key, w === S.rep.img.width && h === S.rep.img.height ? Promise.resolve(S.rep.img) : grabFrame(m.video, S.rep.t, w, h));
  }
  return PV.frames.get(key);
}
function scheduleAdjPreview() {
  if (!S.rep || $('adjustCard').hidden) return;
  clearTimeout(PV.finalTimer);
  if (PV.fast) { requestRender(true); PV.finalTimer = setTimeout(() => requestRender(false), 250); }
  else requestRender(false);
}
function requestRender(draft) { PV.pending = { draft }; if (!PV.busy) pumpRenders(); }
async function pumpRenders() {
  PV.busy = true;
  try {
    while (PV.pending) {
      const job = PV.pending; PV.pending = null;
      const seq = ++adjSeq;
      try {
        await S.grade.ready;
        const side = job.draft ? Math.max(320, PV_SIDE[PV.q] / 2) : PV_SIDE[PV.q];
        const src = await previewFrame(side), t0 = performance.now();
        const intent = S.intent.confirmed ? serializeIntent(S.intent.confirmed) : null;
        const r = await engine.render(src, { intent, seg: S.rep.seg, grade: true, look: S.look.draft, draft: job.draft });
        if (seq !== adjSeq) continue;
        const ms = performance.now() - t0;
        adjRendered = r.imageData;
        S.preview = { src, out: r.imageData, w: src.width, h: src.height, draft: job.draft, ms };
        drawAdj();
        $('pvInfo').textContent = `${src.width}×${src.height}・${job.draft ? '快速（17³）' : '完整品質（41³）'}・${Math.round(ms)} ms`;
        if (job.draft) log('adjust-preview-draft', { w: src.width, ms: Math.round(ms) });
        else { drawScopes(); log('adjust-preview', { look: { ...S.look.draft }, w: src.width, ms: Math.round(ms) }); }
      } catch (err) { $('adjState').textContent = `預覽失敗：${err.message}`; }
    }
  } finally { PV.busy = false; }
}
function drawAdj() {
  const p = S.preview;
  if (!S.rep || !p) return;
  const c = $('adjCanvas'), w = p.w, h = p.h, x = Math.round($('adjWipe').value / 100 * w);
  c.width = w; c.height = h;
  const ctx = c.getContext('2d');
  ctx.putImageData(p.src, 0, 0);
  ctx.putImageData(p.out, 0, 0, x, 0, w - x, h);
  ctx.fillStyle = '#fff'; ctx.fillRect(x - 1, 0, 2, h);
}
function syncPreviewControls() {
  for (const b of document.querySelectorAll('[data-pvq]')) b.setAttribute('aria-checked', String(b.dataset.pvq === PV.q));
  $('pvFast').checked = PV.fast;
}
for (const b of document.querySelectorAll('[data-pvq]')) b.onclick = () => {
  PV.q = b.dataset.pvq; syncPreviewControls();
  try { localStorage.setItem('zhaoyang-preview', JSON.stringify({ q: PV.q, fast: PV.fast })); } catch {}
  log('preview-quality', { q: PV.q });
  clearTimeout(PV.finalTimer); requestRender(false);
};
$('pvFast').addEventListener('change', (e) => {
  PV.fast = e.target.checked;
  try { localStorage.setItem('zhaoyang-preview', JSON.stringify({ q: PV.q, fast: PV.fast })); } catch {}
  log('preview-fast', { on: PV.fast });
});
syncPreviewControls();
// ---- scopes: current result (or the original) beside the reference, redrawn with every preview
let refScopeImg = null;
function refCentre() { // the part of the reference the grade analyses (central 70 %)
  const r = S.ref && S.ref.imageData;
  if (!r) return null;
  if (refScopeImg && refScopeImg.src === r) return refScopeImg.img;
  const x0 = Math.floor(r.width * 0.15), y0 = Math.floor(r.height * 0.15), w = r.width - 2 * x0, h = r.height - 2 * y0, d = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) d.set(r.data.subarray(((y + y0) * r.width + x0) * 4, ((y + y0) * r.width + x0 + w) * 4), y * w * 4);
  refScopeImg = { src: r, img: { data: d, width: w, height: h } };
  return refScopeImg.img;
}
function drawScope(canvas, img) {
  const kind = $('scopeKind').value, ctx = canvas.getContext('2d'), step = Math.max(1, Math.floor(Math.sqrt((img.width * img.height) / 90000)));
  if (kind === 'vector') {
    canvas.width = 256; canvas.height = 256;
    drawVectorscope(ctx, vectorscopeData(img.data, img.width, img.height, { size: 256, step }), 256);
  } else {
    canvas.width = 384; canvas.height = 216;
    drawWaveform(ctx, waveformData(img.data, img.width, img.height, { mode: kind, cols: kind === 'parade' ? 128 : 256, levels: 108, step }), 384, 216);
  }
}
function drawScopes() {
  if (!$('scopes').open || !S.rep) return;
  const orig = $('scopeOrig').checked, img = orig || !adjRendered ? S.rep.img : adjRendered;
  drawScope($('scopeMain'), img);
  $('scopeMainCap').textContent = orig || !adjRendered ? '原片' : '目前結果';
  const ref = refCentre(); if (ref) drawScope($('scopeRef'), ref);
  const c = clipShares(img.data), pc = (v) => `${(v * 100).toFixed(1)}%`;
  $('scopeInfo').textContent = `${orig ? '原片' : '目前結果'}：過曝（任一色版 ≥ 254）${pc(c.hi)}，死黑 ${pc(c.lo)}。${c.hi > 0.02 ? '亮部有明顯裁切。' : ''}${c.lo > 0.02 ? '暗部有明顯死黑。' : ''}`;
  log('scopes-drawn', { kind: $('scopeKind').value, orig, clip: c });
}
$('scopes').addEventListener('toggle', drawScopes);
$('scopeKind').addEventListener('change', drawScopes);
$('scopeOrig').addEventListener('change', drawScopes);

function showAdjustCard() { setAdjustVisible(true); syncAdjustControls(); scheduleAdjPreview(); renderCands(); }

async function startProcessing() {
  const intent = S.intent.confirmed;
  if (!intent) return;
  const token = ++S.job.token;
  await S.grade.ready;
  // the automatic look includes the local grade: wait for it (bounded) if it is still being fitted
  if (S.look.auto && S.ai.status === 'ready') await Promise.race([fitLocalGrade(), new Promise((r) => setTimeout(r, 20000))]);
  S.look.applied = JSON.parse(JSON.stringify(S.look.draft)); // deep: pro controls edit look.prim in place
  setPhase('processing');
  updateAdjState();
  $('progressWrap').hidden = false;
  for (const id of ['exportBtn', 'reprocess', 'editIntent', 'syncPlay', 'cmpTime']) $(id).disabled = true;
  $('photoNav').hidden = true;
  const regs = effectiveRegions(intent);
  // AI segmentation per frame: for AI regions, and for skin protection of the pro grade
  const ap = S.look.applied.prim;
  const needsSeg = regs.some((r) => KINDS[r.kind].source === 'ai') || !!(ap && (ap.skin > 0 || ap.local));
  if (needsSeg) await initSegmenter();
  if (isPhoto()) { $('progressBar').style.width = '0'; $('progressText').textContent = '處理照片中…'; return processPhotoJob(intent, token, needsSeg); }
  try {
    const res = await processVideo({
      engine, meta: S.video.meta, intentJson: serializeIntent(intent), needsSeg, segmentFn: segmentRestored, maskH: intent.maskH,
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
    $('resultInfo').textContent = `${res.frames} 格，${+res.fps.toFixed(3)} fps，${res.width}×${res.height}，${res.codec}，${(res.blob.size / 1e6).toFixed(1)} MB。`
      + `處理引擎回報的光暗設定雜湊 ${res.intentHash}${res.intentHash === S.output.confirmedHash ? '（與確認的設定一致）' : '（⚠ 不一致）'}，色彩設定雜湊 ${res.lookHash}`
    + `（${describeLook(S.look.applied)}）`
    + `；局部調整區域：${res.jobRegions.length ? res.jobRegions.map((r) => `${KINDS[r.kind].label} ${r.ev > 0 ? '+' : ''}${r.ev.toFixed(2)} 檔`).join('、') : '無'}。`
    + (res.audio.kept ? `音訊：已保留原片音軌（${res.audio.codec}，未重新編碼）。` : `音訊：${res.audio.reason}。`);
    log('processing-done', { frames: res.frames, hash: res.intentHash, size: res.blob.size, audio: res.audio });
    setPhase('done');
    updateAdjState();
    for (const id of ['exportBtn', 'reprocess', 'editIntent', 'syncPlay', 'cmpTime']) $(id).disabled = false;
    $('progressWrap').hidden = true;
    $('tabResult').disabled = false; $('tabWipe').disabled = false;
    showView('result');
    drawWipe();
  } catch (err) {
    if (token !== S.job.token) return;
    log('processing-failed', { message: err.message });
    setPhase('error', err.message);
    $('progressWrap').hidden = true;
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
  if (S.output.kind === 'image') { // photo: original | result at the processed size, no time axis
    const out = S.output, i = out.cur, it = out.items[i];
    if (!out.pair || out.pair.i !== i) out.pair = { i, src: await S.video.meta.video.frameAt(i, 4096), res: await createImageBitmap(it.blob) };
    if (S.output !== out || out.cur !== i) return; // navigated meanwhile
    const c = $('wipeCanvas'), w = it.width, h = it.height, x = Math.round($('wipePos').value / 100 * w);
    c.width = w; c.height = h;
    const ctx = c.getContext('2d');
    ctx.putImageData(out.pair.src, 0, 0);
    ctx.save(); ctx.beginPath(); ctx.rect(x, 0, w - x, h); ctx.clip(); ctx.drawImage(out.pair.res, 0, 0, w, h); ctx.restore();
    ctx.fillStyle = '#fff'; ctx.fillRect(x - 1, 0, 2, h);
    return;
  }
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
  const batch = S.output.kind === 'image' && S.output.items.length > 1;
  const name = batch ? `photos_graded_${S.output.intentHash}.zip` // ASCII: some browsers drop a non-ASCII download name
    : S.output.kind === 'image' ? S.output.items[0].name : `${S.video.file.name.replace(/\.[^.]+$/, '')}_graded_${S.output.intentHash}.mp4`;
  try {
    const blob = batch ? await zipStore(S.output.items.map((it) => ({ name: it.name, data: it.blob }))) : S.output.kind === 'image' ? S.output.items[0].blob : S.output.blob;
    await saveFile(name, blob);
    log('exported', { name, size: blob.size, count: batch ? S.output.items.length : 1 });
    setPhase('exported', name);
  } catch (err) {
    setPhase(err && err.code === 'declined' ? 'done' : 'error', err && err.code === 'declined' ? '已取消下載' : `匯出失敗：${err.message || err.code}`);
  }
};

async function saveFile(name, blob) {
  // inside a claude.ai artifact plain download links are blocked; the platform's save dialog is used instead
  const downloads = window.claude && window.claude.use ? await window.claude.use('downloads') : null;
  if (downloads) { await downloads.save({ filename: name, data: blob }); return; }
  const url = URL.createObjectURL(blob), a = document.createElement('a');
  a.href = url; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60000);
}

// ------------------------------------------------------------------ pages, upload controls, workspace views
S.page = 'upload';
function showPage(p) {
  S.page = p;
  $('pageUpload').hidden = p !== 'upload';
  $('pageWork').hidden = p !== 'work';
  $('backToUpload').hidden = p !== 'work';
  document.body.classList.toggle('is-work', p === 'work');
  log('page', { page: p });
  if (p === 'work') {
    if (!$('adjustCard').hidden) scheduleAdjPreview(); // the canvas was not drawn while the page was hidden
    promptOnce();
  }
}
function updateStart() {
  const ready = !!pairKey(), loading = S.ref.status === 'loading' || S.video.status === 'loading';
  $('startBtn').disabled = !ready;
  $('startHint').textContent = ready ? '素材已就緒，可以開始調色。' : loading ? '素材載入中…'
    : S.ref.status !== 'ready' && S.video.status !== 'ready' ? '請上傳參考圖片，以及要調色的影片或照片。'
    : S.ref.status !== 'ready' ? '還需要一張參考圖片。' : '還需要要調色的影片或照片。';
  for (const k of ['ref', 'video']) $(`${k}Actions`).hidden = S[k].status !== 'ready' && S[k].status !== 'error';
  $('refDrop').classList.toggle('has-file', S.ref.status === 'ready');
  $('videoDrop').classList.toggle('has-file', S.video.status === 'ready');
}
$('startBtn').onclick = () => { if (pairKey()) showPage('work'); };
$('backToUpload').onclick = () => showPage('upload');
$('stageIntent').onclick = () => openDialog();
for (const k of ['ref', 'video']) {
  const input = $(`${k}Input`), zone = $(`${k}Drop`);
  $(`${k}Replace`).onclick = () => input.click();
  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('drag'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag'));
  zone.addEventListener('drop', (e) => { // a dropped file goes through the same loader as a picked one
    e.preventDefault(); zone.classList.remove('drag');
    if (!e.dataTransfer.files.length) return;
    const dt = new DataTransfer(); for (const f of (k === 'video' ? [...e.dataTransfer.files] : [e.dataTransfer.files[0]])) dt.items.add(f);
    input.files = dt.files;
    input.dispatchEvent(new Event('change'));
  });
}
for (const ev of ['dragover', 'drop']) window.addEventListener(ev, (e) => e.preventDefault()); // a stray drop must not navigate away
$('videoPreview').addEventListener('loadedmetadata', (e) => { if (e.target.currentTime === 0) e.target.currentTime = Math.min(0.1, e.target.duration / 2); }); // show a frame, not black
$('refRemove').onclick = () => {
  S.ref = { status: 'empty', id: S.ref.id + 1 };
  $('refInput').value = ''; $('refPreview').hidden = true; $('refPreview').removeAttribute('src'); $('refInfo').textContent = '';
  log('ref-removed');
  assetsChanged('ref');
};
$('videoRemove').onclick = () => {
  S.job.token++; // drop any processing still running for this video
  S.video = { status: 'empty', id: S.video.id + 1 };
  S.output = null; S.intent.confirmed = null; S.rep = null; clearAiPair();
  setAdjustVisible(false); resetResultViews(); updateIntentCard();
  showTargetPreview(null);
  $('videoInput').value = ''; $('videoInfo').textContent = '';
  log('video-removed');
  assetsChanged('video');
};
function setAdjustVisible(on) {
  $('adjustCard').hidden = !on;
  $('stageEmpty').hidden = on;
  $('adjCanvas').hidden = !on;
}
function resetResultViews() {
  $('tabResult').disabled = true; $('tabWipe').disabled = true; $('progressWrap').hidden = true;
  $('photoNav').hidden = true;
  $('exportBtn').disabled = true;
  showView('preview');
}
const VIEWS = { preview: ['tabPreview', 'viewPreview'], result: ['tabResult', 'resultCard'], wipe: ['tabWipe', 'wipe'] };
function showView(name) {
  for (const [k, [tab, view]] of Object.entries(VIEWS)) { $(view).hidden = k !== name; $(tab).setAttribute('aria-selected', String(k === name)); }
  if (name !== 'result') for (const id of ['origVideo', 'outVideo']) $(id).pause();
  if (name === 'wipe') drawWipe();
  if (name === 'preview') drawAdj();
}
for (const [k, [tab]] of Object.entries(VIEWS)) $(tab).onclick = () => showView(k);
// changing a control while a processed result is shown: switch back to the live preview so the change is visible
$('adjustCard').addEventListener('input', (e) => { if (e.target.matches('input[type=range], input[type=checkbox], input[type=radio]') && $('viewPreview').hidden) showView('preview'); });
$('candBox').addEventListener('click', () => { if ($('viewPreview').hidden) showView('preview'); });
// ---- photos as the target: one frame, same engine job, JPEG out
const isPhoto = () => !!(S.video.meta && S.video.meta.kind === 'image');
function showTargetPreview(meta) {
  const v = $('videoPreview'), im = $('photoPreview'), photo = !!(meta && meta.kind === 'image');
  if (!meta || photo) { v.pause(); v.hidden = true; if (v.getAttribute('src')) { v.removeAttribute('src'); v.load(); } }
  else { v.src = meta.url; v.hidden = false; }
  if (photo) { im.src = meta.url; im.hidden = false; } else { im.hidden = true; im.removeAttribute('src'); }
  const strip = $('photoStrip'); // thumbnails of a batch (first 11, then a count)
  for (const el of strip.querySelectorAll('img')) URL.revokeObjectURL(el.src);
  strip.replaceChildren();
  if (photo && meta.count > 1) {
    for (const f of S.video.files.slice(0, 11)) { const t = document.createElement('img'); t.src = URL.createObjectURL(f); t.alt = f.name; t.title = f.name; strip.appendChild(t); }
    if (meta.count > 11) { const more = document.createElement('span'); more.textContent = `＋${meta.count - 11} 張`; strip.appendChild(more); }
  }
  strip.hidden = !(photo && meta.count > 1);
  syncKindLabels();
}
function syncKindLabels() { // wording and controls that only make sense for one kind of target
  const photo = isPhoto(), n = photo ? S.video.meta.count : 0;
  $('frameTimeRow').hidden = photo && n < 2;
  $('frameTimeLbl').textContent = photo ? '代表照片' : '代表影格時間';
  $('tabResult').textContent = photo ? '處理結果照片' : '處理結果影片';
  $('origCap').textContent = photo ? '原始照片' : '原始影片'; $('outCap').textContent = photo ? '調色後照片' : '處理後影片';
  for (const id of ['origVideo', 'outVideo']) $(id).hidden = photo;
  for (const id of ['origPhoto', 'outPhoto']) $(id).hidden = !photo;
  $('syncPlay').hidden = photo; $('cmpTimeRow').hidden = photo;
  $('exportBtn').textContent = !photo ? '匯出影片' : n > 1 ? '匯出全部照片' : '匯出照片';
  $('adjApply').textContent = !photo ? '套用並處理影片' : n > 1 ? `套用並處理 ${n} 張` : '套用並處理照片';
  $('exportKindTitle').textContent = photo ? '照片' : '影片';
  $('exportVideoInfo').hidden = photo; $('exportPhotoInfo').hidden = !photo;
  $('reprocess').textContent = '以目前設定重新處理';
}
async function processPhotoJob(intent, token, needsSeg) {
  const m = S.video.meta, n = m.count, items = [], names = new Set();
  let first = null;
  try {
    for (let i = 0; i < n; i++) { // one after another (in parallel they only compete for the CPU and memory)
      if (token !== S.job.token) return;
      $('progressBar').style.width = `${(i / n) * 100}%`; $('progressText').textContent = `${i} / ${n} 張`;
      const res = await processPhoto({ engine, meta: m, index: i, keepPixels: false, intentJson: serializeIntent(intent), needsSeg, segmentFn: segmentRestored, maskH: intent.maskH, look: S.look.applied });
      if (res.intentHash !== hashIntent(intent)) throw new Error(`第 ${i + 1} 張：處理引擎收到的設定與確認的設定不一致`);
      let name = `${S.video.files[i].name.replace(/\.[^.]+$/, '')}_graded.jpg`;
      for (let k = 2; names.has(name); k++) name = `${S.video.files[i].name.replace(/\.[^.]+$/, '')}_graded_${k}.jpg`;
      names.add(name);
      items.push({ name, blob: res.blob, width: res.width, height: res.height });
      first = first || res;
    }
    if (token !== S.job.token) return;
    if (S.output) for (const it of S.output.items || [S.output]) if (it.url) URL.revokeObjectURL(it.url);
    for (const it of items) it.url = URL.createObjectURL(it.blob);
    if (n === 1) items[0].name = `${S.video.file.name.replace(/\.[^.]+$/, '')}_graded_${first.intentHash}.jpg`;
    S.output = { kind: 'image', items, cur: 0, intentHash: first.intentHash, lookHash: first.lookHash, jobRegions: first.jobRegions, confirmedHash: hashIntent(intent), width: items[0].width, height: items[0].height };
    const mb = items.reduce((t, it) => t + it.blob.size, 0) / 1e6;
    $('resultInfo').textContent = `${n > 1 ? `${n} 張照片，同一組調色設定` : `照片 ${items[0].width}×${items[0].height}`}，JPEG，共 ${mb.toFixed(1)} MB。`
      + `處理引擎回報的光暗設定雜湊 ${first.intentHash}（與確認的設定一致），色彩設定雜湊 ${first.lookHash}（${describeLook(S.look.applied)}）`
      + `；局部調整區域：${first.jobRegions.length ? first.jobRegions.map((r) => `${KINDS[r.kind].label} ${r.ev > 0 ? '+' : ''}${r.ev.toFixed(2)} 檔`).join('、') : '無'}。`;
    log('processing-done', { frames: n, photo: true, count: n, hash: first.intentHash, size: Math.round(mb * 1e6) });
    setPhase('done');
    updateAdjState();
    for (const id of ['exportBtn', 'reprocess', 'editIntent']) $(id).disabled = false;
    $('progressWrap').hidden = true;
    $('tabResult').disabled = false; $('tabWipe').disabled = false;
    showPhoto(0);
    showView('result');
  } catch (err) {
    if (token !== S.job.token) return;
    log('processing-failed', { message: err.message });
    setPhase('error', err.message);
    $('progressWrap').hidden = true;
    for (const id of ['reprocess', 'editIntent']) $(id).disabled = false;
  }
}
let origPhotoUrl = null;
function showPhoto(i) { // the result views of a photo (set): original | graded photo i, navigation for a batch
  const out = S.output;
  if (!out || out.kind !== 'image') { $('photoNav').hidden = true; return; }
  out.cur = Math.max(0, Math.min(out.items.length - 1, i));
  const it = out.items[out.cur];
  if (origPhotoUrl) URL.revokeObjectURL(origPhotoUrl);
  origPhotoUrl = URL.createObjectURL(S.video.files[out.cur]);
  $('origPhoto').src = origPhotoUrl; $('outPhoto').src = it.url;
  out.width = it.width; out.height = it.height;
  $('photoNav').hidden = false;
  $('photoPos').textContent = out.items.length > 1 ? `第 ${out.cur + 1} / ${out.items.length} 張　${S.video.files[out.cur].name}` : S.video.files[0].name;
  $('photoPrev').disabled = out.cur === 0; $('photoNext').disabled = out.cur === out.items.length - 1;
  for (const id of ['photoPrev', 'photoNext']) $(id).hidden = out.items.length < 2;
  if (!$('wipe').hidden) drawWipe();
  log('photo-shown', { i: out.cur });
}
$('photoPrev').onclick = () => showPhoto(S.output.cur - 1);
$('photoNext').onclick = () => showPhoto(S.output.cur + 1);
$('photoOne').onclick = async () => { const it = S.output.items[S.output.cur]; await saveFile(it.name, it.blob); log('exported-one', { name: it.name }); };

// ---- curve editor: drag the five knots of the engine's curves (fixed x at 0/25/50/75/100 %) up and down
const CURVE_COL = { luma: '#e8e8ec', red: '#ff6b6b', green: '#5fd38d', blue: '#62a8ff' };
const curve = { ch: 'luma', sel: null, drag: false, hist: null, histFor: null };
function curveHist() { // tone distribution of the original representative frame, for the editor's background
  const img = S.rep && S.rep.img;
  if (!img) return null;
  const key = `${curve.ch}|${S.rep.t}|${img.width}`;
  if (curve.histFor === key) return curve.hist;
  const h = new Float32Array(64), d = img.data, l = [0, 0, 0], k = { red: 0, green: 1, blue: 2 }[curve.ch];
  for (let i = 0; i < d.length; i += 16) {
    const v = k === undefined ? srgbToOklab(d[i] / 255, d[i + 1] / 255, d[i + 2] / 255, l)[0] : d[i + k] / 255;
    h[Math.min(63, Math.floor(v * 64))]++;
  }
  const mx = Math.max(...h) || 1; for (let i = 0; i < 64; i++) h[i] = Math.sqrt(h[i] / mx);
  curve.hist = h; curve.histFor = key; return h;
}
function drawCurve() {
  const c = $('curveCanvas');
  if (!c || !c.offsetWidth) return; // collapsed: drawn when opened
  const dpr = window.devicePixelRatio || 1, W = Math.round(c.offsetWidth * dpr);
  if (c.width !== W) { c.width = W; c.height = W; }
  const ctx = c.getContext('2d'), p = S.look.draft.prim || neutralPrim(), ys = p[curve.ch], pad = 14 * dpr, s = W - 2 * pad;
  const X = (x) => pad + x * s, Y = (y) => pad + (1 - y) * s;
  ctx.clearRect(0, 0, W, W);
  const h = curveHist();
  if (h) { ctx.fillStyle = 'rgba(255,255,255,0.10)'; for (let i = 0; i < 64; i++) { const bh = h[i] * s; ctx.fillRect(X(i / 64), pad + s - bh, s / 64 + 0.5, bh); } }
  ctx.strokeStyle = 'rgba(255,255,255,0.14)'; ctx.lineWidth = dpr;
  for (const t of CURVE_X) { ctx.beginPath(); ctx.moveTo(X(t), Y(0)); ctx.lineTo(X(t), Y(1)); ctx.moveTo(X(0), Y(t)); ctx.lineTo(X(1), Y(t)); ctx.stroke(); }
  ctx.setLineDash([4 * dpr, 4 * dpr]); ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(1), Y(1)); ctx.stroke(); ctx.setLineDash([]);
  const f = curveFn(ys); // exactly the function the engine applies (monotone cubic through the knots)
  ctx.strokeStyle = CURVE_COL[curve.ch]; ctx.lineWidth = 2 * dpr; ctx.beginPath();
  for (let i = 0; i <= 128; i++) { const x = i / 128, y = Math.min(1.05, Math.max(-0.05, f(x))); i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)); }
  ctx.stroke();
  CURVE_X.forEach((x, i) => {
    ctx.beginPath(); ctx.arc(X(x), Y(Math.min(1.05, Math.max(-0.05, ys[i]))), (curve.sel === i ? 7 : 5) * dpr, 0, Math.PI * 2);
    ctx.fillStyle = curve.sel === i ? CURVE_COL[curve.ch] : '#1b1b1e'; ctx.fill(); ctx.strokeStyle = CURVE_COL[curve.ch]; ctx.lineWidth = 2 * dpr; ctx.stroke();
  });
}
function setKnot(i, v) { // same path as the curve sliders: look.prim, auto-follow off, live preview
  const p = proPrim(), ys = p[curve.ch];
  const lo = i > 0 ? ys[i - 1] : -0.05, hi = i < 4 ? ys[i + 1] : 1.05; // keep the curve monotone (the engine flattens a dip anyway)
  ys[i] = Math.round(Math.min(hi, Math.max(lo, Math.min(1.05, Math.max(-0.05, v)))) * 100) / 100;
  S.look.auto = false;
  syncPro(); updateAdjState(); scheduleAdjPreview();
  log('curve-edited', { ch: curve.ch, i, v: ys[i] });
}
function curvePos(e) { const b = $('curveCanvas').getBoundingClientRect(), s = b.width - 28; return { x: (e.clientX - b.left - 14) / s, y: 1 - (e.clientY - b.top - 14) / s }; } // 14 px inset, as drawn
$('curveCanvas').addEventListener('pointerdown', (e) => {
  const { x } = curvePos(e);
  let best = 0; CURVE_X.forEach((cx, i) => { if (Math.abs(cx - x) < Math.abs(CURVE_X[best] - x)) best = i; });
  curve.sel = best; curve.drag = true; $('curveCanvas').setPointerCapture(e.pointerId); $('curveCanvas').focus({ preventScroll: true }); // a scroll would move the curve under the pointer
  setKnot(best, curvePos(e).y);
});
$('curveCanvas').addEventListener('pointermove', (e) => { if (curve.drag) setKnot(curve.sel, curvePos(e).y); });
$('curveCanvas').addEventListener('pointerup', () => { curve.drag = false; });
$('curveCanvas').addEventListener('keydown', (e) => {
  if (curve.sel === null) curve.sel = 2;
  const d = { ArrowUp: 0.01, ArrowDown: -0.01, PageUp: 0.05, PageDown: -0.05 }[e.key];
  if (d) { e.preventDefault(); setKnot(curve.sel, proPrim()[curve.ch][curve.sel] + d); }
  else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { e.preventDefault(); curve.sel = Math.max(0, Math.min(4, curve.sel + (e.key === 'ArrowRight' ? 1 : -1))); drawCurve(); }
});
const resetCurve = () => { proPrim()[curve.ch] = CURVE_X.slice(); S.look.auto = false; syncPro(); updateAdjState(); scheduleAdjPreview(); log('curve-reset', { ch: curve.ch }); };
$('curveCanvas').addEventListener('dblclick', resetCurve);
$('curveReset').onclick = resetCurve;
for (const b of document.querySelectorAll('[data-curve]')) b.onclick = () => {
  curve.ch = b.dataset.curve; curve.sel = null;
  for (const o of document.querySelectorAll('[data-curve]')) o.setAttribute('aria-checked', String(o === b));
  drawCurve();
};
$('catCurves').addEventListener('toggle', drawCurve);
window.addEventListener('resize', drawCurve);

// ---- interface colour: off-white or dark grey; the viewer's choice is remembered on this device
const THEME_KEY = 'zhaoyang-theme';
function applyTheme(choice) {
  if (choice === 'light' || choice === 'dark') document.documentElement.dataset.theme = choice;
  else delete document.documentElement.dataset.theme;
  const eff = choice || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  for (const b of document.querySelectorAll('[data-theme-choice]')) b.setAttribute('aria-checked', String(b.dataset.themeChoice === eff));
}
for (const b of document.querySelectorAll('[data-theme-choice]')) b.onclick = () => {
  applyTheme(b.dataset.themeChoice);
  try { localStorage.setItem(THEME_KEY, b.dataset.themeChoice); } catch {}
  log('theme', { theme: b.dataset.themeChoice });
};
applyTheme((() => { try { return localStorage.getItem(THEME_KEY); } catch { return null; } })());

setAdjustVisible(false);
updateStart();
setPhase('idle');
