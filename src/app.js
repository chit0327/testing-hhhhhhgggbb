// Page controller: asset loading, the Lighting Intent dialog (draft -> confirmed), processing and export.
// Pixels are only ever produced by the worker (src/worker.js) from the serialised CONFIRMED intent.
import {
  createIntent, newRegion, cloneIntent, confirmIntent, canConfirm, effectiveRegions, markEdited,
  serializeIntent, hashIntent, MODES, KINDS, MASK_W,
} from './engine/intent.js';
import { rawMask } from './engine/regionmasks.js';
import { applyEdits, feather, lightnessGrid } from './engine/masks.js';
import { EngineClient, probeVideo, probeImages, PHOTO_MAX, grabFrame, fitSize, loadImage, imageDataOf, processVideo, processPhoto, seek, frameTime, canvas2d, primeVideo } from './pipeline.js';
import { initSegmenter, segment } from './segment.js';
import { DEFAULT_LOOK, measure, toCube, gradeColour } from './engine/grade.js';
import { MOODS, MOOD_GROUPS, moodLook } from './engine/moods.js';
import { featherCells, paintStroke, fillPolygon, combineSelection, labGrid, magicWand, stampSelection, lumaSmall, pyramid, estimateMotion, warpLayer, warpPoint, toN, fromN, IDENTITY, brushRadius } from './engine/blur.js';
import { waveformData, vectorscopeData, drawWaveform, drawVectorscope, clipShares } from './engine/scopes.js';
import { neutralPrim, isNeutralPrim, sanitizePrim, compilePrim, curveFn, HUE_ANCHORS, HUE_NAMES, CURVE_X, LOCAL_NAMES } from './engine/primary.js';
import { bandColours, bandDeltaE, structureScore, clipping, maskedDeltaE } from './engine/metrics.js';
import { claudeAvailable, claudeGrade, toJpeg } from './claudegrade.js';
import { suggestLook } from './engine/suggest.js';
import { measureFilm, FILM_KEYS, FILM_DEFAULT } from './engine/film.js';
import { findBars, findPicture } from './engine/bars.js';
import { refAirRGB } from './engine/render.js';
import { dofPlan, personRel } from './engine/dof.js';
import { initDepth, depthOf, depthInfo, onDepthProgress, prefetchDepth, depthStored } from './depth.js';
import { zipStore } from './zip.js';
import { oklabToSrgb, srgbToOklab } from './engine/color.js';

const $ = (id) => document.getElementById(id);
const engine = new EngineClient();

const PHASE_TEXT = {
  idle: '請先上傳要調色的影片或照片（參考圖片可選）', loading: '素材載入中…', analysing: '正在分析參考圖片與影片…',
  draft: '草稿設定：請在光暗設定視窗中確認', unset: '尚未確認光暗設定（按「重新設定光暗」開始）',
  confirmed: '調色中（尚未處理）', processing: '正在處理…', done: '已完成處理',
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
// long help paragraphs (the fixed ones in the page, not the status lines the app writes): two lines, the rest on a tap
for (const p of document.querySelectorAll('.cat-body p.info:not([id]), .intent-box p.info:not([id]), .ref-video p.info:not([id])')) {
  if (p.textContent.trim().length < 70) continue;
  p.classList.add('clamp'); p.title = '按一下展開／收起';
  p.addEventListener('click', () => p.classList.toggle('open'));
}
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

// no reference (none chosen, or it failed to load): the look comes from a mood preset or the sliders alone
const moodOnly = () => S.ref.status === 'empty' || S.ref.status === 'error';
const pairKey = () => (S.video.status !== 'ready' ? null : S.ref.status === 'ready' ? `${S.ref.id}|${S.video.id}` : moodOnly() ? `mood|${S.video.id}` : null);
const moodBase = () => ({ ...DEFAULT_LOOK, strength: 0, match: 0, source: 'mood' }); // no reference transfer
const maskH = () => Math.round(MASK_W * S.video.meta.height / S.video.meta.width);

// ------------------------------------------------------------------ assets
$('refInput').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const id = ++S.ref.id;
  S.ref = { status: 'loading', id };
  if (typeof LM !== 'undefined') { LM.user = null; $('lmSuggest').hidden = true; } // (a new reference: 依深度分層匹配 is suggested afresh)
  $('refInfo').textContent = '載入中…';
  if (S.refVideo) { URL.revokeObjectURL(S.refVideo.url); S.refVideo.video.remove(); S.refVideo = null; } $('refVideoBox').hidden = true;
  if (file.type.startsWith('video/') || /\.(mp4|mov|m4v|webm)$/i.test(file.name)) { await loadRefVideo(file, id); assetsChanged('ref'); return; }
  try {
    if (!file.type.startsWith('image/')) throw new Error(`不支援的檔案類型（${file.type || '未知'}），請選擇圖片或影片`);
    const bmp = await loadImage(file);
    if (id !== S.ref.id) return;
    // bars around the picture (letterbox, or a video player's background in a screenshot) and panels (a colour-palette
    // strip under a film still, a white frame) are cut off before any analysis
    const probe = imageDataOf(bmp, 512), b = findPicture(probe), k = bmp.width / probe.width;
    const crop = b.w < probe.width || b.h < probe.height ? { x: Math.round(b.x * k), y: Math.round(b.y * k), w: Math.round(b.w * k), h: Math.round(b.h * k) } : null;
    S.ref = { status: 'ready', id, file, bitmap: bmp, crop, imageData: crop ? refPixels(bmp, crop, 512) : probe };
    const img = $('refPreview');
    img.src = URL.createObjectURL(file); img.hidden = false;
    setPipSrc(img.src);
    $('refInfo').textContent = `${file.name}　${bmp.width}×${bmp.height}${crop ? `　已去除${b.panels ? '色板／邊框' : '邊框'}，分析範圍 ${crop.w}×${crop.h}` : ''}`;
    if (crop) log('ref-bars-removed', { crop, panels: !!b.panels });
    log('ref-loaded', { name: file.name });
  } catch (err) {
    S.ref = { status: 'error', id };
    $('refInfo').textContent = `無法載入圖片：${err.message}`;
  }
  assetsChanged('ref');
});

// ---- a reference VIDEO (a drama clip, a screen recording): 整段 = 12 frames spread over it (black / blank ones
// skipped), each with its own bars cut off, tiled into one picture that the analysis reads like a reference image (so
// one warm or dark shot does not decide the look); 選一格 = one frame, chosen with a slider. The 參考圖 picture shows
// one frame (the middle one for 整段).
const REF_FRAMES = 12;
// a <video> that iPad Safari will decode for drawing: in the page (hidden, tiny), playsinline / muted attributes, and
// played once before frames are read — as the work video (pipeline.js probeVideo); a detached video drew black frames
function iosReadyVideo() {
  const v = document.createElement('video');
  v.muted = true; v.playsInline = true; v.preload = 'auto';
  v.setAttribute('playsinline', ''); v.setAttribute('webkit-playsinline', ''); v.setAttribute('muted', ''); v.setAttribute('aria-hidden', 'true');
  v.style.cssText = 'position:fixed;left:0;top:0;width:4px;height:4px;opacity:0.01;pointer-events:none;z-index:-1';
  document.body.appendChild(v);
  return v;
}
function releaseVideo(v) { v.pause(); v.removeAttribute('src'); v.load(); }
async function reloadVideo(v, url) {
  if (v.getAttribute('src')) return;
  v.src = url;
  await new Promise((res, rej) => { const tm = setTimeout(() => rej(new Error('重新讀取參考影片逾時')), 30000); v.addEventListener('loadeddata', () => { clearTimeout(tm); res(); }, { once: true }); v.addEventListener('error', () => { clearTimeout(tm); rej(new Error('無法重新讀取參考影片')); }, { once: true }); });
  await v.play().catch(() => {}); v.pause(); // (iOS: decode after a play)
}
async function loadRefVideo(file, id) {
  $('refInfo').textContent = '讀取參考影片中…';
  const url = URL.createObjectURL(file), v = iosReadyVideo();
  v.src = url;
  try {
    await new Promise((res, rej) => {
      const tm = setTimeout(() => rej(new Error('讀取參考影片逾時')), 30000);
      v.addEventListener('loadeddata', () => { clearTimeout(tm); res(); }, { once: true });
      v.addEventListener('error', () => { clearTimeout(tm); rej(new Error('這個瀏覽器無法讀取這個影片的格式（例如 iPhone／iPad 錄影常用的 HEVC，部分電腦的 Chrome 不支援）。請改用 Safari，或先把影片轉成 MP4（H.264）。')); }, { once: true });
    });
    if (id !== S.ref.id) return;
    await v.play().catch(() => {}); v.pause(); // iOS decodes pictures only after a play (without it every frame read black on an iPad)
    const dur = v.duration, W = v.videoWidth, H = v.videoHeight;
    if (!W || !Number.isFinite(dur) || dur <= 0) throw new Error('讀不到影片的長度或畫面大小');
    const { w, h } = fitSize(W, H, 960), frames = [];
    let skipped = 0;
    for (let i = 0; i < REF_FRAMES; i++) {
      const t = dur * (0.04 + 0.92 * i / (REF_FRAMES - 1));
      $('refInfo').textContent = `讀取參考影片中…（${i + 1} / ${REF_FRAMES} 格）`;
      const img = await grabFrame(v, t, w, h);
      if (id !== S.ref.id) return;
      const f = refFrame(img, t);
      if (f) frames.push(f); else skipped++;
    }
    if (!frames.length) throw new Error('讀到的畫面全是黑色：這部裝置的瀏覽器未能解出這段影片的畫面（例如格式或解碼器不支援）。可以改為上傳影片的截圖作參考圖，或先把影片轉成 MP4（H.264）再試');
    releaseVideo(v); // (the frames are read: an iPad decodes few videos at once, and a loaded reference stalled the target video)
    if (S.video.meta && S.video.meta.video && !isPhoto()) await primeVideo(S.video.meta.video); // (playing the reference took the decoder from the video to grade: give it back)
    S.refVideo = { url, video: v, file, dur, W, H, w, h, frames, skipped, mode: 'all', t: frames[frames.length >> 1].t };
    $('refVideoBox').hidden = false;
    for (const el of document.querySelectorAll('input[name=refMode]')) el.checked = el.value === 'all';
    $('refTimeRow').hidden = true;
    await applyRefVideo(id);
  } catch (err) {
    URL.revokeObjectURL(url); v.remove();
    if (id !== S.ref.id) return;
    S.ref = { status: 'error', id }; S.refVideo = null; $('refVideoBox').hidden = true;
    $('refInfo').textContent = `無法載入參考影片：${err.message}`;
  }
}
/** one grabbed frame → {t, canvas (bars cut off)}; null for a black / blank frame (fades, cuts) */
function refFrame(img, t) {
  const b = findBars(img), d = img.data;
  let s = 0, s2 = 0, n = 0;
  for (let y = b.y; y < b.y + b.h; y += 4) for (let x = b.x; x < b.x + b.w; x += 4) { const i = 4 * (y * img.width + x), l = 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2]; s += l; s2 += l * l; n++; }
  const mean = s / n, sd = Math.sqrt(Math.max(0, s2 / n - mean * mean));
  if (mean < 14 || sd < 5) return null;
  const c = canvas2d(b.w, b.h); c.getContext('2d').putImageData(img, -b.x, -b.y);
  return { t, c, bars: b.w < img.width || b.h < img.height };
}
async function applyRefVideo(id) { // S.refVideo → S.ref (and the pictures)
  const R = S.refVideo;
  let pic, show, note;
  if (R.mode === 'one') {
    await reloadVideo(R.video, R.url);
    let img; try { img = await grabFrame(R.video, R.t, R.w, R.h); } finally { releaseVideo(R.video); }
    if (id !== S.ref.id) return;
    const f = refFrame(img, R.t) || { c: (() => { const c = canvas2d(R.w, R.h); c.getContext('2d').putImageData(img, 0, 0); return c; })(), bars: false };
    pic = f.c; show = f.c; note = `選了 ${R.t.toFixed(1)} 秒的畫面${f.bars ? '，已去除黑邊' : ''}`;
  } else { // tile the frames (a full grid: the first frames fill any empty places)
    const fr = R.frames, n = fr.length, cols = Math.ceil(Math.sqrt(n)), rows = Math.ceil(n / cols), TW = 480;
    const ar = fr.map((f) => f.c.height / f.c.width).sort((a, b) => a - b)[n >> 1], TH = Math.round(TW * ar);
    pic = canvas2d(cols * TW, rows * TH);
    const g = pic.getContext('2d');
    for (let k = 0; k < cols * rows; k++) g.drawImage(fr[k % n].c, (k % cols) * TW, Math.floor(k / cols) * TH, TW, TH);
    show = fr[n >> 1].c;
    note = `整段：取了 ${n} 格${R.skipped ? `（略過 ${R.skipped} 格黑畫面）` : ''}，每格已去除黑邊`;
  }
  const bmp = await createImageBitmap(pic);
  if (id !== S.ref.id) return;
  S.ref = { status: 'ready', id, file: R.file, bitmap: bmp, crop: null, imageData: refPixels(bmp, null, R.mode === 'one' ? 512 : 1024), video: true };
  const jpeg = (c) => (c.convertToBlob ? c.convertToBlob({ type: 'image/jpeg', quality: 0.9 }) : new Promise((res) => c.toBlob(res, 'image/jpeg', 0.9))); // (OffscreenCanvas or <canvas>)
  const blob = await jpeg(R.mode === 'one' ? show : pic), pip = await jpeg(show);
  if (id !== S.ref.id) return;
  for (const u of [R.previewUrl, R.pipUrl]) if (u) URL.revokeObjectURL(u);
  R.previewUrl = URL.createObjectURL(blob); R.pipUrl = URL.createObjectURL(pip);
  $('refPreview').src = R.previewUrl; $('refPreview').hidden = false;
  setPipSrc(R.pipUrl);
  $('refTime').value = Math.round(R.t / R.dur * 1000); $('refTime-v').textContent = `${R.t.toFixed(1)} 秒`;
  $('refInfo').textContent = `${R.file.name}　影片 ${R.W}×${R.H}，${R.dur.toFixed(1)} 秒　${note}`;
  log('ref-loaded', { name: R.file.name, video: true, mode: R.mode, frames: R.mode === 'one' ? 1 : R.frames.length, skipped: R.skipped, t: R.mode === 'one' ? +R.t.toFixed(2) : null });
}
async function refVideoChanged() { // 整段 ↔ 選一格, or another frame: a new reference (the look is fitted again)
  if (!S.refVideo) return;
  const id = ++S.ref.id;
  S.ref = { status: 'loading', id };
  await applyRefVideo(id);
  assetsChanged('ref');
}
for (const el of document.querySelectorAll('input[name=refMode]')) el.addEventListener('change', () => { if (!S.refVideo || !el.checked) return; S.refVideo.mode = el.value; $('refTimeRow').hidden = el.value !== 'one'; refVideoChanged(); });
$('refTime').addEventListener('input', (e) => { if (S.refVideo) $('refTime-v').textContent = `${(e.target.value / 1000 * S.refVideo.dur).toFixed(1)} 秒`; });
$('refTime').addEventListener('change', (e) => { if (!S.refVideo) return; S.refVideo.t = Math.min(S.refVideo.dur - 0.05, e.target.value / 1000 * S.refVideo.dur); refVideoChanged(); });

/** The reference's pixels inside `crop` (bars removed; whole picture when null), long side ≤ maxSide. */
function refPixels(bmp, crop, maxSide) {
  const c0 = crop || { x: 0, y: 0, w: bmp.width, h: bmp.height }, { w, h } = fitSize(c0.w, c0.h, maxSide);
  const c = canvas2d(w, h), ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bmp, c0.x, c0.y, c0.w, c0.h, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}
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
  syncRefMode();
  if (key.startsWith('mood|')) { // nothing to fit without a reference
    S.grade = { pair: key, ready: Promise.resolve(key), cands: [], stats: null, refBands: null };
    S.candMode = null; S.candRun = (S.candRun || 0) + 1; // cancels a candidate pass of the previous reference
    // the previous reference's look cannot be rendered any more: start from the original, or keep the user's edits
    S.look.draft = S.look.auto ? { ...moodBase(), ...kept() } : { ...S.look.draft, strength: 0, match: 0, source: 'mood', pairId: undefined };
    $('candBox').replaceChildren(); $('candInfo').textContent = '';
    if (!$('adjustCard').hidden) { syncAdjustControls(); scheduleAdjPreview(); }
    if (!['processing', 'exporting'].includes(S.phase)) setPhase(S.intent.confirmed ? 'confirmed' : 'draft');
    log('grade-fitted', { mood: true });
    return S.grade.ready;
  }
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
    S.grade.refGrad = r.refGrad || null; drawGradStrip();
    S.grade.autoSkinFollow = r.skinFollow || 0; // a reference whose whole picture is one colour (blue hour): skin follows
    S.grade.cands = (r.candidates || []).map((c) => (SKIN_KEEP ? { ...c, prim: { ...c.prim, skinLine: 0 } } : c));
    S.grade.refBands = bandColours(S.ref.imageData.data, S.ref.imageData.width, S.ref.imageData.height, { centre: 0.7, skipWhite: true });
    log('autofit', { ms: Math.round(r.autofitMs || 0), modes: S.grade.cands.map((c) => c.mode) });
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

$('repRetry').onclick = () => loadRepFrame(S.rep ? S.rep.t : frameTime(S.video.meta, Math.floor(S.video.meta.frameCount / 2)));
async function openDialog() {
  stopPlay(); // (playing uses the video these need to seek)
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
  stopPlay(); // (playing uses the video these need to seek)
  const m = S.video.meta, { w, h } = fitSize(m.width, m.height, 640);
  // reading the frame can fail (a tablet's video decoder busy or stuck): say so and offer a retry, instead of leaving
  // the window black with 確認 greyed out (an iPad stayed like that)
  $('repRetryRow').hidden = true;
  const slow = setTimeout(() => { $('repMsg').hidden = false; $('repMsg').textContent = '正在讀取代表影格…（較慢的裝置或大型影片需要多幾秒）'; }, 2500);
  let img;
  try { if (!isPhoto()) await primeVideo(m.video); img = await grabFrame(m.video, t, w, h); } catch (err) {
    clearTimeout(slow);
    $('repMsg').hidden = false; $('repMsg').textContent = `讀不到影片畫面：${err.message}`; $('repRetryRow').hidden = false;
    log('rep-frame-failed', { message: err.message });
    return;
  }
  clearTimeout(slow); $('repMsg').hidden = true;
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

/** AI segmentation per frame is needed: AI regions, skin protection / local grade, AI background blur, person / skin sliders, haze depth */
function lookNeedsSeg(look, regs) {
  const ap = look.prim, bl = look.blur;
  return !!(look.dof && look.dof.on && look.dof.person) || !!(look.layerMatch && look.layerMatch.on) || !!(look.classMatch && look.classMatch.on) || !!(look.source === 'preset' && look.presetLayers && look.presetLayers.length) || regs.some((r) => KINDS[r.kind].source === 'ai') || !!(ap && (ap.skin > 0 || ap.local)) || !!(bl && bl.on && bl.ai) || !!(look.personEv || look.bgEv || look.skinSat || look.skinWarm || look.skinHue || (look.haze > 0 && (look.hazeDepth > 0 || look.hazeWhere === 1 || look.hazeWhere === 2)));
}

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
  // no processing yet: the user grades on the preview first; 套用並處理 starts it (processing in the background
  // while grading slowed the preview, used the tablet's memory, and was thrown away at the first adjustment)
  updateAdjState();
};
$('openIntent').onclick = () => openDialog();
$('editIntent').onclick = () => openDialog();

function describeLook(k) {
  const mood = k.source === 'mood' && MOODS.find((m) => m.id === k.mood);
  const parts = k.source === 'mood' ? (mood ? [`氛圍「${mood.name}」`] : []) : k.source === 'ai' ? [`強度 ${Math.round(k.strength * 100)}%`] : [`接近參考 ${Math.round(k.match * 100)}%`, `強度 ${Math.round(k.strength * 100)}%`];
  for (const [key, name] of [['exposure', '曝光'], ['contrast', '對比'], ['highlights', '高光'], ['shadows', '陰影'], ['saturation', '飽和'], ['temperature', '色溫'], ['tint', '色調']]) {
    if (k[key]) parts.push(`${name} ${k[key] > 0 ? '+' : ''}${key === 'exposure' ? k[key].toFixed(2) : Math.round(k[key] * 100)}`);
  }
  if (k.black) parts.push(`黑位 +${Math.round(k.black * 100)}`);
  if (k.white !== 1) parts.push(`白位 ${Math.round(k.white * 100)}`);
  for (const [key, name] of [['shadow', '陰影色'], ['mid', '中間調色'], ['hi', '亮部色']]) {
    if (k[`${key}Amt`]) parts.push(`${name} ${Math.round(k[`${key}Hue`])}° ${Math.round(k[`${key}Amt`] * 100)}`);
  }
  if (k.source === 'ai') parts.unshift('AI 調色圖');
  if (k.source === 'preset') parts.unshift(`設定檔「${k.presetName || '未命名'}」`);
  if (k.amount !== undefined && k.amount !== 1) parts.unshift(`整體強度 ${Math.round(k.amount * 100)}%`);
  if (k.prim && !isNeutralPrim(k.prim)) parts.push(`專業調色${S.candMode && S.look.auto ? `（${{ look: '風格', lookStrong: '風格・濃', infer: '推斷風格', natural: '照搬・自然', faithful: '照搬・忠實', bold: '照搬・濃烈' }[S.candMode]}）` : ''}${k.prim.skin ? `、膚色保護 ${Math.round(k.prim.skin * 100)}` : ''}${k.prim.local && (k.prim.localAmt ?? 1) > 0 ? `、局部調色 ${Math.round((k.prim.localAmt ?? 1) * 100)}` : ''}${k.prim.norm && k.prim.norm.amt > 0.02 ? `、LOG 還原 ${Math.round(k.prim.norm.amt * 100)}` : ''}`);
  if (k.glow) parts.push(`柔光 ${Math.round(k.glow * 100)}`);
  if (k.colorPairs && k.colorPairs.on && k.colorPairs.items && k.colorPairs.items.length) parts.push(`點選對應顏色 ${k.colorPairs.items.length} 對`);
  if (k.classMatch && k.classMatch.on && k.classMatch.fit) parts.push(`同類別匹配 ${Math.round((k.classMatch.amt ?? 0.8) * 100)}%`);
  if (k.layerMatch && k.layerMatch.on && k.layerMatch.fit) parts.push(`依深度分層匹配 ${Math.round((k.layerMatch.amt ?? 0.8) * 100)}%${k.layerMatch.bgSat ? `、背景飽和度 ${k.layerMatch.bgSat > 0 ? '+' : ''}${Math.round(k.layerMatch.bgSat * 100)}` : ''}`);
  if (k.dof && k.dof.on) parts.push(`對焦（景深）${Math.round((k.dof.amount ?? 0.5) * 100)}、對焦在${k.dof.person ? '人物' : `點選位置（焦點範圍 ${Math.round((k.dof.area ?? 0.35) * 100)}）`}、光斑 ${Math.round((k.dof.glow ?? 0.5) * 100)}`);
  if (k.fringe) parts.push(`色散 ${Math.round(k.fringe * 100)}${(k.fringeFocus ?? 1) < 1 ? `（失焦處 ${Math.round((k.fringeFocus ?? 1) * 100)}）` : ''}`);
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
  haze: ['haze', 100, (v) => `${v}`], hazeRadius: ['hazeRadius', 100, (v) => `${v}`],
  hazeSoft: ['hazeSoft', 100, (v) => `${v}`], hazeBloom: ['hazeBloom', 100, (v) => `${v}`], hazeVeil: ['hazeVeil', 100, (v) => `${v}`],
  personEv: ['personEv', 100, (v) => `${v > 0 ? '+' : ''}${(v / 100).toFixed(2)} 檔`], bgEv: ['bgEv', 100, (v) => `${v > 0 ? '+' : ''}${(v / 100).toFixed(2)} 檔`],
  skinFollow: ['skinFollow', 100, (v) => `${v}%`], skinSat: ['skinSat', 100, (v) => `${v > 0 ? '+' : ''}${v}`], skinWarm: ['skinWarm', 100, (v) => (v > 0 ? `暖 ${v}` : v < 0 ? `冷 ${-v}` : '0')], skinHue: ['skinHue', 100, (v) => (v > 0 ? `黃 ${v}` : v < 0 ? `粉 ${-v}` : '0')], gradMap: ['gradMap', 100, (v) => `${v}%`], follow: ['follow', 100, (v) => `${v}%`],
  vignette: ['vignette', 100, (v) => `${v > 0 ? '+' : ''}${v}`], vignetteSize: ['vignetteSize', 100, (v) => `${v}`],
  hazeDepth: ['hazeDepth', 100, (v) => `${v}`],
  fringe: ['fringe', 100, (v) => `${v}`], fringeFocus: ['fringeFocus', 100, (v) => `${v}`],
  filmGrain: ['filmGrain', 100, (v) => `${v}`], filmGrainSize: ['filmGrainSize', 100, (v) => `${v}`], halation: ['halation', 100, (v) => `${v}`], filmSoft: ['filmSoft', 100, (v) => `${v}`],
};
// settings that belong to the user's finishing, not to a colour source: kept when the source replaces the look
const KEEP = ['amount', 'follow', 'gradMap', 'personEv', 'bgEv', 'skinFollow', 'skinSat', 'skinWarm', 'skinHue', 'glow', 'glowRadius', 'glowHue', 'glowTint', 'haze', 'hazeRadius', 'hazeSoft', 'hazeBloom', 'hazeVeil', 'vignette', 'vignetteSize', 'filmGrain', 'filmGrainSize', 'halation', 'filmSoft', 'hazeAir', 'hazeDepth', 'hazeWhere', 'fringe', 'fringeFocus', 'dof', 'layerMatch', 'classMatch', 'colorPairs', 'blur'];
const kept = () => Object.fromEntries(KEEP.map((k) => [k, S.look.draft[k] ?? DEFAULT_LOOK[k]]));
// hue sliders show the colour they add (OKLab hue at a mid lightness, so the gradient matches the effect)
{
  const stops = [];
  for (let h = 0; h <= 360; h += 30) { const c = oklabToSrgb(0.72, 0.12 * Math.cos(h * Math.PI / 180), 0.12 * Math.sin(h * Math.PI / 180)); stops.push(`rgb(${c.map((v) => Math.round(v * 255)).join(',')}) ${(h / 360 * 100).toFixed(1)}%`); }
  for (const el of document.querySelectorAll('input.hue')) el.style.setProperty('--hue-gradient', `linear-gradient(90deg, ${stops.join(', ')})`);
}
function syncAdjustControls() {
  syncDof(); syncLm(); syncCm(); syncCp();
  $('hazeFromRef').checked = Array.isArray(S.look.draft.hazeAir);
  syncHazeWhere();
  $('hazeFromRef').disabled = !(S.ref.status === 'ready' && S.ref.imageData);
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
  markMood();
  syncBlurControls();
  drawGlowWheel(); markHazePreset();
}
// ---- 復原 / 重做: every settled change of the settings (the whole look: sliders, moods, candidates, effects, 對焦,
// matching, colour pairs, a loaded 設定檔) is one step. A change is recorded 0.5 s after the last update (a slider drag
// is one step, not one per pixel); undo / redo first records a change still pending. Kept per video, at most 60 steps.
// Not covered: the painted background-blur selection itself (its strokes are not part of the settings) and the
// lighting intent (its own dialog).
const HIST = { stack: [], at: -1, timer: 0, restoring: false, video: null, touched: false };
// until the user first touches the workspace, what the site sets by itself (the automatic match, its suggestions) only
// moves the starting point: undoing back to an ungraded picture made no sense
for (const ev of ['pointerdown', 'keydown']) document.addEventListener(ev, (e) => { if (e.isTrusted && !$('pageWork').hidden) HIST.touched = true; }, true);
const histSnap = () => JSON.stringify({ draft: S.look.draft, auto: S.look.auto, candMode: S.candMode ?? null });
function histRecord() {
  clearTimeout(HIST.timer); HIST.timer = 0;
  if (!S.rep || HIST.restoring) return;
  if (HIST.video !== S.video.id) { HIST.video = S.video.id; HIST.stack = []; HIST.at = -1; HIST.touched = false; }
  const snap = histSnap();
  if (HIST.at >= 0 && HIST.stack[HIST.at] === snap) return;
  if (!HIST.touched) { HIST.stack = [snap]; HIST.at = 0; syncHist(); return; } // (still the starting point)
  HIST.stack = HIST.stack.slice(0, HIST.at + 1); HIST.stack.push(snap);
  if (HIST.stack.length > 60) HIST.stack.shift();
  HIST.at = HIST.stack.length - 1;
  syncHist();
}
function histNote() { if (HIST.restoring || !S.rep) return; clearTimeout(HIST.timer); HIST.timer = setTimeout(histRecord, 500); }
function syncHist() {
  const same = HIST.video === (S.video && S.video.id);
  $('undoBtn').disabled = !same || HIST.at <= 0; $('redoBtn').disabled = !same || HIST.at < 0 || HIST.at >= HIST.stack.length - 1;
}
function histGo(d) {
  if (HIST.timer) histRecord(); // (a change still settling counts as a step first)
  const to = HIST.at + d;
  if (to < 0 || to >= HIST.stack.length) return;
  HIST.at = to; HIST.restoring = true;
  try {
    const o = JSON.parse(HIST.stack[to]);
    S.look.draft = o.draft; S.look.auto = o.auto; S.candMode = o.candMode ?? null;
    for (const el of $('candBox').children) el.classList.toggle('chosen', !!S.candMode && el.dataset.mode === S.candMode);
    DOF.picking = false; CP.step = null; CP.pending = null;
    syncAdjustControls(); scheduleAdjPreview();
    log(d < 0 ? 'undo' : 'redo', { at: to, of: HIST.stack.length });
  } finally { HIST.restoring = false; }
  syncHist();
}
$('undoBtn').onclick = () => histGo(-1);
$('redoBtn').onclick = () => histGo(1);
document.addEventListener('keydown', (e) => {
  if (!(e.metaKey || e.ctrlKey) || e.altKey || $('pageWork').hidden) return;
  const t = e.target, typing = t && (t.isContentEditable || (t.tagName === 'INPUT' && !/^(range|checkbox|radio|button)$/.test(t.type)) || t.tagName === 'TEXTAREA');
  if (typing || document.querySelector('dialog[open]')) return;
  const k = e.key.toLowerCase();
  if (k === 'z' && !e.shiftKey) { e.preventDefault(); histGo(-1); } else if ((k === 'z' && e.shiftKey) || k === 'y') { e.preventDefault(); histGo(1); }
});
function updateAdjState() {
  histNote();
  const dirty = !S.look.applied || JSON.stringify(S.look.applied) !== JSON.stringify(S.look.draft);
  $('adjState').textContent = !S.intent.confirmed ? '確認光暗設定後即可處理影片。'
    : upToDate() ? `${isPhoto() ? '照片' : '影片'}已使用目前的調整：按「${$('exportBtn').textContent}」直接下載。`
    : `按「${$('exportBtn').textContent}」會先匹配顏色和效果，完成後${needsTapToSave() ? '按「下載」儲存' : '自動下載'}。`;
  $('adjApply').disabled = !S.intent.confirmed || S.phase === 'processing';
  $('exportBtn').disabled = $('adjApply').disabled;
  $('tabResult').disabled = $('tabWipe').disabled = $('adjApply').disabled; // (no result yet: asks to match first)
  syncStaleNote();
}
// the result tabs show the last processed file: say so when the settings have changed since (e.g. blur turned on)
function syncStaleNote() {
  const dirty = !!S.output && !!S.look.applied && JSON.stringify(S.look.applied) !== JSON.stringify(S.look.draft);
  $('staleNote').hidden = !(dirty && S.phase !== 'processing' && ($('viewPreview').hidden));
  $('staleApply').textContent = $('exportBtn').textContent;
  $('staleApply').disabled = $('adjApply').disabled;
}
$('staleApply').onclick = () => $('exportBtn').click();
// 朦朧預設: haze / glow settings only (colour untouched), then the sliders fine-tune them
const HAZE_PRESETS = {
  dreamy: { glow: 0, haze: 0.6, hazeRadius: 0.6, hazeSoft: 0.5, hazeBloom: 0.5, hazeVeil: 0.5, glowTint: 0 },
  // 電影柔光 (Black Pro-Mist in afternoon sun, the user's film still): tight glow, deep blacks (no veil), warm light
  promist: { glow: 0, haze: 0.45, hazeRadius: 0.3, hazeSoft: 0.3, hazeBloom: 0.6, hazeVeil: 0, glowHue: 75, glowTint: 0.35 },
  off: { glow: 0, haze: 0 },
};
const hazeKeys = ['glow', 'haze', 'hazeRadius', 'hazeSoft', 'hazeBloom', 'hazeVeil', 'glowHue', 'glowTint'];
function markHazePreset() {
  const d = S.look.draft, on = (p) => Object.entries(HAZE_PRESETS[p]).every(([k, v]) => Math.abs((d[k] ?? DEFAULT_LOOK[k]) - v) < 1e-6);
  for (const b of document.querySelectorAll('[data-haze]')) b.setAttribute('aria-pressed', String(b.dataset.haze !== 'off' && on(b.dataset.haze)));
}
for (const b of document.querySelectorAll('[data-haze]')) b.onclick = () => {
  S.look.auto = false;
  const p = HAZE_PRESETS[b.dataset.haze];
  for (const k of hazeKeys) if (k in p) S.look.draft[k] = p[k]; else if (b.dataset.haze !== 'off') S.look.draft[k] = DEFAULT_LOOK[k];
  log('haze-preset', { preset: b.dataset.haze });
  syncAdjustControls(); scheduleAdjPreview();
};
// 霧色取自參考圖: the haze veil in the colour of the reference's air (read once per reference)
$('hazeFromRef').addEventListener('change', (e) => {
  const ref = S.ref.status === 'ready' && S.ref.imageData;
  S.look.auto = false;
  S.look.draft.hazeAir = e.target.checked && ref ? (S.ref.air ??= refAirRGB(ref)) : null;
  log('haze-air', { on: !!S.look.draft.hazeAir, rgb: S.look.draft.hazeAir });
  syncAdjustControls(); scheduleAdjPreview();
});
// 膠片模板: grain / halation / softness set from the reference (how film-like its tone is), or a typical film setting
$('filmAuto').onclick = () => {
  const ref = S.ref.status === 'ready' && S.ref.bitmap ? refPixels(S.ref.bitmap, S.ref.crop, 1600) : null; // (full-ish resolution: grain; bars cut off)
  const r = ref ? measureFilm(ref) : { ...FILM_DEFAULT };
  S.look.auto = false;
  for (const k of FILM_KEYS) S.look.draft[k] = r[k];
  $('filmInfo').textContent = ref
    ? `已依參考圖設定：膠片感 ${Math.round(r.measured.film * 100)}%（暗部亮度 ${r.measured.black}、明暗範圍 ${r.measured.range}）→ 顆粒 ${Math.round(r.filmGrain * 100)}、光暈 ${Math.round(r.halation * 100)}、柔化 ${Math.round(r.filmSoft * 100)}。可以再用滑桿微調。`
    : `沒有參考圖：套用一般的膠片設定（顆粒 ${Math.round(r.filmGrain * 100)}、光暈 ${Math.round(r.halation * 100)}、柔化 ${Math.round(r.filmSoft * 100)}）。`;
  log('film-template', { fromRef: !!ref, values: Object.fromEntries(FILM_KEYS.map((k) => [k, r[k]])), measured: r.measured || null });
  syncAdjustControls(); scheduleAdjPreview();
};
$('filmOff').onclick = () => { S.look.auto = false; for (const k of FILM_KEYS) S.look.draft[k] = DEFAULT_LOOK[k]; log('film-template', { off: true }); syncAdjustControls(); scheduleAdjPreview(); };
// 光芒顏色 wheel: OKLab hue plane (a →, b ↑), radius = tint 0..1
function drawGlowWheel() {
  const c = $('glowWheel'), ctx = c.getContext('2d'), N = c.width, R = N / 2 - 6, id = ctx.createImageData(N, N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const dx = (x - N / 2) / R, dy = (N / 2 - y) / R, r = Math.hypot(dx, dy), i = (y * N + x) * 4;
    if (r > 1) continue;
    const rgb = oklabToSrgb(0.78, 0.13 * dx, 0.13 * dy);
    id.data[i] = Math.min(1, Math.max(0, rgb[0])) * 255; id.data[i + 1] = Math.min(1, Math.max(0, rgb[1])) * 255; id.data[i + 2] = Math.min(1, Math.max(0, rgb[2])) * 255;
    id.data[i + 3] = 255 * Math.min(1, (1 - r) * R); // anti-aliased rim
  }
  ctx.putImageData(id, 0, 0);
  const d = S.look.draft, t = d.glowTint ?? 0, h = ((d.glowHue ?? 60) * Math.PI) / 180, mx = N / 2 + Math.cos(h) * t * R, my = N / 2 - Math.sin(h) * t * R;
  ctx.lineWidth = 2; ctx.strokeStyle = '#fff'; ctx.beginPath(); ctx.arc(mx, my, 6, 0, 2 * Math.PI); ctx.stroke();
  ctx.strokeStyle = '#000'; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(mx, my, 7.5, 0, 2 * Math.PI); ctx.stroke();
  $('glowWheel-v').textContent = t < 0.01 ? '原本的光色' : `色相 ${Math.round(d.glowHue ?? 60)}°・濃度 ${Math.round(t * 100)}%`;
}
function setGlowColour(hue, tint) {
  S.look.auto = false;
  S.look.draft.glowHue = Math.round(((hue % 360) + 360) % 360); S.look.draft.glowTint = Math.round(Math.min(1, Math.max(0, tint)) * 100) / 100;
  drawGlowWheel(); markHazePreset(); updateAdjState(); scheduleAdjPreview();
}
{
  const c = $('glowWheel');
  const pick = (e) => { const r = c.getBoundingClientRect(), R = r.width / 2 - 6 * r.width / c.width, dx = e.clientX - r.left - r.width / 2, dy = r.top + r.height / 2 - e.clientY;
    setGlowColour((Math.atan2(dy, dx) * 180) / Math.PI, Math.hypot(dx, dy) / R); };
  c.addEventListener('pointerdown', (e) => { c.setPointerCapture(e.pointerId); pick(e); });
  c.addEventListener('pointermove', (e) => { if (c.hasPointerCapture(e.pointerId)) pick(e); });
  c.addEventListener('pointerup', () => log('glow-colour', { hue: S.look.draft.glowHue, tint: S.look.draft.glowTint }));
  c.addEventListener('keydown', (e) => { // arrows: ←/→ hue, ↑/↓ strength
    const d = S.look.draft, k = { ArrowLeft: [-10, 0], ArrowRight: [10, 0], ArrowUp: [0, 0.05], ArrowDown: [0, -0.05] }[e.key];
    if (k) { e.preventDefault(); setGlowColour((d.glowHue ?? 60) + k[0], (d.glowTint ?? 0) + k[1]); }
  });
  $('glowWheelReset').onclick = () => setGlowColour(DEFAULT_LOOK.glowHue, 0);
}
$('adj-skinFollow').addEventListener('input', () => { S.look.skinFollowSet = true; });
// 漸層映射: the reference's gradient (its colour at each lightness, dark → bright), drawn at full colour
function drawGradStrip() {
  const c = $('gradStrip'), g = S.grade && S.grade.refGrad, ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  if (!g) return;
  const N = g.a.length, k = 2.2; // chroma exaggerated a little so the gradient's hues read clearly
  for (let x = 0; x < c.width; x++) {
    const L = 0.08 + 0.88 * x / (c.width - 1), t = L * (N - 1), i = Math.min(N - 2, Math.floor(t)), f = t - i;
    const a = g.a[i] + f * (g.a[i + 1] - g.a[i]), b = g.b[i] + f * (g.b[i + 1] - g.b[i]), rgb = oklabToSrgb(L, a * k, b * k);
    ctx.fillStyle = `rgb(${rgb.map((v) => Math.round(Math.min(1, Math.max(0, v)) * 255)).join(',')})`; ctx.fillRect(x, 0, 1, c.height);
  }
}
for (const [id, [key, scale, fmt]] of Object.entries(ADJ)) {
  $(`adj-${id}`).addEventListener('input', (e) => { S.look.auto = false; S.look.draft[key] = e.target.value / scale; if (fmt) $(`adj-${id}-v`).textContent = fmt(+e.target.value); updateAdjState(); scheduleAdjPreview(); });
  $(`adj-${id}`).addEventListener('dblclick', () => { S.look.auto = false; S.look.draft[key] = DEFAULT_LOOK[key]; syncAdjustControls(); scheduleAdjPreview(); });
}
$('adj-protect').addEventListener('change', (e) => { S.look.auto = false; S.look.draft.protect = e.target.checked; updateAdjState(); scheduleAdjPreview(); });
$('adjReset').onclick = () => { S.look.auto = false; S.look.draft = moodOnly() ? moodBase() : { ...DEFAULT_LOOK }; $('moodInfo').textContent = MOOD_INFO; syncAdjustControls(); scheduleAdjPreview(); markMood(); };
// 膚色偏向參考圖 starts at the reference's automatic value until the user moves that slider
const candLook = (c) => ({ ...DEFAULT_LOOK, strength: 0, match: 0, source: 'reference', prim: sanitizePrim(c.prim), ...kept(), ...(S.look.skinFollowSet ? {} : { skinFollow: S.grade.autoSkinFollow || 0 }) });
function applyAutoLook(mode = 'look') { // 風格: the stable default (same look on any footage)
  const st = S.grade.stats;
  if (!st) return;
  const c = (S.grade.cands || []).find((x) => x.mode === mode);
  S.look.draft = c ? candLook(c) : { ...suggestLook(st.src, st.ref), source: 'reference', ...kept() };
  S.candMode = c ? mode : null;
  S.look.auto = true;
  $('suggestInfo').textContent = `已依參考圖自動匹配：${describeLook(S.look.draft)}。可以再用滑桿微調。`;
  log('suggested', { look: { ...S.look.draft } });
  if (!$('adjustCard').hidden) { syncAdjustControls(); scheduleAdjPreview(); }
  scheduleLmAuto(); scheduleCmAuto();
}
$('adjSuggest').onclick = async () => { await S.grade.ready; applyAutoLook(S.candMode || 'look'); syncAdjustControls(); scheduleAdjPreview(); renderCands(); };

// ---- background blur (背景模糊): the selection is the AI background and/or a layer painted on one frame (the
// anchor: brush, eraser, lasso), kept in the anchor's coordinates. On other frames the layer is placed with the
// camera motion from that frame to the anchor (the worker does the same for every exported frame).
// angle: degrees counter-clockwise, 0 = →, 90 = ↑ (as people read a direction); feather: edge softness 0..1
const BLUR_DEFAULT = { on: false, ai: true, invert: false, type: 'gauss', amount: 0.5, angle: 0, feather: 0.04, bokehGlow: 0.5, bokehRange: 0.5, selId: null };
const ARROWS = ['→', '↗', '↑', '↖', '←', '↙', '↓', '↘'], arrowOf = (a) => ARROWS[Math.round((((a % 360) + 360) % 360) / 45) % 8];
const BL = { tol: 12, labFor: null, lab: null, layer: null, w: 0, h: 0, anchorT: null, anchorLuma: null, anchorPyr: null, curSel: null, seq: 0, tool: 'none', brush: 76, soft: 0, dir: false, motion: new Map(), drag: null, hover: null };
window.__app.BL = BL;
const blurOf = () => ({ ...BLUR_DEFAULT, ...(S.look.draft.blur || {}) });
function editBlur(changes) {
  const next = { ...blurOf(), ...changes };
  if (BL.layer && !next.selId) next.selId = BL.curSel;
  S.look.draft.blur = next;
  syncBlurControls(); syncDof(); updateAdjState(); scheduleAdjPreview(); // (syncDof: the note when background blur and 對焦 are both on)
}
function syncBlurControls() {
  const b = blurOf();
  $('blurOn').checked = b.on; $('blurAi').checked = b.ai; $('blurInvert').checked = b.invert;
  for (const el of document.querySelectorAll('input[name=blurType]')) el.checked = el.value === b.type;
  $('blurAmount').value = Math.round(b.amount * 100); $('blurAmount-v').textContent = Math.round(b.amount * 100);
  $('blurAngle').value = Math.round(b.angle) % 360; $('blurAngle-v').textContent = `${Math.round(b.angle) % 360}° ${arrowOf(b.angle)}`;
  $('blurFeather').value = Math.round(b.feather * 100); $('blurFeather-v').textContent = Math.round(b.feather * 100);
  $('blurDirRow').hidden = b.type !== 'motion';
  $('blurBokehRow').hidden = b.type !== 'bokeh';
  $('blurBokehGlow').value = Math.round(b.bokehGlow * 100); $('blurBokehGlow-v').textContent = Math.round(b.bokehGlow * 100);
  $('blurBokehRange').value = Math.round(b.bokehRange * 100); $('blurBokehRange-v').textContent = Math.round(b.bokehRange * 100);
  $('blurDirBtn').setAttribute('aria-pressed', String(BL.dir));
  $('adjCanvas').classList.toggle('editing', blurEditing());
  if (!blurEditing() && VIEW.s !== 1) setView(1, 0, 0); // (zoom is for painting only)
  syncRefPip();
}
// (the 效果 tab must be the one shown: the tabs hide the other categories with CSS, so catBlur can be open and not hidden
// while the user is on 簡易調色 — the canvas stayed in brush mode there and the reference picture stayed hidden)
const blurEditing = () => (BL.tool !== 'none' || BL.dir) && !$('catBlur').hidden && $('catBlur').open && $('adjustCard').dataset.tab === $('catBlur').dataset.group && !$('viewPreview').hidden;
$('blurOn').addEventListener('change', (e) => { if (e.target.checked) $('blurShow').checked = false; editBlur({ on: e.target.checked }); }); // turning blur on: show the result
// 顯示遮罩 is separate from 開啟模糊: while the mask is shown, the preview is rendered WITHOUT the blur (sharp picture,
// so the selection can be checked and erased precisely); the look itself (and the processed file) is unchanged
const maskShown = () => $('blurShow').checked && !$('catBlur').hidden && $('catBlur').open && $('adjustCard').dataset.tab === $('catBlur').dataset.group && !$('viewPreview').hidden; // (the 效果 tab shown: the tabs hide the other categories with CSS)
function previewLook(d = S.look.draft) { return maskShown() && d.blur && d.blur.on ? { ...d, blur: { ...d.blur, on: false } } : d; }
$('blurAi').addEventListener('change', (e) => { editBlur({ ai: e.target.checked }); if (e.target.checked && !(S.rep && S.rep.seg)) $('blurInfo').textContent = 'AI 背景偵測還沒有結果（請先完成光暗設定的 AI 辨識），可以先用筆刷或套索。'; });
$('blurInvert').addEventListener('change', (e) => editBlur({ invert: e.target.checked }));
$('blurShow').addEventListener('change', () => { drawAdj(); scheduleAdjPreview(); });
for (const el of document.querySelectorAll('input[name=blurType]')) el.addEventListener('change', () => editBlur({ type: el.value }));
$('blurAmount').addEventListener('input', (e) => { $('blurAmount-v').textContent = e.target.value; editBlur({ amount: e.target.value / 100 }); });
$('blurAngle').addEventListener('input', (e) => { editBlur({ angle: +e.target.value }); });
$('blurFeather').addEventListener('input', (e) => { editBlur({ feather: e.target.value / 100 }); });
$('blurBokehGlow').addEventListener('input', (e) => { editBlur({ bokehGlow: e.target.value / 100 }); });
$('blurBokehRange').addEventListener('input', (e) => { editBlur({ bokehRange: e.target.value / 100 }); });
// narrow screens: the panel is below the picture, so bring the picture into view to draw on it
const showCanvasForEditing = () => { if (window.innerWidth <= 1024 && blurEditing()) $('adjCanvas').scrollIntoView({ block: 'center', behavior: 'instant' }); };
for (const el of document.querySelectorAll('input[name=blurTool]')) el.addEventListener('change', () => { if (el.value !== 'none' && (PLAY.on || PLAY.viewT !== null)) stopPlay(true); BL.tool = el.value; BL.dir = false; if (BL.tool !== 'none' && !$('blurShow').checked) { $('blurShow').checked = true; scheduleAdjPreview(); } /* selecting: show the mask */ $('blurTolRow').hidden = !BL.tool.startsWith('wand'); $('blurSizeRow').hidden = !(BL.tool === 'brush' || BL.tool === 'erase'); syncBlurControls(); drawAdj(); showCanvasForEditing(); });
$('blurTol').addEventListener('input', (e) => { BL.tol = +e.target.value; $('blurTol-v').textContent = e.target.value; });
$('blurBrush').addEventListener('input', (e) => { BL.brush = +e.target.value; $('blurBrush-v').textContent = BL.brush; drawAdj(); });
$('blurBrushSoft').addEventListener('input', (e) => { BL.soft = e.target.value / 100; $('blurBrushSoft-v').textContent = e.target.value; drawAdj(); });
const brushR = () => brushRadius(BL.brush) / VIEW.s; // share of the frame width (zoomed in: finer, same size on screen)
$('blurDirBtn').onclick = () => { BL.dir = !BL.dir; syncBlurControls(); drawAdj(); showCanvasForEditing(); };
$('catBlur').addEventListener('toggle', () => { syncBlurControls(); drawAdj(); if (blurOf().on) scheduleAdjPreview(); }); // (mask shown only while the panel is open)
$('blurClear').onclick = () => { BL.layer = null; BL.curSel = null; BL.motion.clear(); editBlur({ selId: null }); log('blur-cleared'); };

// motion from the current preview frame to the anchor (cached per frame time)
function motionToAnchor() {
  if (!BL.layer || !S.rep || S.rep.t === BL.anchorT || isPhoto()) return IDENTITY;
  if (!BL.motion.has(S.rep.t)) {
    const r = estimateMotion(BL.anchorPyr, pyramid(lumaSmall(S.rep.img.data, S.rep.img.width, S.rep.img.height)));
    BL.motion.set(S.rep.t, r.ok ? r.p : IDENTITY);
    if (!r.ok) $('blurInfo').textContent = '這一格和塗抹的那一格差太遠，追蹤不到；筆刷範圍暫時放在原來的位置。';
  }
  return BL.motion.get(S.rep.t);
}
function ensureLayer() {
  if (BL.layer) return;
  BL.w = MASK_W; BL.h = maskH(); BL.layer = new Float32Array(BL.w * BL.h);
  BL.anchorT = S.rep.t; BL.anchorLuma = lumaSmall(S.rep.img.data, S.rep.img.width, S.rep.img.height); BL.anchorPyr = pyramid(BL.anchorLuma); BL.motion.clear();
}
// picture position (0..1 of width / height) → layer grid of the ANCHOR frame
function toAnchorGrid(fx, fy) {
  const gx = fx * BL.w, gy = fy * BL.h;
  const [u, v] = toN(gx - 0.5, gy - 0.5, BL.w, BL.h), [X, Y] = fromN(...warpPoint(motionToAnchor(), u, v), BL.w, BL.h);
  return [X + 0.5, Y + 0.5];
}
// pointer → position in the picture as 0..1 of its width / height (the canvas is letterboxed by object-fit: contain,
// and its pixel size changes between the quick and the full-quality preview, so canvas pixels are not stable)
/** the AI depth of every frame is needed: 對焦（景深） or 依深度分層匹配 */
function needsDepthFor(look) { const d = dofPlan(look.dof); return !!((d && d.mode !== 'tilt') || (look.layerMatch && look.layerMatch.on && look.layerMatch.fit) || (look.classMatch && look.classMatch.on && look.classMatch.fit)); } // (移軸 needs no depth model)

// ---- 依深度分層匹配: the background, split by depth into near and far, each matched to the reference's same layer.
// The fit (worker 'fit-layers') uses the representative frame graded with the current look, its AI person mask and
// depth, and the reference's; it is redone when switched on and when the reference or the candidate changes.
const LM = { busy: false, refDepth: null, user: null, timer: null, suggestedFor: null }; // user: the user switched it by hand
/**
 * Automatically after the reference matching: fitted in the background and SUGGESTED (a 建議開啟 button) when the
 * reference's background layers clearly differ from the video's after the grade (brightness ≥ 0.04, contrast or the
 * vivid colours ≥ 20 %). It never switches itself on: switched on, every exported frame needs its AI depth and person
 * mask (an export took ~40 % longer here), and a picture that changed by itself seconds after it appeared raced with
 * exporting. Once per reference and representative frame; not after the user switched it by hand.
 */
function scheduleLmAuto() {
  clearTimeout(LM.timer);
  if (LM.user !== null || !S.rep || !(S.ref.status === 'ready' && S.ref.imageData) || (S.look.draft.layerMatch && S.look.draft.layerMatch.on)) return;
  const key = `${S.ref.id}|${S.video.id}|${S.rep.t}`;
  if (LM.suggestedFor === key) return;
  LM.timer = setTimeout(() => { if (!LM.busy) { LM.suggestedFor = key; fitLm({ auto: true }); } }, 400);
}
const lmWorth = (fit) => ['near', 'far'].some((k) => { const f = fit && fit[k]; if (!f) return false; const vivid = f.cs ? f.cr[3] / Math.max(f.cs[3], 1e-4) : 1; return Math.abs(f.dL) >= 0.04 || f.k <= 0.8 || f.k >= 1.25 || vivid <= 0.8; });
function syncLm() {
  const l = S.look.draft.layerMatch, on = !!(l && l.on);
  $('lmOn').checked = on; $('lmAmt').disabled = !on;
  $('lmAmt').value = Math.round(((l && l.amt) ?? 0.8) * 100); $('lmAmt-v').textContent = `${$('lmAmt').value}%`;
  $('lmSat').disabled = !on; $('lmSat').value = Math.round(((l && l.bgSat) ?? 0) * 100); $('lmSat-v').textContent = `${+$('lmSat').value > 0 ? '+' : ''}${$('lmSat').value}`;
  $('lmOn').disabled = !(S.ref.status === 'ready' && S.ref.imageData);
}
const fmtAdj = (f) => (f ? `${f.dL > 0.005 ? '亮 +' + Math.round(f.dL * 100) : f.dL < -0.005 ? '暗 ' + Math.round(f.dL * 100) : '亮度不變'}、對比 ×${f.k.toFixed(2)}、鮮艷處 ×${(f.cs ? f.cr[3] / Math.max(f.cs[3], 1e-4) : f.sat ?? 1).toFixed(2)}${f.cb && f.cb[3] != null ? `、亮部飽和 ×${f.cb.slice(3).filter((v) => v != null).map((v) => v.toFixed(2)).join('／')}` : ''}` : '面積太小，不調整');
async function fitLm({ auto = false } = {}) {
  if (!S.rep || !(S.ref.status === 'ready' && S.ref.imageData)) return;
  LM.busy = true; $('lmInfo').textContent = 'AI 深度和人物偵測中…';
  try {
    await initSegmenter(); if (!S.rep.seg) await runSegmentation();
    const refSeg = await referenceSeg(), seg = { w: MASK_W, h: maskH(), ...S.rep.seg };
    const repD = await repDepth();
    if (!LM.refDepth || LM.refDepth.for !== S.ref.imageData) { const bm = await createImageBitmap(S.ref.imageData); LM.refDepth = { for: S.ref.imageData, depth: await depthOf(bm, bm.width, bm.height) }; }
    const intent = S.intent.confirmed ? serializeIntent(S.intent.confirmed) : null, base = { ...S.look.draft, layerMatch: null };
    const r = await engine.fitLayers(S.rep.img, seg, repD, S.ref.imageData, refSeg, LM.refDepth.depth, base, intent);
    if (auto) { // only a suggestion: the picture is not changed
      const worth = lmWorth(r.fit);
      $('lmSuggest').hidden = !worth;
      $('lmInfo').textContent = worth ? `參考圖的背景分層和影片明顯不同（遠景：${fmtAdj(r.fit.far)}；近景：${fmtAdj(r.fit.near)}）。建議開啟：` : '參考圖和影片的背景分層差不多，不需要開啟（需要時可以自己勾選）。';
      if (worth && !/建議開啟下面/.test($('suggestInfo').textContent)) $('suggestInfo').textContent += ' 背景的遠近分層和參考圖明顯不同，建議開啟下面的「依深度分層匹配」。';
      log('layers-fitted', { fit: r.fit, src: r.src, ref: r.ref, ms: Math.round(r.ms), auto, suggested: worth });
      return;
    }
    $('lmSuggest').hidden = true;
    const prev = S.look.draft.layerMatch;
    S.look.draft.layerMatch = { on: true, amt: (prev && prev.amt) ?? 0.8, bgSat: (prev && prev.bgSat) || 0, fit: r.fit };
    $('lmInfo').textContent = `遠景：${fmtAdj(r.fit.far)}；近景：${fmtAdj(r.fit.near)}（人物不變）。`;
    log('layers-fitted', { fit: r.fit, src: r.src, ref: r.ref, ms: Math.round(r.ms), auto, on: true });
    syncLm(); updateAdjState(); scheduleAdjPreview();
  } catch (err) { $('lmInfo').textContent = `不能分層匹配：${err.message}`; if (!auto) S.look.draft.layerMatch = null; syncLm(); }
  finally { LM.busy = false; }
}
// ---- 同類別匹配 (engine/classes.js + layermatch.js): sky / plants / rest of the background, each towards the same class of
// the reference. Like 依深度分層匹配: fitted in the background after the reference matching and only SUGGESTED (a
// 建議開啟 button) when some class clearly differs; on by hand at any time.
const CM = { busy: false, user: null, suggestedFor: '', timer: 0, refDepth: null };
const CM_NAMES = { sky: '天空', plants: '植物', rest: '建築地面' };
// suggested only when the sky or the plants can be matched (both pictures have them) and differ: with only 「建築地面」
// in common it is close to the whole-picture grade (it was suggested for 10 of 11 test pairs that way)
const cmWorth = (fit) => ['sky', 'plants'].some((k) => { const f = fit && fit[k]; if (!f) return false; const vivid = f.cs ? f.cr[3] / Math.max(f.cs[3], 1e-4) : 1; return Math.abs(f.dL) >= 0.04 || f.k <= 0.8 || f.k >= 1.25 || vivid <= 0.8 || Math.hypot(f.da, f.db) >= 0.012; });
const cmText = (r) => ['sky', 'plants', 'rest'].map((k) => `${CM_NAMES[k]}：${r.fit[k] ? fmtAdj(r.fit[k]) + `${Math.hypot(r.fit[k].da, r.fit[k].db) >= 0.006 ? '、色調移向參考' : ''}` : (r.shares.src[k] < 0.04 ? '影片沒有' : '參考圖沒有') + '，不調整'}`).join('；');
function scheduleCmAuto() {
  clearTimeout(CM.timer);
  if (CM.user !== null || !S.rep || !(S.ref.status === 'ready' && S.ref.imageData) || (S.look.draft.classMatch && S.look.draft.classMatch.on)) return;
  const key = `${S.ref.id}|${S.video.id}|${S.rep.t}`;
  if (CM.suggestedFor === key) return;
  CM.timer = setTimeout(() => { if (!CM.busy && !LM.busy) { CM.suggestedFor = key; fitCm({ auto: true }); } else { CM.suggestedFor = ''; scheduleCmAuto(); } }, 900);
}
function syncCm() {
  const c = S.look.draft.classMatch, on = !!(c && c.on);
  $('cmOn').checked = on; $('cmAmt').disabled = !on;
  $('cmAmt').value = Math.round(((c && c.amt) ?? 0.8) * 100); $('cmAmt-v').textContent = `${$('cmAmt').value}%`;
  $('cmOn').disabled = !(S.ref.status === 'ready' && S.ref.imageData);
}
async function fitCm({ auto = false } = {}) {
  if (!S.rep || !(S.ref.status === 'ready' && S.ref.imageData)) return;
  CM.busy = true; $('cmInfo').textContent = 'AI 深度和人物偵測中…';
  try {
    await initSegmenter(); if (!S.rep.seg) await runSegmentation();
    const refSeg = await referenceSeg(), seg = { w: MASK_W, h: maskH(), ...S.rep.seg };
    const repD = await repDepth();
    if (!LM.refDepth || LM.refDepth.for !== S.ref.imageData) { const bm = await createImageBitmap(S.ref.imageData); LM.refDepth = { for: S.ref.imageData, depth: await depthOf(bm, bm.width, bm.height) }; }
    const intent = S.intent.confirmed ? serializeIntent(S.intent.confirmed) : null, base = { ...S.look.draft, classMatch: null };
    const r = await engine.fitClasses(S.rep.img, seg, repD, S.ref.imageData, refSeg, LM.refDepth.depth, base, intent);
    log('classes-fitted', { fit: r.fit, shares: r.shares, ms: Math.round(r.ms), auto });
    if (auto) {
      const worth = cmWorth(r.fit);
      $('cmSuggest').hidden = !worth;
      $('cmInfo').textContent = worth ? `參考圖的天空、植物或建築地面和影片明顯不同（${cmText(r)}）。建議開啟：` : '影片和參考圖同類的地方差不多，不需要開啟（需要時可以自己勾選）。';
      return;
    }
    $('cmSuggest').hidden = true;
    const prev = S.look.draft.classMatch;
    S.look.draft.classMatch = { on: true, amt: (prev && prev.amt) ?? 0.8, fit: r.fit };
    $('cmInfo').textContent = `${cmText(r)}（人物不變）。`;
    syncCm(); updateAdjState(); scheduleAdjPreview();
  } catch (err) { $('cmInfo').textContent = `不能同類別匹配：${err.message}`; if (!auto) S.look.draft.classMatch = null; syncCm(); }
  finally { CM.busy = false; }
}
$('cmSuggest').onclick = () => { CM.user = true; $('cmSuggest').hidden = true; $('cmOn').checked = true; S.look.auto = false; log('classes-suggestion-taken'); fitCm(); };
$('cmOn').addEventListener('change', (e) => {
  $('cmSuggest').hidden = true; S.look.auto = false; CM.user = e.target.checked;
  if (e.target.checked) fitCm();
  else { if (S.look.draft.classMatch) S.look.draft.classMatch = { ...S.look.draft.classMatch, on: false }; syncCm(); updateAdjState(); scheduleAdjPreview(); log('classes-off'); }
});
$('cmAmt').addEventListener('input', (e) => { const c = S.look.draft.classMatch; if (!c || !c.on) return; S.look.auto = false; S.look.draft.classMatch = { ...c, amt: e.target.value / 100 }; $('cmAmt-v').textContent = `${e.target.value}%`; updateAdjState(); scheduleAdjPreview(); });
// ---- 點選對應顏色 (engine/colorpairs.js, baked into the LUT): tap the reference's colour, then the video's matching place.
// The video's colour is read from the ORIGINAL representative frame (before grading: the bake grades it with the rest of
// the look, so the pair stays right while other settings change); the reference's as it is (OKLab), 7×7 px averaged.
const CP = { step: null, pending: null };
const meanAt = (img, fx, fy) => { const x0 = Math.round(fx * (img.width - 1)), y0 = Math.round(fy * (img.height - 1)), r = Math.max(2, Math.round(img.width / 160)); let s = [0, 0, 0], n = 0;
  for (let y = Math.max(0, y0 - r); y <= Math.min(img.height - 1, y0 + r); y++) for (let x = Math.max(0, x0 - r); x <= Math.min(img.width - 1, x0 + r); x++) { const p = 4 * (y * img.width + x); s[0] += img.data[p]; s[1] += img.data[p + 1]; s[2] += img.data[p + 2]; n++; }
  return s.map((v) => v / n / 255); };
const rgbCss = (c) => `rgb(${c.map((v) => Math.round(Math.min(1, Math.max(0, v)) * 255)).join(',')})`;
function drawCpRef() {
  const c = $('cpRef'), img = S.ref.imageData;
  if (!img) { c.hidden = true; return; }
  c.hidden = false; c.width = 320; c.height = Math.round(320 * img.height / img.width);
  const t = canvas2d(img.width, img.height); t.getContext('2d').putImageData(img, 0, 0);
  const x = c.getContext('2d'); x.drawImage(t, 0, 0, c.width, c.height);
  const items = (S.look.draft.colorPairs && S.look.draft.colorPairs.items) || [];
  const marks = items.map((it) => it.ref).concat(CP.pending ? [CP.pending.ref] : []);
  marks.forEach((m, i) => { if (!m) return; x.beginPath(); x.arc(m[0] * c.width, m[1] * c.height, 6, 0, 7); x.lineWidth = 2; x.strokeStyle = '#fff'; x.stroke(); x.fillStyle = '#000'; x.font = '10px sans-serif'; x.fillText(String(i + 1), m[0] * c.width + 8, m[1] * c.height + 4); });
}
function syncCp() {
  const cp = S.look.draft.colorPairs, items = (cp && cp.items) || [], has = !!(cp && cp.on && items.length), ok = S.ref.status === 'ready' && !!S.ref.imageData;
  $('cpAdd').disabled = !ok || items.length >= 6; $('cpClear').disabled = !items.length;
  $('cpRange').disabled = !has; $('cpAmt').disabled = !has;
  $('cpRange').value = Math.round(((cp && cp.range) ?? 0.5) * 100); $('cpRange-v').textContent = $('cpRange').value;
  $('cpAmt').value = Math.round(((cp && cp.amt) ?? 1) * 100); $('cpAmt-v').textContent = `${$('cpAmt').value}%`;
  $('cpAdd').textContent = CP.step === 'ref' ? '在參考圖點想要的顏色…（再按取消）' : CP.step === 'src' ? '在預覽畫面點對應的地方…（再按取消）' : '新增一對顏色';
  $('cpRef').classList.toggle('picking', CP.step === 'ref'); $('adjCanvas').classList.toggle('cp-picking', CP.step === 'src');
  const list = $('cpList'); list.textContent = '';
  items.forEach((it, i) => {
    const row = document.createElement('div'); row.className = 'cp-item';
    const a = document.createElement('span'); a.className = 'cp-sw'; a.style.background = rgbCss(it.s0); a.title = '影片原本的顏色';
    const b = document.createElement('span'); b.className = 'cp-sw'; b.style.background = rgbCss(it.rgb || [0, 0, 0]); b.title = '參考圖的顏色';
    const t = document.createElement('span'); t.textContent = `${i + 1}. 影片 → 參考圖`;
    const del = document.createElement('button'); del.type = 'button'; del.textContent = '刪除'; del.onclick = () => { const c = S.look.draft.colorPairs; S.look.auto = false; S.look.draft.colorPairs = { ...c, items: c.items.filter((_, k) => k !== i) }; log('color-pair-removed', { i }); syncCp(); updateAdjState(); scheduleAdjPreview(); };
    row.append(t, a, document.createTextNode('→'), b, del); list.appendChild(row);
  });
  drawCpRef();
}
function cpPickSource(pt) {
  const [fx, fy] = pt.map((v) => Math.min(1, Math.max(0, v))), prev = S.look.draft.colorPairs || { on: true, amt: 1, range: 0.5, items: [] };
  const it = { s0: meanAt(S.rep.img, fx, fy), r: CP.pending.r, rgb: CP.pending.rgb, ref: CP.pending.ref, src: [+fx.toFixed(4), +fy.toFixed(4)] };
  S.look.auto = false;
  S.look.draft.colorPairs = { ...prev, on: true, items: [...prev.items, it].slice(-6) };
  CP.step = null; CP.pending = null;
  $('cpInfo').textContent = `已加入第 ${S.look.draft.colorPairs.items.length} 對：影片中這個顏色和相近的顏色會變成參考圖的顏色。「影響範圍」越大，越多相近顏色一起改變。`;
  log('color-pair-added', { s0: it.s0, r: it.r });
  syncCp(); updateAdjState(); scheduleAdjPreview();
}
$('cpAdd').onclick = () => { if (CP.step) { CP.step = null; CP.pending = null; } else { CP.step = 'ref'; DOF.picking = false; stopBlurPainting(); syncDof(); $('cpInfo').textContent = '在上面的參考圖點想要的顏色（例如天空、牆、草地）。'; } syncCp(); };
$('cpRef').addEventListener('pointerdown', (e) => {
  if (CP.step !== 'ref' || !S.ref.imageData) return;
  e.preventDefault();
  const r = $('cpRef').getBoundingClientRect(), fx = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), fy = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height));
  const rgb = meanAt(S.ref.imageData, fx, fy), lab = srgbToOklab(rgb[0], rgb[1], rgb[2], [0, 0, 0]);
  CP.pending = { r: lab.map((v) => +v.toFixed(5)), rgb, ref: [+fx.toFixed(4), +fy.toFixed(4)] }; CP.step = 'src';
  $('cpInfo').textContent = '現在在預覽畫面點影片中對應的地方（例如影片的天空）。';
  if ($('viewPreview').hidden) showView('preview');
  syncCp();
});
$('cpClear').onclick = () => { S.look.auto = false; S.look.draft.colorPairs = null; CP.step = null; CP.pending = null; log('color-pairs-cleared'); syncCp(); updateAdjState(); scheduleAdjPreview(); };
for (const [id, k, fmt] of [['cpRange', 'range', (v) => v], ['cpAmt', 'amt', (v) => `${v}%`]]) $(id).addEventListener('input', (e) => { const c = S.look.draft.colorPairs; if (!c) return; S.look.auto = false; S.look.draft.colorPairs = { ...c, [k]: e.target.value / 100 }; $(`${id}-v`).textContent = fmt(e.target.value); updateAdjState(); scheduleAdjPreview(); });
// ---- 設定檔 (.preset): the grade as baked LUTs (the main one and the skin / hair-clothes layers, 33³, 16-bit) plus the
// settings that are not part of the LUT (effects, 對焦 strengths, 人物／背景亮度). Loaded, it becomes the colour source
// (like an AI-graded frame's LUT): no reference needed; the sliders still work on top. Things tied to the original
// video (focus point, blur selection, layer / class fits, colour pairs' positions) are not kept — the colour pairs and
// everything else in the grade are inside the LUT already.
const PRESET_FORMAT = 'zhaoyang-preset';
const PRESET_KEEP = ['personEv', 'bgEv', 'glow', 'glowRadius', 'glowHue', 'glowTint', 'haze', 'hazeRadius', 'hazeSoft', 'hazeBloom', 'hazeVeil', 'vignette', 'vignetteSize', 'filmGrain', 'filmGrainSize', 'halation', 'filmSoft', 'hazeAir', 'hazeDepth', 'hazeWhere', 'fringe', 'fringeFocus'];
const b64of = (f32) => { const u = new Uint16Array(f32.length); for (let i = 0; i < f32.length; i++) u[i] = Math.round(Math.min(1, Math.max(0, f32[i])) * 65535); const b = new Uint8Array(u.buffer); let s = ''; for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode.apply(null, b.subarray(i, i + 0x8000)); return btoa(s); };
const f32of = (b64, n) => { const s = atob(b64), b = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) b[i] = s.charCodeAt(i); const u = new Uint16Array(b.buffer); if (u.length !== n) throw new Error('調色表大小不符'); const f = new Float32Array(n); for (let i = 0; i < n; i++) f[i] = u[i] / 65535; return f; };
function presetSettings(look) {
  const o = {};
  for (const k of PRESET_KEEP) if (look[k] !== undefined && look[k] !== DEFAULT_LOOK[k]) o[k] = look[k];
  const d = look.dof;
  if (d && d.on) { // strengths kept; the tapped point only for 移軸 (a height across the frame, not a thing in the picture)
    const { x, y, person, rel, ...rest } = d;
    o.dof = d.mode === 'tilt' ? { ...rest, x, y, on: true } : { ...rest, on: false };
  }
  return o;
}
async function savePreset() {
  if (!S.rep) return;
  $('presetSave').disabled = true; $('presetInfo').textContent = '正在產生設定檔…';
  try {
    const intent = S.intent.confirmed ? serializeIntent(S.intent.confirmed) : null, look = S.look.draft, b = await engine.bakePreset(look, intent);
    const name = $('presetName').value.trim() || (look.source === 'preset' && look.presetName) || '我的調色';
    const file = { format: PRESET_FORMAT, version: 1, app: '照樣調色', name, created: new Date().toISOString(), from: describeLook(look),
      lut: { size: b.size, encoding: 'uint16-base64', main: b64of(b.main), layers: Object.fromEntries(Object.entries(b.layers).map(([k, v]) => [k, b64of(v)])) },
      settings: presetSettings(look) };
    const blob = new Blob([JSON.stringify(file)], { type: 'application/octet-stream' }) /* (no type: Safari would add .json) */, fname = `${name.replace(/[\\/:*?"<>|]+/g, '_')}.preset`;
    await saveBlob(blob, fname);
    $('presetInfo').textContent = `已儲存「${fname}」（${(blob.size / 1024).toFixed(0)} KB）：色調${Object.keys(b.layers).length ? '、膚色保護' : ''}和效果設定。下次按「載入設定檔」選它即可。`;
    log('preset-saved', { name, size: blob.size, layers: Object.keys(b.layers), settings: Object.keys(file.settings) });
  } catch (err) { $('presetInfo').textContent = `不能儲存設定檔：${err.message}`; }
  finally { $('presetSave').disabled = false; }
}
async function loadPreset(f) {
  try {
    const j = JSON.parse(await f.text());
    if (!j || j.format !== PRESET_FORMAT || !j.lut || !j.lut.main) throw new Error('這不是照樣調色的 .preset 設定檔');
    if (j.version > 1) throw new Error('這個設定檔來自較新版本的網站，請重新整理頁面再試');
    const N = j.lut.size | 0, n = N * N * N * 3;
    if (!(N >= 2 && N <= 65)) throw new Error('調色表大小不正確');
    const main = f32of(j.lut.main, n), layers = Object.fromEntries(Object.entries(j.lut.layers || {}).filter(([k]) => k === 'skin' || k === 'hair').map(([k, v]) => [k, f32of(v, n)]));
    let h = 2166136261; for (let i = 0; i < j.lut.main.length; i += 97) h = Math.imul(h ^ j.lut.main.charCodeAt(i), 16777619);
    const id = `p${(h >>> 0).toString(36)}${N}`;
    await engine.setPreset(id, N, main, layers);
    const st = j.settings || {}, keep = Object.fromEntries(PRESET_KEEP.filter((k) => st[k] !== undefined).map((k) => [k, st[k]]));
    S.look.auto = false; S.candMode = null;
    S.look.draft = { ...DEFAULT_LOOK, ...keep, ...(st.dof ? { dof: st.dof } : {}), source: 'preset', presetId: id, presetName: String(j.name || '未命名').slice(0, 40), presetLayers: Object.keys(layers), strength: 1, match: 0 };
    for (const el of $('candBox').children) el.classList.remove('chosen');
    $('presetName').value = S.look.draft.presetName;
    $('presetInfo').textContent = `已載入「${S.look.draft.presetName}」${j.from ? `（${j.from}）` : ''}。滑桿可以在這個色調上再微調；「強度」調整設定檔色調的程度。${st.dof && !st.dof.on ? '對焦的強度已恢復，請按「選對焦點」重新點一下。' : ''}`;
    log('preset-loaded', { name: S.look.draft.presetName, layers: Object.keys(layers), settings: Object.keys(st) });
    markMood(); syncAdjustControls(); updateAdjState(); scheduleAdjPreview();
    if ($('viewPreview').hidden) showView('preview');
  } catch (err) { $('presetInfo').textContent = `不能載入設定檔：${err.message}`; log('preset-failed', { message: err.message }); }
}
$('presetSave').onclick = () => savePreset();
$('presetLoad').onclick = () => $('presetInput').click();
$('presetInput').addEventListener('change', (e) => { const f = e.target.files && e.target.files[0]; e.target.value = ''; if (f) loadPreset(f); });
// 朦朧的作用範圍: everywhere, only on the people (前景) or only on the background — by the AI person mask (soft edge,
// following the people from frame to frame, like 背景霧更濃)
function syncHazeWhere() {
  const v = S.look.draft.hazeWhere || 0;
  for (const r of document.querySelectorAll('input[name=hazeWhere]')) r.checked = +r.value === v;
  $('adj-hazeDepth').closest('label').hidden = v !== 0; // (背景霧更濃 only matters when the haze is everywhere)
  const info = $('hazeWhereInfo');
  info.hidden = !v;
  if (v) info.textContent = !(S.rep && S.rep.seg) ? '需要 AI 人物偵測：確認光暗設定後會自動計算。' : `${v === 1 ? '霧只加在人物上，背景保持清晰。' : '霧只加在背景，人物保持清晰。'}用 AI 人物偵測分開，邊緣柔和過渡；畫面中沒有人時，${v === 1 ? '不會有霧' : '整個畫面都有霧'}。`;
}
for (const r of document.querySelectorAll('input[name=hazeWhere]')) r.addEventListener('change', () => {
  S.look.auto = false; S.look.draft.hazeWhere = +r.value; syncHazeWhere(); updateAdjState(); scheduleAdjPreview(); log('haze-where', { v: +r.value });
});
$('lmSuggest').onclick = () => { LM.user = true; $('lmSuggest').hidden = true; $('lmOn').checked = true; S.look.auto = false; log('layers-suggestion-taken'); fitLm(); };
$('lmOn').addEventListener('change', (e) => {
  $('lmSuggest').hidden = true;
  S.look.auto = false; LM.user = e.target.checked;
  if (e.target.checked) fitLm();
  else { if (S.look.draft.layerMatch) S.look.draft.layerMatch = { ...S.look.draft.layerMatch, on: false }; syncLm(); updateAdjState(); scheduleAdjPreview(); log('layers-off'); }
});
$('lmSat').addEventListener('input', (e) => { const l = S.look.draft.layerMatch; if (!l || !l.on) return; S.look.auto = false; S.look.draft.layerMatch = { ...l, bgSat: e.target.value / 100 }; $('lmSat-v').textContent = `${+e.target.value > 0 ? '+' : ''}${e.target.value}`; updateAdjState(); scheduleAdjPreview(); });
$('lmAmt').addEventListener('input', (e) => { const l = S.look.draft.layerMatch; if (!l || !l.on) return; S.look.auto = false; S.look.draft.layerMatch = { ...l, amt: e.target.value / 100 }; $('lmAmt-v').textContent = `${e.target.value}%`; updateAdjState(); scheduleAdjPreview(); });

// ---- 對焦（景深）: tap the preview to set the focus; the AI depth model (src/depth.js, loaded on first use) gives each
// frame's depth. On a person (AI mask at the tapped point) the focus follows that person; elsewhere it stays on the
// depth at the tapped point. The representative frame's depth is kept (repDepth) for the preview.
const DOF = { picking: false, rep: null, loading: false, mode: 'lens' };
const dofTilt = () => { const d = S.look.draft.dof; return d && d.on ? d.mode === 'tilt' : DOF.mode === 'tilt'; };
function syncDof() {
  const d = S.look.draft.dof, on = !!(d && d.on);
  $('dofAmount').value = Math.round(((d && d.amount) ?? 0.5) * 100); $('dofAmount-v').textContent = $('dofAmount').value;
  $('dofAmount').disabled = !on; $('dofOff').disabled = !on;
  for (const [id, k] of [['dofGlow', 'glow'], ['dofRange', 'range']]) { $(id).value = Math.round(((d && d[k]) ?? 0.5) * 100); $(`${id}-v`).textContent = $(id).value; $(id).disabled = !on; }
  const tilt = dofTilt();
  for (const r of document.querySelectorAll('input[name=dofMode]')) r.checked = r.value === (tilt ? 'tilt' : 'lens');
  $('dofArea').value = Math.round(((d && d.area) ?? (tilt ? 0.3 : 0.35)) * 100); $('dofArea-v').textContent = $('dofArea').value; $('dofArea').disabled = !on || !!(d && d.person); // (on a person: the focus follows the person)
  $('dofAreaLbl').textContent = tilt ? '　清楚範圍' : '　焦點範圍';
  $('dofAngleRow').hidden = !tilt; $('dofAngle').disabled = !on; $('dofAngle').value = Math.round(((d && d.angle) || 0) * 45); $('dofAngle-v').textContent = `${$('dofAngle').value}°`;
  $('dofGrain').disabled = !on; $('dofGrain').value = Math.round(((d && d.grain) ?? 0.5) * 100); $('dofGrain-v').textContent = $('dofGrain').value;
  $('dofPick').textContent = DOF.picking ? '在預覽上點一下…（再按取消）' : tilt ? (on ? '重新選清楚的位置' : '選清楚的位置') : on ? '重新選對焦點' : '選對焦點';
  $('dofPick').setAttribute('aria-pressed', DOF.picking ? 'true' : 'false');
  $('adjCanvas').classList.toggle('picking', DOF.picking);
  const bl = S.look.draft.blur, both = on && bl && bl.on && bl.amount > 0;
  if (!DOF.loading && tilt) { $('dofInfo').textContent = on ? `移軸：點選位置的一條橫帶保持清楚（清楚範圍、角度），上下越遠越模糊，像移軸鏡頭拍出的「微縮模型」效果。不需要 AI 深度模型。模糊處噪點：把原片的噪點加回模糊的地方，令模糊部分不會過份平滑。${both ? '　注意：「背景模糊」也開着，兩者會疊加。' : ''}` : '按「選清楚的位置」，再在預覽畫面點一下要保持清楚的高度（例如街道上的人和車）。不需要下載 AI 模型。'; return; }
  if (!DOF.loading) $('dofInfo').textContent = on ? `對焦在${d.person ? '人物（人物移動時焦點跟着）' : '點選的位置（焦點範圍：只有點選位置附近清楚，越遠越模糊；100 = 同一距離的地方全部清楚）'}；比它近或遠的地方都會按距離逐漸變模糊，背景的小光點會散開成圓形光斑（光斑亮度、光斑範圍）。${depthInfo() ? `深度計算：${depthInfo().ep === 'gpu' ? '顯示卡' : '處理器'}。` : ''}${both ? '　注意：「背景模糊」也開着，兩者會疊加（先背景模糊、再景深），背景被模糊兩次，背景模糊的光斑會被再糊開；用景深時通常關掉背景模糊、改用這裏的光斑。' : ''}` : '按「選對焦點」，再在預覽畫面點一下要清楚的地方（例如人的臉）。第一次使用會下載 AI 深度模型（約 27 MB）。';
}
async function repDepth() { // the representative frame's AI depth (cached per frame)
  const key = `${S.video.id}|${S.rep.t}`;
  if (DOF.rep && DOF.rep.key === key) return DOF.rep.depth;
  DOF.loading = true; $('dofInfo').textContent = depthInfo() ? 'AI 深度計算中…' : '載入 AI 深度模型（約 27 MB，只在第一次）…';
  const off = onDepthProgress((st) => { $('dofInfo').textContent = `${st}…（只在第一次）`; });
  try {
    const bm = await createImageBitmap(S.rep.img), depth = await depthOf(bm, bm.width, bm.height);
    DOF.rep = { key, depth }; log('dof-depth', { ms: depth.ms, w: depth.w, h: depth.h, ep: depthInfo() && depthInfo().ep });
    return depth;
  } catch (err) { DOF.error = err.message; throw err; }
  finally { off(); DOF.loading = false; syncDof(); if (DOF.error) { $('dofInfo').textContent = `不能計算深度：${DOF.error}`; DOF.error = null; } }
}
function setFocus(pt) {
  const [x, y] = pt.map((v) => Math.min(1, Math.max(0, v)));
  if (dofTilt()) { // 移軸: the sharp band goes through the tapped point
    const prev = S.look.draft.dof && S.look.draft.dof.mode === 'tilt' ? S.look.draft.dof : null;
    S.look.auto = false;
    S.look.draft.dof = { on: true, mode: 'tilt', amount: (prev && prev.amount) ?? 0.6, x: +x.toFixed(4), y: +y.toFixed(4), area: (prev && prev.area) ?? 0.3, angle: (prev && prev.angle) || 0, grain: (prev && prev.grain) ?? 0.5, glow: (prev && prev.glow) ?? 0.5, range: (prev && prev.range) ?? 0.5 };
    DOF.picking = false; syncDof(); updateAdjState(); scheduleAdjPreview();
    log('dof-focus', { x: +x.toFixed(3), y: +y.toFixed(3), tilt: true });
    return;
  }
  const seg = S.rep.seg, mh = seg && seg.person ? seg.person.length / MASK_W : 0;
  const rel = mh ? personRel(x, y, { person: seg.person, mw: MASK_W, mh }) : null, onPerson = rel !== null;
  const prev = S.look.draft.dof;
  S.look.auto = false;
  S.look.draft.dof = { ...(prev && prev.mode !== 'tilt' ? { amount: prev.amount, glow: prev.glow, range: prev.range, area: prev.area, grain: prev.grain } : prev ? { amount: prev.amount, glow: prev.glow, range: prev.range, grain: prev.grain } : {}), on: true, amount: (prev && prev.amount) ?? 0.5, x: +x.toFixed(4), y: +y.toFixed(4), person: onPerson, ...(onPerson ? { rel: +rel.toFixed(3) } : {}) };
  DOF.picking = false; syncDof(); updateAdjState(); scheduleAdjPreview();
  log('dof-focus', { x: +x.toFixed(3), y: +y.toFixed(3), person: onPerson, rel });
}
$('dofPick').onclick = () => { if (!S.rep) return; DOF.picking = !DOF.picking; if (DOF.picking) { stopBlurPainting(); if (!dofTilt()) initDepth().catch(() => {}); } syncDof(); };
for (const r of document.querySelectorAll('input[name=dofMode]')) r.addEventListener('change', () => { // switching keeps the picked point
  DOF.mode = r.value; const d = S.look.draft.dof;
  if (d && d.on) { S.look.draft.dof = { ...d, mode: r.value, area: undefined, angle: undefined }; setFocus([d.x ?? 0.5, d.y]); log('dof-mode', { mode: r.value }); } else syncDof();
});
$('dofAngle').addEventListener('input', (e) => { const d = S.look.draft.dof; if (!d || !d.on) return; S.look.auto = false; S.look.draft.dof = { ...d, angle: e.target.value / 45 }; $('dofAngle-v').textContent = `${e.target.value}°`; updateAdjState(); scheduleAdjPreview(); });
$('dofOff').onclick = () => { if (S.look.draft.dof) S.look.draft.dof = { ...S.look.draft.dof, on: false }; DOF.picking = false; syncDof(); updateAdjState(); scheduleAdjPreview(); log('dof-off'); };
for (const [id, k] of [['dofGlow', 'glow'], ['dofRange', 'range'], ['dofArea', 'area'], ['dofGrain', 'grain']]) $(id).addEventListener('input', (e) => { const d = S.look.draft.dof; if (!d || !d.on) return; S.look.auto = false; S.look.draft.dof = { ...d, [k]: e.target.value / 100 }; $(`${id}-v`).textContent = e.target.value; updateAdjState(); scheduleAdjPreview(); });
$('dofAmount').addEventListener('input', (e) => { const d = S.look.draft.dof; if (!d || !d.on) return; S.look.auto = false; S.look.draft.dof = { ...d, amount: e.target.value / 100 }; $('dofAmount-v').textContent = e.target.value; updateAdjState(); scheduleAdjPreview(); });

function canvasPoint(e) {
  const c = $('adjCanvas'), r = c.getBoundingClientRect(), k = Math.min(r.width / c.width, r.height / c.height);
  return [(e.clientX - r.left - (r.width - c.width * k) / 2) / (c.width * k), (e.clientY - r.top - (r.height - c.height * k) / 2) / (c.height * k)];
}
function applyWand(pt) { // one tap: the connected area of similar colour on THIS frame
  const key = `${S.video.id}|${S.rep.t}|${S.rep.img.width}`;
  if (BL.labFor !== key) { BL.lab = labGrid(S.rep.img.data, S.rep.img.width, S.rep.img.height, BL.w, BL.h); BL.labFor = key; }
  const sel = magicWand(BL.lab, BL.w, BL.h, pt[0] * BL.w, pt[1] * BL.h, BL.tol);
  stampSelection(BL.layer, BL.w, BL.h, sel, motionToAnchor(), BL.tool === 'wand' ? 1 : -1);
  log('blur-wand', { tol: BL.tol, cells: sel.reduce((n, v) => n + v, 0) });
  commitSelection(); drawAdj();
}
async function commitSelection() {
  const selId = `blur${++BL.seq}`;
  await engine.setBlurSelection(selId, { data: BL.layer, w: BL.w, h: BL.h }, BL.anchorLuma);
  BL.curSel = selId;
  const b = blurOf();
  editBlur({ selId, on: true, ...(b.amount ? {} : { amount: 0.5 }) });
  log('blur-selection', { selId, painted: BL.layer.reduce((n, v) => n + (v > 0.5), 0) });
}
{
  const cv = $('adjCanvas');
  // fingers on the picture while painting a selection. A second finger is never paint: whatever the first one started is
  // undone (the layer as it was before the stroke) and the two fingers zoom (pinch) and move (drag) the picture instead;
  // nothing paints again until all fingers are lifted. The magic wand applies when a finger lifts (a tap), so a finger
  // that turns into a pinch does not select.
  const touches = new Map();
  const startStroke = (pt, e) => {
    if (BL.dir) { BL.drag = { kind: 'dir', from: pt, to: pt }; return; }
    ensureLayer();
    if (BL.tool === 'wand' || BL.tool === 'wandErase') {
      if (e.pointerType !== 'mouse') { BL.drag = { kind: 'wand', pt }; return; } // (applied on lift)
      applyWand(pt); return;
    }
    if (BL.tool === 'brush' || BL.tool === 'erase') {
      const a = toAnchorGrid(...pt);
      BL.drag = { kind: 'brush', last: a, before: BL.layer.slice() };
      paintStroke(BL.layer, BL.w, BL.h, a[0], a[1], a[0], a[1], brushR() * BL.w, BL.tool === 'brush' ? 1 : -1, BL.soft);
    } else BL.drag = { kind: 'lasso', pts: [pt] };
    drawAdj();
    if (e.pointerType !== 'mouse') drawLoupe(pt);
  };
  const cancelStroke = () => { // a second finger: the stroke never happened
    const d = BL.drag; BL.drag = null; $('loupe').hidden = true;
    if (d && d.kind === 'brush' && d.before) BL.layer.set(d.before);
    if (d) log('blur-stroke-cancelled', { kind: d.kind });
    drawAdj();
  };
  cv.addEventListener('pointerdown', (e) => {
    if (DOF.picking && S.rep) { e.preventDefault(); setFocus(canvasPoint(e)); return; } // 對焦（景深）: the tap is the focus
    if (CP.step === 'src' && S.rep) { e.preventDefault(); cpPickSource(canvasPoint(e)); return; } // 點選對應顏色: the video's colour
    if (!blurEditing() || !S.rep) return;
    e.preventDefault(); try { cv.setPointerCapture(e.pointerId); } catch { /* (no capture: painting still works while the pointer stays on the picture) */ }
    if (e.pointerType !== 'mouse') touches.set(e.pointerId, [e.clientX, e.clientY]);
    if (touches.size >= 2) { // pinch / pan
      cancelStroke();
      BL.noPaint = true;
      startPinch();
      return;
    }
    if (BL.noPaint) return;
    startStroke(canvasPoint(e), e);
  });
  cv.addEventListener('pointermove', (e) => {
    if (!blurEditing()) return;
    if (touches.has(e.pointerId)) touches.set(e.pointerId, [e.clientX, e.clientY]);
    if (BL.pinch) { if (touches.size >= 2) movePinch(); return; }
    if (BL.noPaint) return;
    const pt = canvasPoint(e); BL.hover = pt;
    const d = BL.drag;
    if (d && d.kind === 'brush') {
      const a = toAnchorGrid(...pt);
      paintStroke(BL.layer, BL.w, BL.h, d.last[0], d.last[1], a[0], a[1], brushR() * BL.w, BL.tool === 'brush' ? 1 : -1, BL.soft);
      d.last = a;
    } else if (d && d.kind === 'lasso') d.pts.push(pt);
    else if (d && d.kind === 'dir') d.to = pt;
    else if (d && d.kind === 'wand') d.pt = pt;
    drawAdj();
    if (d && (d.kind === 'brush' || d.kind === 'lasso') && e.pointerType !== 'mouse') drawLoupe(pt);
  });
  const end = (e) => {
    touches.delete(e.pointerId);
    if (BL.pinch) { if (touches.size < 2) { BL.pinch = null; log('blur-zoom', { scale: +VIEW.s.toFixed(2) }); } }
    if (BL.noPaint) { if (touches.size === 0) BL.noPaint = false; return; }
    $('loupe').hidden = true;
    const d = BL.drag; BL.drag = null;
    if (!d) return;
    if (d.kind === 'wand') { applyWand(d.pt); return; }
    if (d.kind === 'dir') {
      const c = $('adjCanvas'), dx = (d.to[0] - d.from[0]) * c.width, dy = (d.to[1] - d.from[1]) * c.height;
      if (Math.hypot(dx / c.width, dy / c.width) > 0.01) { const ang = ((Math.atan2(-dy, dx) * 180 / Math.PI) % 360 + 360) % 360; /* screen y points down; angles count counter-clockwise */ editBlur({ angle: Math.round(ang), type: 'motion', on: true }); log('blur-direction', { angle: Math.round(ang) }); }
      drawAdj(); return;
    }
    if (d.kind === 'lasso' && d.pts.length > 2) fillPolygon(BL.layer, BL.w, BL.h, d.pts.map((p) => toAnchorGrid(...p)), BL.tool === 'lassoErase' ? -1 : 1);
    commitSelection();
  };
  cv.addEventListener('pointerup', end); cv.addEventListener('pointercancel', end);
  cv.addEventListener('pointerleave', () => { BL.hover = null; if (blurEditing()) drawAdj(); });
  // ---- pinch zoom / two-finger pan of the picture (CSS transform of the canvas; canvasPoint reads the transformed box,
  // so painting lands where the finger is). 1–6×, the picture always fills its box.
  const base = () => { const r = cv.getBoundingClientRect(); return { x: r.left - VIEW.tx, y: r.top - VIEW.ty, w: cv.offsetWidth, h: cv.offsetHeight }; };
  const two = () => { const [a, b] = [...touches.values()]; return { d: Math.hypot(a[0] - b[0], a[1] - b[1]) || 1, mx: (a[0] + b[0]) / 2, my: (a[1] + b[1]) / 2 }; };
  function startPinch() { const t = two(), b = base(); BL.pinch = { d0: t.d, s0: VIEW.s, ux: (t.mx - b.x - VIEW.tx) / VIEW.s, uy: (t.my - b.y - VIEW.ty) / VIEW.s }; }
  function movePinch() {
    const P = BL.pinch, t = two(), b = base();
    const s = Math.min(6, Math.max(1, P.s0 * t.d / P.d0));
    setView(s, t.mx - b.x - s * P.ux, t.my - b.y - s * P.uy);
  }
}
// zoom / pan of the preview while painting a selection (two fingers); the brush keeps its size on the screen, so zoomed
// in it paints finer
const VIEW = { s: 1, tx: 0, ty: 0 };
function setView(s, tx, ty) {
  const cv = $('adjCanvas'), w = cv.offsetWidth, h = cv.offsetHeight;
  VIEW.s = s; VIEW.tx = Math.min(0, Math.max(w * (1 - s), tx)); VIEW.ty = Math.min(0, Math.max(h * (1 - s), ty));
  cv.style.transformOrigin = '0 0';
  cv.style.transform = s === 1 ? '' : `translate(${VIEW.tx}px, ${VIEW.ty}px) scale(${s})`;
  $('pvResetZoom').hidden = s === 1;
}
$('pvResetZoom').onclick = () => { setView(1, 0, 0); drawAdj(); };
// 放大鏡 while painting with a finger (the finger hides what it paints): the preview around the touch point at 2.5×, as
// drawn (the red stripes of the selection included), with a crosshair and the brush outline. Top-left of the picture;
// top-right while the finger is in the top-left part.
const LOUPE_ZOOM = 2.5;
function drawLoupe(pt) {
  const c = $('adjCanvas'), lp = $('loupe'), r = c.getBoundingClientRect(), k = Math.min(r.width / c.width, r.height / c.height);
  lp.hidden = false;
  lp.classList.toggle('right', pt[0] < 0.45 && pt[1] < 0.55);
  const css = lp.clientWidth || 132, dpr = window.devicePixelRatio || 1, n = Math.round(css * dpr);
  if (lp.width !== n) { lp.width = n; lp.height = n; }
  const g = lp.getContext('2d'), src = css / LOUPE_ZOOM / k; // source square in canvas pixels
  const cx = pt[0] * c.width, cy = pt[1] * c.height;
  g.save(); g.clearRect(0, 0, n, n); g.beginPath(); g.arc(n / 2, n / 2, n / 2, 0, 2 * Math.PI); g.clip();
  g.fillStyle = '#111'; g.fillRect(0, 0, n, n);
  g.imageSmoothingEnabled = true;
  g.drawImage(c, cx - src / 2, cy - src / 2, src, src, 0, 0, n, n);
  const s = n / src; // canvas pixel → loupe pixel
  if (BL.tool === 'brush' || BL.tool === 'erase') { g.strokeStyle = BL.tool === 'brush' ? '#fff' : '#ff8080'; g.lineWidth = 1.5 * dpr; g.setLineDash([4 * dpr, 3 * dpr]); g.beginPath(); g.arc(n / 2, n / 2, brushR() * c.width * s, 0, 2 * Math.PI); g.stroke(); g.setLineDash([]); }
  g.strokeStyle = 'rgba(255,255,255,.9)'; g.lineWidth = 1.5 * dpr; g.beginPath(); g.moveTo(n / 2 - 7 * dpr, n / 2); g.lineTo(n / 2 + 7 * dpr, n / 2); g.moveTo(n / 2, n / 2 - 7 * dpr); g.lineTo(n / 2, n / 2 + 7 * dpr); g.stroke();
  g.restore();
}
function drawBlurOverlay(ctx, w, h) {
  const b = blurOf(), panel = !$('catBlur').hidden && $('catBlur').open && $('adjustCard').dataset.tab === $('catBlur').dataset.group; // (the 效果 tab shown)
  if (!panel) return;
  if ($('blurShow').checked) { // the selection as thick red diagonal stripes (shown whether or not the blur is on)
    const p = motionToAnchor(), bg = b.ai && S.rep.seg ? S.rep.seg.background : null;
    const lay = BL.layer ? warpLayer(BL.layer, BL.w, BL.h, p) : null, mw = BL.layer ? BL.w : MASK_W, mh = BL.layer ? BL.h : maskH();
    if ((lay || bg) && (!bg || bg.length === mw * mh)) {
      const m = feather(combineSelection(lay, bg, { ai: b.ai, invert: b.invert }, mw * mh), mw, mh, featherCells(b.feather)), id = new ImageData(mw, mh);
      for (let i = 0; i < m.length; i++) { id.data[4 * i] = 255; id.data[4 * i + 1] = 255; id.data[4 * i + 2] = 255; id.data[4 * i + 3] = m[i] * 255; }
      const t = canvas2d(mw, mh); t.getContext('2d').putImageData(id, 0, 0);
      const s = canvas2d(w, h), sc = s.getContext('2d'); // the mask at picture size, then stripes kept only inside it
      sc.imageSmoothingEnabled = true; sc.drawImage(t, 0, 0, w, h);
      ctx.save(); ctx.globalAlpha = 0.18; ctx.imageSmoothingEnabled = true; ctx.drawImage(s, 0, 0); ctx.restore(); // (white mask, faint: the whole area a little lighter)
      sc.globalCompositeOperation = 'source-in';
      const gap = Math.max(10, Math.round(w / 45)), lw = Math.max(3, gap * 0.38);
      sc.strokeStyle = 'rgba(230, 20, 30, 0.85)'; sc.lineWidth = lw; sc.beginPath();
      for (let k = -h; k < w + h; k += gap) { sc.moveTo(k, h); sc.lineTo(k + h, 0); } // 45° stripes
      sc.stroke();
      ctx.drawImage(s, 0, 0);
    }
  }
  const d = BL.drag;
  ctx.lineWidth = Math.max(1.5, w / 400); ctx.strokeStyle = '#fff'; ctx.setLineDash([]);
  if (d && d.kind === 'lasso') { ctx.beginPath(); d.pts.forEach(([x, y], i) => (i ? ctx.lineTo(x * w, y * h) : ctx.moveTo(x * w, y * h))); ctx.stroke(); }
  if (BL.dir || b.type === 'motion' && b.on && BL.tool === 'none') { // the motion direction as an arrow (dragging: the drag itself)
    const ca = Math.cos(b.angle * Math.PI / 180), sa = -Math.sin(b.angle * Math.PI / 180); // on screen (y down)
    const from = d && d.kind === 'dir' ? [d.from[0] * w, d.from[1] * h] : [w / 2 - ca * w * 0.15, h / 2 - sa * w * 0.15];
    const to = d && d.kind === 'dir' ? [d.to[0] * w, d.to[1] * h] : [w / 2 + ca * w * 0.15, h / 2 + sa * w * 0.15];
    if (BL.dir || d) {
      const a = Math.atan2(to[1] - from[1], to[0] - from[0]), hl = w / 40;
      ctx.strokeStyle = '#ffd34d'; ctx.lineWidth = Math.max(2, w / 300);
      ctx.beginPath(); ctx.moveTo(...from); ctx.lineTo(...to);
      ctx.lineTo(to[0] - hl * Math.cos(a - 0.4), to[1] - hl * Math.sin(a - 0.4)); ctx.moveTo(...to); ctx.lineTo(to[0] - hl * Math.cos(a + 0.4), to[1] - hl * Math.sin(a + 0.4)); ctx.stroke();
    }
  }
  if (BL.hover && (BL.tool === 'brush' || BL.tool === 'erase') && !BL.dir && blurEditing()) { // brush outline
    const rr = brushR() * w;
    ctx.strokeStyle = BL.tool === 'brush' ? '#fff' : '#ff8080'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.arc(BL.hover[0] * w, BL.hover[1] * h, rr, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
  }
}

// ---- mood presets (氛圍): ready-made looks, usable with or without a reference. Each card shows the user's own
// frame graded by the mood (quick 17³ preview) and a colour chart: sample colours before (thin top band) and after.
const MOOD_SAMPLES = [[0.86, 0.64, 0.52], [0.5, 0.7, 0.9], [0.38, 0.55, 0.25], [0.85, 0.25, 0.22], [0.95, 0.94, 0.9], [0.55, 0.55, 0.55], [0.15, 0.15, 0.16]]; // skin, sky, foliage, red, white, grey, shadow
function moodSwatch(m) {
  const look = moodLook(m), n = MOOD_SAMPLES.length, cw = 24, c = document.createElement('canvas'), o = [0, 0, 0];
  c.width = n * cw; c.height = 24;
  const ctx = c.getContext('2d');
  MOOD_SAMPLES.forEach((rgb, i) => {
    const css = (v) => `rgb(${v.map((x) => Math.round(Math.min(1, Math.max(0, x)) * 255)).join(',')})`;
    ctx.fillStyle = css(rgb); ctx.fillRect(i * cw, 0, cw, 6);
    ctx.fillStyle = css(gradeColour(null, rgb[0], rgb[1], rgb[2], look, o)); ctx.fillRect(i * cw, 6, cw, 18);
  });
  c.className = 'mood-swatch'; c.setAttribute('aria-hidden', 'true');
  return c;
}
const MOOD_INFO = $('moodInfo').textContent; // the explanation shown while no mood is applied
function buildMoods() {
  const box = $('moodBox');
  MOOD_GROUPS.forEach((g, gi) => {
    const h = document.createElement('h3'); h.textContent = g; box.appendChild(h);
    const grid = document.createElement('div'); grid.className = 'mood-grid';
    for (const m of MOODS.filter((x) => x.group === gi)) {
      const b = document.createElement('button'); b.type = 'button'; b.className = 'mood'; b.dataset.mood = m.id; b.title = m.desc;
      b.setAttribute('aria-pressed', 'false');
      const th = document.createElement('canvas'); th.className = 'mood-thumb'; th.width = 16; th.height = 9; th.setAttribute('aria-hidden', 'true');
      const name = document.createElement('span'); name.className = 'mood-name'; name.textContent = m.name;
      b.append(th, moodSwatch(m), name);
      b.onclick = () => applyMood(m);
      grid.appendChild(b);
    }
    box.appendChild(grid);
  });
}
buildMoods();
function markMood() {
  const id = S.look.draft.source === 'mood' ? S.look.draft.mood : null;
  for (const b of document.querySelectorAll('#moodBox .mood')) b.setAttribute('aria-pressed', String(b.dataset.mood === id));
}
function applyMood(m) {
  S.look.auto = false; S.candMode = null;
  S.look.draft = moodLook(m, { amount: S.look.draft.amount ?? 1, blur: S.look.draft.blur });
  for (const el of $('candBox').children) el.classList.remove('chosen');
  $('moodInfo').textContent = `已套用「${m.name}」：${m.desc}。下面的滑桿已按這個氛圍設定，可以再微調${moodOnly() ? '' : '；按「恢復自動匹配」可回到參考圖的調色'}。`;
  log('mood-chosen', { mood: m.id });
  markMood(); syncAdjustControls(); scheduleAdjPreview();
  if ($('viewPreview').hidden) showView('preview');
}
async function renderMoodThumbs() {
  if (!S.rep || $('adjustCard').hidden || !$('catMood').open) return; // collapsed: no worker time spent on thumbnails
  const run = S.moodRun = (S.moodRun || 0) + 1, key = `${S.video.id}|${S.rep.t}|${S.intent.confirmed ? hashIntent(S.intent.confirmed) : ''}`;
  if (S.moodThumbsFor === key) return;
  const src = await previewFrame(240), intent = S.intent.confirmed ? serializeIntent(S.intent.confirmed) : null;
  for (const m of MOODS) {
    const r = await engine.render(src, { intent, seg: S.rep.seg, grade: true, look: moodLook(m), draft: true });
    if (run !== S.moodRun) return;
    const c = document.querySelector(`#moodBox .mood[data-mood="${m.id}"] .mood-thumb`);
    c.width = src.width; c.height = src.height; c.getContext('2d').putImageData(r.imageData, 0, 0);
  }
  S.moodThumbsFor = key;
  log('mood-thumbs', { n: MOODS.length });
}
$('catMood').addEventListener('toggle', () => renderMoodThumbs());

// ---- 快速微調（四選一）: four variations of the current look side by side; picking one applies it and offers four
// finer ones around it (like an eye test: "this one or that one"), so the user's eye decides, not a guess
const PICK = { step: 1, round: 1, run: 0, done: 0, shownFor: '' };
function pickVariants(d, st) {
  const c = (v, lo, hi) => Math.min(hi, Math.max(lo, v)), ref = d.source === 'reference' && !moodOnly();
  return [
    { id: 'warm', name: '更暖', look: { ...d, temperature: c(d.temperature + 0.15 * st, -1, 1), tint: c(d.tint + 0.03 * st, -1, 1) } },
    { id: 'cool', name: '更冷', look: { ...d, temperature: c(d.temperature - 0.15 * st, -1, 1) } },
    { id: 'moody', name: '更暗、更有氣氛', look: { ...d, exposure: c(d.exposure - 0.2 * st, -2, 2), contrast: c(d.contrast + 0.08 * st, -1, 1), highlights: c(d.highlights - 0.1 * st, -1, 1) } },
    ref ? { id: 'ref', name: '更像參考圖', look: { ...d, follow: c((d.follow || 0) + 0.3 * st, 0, 1) } }
      : { id: 'bright', name: '更亮、更清爽', look: { ...d, exposure: c(d.exposure + 0.2 * st, -2, 2), shadows: c(d.shadows + 0.1 * st, -1, 1) } },
  ];
}
const pickKey = () => JSON.stringify(S.look.draft) + '|' + PICK.step;
// The four variations are rendered when 四選一 is opened, after a pick and on 重新開始 / 按目前設定更新 — not after every
// slider move (four extra renders each time, and the pictures kept changing while adjusting). After a slider move the old
// four are dimmed with a button to refresh them; a refresh renders off-screen and swaps all four at once (no blank flash).
function markPicksStale() {
  const stale = !!PICK.shownFor && PICK.done === PICK.run && PICK.shownFor !== pickKey(); // (not while four new ones are on their way)
  $('pickGrid').classList.toggle('stale', stale); $('pickRefresh').hidden = !stale;
}
async function renderPicks() {
  if (!S.rep || $('adjustCard').hidden || !$('catPick').open) return;
  const key = pickKey();
  if (PICK.shownFor === key) { markPicksStale(); return; }
  const run = ++PICK.run, src = await previewFrame(320), intent = S.intent.confirmed ? serializeIntent(S.intent.confirmed) : null;
  await S.grade.ready;
  const vs = pickVariants(S.look.draft, PICK.step), btns = [];
  for (const v of vs) {
    const b = document.createElement('button'); b.type = 'button'; b.dataset.pick = v.id;
    const cv = document.createElement('canvas'); cv.width = src.width; cv.height = src.height;
    const lab = document.createElement('span'); lab.textContent = v.name;
    b.append(cv, lab); btns.push(b);
    b.onclick = () => { // apply, then four finer variations around it
      S.look.auto = false; S.look.draft = v.look; PICK.step = Math.max(0.25, PICK.step * 0.6); PICK.round++;
      $('pickRound').textContent = `第 ${PICK.round} 輪`;
      log('pick', { id: v.id, round: PICK.round - 1 });
      syncAdjustControls(); scheduleAdjPreview(); renderPicks();
    };
    const r = await engine.render(src, { intent, seg: S.rep.seg, grade: true, look: v.look, draft: true });
    if (run !== PICK.run) return;
    cv.getContext('2d').putImageData(r.imageData, 0, 0);
  }
  $('pickGrid').replaceChildren(...btns);
  PICK.shownFor = key; PICK.done = run; markPicksStale();
  log('pick-rendered', { round: PICK.round, step: PICK.step });
}
$('catPick').addEventListener('toggle', () => renderPicks());
$('pickRefresh').onclick = () => renderPicks();
$('pickRestart').onclick = () => { PICK.step = 1; PICK.round = 1; $('pickRound').textContent = '第 1 輪'; renderPicks(); };
function syncRefMode() { // reference-only controls are hidden without a reference; the moods open instead
  const none = moodOnly();
  $('catMatch').hidden = none;
  $('catMood').open = none || $('catMood').open;
  $('moodLead').textContent = none ? '沒有參考圖片時，可以直接選一種氛圍；之後仍可用下面的滑桿微調。' : '不想跟參考圖？也可以直接選一種氛圍（會取代參考圖的調色）。';
}

// ---- automatic candidates: render each on the representative frame and measure it
function skinMaskFor(img) { // AI skin grid resampled to the frame (for the skin ΔE report), or null
  const seg = S.rep && S.rep.seg;
  if (!seg || !seg.skin) return null;
  const mw = MASK_W, mh = Math.round(seg.skin.length / mw), w = img.width, h = img.height, m = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) m[y * w + x] = seg.skin[Math.min(mh - 1, Math.floor(y * mh / h)) * mw + Math.min(mw - 1, Math.floor(x * mw / w))];
  return m;
}
const CAND_NOTE = {
  look: '（推薦：讀出參考圖的風格，同一張參考圖套在任何畫面都一致）',
  lookStrong: '（同上，效果更強）',
  natural: '（直接照搬參考圖的顏色：效果較強，但會隨畫面內容變化）',
};
async function renderCands() {
  const box = $('candBox');
  if (!S.rep || !(S.grade.cands || []).length || $('adjustCard').hidden) return;
  await S.grade.ready;
  if (!(S.grade.cands || []).length || !S.grade.refBands) return; // the reference was removed meanwhile
  const run = S.candRun = (S.candRun || 0) + 1; // a newer run (e.g. after the local fit) replaces this one
  const intent = S.intent.confirmed ? serializeIntent(S.intent.confirmed) : null;
  const img = S.rep.img, skinM = skinMaskFor(img);
  const results = [], cards = [];
  for (const c of S.grade.cands) {
    const look = candLook(c);
    const r = await engine.render(img, { intent, seg: S.rep.seg, grade: true, look });
    if (run !== S.candRun || !S.grade.refBands) return;
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
    p.textContent = `${c.label}${CAND_NOTE[c.mode] || ''}：與參考圖色差 ΔE ${m.de.toFixed(1)}｜結構保留 ${m.structure.toFixed(3)}${m.skin !== null ? `｜膚色變化 ΔE ${m.skin.toFixed(1)}` : ''}${m.clip > 0.01 ? `｜裁切 ${(m.clip * 100).toFixed(1)}%` : ''}`;
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
  $('candInfo').textContent = `${restore}${localTxt}原片與參考圖色差 ΔE ${ref0.toFixed(1)}。ΔE 越小越接近參考圖的明暗色調（「照搬」三個會追求接近參考圖的顏色分佈，ΔE 通常較小，但結果會隨畫面內容變化）；結構保留 1.000 代表構圖與紋理完全未變；膚色變化是和原片相比（LOG 還原和局部調色會刻意把膚色調向參考圖的自然膚色，所以不是越小越好，請以畫面判斷）。`;
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
  PRO.push({ el, get, label });
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
// ↺ reset button on every slider row of the panel: puts the row's sliders back to their defaults (DEFAULT_LOOK, the
// neutral pro grade, the blur defaults) through their own input handlers, so state, labels and preview update as usual
{
  const def = new Map();
  for (const [id, [key, scale]] of Object.entries(ADJ)) def.set($(`adj-${id}`), () => Math.round(DEFAULT_LOOK[key] * scale));
  for (const c of PRO) def.set(c.el, /膚色保護/.test(c.label || '') ? () => (S.look.draft.source === 'reference' ? 100 : 0) : () => Math.round(c.get(neutralPrim()))); // skin protection is on by default for reference grading
  for (const [id, v] of [['blurAmount', BLUR_DEFAULT.amount * 100], ['blurFeather', BLUR_DEFAULT.feather * 100], ['blurBokehGlow', BLUR_DEFAULT.bokehGlow * 100], ['blurBokehRange', BLUR_DEFAULT.bokehRange * 100], ['blurAngle', BLUR_DEFAULT.angle], ['blurTol', 12], ['dofAmount', 50], ['dofArea', 35], ['dofAngle', 0], ['dofGrain', 50], ['dofGlow', 50], ['dofRange', 50], ['lmAmt', 80], ['lmSat', 0], ['cmAmt', 80], ['cpRange', 50], ['cpAmt', 100]]) def.set($(id), () => v);
  for (const row of $('adjustCard').querySelectorAll('label.slider, .tone-row, .wheel-row, .curve-row, .hue-row')) {
    const inputs = [...row.querySelectorAll('input[type=range]')].filter((el) => def.has(el));
    if (!inputs.length) continue;
    const b = document.createElement('button'); b.type = 'button'; b.className = 'reset'; b.textContent = '↺';
    const name = (row.querySelector('span') || {}).textContent || ''; b.title = b.ariaLabel = `重設${name.trim() ? `「${name.trim()}」` : ''}`;
    b.addEventListener('click', (e) => {
      e.preventDefault(); e.stopPropagation(); // inside a <label>: do not also focus / activate the slider
      for (const el of inputs) { el.value = def.get(el)(); el.dispatchEvent(new Event('input', { bubbles: true })); }
      log('slider-reset', { ids: inputs.map((el) => el.id || el.closest('[class$=row]')?.className || '') });
    });
    row.appendChild(b);
  }
}
function syncPro() {
  const p = S.look.draft.prim || neutralPrim();
  for (const c of PRO) c.el.value = Math.round(c.get(p));
  drawCurve();
}
$('gpuBench').onclick = async () => {
  if (!S.rep || !S.rep.img) { $('gpuInfo').textContent = '請先完成光暗設定，讓預覽出現後再測試。'; return; }
  $('gpuBench').disabled = true; $('gpuInfo').textContent = '測試中…（4K 用處理器計算一格要幾秒）';
  try {
    const intent = S.intent.confirmed ? serializeIntent(S.intent.confirmed) : null;
    const r = await engine.gpuBench(S.rep.img, S.look.draft, intent);
    const ms = (v) => (v == null ? '—' : `${Math.round(v)} ms`);
    $('gpuInfo').textContent = !r.renderer ? `這部裝置不能用顯示卡調色：${r.why}。處理器每格：${r.sizes.map((s) => `${s.w}×${s.h} ${ms(s.cpu)}`).join('，')}。`
      : `顯示卡：${r.renderer}。` + r.sizes.map((s) => `${s.w}×${s.h}：處理器 ${ms(s.cpu)}／顯示卡 ${ms(s.gpu)}${s.gpu ? (s.cpu >= s.gpu ? `（顯示卡快 ${(s.cpu / s.gpu).toFixed(1)} 倍）` : `（顯示卡慢 ${(s.gpu / s.cpu).toFixed(1)} 倍）`) : ''}${s.maxDiff != null ? `，結果差異最多 ${s.maxDiff}／255（平均 ${s.meanDiff.toFixed(2)}）` : ''}${s.error ? `，錯誤：${s.error}` : ''}${s.fxCpu != null ? `；效果：處理器 ${ms(s.fxCpu)}／顯示卡 ${ms(s.fxGpu)}${s.fxMaxDiff != null ? `，差異最多 ${s.fxMaxDiff}／255` : ''}${s.fxError ? `，錯誤：${s.fxError}` : ''}` : ''}`).join('；') + `。${r.sizes.some((s) => s.fxCpu != null) ? '' : '（目前的設定沒有效果，只計調色一步。）'}`;
    log('gpu-bench', r);
  } catch (err) { $('gpuInfo').textContent = `測試失敗：${err.message}`; }
  $('gpuBench').disabled = false;
};
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
  if (!S.claude || !S.rep || S.claudeBusy || moodOnly()) return false; // Claude compares with the reference
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
  if (S.look.draft.source === 'ai') S.look.draft = { ...S.look.draft, source: 'reference', pairId: undefined };
}
$('adjApply').onclick = () => startProcessing();
$('reprocess').onclick = () => startProcessing();
// the 原始／調色 split is not drawn while painting a blur selection (the whole result is shown to paint on): moving the
// split slider means comparing, so painting ends there (the split was "gone" after an export with a tool still picked)
function stopBlurPainting() {
  if (BL.tool === 'none' && !BL.dir) return false;
  const none = document.querySelector('input[name=blurTool][value=none]');
  if (none) { none.checked = true; none.dispatchEvent(new Event('change', { bubbles: true })); }
  if ($('blurShow').checked) { $('blurShow').checked = false; scheduleAdjPreview(); } // (the stripes came on with the tool; comparing needs the picture)
  BL.dir = false; syncBlurControls(); drawAdj(); log('blur-painting-ended');
  return true;
}
$('adjWipe').addEventListener('input', () => { if (blurEditing()) stopBlurPainting(); drawAdj(); });

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
  if (PLAY.on) return; // (playing: every shown frame is rendered with the current settings)
  if (PLAY.viewT !== null) { videoFrame(); return; } // (paused on another frame: that frame follows the settings)
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
        const used = JSON.parse(JSON.stringify(S.look.draft)); // (the settings this picture is made with: logged below, not whatever the sliders say by the time it is back)
        const depth = needsDepthFor(used) ? await repDepth() : null;
        const r = await engine.render(src, { intent, seg: S.rep.seg, depth, grade: true, look: previewLook(used), draft: job.draft, cacheKey: `${S.video.id}|${S.rep.t}|${S.rep.seg ? 's' : ''}|${depth ? 'd' : ''}` });
        if (seq !== adjSeq) continue;
        const ms = performance.now() - t0;
        adjRendered = r.imageData;
        S.preview = { src, out: r.imageData, w: src.width, h: src.height, draft: job.draft, ms };
        drawAdj();
        $('pvInfo').textContent = `${src.width}×${src.height}・${job.draft ? '快速（17³）' : '完整品質（41³）'}・${Math.round(ms)} ms`;
        if (job.draft) log('adjust-preview-draft', { w: src.width, ms: Math.round(ms), stages: r.timing });
        else { drawScopes(); log('adjust-preview', { look: used, w: src.width, ms: Math.round(ms), stages: r.timing }); markPicksStale(); } // (四選一 is not re-rendered on every slider move: it waits for 更新)
      } catch (err) { $('adjState').textContent = `預覽失敗：${err.message}`; }
    }
  } finally { PV.busy = false; }
}
// ---- 播放預覽: the video plays and each shown frame is graded with the current settings (colour, effects), so effects
// are judged in motion instead of on one frame. A frame is rendered only when the previous one is back (frames in
// between are skipped: the picture keeps up with the video), at most 640 px wide. AI segmentation runs per frame when
// the look needs it (skin protection, AI regions); a painted blur selection belongs to the representative frame.
// The timeline under the preview shows where it is and can be dragged (playing or paused); ❚❚ keeps the frame it
// stopped on (PLAY.viewT) and sliders then re-render that frame; 代表影格 returns to the representative frame (the
// one the settings were fitted on and the masks are painted on).
var PLAY = { on: false, busy: false, pending: false, n: 0, t0: 0, ms: 0, viewT: null, drag: false }; // (var: stopPlay may be called by showPage while the page starts)
function playSide() { return Math.min(640, PV_SIDE[PV.q]); }
const fmtT = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`;
function syncTimeline(t) {
  const m = S.video.meta; if (!m || !m.duration) return;
  if (!PLAY.drag) $('pvTime').value = Math.round(t / m.duration * 1000);
  $('pvTimeLbl').textContent = `${fmtT(t)} / ${fmtT(m.duration)}`;
}
/** grade the frame the video shows now (playing, or paused on PLAY.viewT); one at a time, the latest wins */
async function videoFrame() {
  const v = S.video.meta && S.video.meta.video;
  if (!v || (!PLAY.on && PLAY.viewT === null)) return;
  if (PLAY.busy) { if (!PLAY.on) PLAY.pending = true; return; }
  PLAY.busy = true;
  try {
    if (!PLAY.on && Math.abs(v.currentTime - PLAY.viewT) > 0.02) await seek(v, PLAY.viewT); // (another read of the video moved it)
    const m = S.video.meta, { w, h } = fitSize(m.width, m.height, playSide());
    const c = PLAY.canvas && PLAY.canvas.width === w && PLAY.canvas.height === h ? PLAY.canvas : (PLAY.canvas = canvas2d(w, h));
    const ctx = c.getContext('2d', { willReadFrequently: true }); ctx.drawImage(v, 0, 0, w, h);
    const src = ctx.getImageData(0, 0, w, h), t = v.currentTime, t0 = performance.now();
    const intent = S.intent.confirmed ? serializeIntent(S.intent.confirmed) : null, look = previewLook();
    let seg = null;
    if (S.ai.status === 'ready' && lookNeedsSeg(look, S.intent.confirmed ? effectiveRegions(S.intent.confirmed) : [])) {
      const sc = canvas2d(256, maskH()); sc.getContext('2d').drawImage(c, 0, 0, sc.width, sc.height);
      seg = segment(restoreCanvas(sc, currentNorm()), MASK_W, maskH());
    }
    const depth = needsDepthFor(look) ? await depthOf(c, w, h) : null;
    await S.grade.ready;
    const r = await engine.render(src, { intent, seg, depth, grade: true, look, draft: false });
    if ((!PLAY.on && PLAY.viewT === null) || $('adjustCard').hidden) return;
    const ms = performance.now() - t0;
    S.preview = { src, out: r.imageData, w, h, draft: false, ms, t }; // (the split compares this frame, original | graded)
    drawAdj(); syncTimeline(t);
    if (PLAY.on) {
      PLAY.n++; PLAY.ms = PLAY.ms ? PLAY.ms * 0.8 + ms * 0.2 : ms;
      const fps = PLAY.n / ((performance.now() - PLAY.t0) / 1000);
      $('pvInfo').textContent = `播放中 ${w}×${h}・每格 ${Math.round(PLAY.ms)} ms・約 ${fps.toFixed(1)} 格/秒${seg ? '・含 AI 人物偵測' : ''}`;
      log('play-frame', { t: +t.toFixed(3), ms: Math.round(ms), seg: !!seg });
    } else {
      $('pvInfo').textContent = `${fmtT(t)} 的畫面 ${w}×${h}・${Math.round(ms)} ms（按「代表影格」回到調色用的那一格）`;
      log('view-frame', { t: +t.toFixed(3), ms: Math.round(ms) });
    }
  } catch (err) { $('adjState').textContent = `播放預覽失敗：${err.message}`; stopPlay(); }
  finally {
    PLAY.busy = false;
    if (PLAY.pending && !PLAY.on) { PLAY.pending = false; videoFrame(); }
  }
}
function playLoop() {
  const v = S.video.meta && S.video.meta.video;
  if (!PLAY.on || !v) return;
  const next = () => { if (!PLAY.on) return; if (v.requestVideoFrameCallback) v.requestVideoFrameCallback(tick); else requestAnimationFrame(tick); };
  const tick = () => { if (!PLAY.on) return; videoFrame(); next(); };
  next();
}
async function startPlay() {
  const v = S.video.meta && S.video.meta.video;
  if (PLAY.on || !v || isPhoto() || !S.rep) return;
  stopBlurPainting();
  const from = PLAY.viewT !== null ? PLAY.viewT : S.rep.t; // (from the frame on screen)
  PLAY.on = true; PLAY.n = 0; PLAY.ms = 0; PLAY.t0 = performance.now(); PLAY.viewT = null;
  $('pvPlay').setAttribute('aria-pressed', 'true'); $('pvPlay').textContent = '❚❚ 暫停'; $('pvRep').hidden = false;
  v.muted = true; v.loop = true;
  try { if (Math.abs(v.currentTime - from) > 0.05) v.currentTime = from; await v.play(); }
  catch (err) { $('adjState').textContent = `不能播放影片：${err.message}`; stopPlay(true); return; }
  log('play-start', { t: v.currentTime });
  playLoop();
}
/** pause: stay on the frame it stopped on (toRep: back to the representative frame, e.g. before the lighting dialog) */
function stopPlay(toRep = true) {
  if (!PLAY) return;
  const v = S.video.meta && S.video.meta.video;
  if (PLAY.on) {
    PLAY.on = false;
    if (v) { v.pause(); v.loop = false; }
    $('pvPlay').setAttribute('aria-pressed', 'false'); $('pvPlay').textContent = '▶ 播放';
    log('play-stop', { frames: PLAY.n });
    if (!toRep && v) { PLAY.viewT = v.currentTime; videoFrame(); return; }
  }
  if (toRep && PLAY.viewT !== null) PLAY.viewT = null;
  if (toRep) { $('pvRep').hidden = true; if (S.rep && !$('adjustCard').hidden && S.preview && S.preview.t !== undefined) { clearTimeout(PV.finalTimer); requestRender(false); } if (S.rep) syncTimeline(S.rep.t); }
}
$('pvPlay').onclick = () => (PLAY.on ? stopPlay(false) : startPlay());
$('pvRep').onclick = () => { stopPlay(true); log('view-rep'); };
// the timeline: dragging seeks (playing: it plays on from there; paused: that frame is graded and shown)
$('pvTime').addEventListener('input', async (e) => {
  const v = S.video.meta && S.video.meta.video, m = S.video.meta;
  if (!v || !m || !m.duration || isPhoto() || !S.rep) return;
  PLAY.drag = true;
  const t = Math.min(m.duration - 0.01, e.target.value / 1000 * m.duration);
  $('pvTimeLbl').textContent = `${fmtT(t)} / ${fmtT(m.duration)}`;
  if (PLAY.on) { v.currentTime = t; return; }
  stopBlurPainting();
  PLAY.viewT = t; $('pvRep').hidden = false;
  try { await seek(v, t); } catch { return; }
  if (PLAY.viewT === t) videoFrame();
});
$('pvTime').addEventListener('change', () => { PLAY.drag = false; });

function drawAdj() {
  $('pvPlay').hidden = $('pvTimeline').hidden = !S.video.meta || isPhoto() || !S.rep;
  if (S.rep && !PLAY.on && PLAY.viewT === null && !$('pvTimeline').hidden) syncTimeline(S.rep.t);
  const p = S.preview;
  if (!S.rep || !p) return;
  const editing = blurEditing(), c = $('adjCanvas'), w = p.w, h = p.h, x = editing ? 0 : Math.round($('adjWipe').value / 100 * w); // editing: the whole result
  c.width = w; c.height = h;
  const ctx = c.getContext('2d');
  ctx.putImageData(p.src, 0, 0);
  ctx.putImageData(p.out, 0, 0, x, 0, w - x, h);
  if ($('pvBw').checked) { // 黑白檢查: both sides as perceived lightness (OKLab L), the painter's value check; display only
    const id = ctx.getImageData(0, 0, w, h), d = id.data, o = [0, 0, 0];
    for (let i = 0; i < d.length; i += 4) { srgbToOklab(d[i] / 255, d[i + 1] / 255, d[i + 2] / 255, o); const v = oklabToSrgb(o[0], 0, 0)[0] * 255; d[i] = d[i + 1] = d[i + 2] = v; }
    ctx.putImageData(id, 0, 0);
  }
  if (!editing) { ctx.fillStyle = '#fff'; ctx.fillRect(x - 1, 0, 2, h); }
  drawBlurOverlay(ctx, w, h);
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
// 參考圖 picture-in-picture on the grading preview: click toggles small / large, drag moves it (kept inside the
// picture area, remembered as a fraction so it stays put when the panel resizes)
const PIP = { x: null, y: null, drag: null };
function syncRefPip() {
  const on = S.ref && S.ref.status === 'ready' && $('pvRef').checked && !!$('refPipImg').getAttribute('src') && !$('adjCanvas').hidden && !blurEditing(); // (hidden while painting a blur selection)
  $('refPip').hidden = !on;
  if (on && PIP.x !== null) placeRefPip();
}
function placeRefPip() {
  const box = $('refPip').parentElement.getBoundingClientRect(), el = $('refPip'), w = el.offsetWidth, h = el.offsetHeight;
  const x = Math.min(Math.max(0, PIP.x * box.width), Math.max(0, box.width - w)), y = Math.min(Math.max(0, PIP.y * box.height), Math.max(0, box.height - h));
  el.style.left = `${x}px`; el.style.top = `${y}px`; el.style.right = 'auto';
}
{
  const el = $('refPip');
  el.addEventListener('pointerdown', (e) => { const r = el.getBoundingClientRect(); PIP.drag = { id: e.pointerId, dx: e.clientX - r.left, dy: e.clientY - r.top, x0: e.clientX, y0: e.clientY, moved: false }; el.setPointerCapture(e.pointerId); });
  el.addEventListener('pointermove', (e) => {
    const d = PIP.drag; if (!d || d.id !== e.pointerId) return;
    if (!d.moved && Math.hypot(e.clientX - d.x0, e.clientY - d.y0) < 4) return;
    d.moved = true; el.classList.add('dragging');
    const box = el.parentElement.getBoundingClientRect();
    PIP.x = (e.clientX - d.dx - box.left) / box.width; PIP.y = (e.clientY - d.dy - box.top) / box.height; placeRefPip();
  });
  const end = (e) => { const d = PIP.drag; if (!d || d.id !== e.pointerId) return; PIP.drag = null; el.classList.remove('dragging');
    if (!d.moved) { el.classList.toggle('big'); if (PIP.x !== null) placeRefPip(); log('ref-pip', { big: el.classList.contains('big') }); } };
  el.addEventListener('pointerup', end); el.addEventListener('pointercancel', end);
  window.addEventListener('resize', () => { if (!el.hidden && PIP.x !== null) placeRefPip(); });
  $('pvRef').addEventListener('change', () => { syncRefPip(); try { localStorage.setItem('zhaoyang-refpip', $('pvRef').checked ? '1' : '0'); } catch {} });
  try { if (localStorage.getItem('zhaoyang-refpip') === '0') $('pvRef').checked = false; } catch {}
}
$('pvBw').addEventListener('change', (e) => { log('bw-check', { on: e.target.checked }); drawAdj(); pipBw(); });
// the reference picture-in-picture follows 黑白檢查: the same perceived lightness (OKLab L) as the preview, so the
// values of the reference and the video can be compared side by side; made once per reference picture
const PIPBW = { for: null, url: null };
function setPipSrc(url) {
  const i = $('refPipImg'); i.dataset.colour = url || '';
  if (url) i.src = url; else i.removeAttribute('src');
  syncRefPip(); pipBw();
}
async function pipBw() {
  const i = $('refPipImg'), colour = i.dataset.colour;
  if (!colour) return;
  if (!$('pvBw').checked) { if (i.src !== colour) i.src = colour; i.dataset.bw = ''; return; }
  if (PIPBW.for !== colour) {
    const bm = await createImageBitmap(await (await fetch(colour)).blob()), k = Math.min(1, 720 / Math.max(bm.width, bm.height));
    const c = canvas2d(Math.round(bm.width * k), Math.round(bm.height * k)), x = c.getContext('2d');
    x.drawImage(bm, 0, 0, c.width, c.height);
    const id = x.getImageData(0, 0, c.width, c.height), d = id.data, o = [0, 0, 0];
    for (let q = 0; q < d.length; q += 4) { srgbToOklab(d[q] / 255, d[q + 1] / 255, d[q + 2] / 255, o); const v = oklabToSrgb(o[0], 0, 0)[0] * 255; d[q] = d[q + 1] = d[q + 2] = v; }
    x.putImageData(id, 0, 0);
    const blob = await new Promise((res) => (c.convertToBlob ? c.convertToBlob({ type: 'image/png' }).then(res) : c.toBlob(res, 'image/png')));
    if (PIPBW.url) URL.revokeObjectURL(PIPBW.url);
    PIPBW.for = colour; PIPBW.url = URL.createObjectURL(blob);
  }
  if ($('pvBw').checked && i.dataset.colour === colour) { i.src = PIPBW.url; i.dataset.bw = '1'; }
}
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

function showAdjustCard() { setAdjustVisible(true); syncAdjustControls(); scheduleAdjPreview(); renderCands(); renderMoodThumbs(); }

async function startProcessing() {
  stopPlay(); // (playing uses the video these need to seek)
  const intent = S.intent.confirmed;
  if (!intent) return;
  const token = ++S.job.token;
  PROC.prevApplied = S.look.applied; PROC.prevPhase = S.phase;
  stopBlurPainting(); // (matching / exporting: done painting; back on 調色預覽 the 原始／調色 split shows again)
  openProc();
  await S.grade.ready;
  // the automatic look includes the local grade: wait for it (bounded) if it is still being fitted
  if (S.look.auto && S.ai.status === 'ready') await Promise.race([fitLocalGrade(), new Promise((r) => setTimeout(r, 20000))]);
  if (token !== S.job.token) return; // left the progress window meanwhile
  S.look.applied = JSON.parse(JSON.stringify(S.look.draft)); // deep: pro controls edit look.prim in place
  setPhase('processing');
  updateAdjState();
  $('progressWrap').hidden = false;
  for (const id of ['exportBtn', 'reprocess', 'editIntent', 'syncPlay', 'cmpTime']) $(id).disabled = true;
  $('photoNav').hidden = true;
  const regs = effectiveRegions(intent);
  // AI segmentation per frame: for AI regions, and for skin protection of the pro grade
  const needsSeg = lookNeedsSeg(S.look.applied, regs);
  if (needsSeg) await initSegmenter();
  // 對焦（景深）: the AI depth of every frame
  const needsDepth = needsDepthFor(S.look.applied);
  if (needsDepth) { $('progressText').textContent = '載入 AI 深度模型（約 27 MB，只在第一次）…'; try { await initDepth(); } catch { /* (the processing below reports it as its error) */ } }
  if (token !== S.job.token) return;
  if (isPhoto()) { $('progressBar').style.width = '0'; $('progressText').textContent = '處理照片中…'; return processPhotoJob(intent, token, needsSeg, needsDepth); }
  // while processing, only the hidden work video decodes the source: the upload preview and the comparison
  // players are unloaded (on iPad every <video> holding a 4K clip takes a hardware decoder and a lot of memory, and
  // four at once gave "Decoder failure" / "out of memory"). They are loaded again afterwards; nothing about the
  // processing or its quality changes.
  for (const id of ['videoPreview', 'origVideo', 'outVideo']) unloadVideo($(id));
  await primeVideo(S.video.meta.video); // (iPad: the decoder back to the work video, should another video have taken it)
  try {
    const res = await processVideo({
      engine, meta: S.video.meta, intentJson: serializeIntent(intent), needsSeg, segmentFn: segmentRestored, segPrepare: (c) => restoreCanvas(c), maskH: intent.maskH, needsDepth, depthFn: depthOf,
      look: S.look.applied, isCancelled: () => token !== S.job.token, keepFrames: window.__keepFrames || [], stream: !window.__noStream, segWorker: !window.__noSegWorker,
      onStage: (text) => { if (token === S.job.token) { $('progressText').textContent = text; log('processing-stage', { text }); } },
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
    + (res.audio.kept ? `音訊：已保留原片音軌（${res.audio.codec}，未重新編碼）。` : `音訊：${res.audio.reason}。`)
    + (res.timing && res.timing.segInfo ? `AI 人物偵測：${res.timing.segInfo.delegate === 'GPU' ? '顯示卡（GPU）' : '處理器（CPU）'}${res.timing.segInfo.cpuMs ? `（實測每格 CPU ${Math.round(res.timing.segInfo.cpuMs)} ms${res.timing.segInfo.gpuMs ? `、GPU ${Math.round(res.timing.segInfo.gpuMs)} ms` : '、GPU 不可用'}）` : ''}。` : '')
    + dofText(res.timing, res.frames)
    + gradeText(res.timing && res.timing.gradeInfo)
    + (res.timing && res.timing.total ? `處理時間 ${(res.timing.total / 1000).toFixed(1)} 秒。` : '')
    + timeText(res.timing, res.frames);
    log('processing-done', { frames: res.frames, hash: res.intentHash, size: res.blob.size, audio: res.audio, timing: res.timing });
    setPhase('done');
    updateAdjState();
    for (const id of ['exportBtn', 'reprocess', 'editIntent', 'syncPlay', 'cmpTime']) $(id).disabled = false;
    $('progressWrap').hidden = true;
    $('tabResult').disabled = false; $('tabWipe').disabled = false;
    showView('result');
    drawWipe();
    finishProc();
  } catch (err) {
    if (token !== S.job.token) return;
    log('processing-failed', { message: err.message });
    closeProc();
    if (S.output && S.output.url) { $('origVideo').src = S.video.meta.url; $('outVideo').src = S.output.url; } // the previous result stays comparable
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
  // the players were just (re)loaded after processing: wait until each has a picture before seeking it (seeking
  // before that could wait forever); a comparison that cannot be drawn is skipped, never an error
  const ready = (v) => (v.readyState >= 2 ? Promise.resolve() : new Promise((res) => { const done = () => { clearTimeout(tm); res(); }; const tm = setTimeout(done, 15000); v.addEventListener('loadeddata', done, { once: true }); }));
  try { await Promise.all([a, b].map(async (v) => { await ready(v); await seek(v, Math.max(0, t - 1e-3)); })); } catch (err) { log('wipe-failed', { message: err.message }); return; }
  const c = $('wipeCanvas'), w = S.output.width, h = S.output.height, x = Math.round($('wipePos').value / 100 * w);
  c.width = w; c.height = h;
  const ctx = c.getContext('2d');
  ctx.drawImage(a, 0, 0, w, h);
  ctx.save(); ctx.beginPath(); ctx.rect(x, 0, w - x, h); ctx.clip(); ctx.drawImage(b, 0, 0, w, h); ctx.restore();
  ctx.fillStyle = '#fff'; ctx.fillRect(x - 1, 0, 2, h);
}
$('cmpTime').addEventListener('input', drawWipe);
$('wipePos').addEventListener('input', drawWipe);

// ---- 匯出: one button. The result is up to date → download it now; otherwise match the colours and effects first
// (progress window), then download. (Two buttons — 套用並處理 and 匯出 — led to matching again by mistake.)
const PROC = { exportAfter: false, viewAfter: null, prevApplied: null, prevPhase: null, doneWhileAsking: false };
// iPad / iPhone: a download not started by a tap may be blocked, so the window offers a 下載 button when done
function needsTapToSave() { return /iPad|iPhone|iPod/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1); }
function upToDate() { return !!S.output && !!S.look.applied && !!S.intent.confirmed && JSON.stringify(S.look.applied) === JSON.stringify(S.look.draft) && S.output.confirmedHash === hashIntent(S.intent.confirmed); }
const procDlg = $('procDialog');
function openProc() {
  $('procMain').hidden = false; $('procConfirm').hidden = true; $('procDoneRow').hidden = true;
  $('procTitle').textContent = '顏色匹配中'; $('procMsg').textContent = '顏色匹配中，請勿退出';
  $('procBar').style.width = '0'; $('procText').textContent = '準備中…';
  if (!procDlg.open) procDlg.showModal();
  log('proc-window', { open: true, exportAfter: PROC.exportAfter });
}
function closeProc() { if (procDlg.open) procDlg.close(); PROC.exportAfter = false; PROC.viewAfter = null; PROC.doneWhileAsking = false; }
async function finishProc() {
  // finished while the leave question is open: wait for the answer (否 → download, 是 → back to grading, no download)
  if (procDlg.open && !$('procConfirm').hidden) { PROC.doneWhileAsking = true; return; }
  if (!PROC.exportAfter) { const v = PROC.viewAfter; closeProc(); if (v) showView(v); return; } // (matched to look at the result: no download)
  PROC.exportAfter = false;
  if (needsTapToSave()) { // done: a tap starts the download
    $('procTitle').textContent = '顏色匹配完成'; $('procMsg').textContent = '按「下載」儲存結果。';
    $('procBar').style.width = '100%'; $('procText').textContent = '完成';
    $('procSave').textContent = $('exportBtn').textContent.replace('匯出', '下載');
    $('procDoneRow').hidden = false;
    return;
  }
  closeProc();
  await exportOutput();
}
$('procSave').onclick = async () => { closeProc(); await exportOutput(); };
// the window mirrors the page's progress bar and text
new MutationObserver(() => { $('procText').textContent = $('progressText').textContent; }).observe($('progressText'), { childList: true, characterData: true, subtree: true });
new MutationObserver(() => { $('procBar').style.width = $('progressBar').style.width; }).observe($('progressBar'), { attributes: true, attributeFilter: ['style'] });
function askLeave() {
  if (!$('procDoneRow').hidden) { closeProc(); return; } // finished: closing only skips the download
  $('procMain').hidden = true; $('procConfirm').hidden = false;
  log('proc-leave-asked');
}
$('procClose').onclick = askLeave;
procDlg.addEventListener('cancel', (e) => { e.preventDefault(); if ($('procConfirm').hidden) askLeave(); else $('procNo').click(); }); // Esc
$('procNo').onclick = () => {
  $('procConfirm').hidden = true; $('procMain').hidden = false; log('proc-leave', { leave: false });
  if (PROC.doneWhileAsking) { PROC.doneWhileAsking = false; finishProc(); }
};
$('procYes').onclick = () => { // stop matching; back to grading with the previous result (if any) as it was
  if (PROC.doneWhileAsking) { log('proc-leave', { leave: true, finished: true }); closeProc(); showView('preview'); return; } // finished meanwhile: keep the new result, skip the download
  S.job.token++;
  S.look.applied = PROC.prevApplied;
  log('proc-leave', { leave: true });
  closeProc();
  $('progressWrap').hidden = true;
  if (S.output && S.output.kind !== 'image' && S.output.url) { $('origVideo').src = S.video.meta.url; $('outVideo').src = S.output.url; }
  for (const id of ['exportBtn', 'reprocess', 'editIntent', 'syncPlay', 'cmpTime']) $(id).disabled = false;
  setPhase(S.output ? 'done' : 'confirmed');
  updateAdjState();
  showView('preview');
};
$('exportBtn').onclick = () => {
  if (S.phase === 'processing' || !S.intent.confirmed) return;
  if (upToDate()) return exportOutput();
  PROC.exportAfter = true;
  startProcessing();
};
async function exportOutput() {
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
}

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
// ---- the control panel's three tabs. Opening a category (by code or a test) brings its tab forward.
function showTab(tab) {
  const card = $('adjustCard');
  if (card.dataset.tab === tab) return;
  card.dataset.tab = tab;
  for (const b of card.querySelectorAll('.panel-tabs [data-tab]')) b.setAttribute('aria-selected', String(b.dataset.tab === tab));
  log('panel-tab', { tab });
  syncBlurControls(); if (S.preview) { drawAdj(); if ($('blurShow').checked) scheduleAdjPreview(); } // leaving / returning to 效果 ends / resumes painting a blur selection (and shows the reference picture again); the mask stripes follow
}
for (const b of document.querySelectorAll('#adjustCard .panel-tabs [data-tab]')) b.addEventListener('click', () => showTab(b.dataset.tab));
// (a MutationObserver rather than 'toggle': it runs right after the change, before the next event, so whatever opened
// the category can use it at once)
const tabObserver = new MutationObserver((list) => { for (const m of list) if (m.target.open) showTab(m.target.dataset.group); });
for (const d of document.querySelectorAll('#adjustCard > details[data-group]')) tabObserver.observe(d, { attributes: true, attributeFilter: ['open'] });
function unloadVideo(v) { v.pause(); if (v.getAttribute('src')) { v.removeAttribute('src'); v.load(); } }
function showPage(p) {
  if (p !== 'work') stopPlay(true);
  S.page = p;
  if (p === 'upload' && S.video.meta && !isPhoto() && !$('videoPreview').getAttribute('src')) $('videoPreview').src = S.video.meta.url; // (unloaded while processing)
  if (p === 'work') unloadVideo($('videoPreview')); // (hidden on the grading page: an iPad decodes few videos at once — only the work video needs one; reloaded on the upload page)
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
  $('startHint').textContent = loading ? '素材載入中…'
    : ready ? (moodOnly() ? '沒有參考圖片：開始後可以在工作區選擇氛圍色調，或用滑桿自行調色。' : '素材已就緒，可以開始調色。')
    : '還需要要調色的影片或照片（參考圖片可選）。';
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
  if (S.refVideo) { URL.revokeObjectURL(S.refVideo.url); S.refVideo.video.remove(); S.refVideo = null; } $('refVideoBox').hidden = true;
  $('refInput').value = ''; $('refPreview').hidden = true; $('refPreview').removeAttribute('src'); $('refInfo').textContent = '';
  setPipSrc(null);
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
  syncRefPip();
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
  document.querySelector('.stage-col').dataset.view = name; // (upright phone: the result views get a taller picture)
  if (name !== 'result') for (const id of ['origVideo', 'outVideo']) $(id).pause();
  if (name === 'wipe') drawWipe();
  if (name === 'preview') drawAdj();
  syncRefPip();
  syncStaleNote();
}
// upright phone: ⤢ enlarges the preview, ⋯ shows the preview options (quality, fast preview, b/w check, reference)
$('pvZoom').onclick = () => { const on = document.body.classList.toggle('pv-big'); $('pvZoom').setAttribute('aria-pressed', String(on)); $('pvZoom').setAttribute('aria-label', on ? '縮小預覽' : '放大預覽'); requestAnimationFrame(() => drawAdj()); };
$('pvMore').onclick = () => { const on = $('pvRow').classList.toggle('open'); $('pvMore').setAttribute('aria-expanded', String(on)); };
for (const [k, [tab]] of Object.entries(VIEWS)) $(tab).onclick = () => (k === 'preview' ? showView(k) : askView(k));
// 處理結果 / 結果分割比較 show the processed file: without one for the current settings, ask to match first
// (好 → the progress window, then that view; no download)
function askView(k) {
  if (S.phase === 'processing' || !S.intent.confirmed) return;
  if (upToDate()) { showView(k); return; }
  PROC.viewAsk = k;
  $('viewAskText').textContent = `如果想要預覽${isPhoto() ? '照片' : '影片'}效果，需要先進行顏色和效果匹配`;
  $('viewAskDialog').showModal();
  log('view-ask', { view: k });
}
$('viewAskBack').onclick = () => { $('viewAskDialog').close(); log('view-ask-answer', { ok: false }); };
$('viewAskOk').onclick = () => { $('viewAskDialog').close(); log('view-ask-answer', { ok: true }); PROC.exportAfter = false; PROC.viewAfter = PROC.viewAsk; startProcessing(); };
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
/** 調色計算：顯示卡／處理器 (the worker's measured choice), for the result info */
/** 對焦深度：顯示卡／處理器 (the depth worker's measured choice) and the time per frame, for the result info */
function dofText(timing, frames = 0) {
  const i = depthInfo();
  if (!needsDepthFor(S.look.applied || {}) || !i) return '';
  const per = timing && timing.depth && frames ? `，每格約 ${Math.round(timing.depth / frames)} ms` : '';
  return `AI 深度計算：${i.ep === 'gpu' ? '顯示卡' : `處理器${i.why ? `（${i.why}）` : ''}`}${per}。`;
}
// 每格時間: where the export's time goes, per frame (ms, averaged). The AI steps run alongside the reading of the next
// frames, so the parts add up to more than the real time per frame; the largest part is what to make faster.
const STAGE_NAMES = { masks: '局部遮罩', layers: '膚色／局部層', lut: '色彩表', grade: '調色', 'layers-depth': '分層匹配', blur: '背景模糊', dof: '景深', effects: '效果' };
function timeText(t, frames) {
  if (!t || !t.total || !frames) return '';
  const per = (v) => Math.round(v / frames), parts = [];
  const how = t.decoder !== 'webcodecs' ? '逐格跳轉' : t.decoderCheck && !t.decoderCheck.ok ? `解碼器的畫面與逐格跳轉不同（差 ${t.decoderCheck.diff}），改用逐格跳轉`
    : t.decoderFallback ? `解碼器在第 ${t.decoderFallback.at + 1} 格出錯，之後改用逐格跳轉` : '解碼器';
  parts.push(`讀取影格 ${per(t.grab)} ms（${how}）`);
  if (t.seg) parts.push(`AI 人物偵測 ${per(t.seg)} ms`);
  if (t.depth) parts.push(`AI 深度 ${per(t.depth)} ms`);
  const st = Object.entries(t.worker || {}).filter(([, v]) => v / frames >= 1).map(([k, v]) => `${STAGE_NAMES[k] || k} ${per(v)}`);
  parts.push(`調色引擎 ${per(t.render)} ms${st.length ? `（${st.join('、')}）` : ''}`);
  parts.push(`等待調色完成 ${per(t.wait)} ms`, `編碼 ${per(t.encode)} ms`);
  return `每格時間：實際每格 ${per(t.total)} ms；其中${parts.join('、')}（AI 步驟與讀取同時進行，所以加起來會多於實際時間）。`;
}
function gradeText(g) {
  if (!g || !g.mode) return '';
  const one = (x) => `${x.mode === 'gpu' ? `顯示卡（${x.renderer || 'GPU'}）` : `處理器${x.why ? `（${x.why}）` : ''}`}${x.cpuMs != null ? `，實測每格處理器 ${Math.round(x.cpuMs)} ms／顯示卡 ${Math.round(x.gpuMs)} ms` : ''}`;
  return `調色計算：${one(g)}。${g.fx && g.fx.mode ? `效果計算：${one(g.fx)}。` : ''}`;
}
async function processPhotoJob(intent, token, needsSeg, needsDepth = false) {
  const m = S.video.meta, n = m.count, items = [], names = new Set();
  let first = null;
  try {
    for (let i = 0; i < n; i++) { // one after another (in parallel they only compete for the CPU and memory)
      if (token !== S.job.token) return;
      $('progressBar').style.width = `${(i / n) * 100}%`; $('progressText').textContent = `${i} / ${n} 張`;
      const res = await processPhoto({ engine, meta: m, index: i, keepPixels: false, intentJson: serializeIntent(intent), needsSeg, segmentFn: segmentRestored, needsDepth, depthFn: depthOf, maskH: intent.maskH, look: S.look.applied });
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
      + `；局部調整區域：${first.jobRegions.length ? first.jobRegions.map((r) => `${KINDS[r.kind].label} ${r.ev > 0 ? '+' : ''}${r.ev.toFixed(2)} 檔`).join('、') : '無'}。`
      + dofText(null)
      + gradeText(await engine.call({ type: 'gpu-info' }).then((r) => r.info).catch(() => null));
    log('processing-done', { frames: n, photo: true, count: n, hash: first.intentHash, size: Math.round(mb * 1e6) });
    setPhase('done');
    updateAdjState();
    for (const id of ['exportBtn', 'reprocess', 'editIntent']) $(id).disabled = false;
    $('progressWrap').hidden = true;
    $('tabResult').disabled = false; $('tabWipe').disabled = false;
    showPhoto(0);
    showView('result');
    finishProc();
  } catch (err) {
    if (token !== S.job.token) return;
    log('processing-failed', { message: err.message });
    closeProc();
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
// switching is animated like iPadOS's own screen changes: the whole page as it was cross-dissolves into the whole page
// as it will be (view transitions: two complete pictures, so nothing passes through muddy in-between colours), 1 s
// with a gentle ease in and out. Browsers without view transitions (Safari before 18) fade the colours and the
// gradient layer instead. A dissolve moves nothing, so it also runs with 減少動態效果 (Reduce Motion) on — iPadOS
// itself uses dissolves then; skipping it there made the switch instant on the user's iPad.
const THEME_MS = 1000;
let themeTimer = 0;
function switchTheme(choice) {
  const root = document.documentElement;
  if (document.startViewTransition) {
    root.classList.add('theme-vt');
    const t = document.startViewTransition(() => applyTheme(choice));
    t.finished.finally(() => root.classList.remove('theme-vt'));
    return 'dissolve';
  }
  root.classList.add('theme-anim');
  applyTheme(choice);
  clearTimeout(themeTimer); themeTimer = setTimeout(() => root.classList.remove('theme-anim'), THEME_MS + 100);
  return 'css';
}
for (const b of document.querySelectorAll('[data-theme-choice]')) b.onclick = () => {
  const how = switchTheme(b.dataset.themeChoice);
  try { localStorage.setItem(THEME_KEY, b.dataset.themeChoice); } catch {}
  log('theme', { theme: b.dataset.themeChoice, transition: how });
};
applyTheme((() => { try { return localStorage.getItem(THEME_KEY); } catch { return null; } })());

setAdjustVisible(false);
updateStart();
setPhase('idle');

// ---- on opening the site: ask to download the AI depth model (對焦（景深）, 依深度分層匹配) now, in the background, so it is
// there when used (~25–30 MB, compressed; kept in the browser afterwards). Only downloaded and stored: the model is
// loaded into memory when a feature needs it. Not asked when it is already stored, nor again in this session after
// 稍後. (Automated browsers are not asked — the dialog would cover the page in every test — unless a test sets
// localStorage zhaoyang-depth-prompt = 'ask'.)
async function askDepthDownload() {
  let force = false; try { force = localStorage.getItem('zhaoyang-depth-prompt') === 'ask'; } catch { /* (no storage) */ }
  if (navigator.webdriver && !force) return;
  try { if (sessionStorage.getItem('zhaoyang-depth-later')) return; } catch { /* (no storage) */ }
  if (await depthStored()) { log('depth-already-stored'); return; }
  const d = $('depthDialog');
  d.showModal(); log('depth-prompt');
  $('depthDlLater').onclick = () => { try { sessionStorage.setItem('zhaoyang-depth-later', '1'); } catch { /* */ } d.close(); log('depth-prompt-later'); };
  $('depthDlHide').onclick = () => { d.close(); log('depth-download-background'); };
  $('depthDlOk').onclick = async () => {
    $('depthDlAsk').hidden = true; $('depthDlBg').hidden = false; $('depthDlProg').hidden = false;
    const off = onDepthProgress((st, m) => { $('depthDlText').textContent = st; if (m && m.total) $('depthDlBar').style.width = `${Math.min(100, m.loaded / m.total * 100)}%`; });
    log('depth-download-allowed');
    try {
      const t0 = performance.now(), st = await prefetchDepth(), secs = (performance.now() - t0) / 1000;
      const name = { 'runtime-wasm': '執行程式', download: '模型' };
      const parts = st.map((x) => `${name[x.what] || x.what} ${(x.bytes / 1e6).toFixed(1)} MB ${(x.ms / 1000).toFixed(1)} 秒${x.cached ? '（已儲存）' : `（${(x.bytes / 1e6 / Math.max(0.05, x.ms / 1000)).toFixed(1)} MB/秒${x.parts > 1 ? `，分 ${x.parts} 段同時下載` : ''}）`}`);
      $('depthDlBar').style.width = '100%'; $('depthDlText').textContent = `已下載並儲存：${parts.join('、')}；共 ${secs.toFixed(1)} 秒。`;
      log('depth-downloaded', { stats: st, secs: +secs.toFixed(1) });
      setTimeout(() => { if (d.open) d.close(); }, 6000); // (long enough to read how long it took)
    } catch (err) { $('depthDlText').textContent = `下載失敗：${err.message}（用到時會再試）`; log('depth-download-failed', { message: err.message }); }
    finally { off(); }
  };
}
askDepthDownload();
