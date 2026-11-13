// GPU (WebGL2) version of the spatial effects after the grade: 色散 (fringe.js applyFringe), 朦朧 / 柔光 / 暗角 (render.js applyEffects) and 膠片質感
// 柔化 / 光暈 / 顆粒 (film.js applyFilm). The same formulas and the same order, with the frame kept in 8-bit between the
// steps as on the processor, so the result matches within rounding (test/e2e/gpufx.cjs).
//
// The work done per full-resolution pixel runs on the graphics card: averaging the frame onto the small grids, the
// composites, the small blur of 柔化, the grain. Only the small grids (~320 / ~640 px) come back to the processor for
// their wide blur (the same box blurs as before — a few ms) and go back up as textures.
//
// gpuEffects() → renderer or null (no WebGL2 / no float render targets). renderer.apply(img, w, h, fx, hazeMap, frame)
// works in place on RGBA like applyEffects + applyFilm and returns img. renderer.lost: the browser took the context back.
import { effectsPlan, veilOf, gridDims, blurGrid } from './render.js';
import { hasFilm, grainGrid, blurChannels } from './film.js';
import { fringePlan, fringeWeights, fringeStep, fringeMaxShift } from './fringe.js';

const VS = `#version 300 es
in vec2 p; void main() { gl_Position = vec4(p, 0.0, 1.0); }`;
const HEAD = `#version 300 es
precision highp float; precision highp int;
out vec4 o;
const vec3 LUM = vec3(0.2126, 0.7152, 0.0722);
float s2l(float c) { return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4); }
vec3 S2L(vec3 c) { return vec3(s2l(c.r), s2l(c.g), s2l(c.b)); }
float l2s(float c) { c = clamp(c, 0.0, 1.0); return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1.0 / 2.4) - 0.055; }
vec3 L2S(vec3 c) { return vec3(l2s(c.r), l2s(c.g), l2s(c.b)); }
float sm(float a, float b, float x) { float t = clamp((x - a) / (b - a), 0.0, 1.0); return t * t * (3.0 - 2.0 * t); }
const float KNEE = 0.8;
float rollOff(float y) { if (y <= KNEE) return y; float t = y - KNEE, r = 1.0 - KNEE; return KNEE + r * t / (t + r); }
`;
// the frame (linear light, weighted by luminance) averaged over f×f blocks: one grid texel per block
const DOWN = HEAD + `
uniform sampler2D img; uniform ivec2 size; uniform int f, mode, k;
float pl(ivec2 p) { return pow(max(dot(S2L(texelFetch(img, clamp(p, ivec2(0), size - 1), 0).rgb), LUM), 0.0), 1.0 / 3.0); }
void main() {
  ivec2 g = ivec2(gl_FragCoord.xy); vec3 acc = vec3(0.0); float n = 0.0;
  if (mode == 4) { // 色散: mean perceptual lightness and mean fine detail (fringe.js focusGrid)
    for (int dy = 0; dy < f; dy++) { int y = g.y * f + dy; if (y >= size.y) break;
      for (int dx = 0; dx < f; dx++) { int x = g.x * f + dx; if (x >= size.x) break;
        ivec2 p = ivec2(x, y); float v = pl(p), m = (pl(p - ivec2(k, 0)) + pl(p + ivec2(k, 0)) + pl(p - ivec2(0, k)) + pl(p + ivec2(0, k))) / 4.0;
        acc.r += v; acc.g += abs(v - m); n += 1.0; } }
    o = vec4(acc / max(n, 1.0), 1.0); return;
  }
  for (int dy = 0; dy < f; dy++) { int y = g.y * f + dy; if (y >= size.y) break;
    for (int dx = 0; dx < f; dx++) { int x = g.x * f + dx; if (x >= size.x) break;
      vec3 l = S2L(texelFetch(img, ivec2(x, y), 0).rgb); float Y = dot(l, LUM);
      if (mode == 0) acc += l * sm(0.2, 0.85, Y); else if (mode == 1) acc += l; else if (mode == 2) acc += l * sm(0.18, 0.75, Y); else acc.r += sm(0.45, 0.95, Y) * Y;
      n += 1.0; } }
  o = vec4(acc / max(n, 1.0), 1.0);
}`;
// 色散: red pushed outwards, blue inwards along the line from the centre, by the out-of-focus weight (fringe.js)
const FRINGE = HEAD + `
uniform sampler2D img, wt; uniform ivec2 size, gs; uniform int f; uniform float ms;
float bil(vec2 s, int c) { // bilinear on the 8-bit levels, as fringe.js
  s = clamp(s, vec2(0.0), vec2(size - 1)); ivec2 a = ivec2(floor(s)), b = min(size - 1, a + 1); vec2 t = s - vec2(a);
  float v00 = texelFetch(img, a, 0)[c], v01 = texelFetch(img, ivec2(b.x, a.y), 0)[c], v10 = texelFetch(img, ivec2(a.x, b.y), 0)[c], v11 = texelFetch(img, b, 0)[c];
  float top = v00 * 255.0 + (v01 - v00) * 255.0 * t.x, bot = v10 * 255.0 + (v11 - v10) * 255.0 * t.x;
  return top + (bot - top) * t.y;
}
void main() {
  ivec2 px = ivec2(gl_FragCoord.xy); vec4 s = texelFetch(img, px, 0); o = s;
  vec2 q = clamp((vec2(px) + 0.5) / float(f) - 0.5, vec2(0.0), vec2(gs - 1));
  ivec2 a = ivec2(floor(q)), b = min(gs - 1, a + 1); vec2 t = q - vec2(a);
  float w = mix(mix(texelFetch(wt, a, 0).r, texelFetch(wt, ivec2(b.x, a.y), 0).r, t.x), mix(texelFetch(wt, ivec2(a.x, b.y), 0).r, texelFetch(wt, b, 0).r, t.x), t.y);
  vec2 c = (vec2(size) - 1.0) / 2.0, d = vec2(px) - c; float r = length(d);
  if (r < 1.0) return;
  float sh = ms * w * (0.3 + 0.7 * r / length(c));
  if (sh < 0.05) return;
  vec2 u = d / r * sh;
  o = vec4(clamp(round(bil(vec2(px) + u, 0)), 0.0, 255.0) / 255.0, s.g, clamp(round(bil(vec2(px) - u, 2)), 0.0, 255.0) / 255.0, s.a);
}`;
// 朦朧 / 柔光 / 暗角 composite (render.js applyEffects, per pixel)
const FX = HEAD + `
uniform sampler2D img, gG, gH, gB, hz; uniform ivec2 size, gs; uniform int f, hasG, hasH, hasHz;
uniform float glow, haze, kSoft, kBloom, kVeil, bk, vig, v0, depth, hasAir, where; uniform vec3 veil, tm, air;
vec3 grid(sampler2D t, ivec2 px) { // bilinear, as render.js sampleGrid
  vec2 g = clamp((vec2(px) + 0.5) / float(f) - 0.5, vec2(0.0), vec2(gs - 1));
  ivec2 a = ivec2(floor(g)), b = min(gs - 1, a + 1); vec2 t2 = g - vec2(a);
  vec3 c0 = mix(texelFetch(t, a, 0).rgb, texelFetch(t, ivec2(b.x, a.y), 0).rgb, t2.x), c1 = mix(texelFetch(t, ivec2(a.x, b.y), 0).rgb, texelFetch(t, b, 0).rgb, t2.x);
  return mix(c0, c1, t2.y);
}
void main() {
  ivec2 px = ivec2(gl_FragCoord.xy);
  vec3 lin = S2L(texelFetch(img, px, 0).rgb);
  float h = haze;
  if (hasHz == 1) {
    ivec2 ms = textureSize(hz, 0);
    vec2 g = clamp((vec2(px) + 0.5) / vec2(size) * vec2(ms) - 0.5, vec2(0.0), vec2(ms - 1));
    ivec2 a = ivec2(floor(g)), b = min(ms - 1, a + 1); vec2 t = g - vec2(a);
    float pr = mix(mix(texelFetch(hz, a, 0).r, texelFetch(hz, ivec2(b.x, a.y), 0).r, t.x), mix(texelFetch(hz, ivec2(a.x, b.y), 0).r, texelFetch(hz, b, 0).r, t.x), t.y);
    float wk = where > 1.5 ? 1.0 - pr : where > 0.5 ? pr : 1.0;
    h = min(1.0, haze * wk * (1.0 - 0.85 * depth * pr) * (1.0 + 0.5 * depth * (1.0 - pr)));
  }
  if (hasH == 1) {
    vec3 bl = grid(gH, px);
    lin += min(1.0, h * kSoft) * (bl - lin);
    bl = grid(gB, px);
    vec3 bt = max(vec3(0.0), bl - 0.6 * lin);
    if (hasAir > 0.5) bt = air * dot(bt, vec3(0.2126, 0.7152, 0.0722));
    vec3 add = min(vec3(1.0), h * tm * (kBloom * bk * bt + kVeil * veil));
    lin = 1.0 - (1.0 - lin) * (1.0 - add);
  }
  if (hasG == 1) lin += glow * 1.2 * tm * grid(gG, px);
  if (vig != 0.0) {
    float m = sm(v0, v0 + 0.6, length((vec2(px) + 0.5) / vec2(size) * 2.0 - 1.0) / sqrt(2.0));
    lin *= vig > 0.0 ? 1.0 - vig * 0.8 * m : 1.0 - vig * 0.6 * m;
  }
  float Y = dot(lin, LUM);
  if (Y > KNEE) lin *= rollOff(Y) / Y;
  float mx = max(lin.r, max(lin.g, lin.b));
  if (mx > 1.0) { float Yn = min(Y, 1.0), t = (mx - 1.0) / max(mx - Yn, 1e-6); lin += (vec3(Yn) - lin) * t; }
  o = vec4(L2S(lin), 1.0);
}`;
// 柔化: one box-blur pass (horizontal or vertical, clamped edges) in linear light; lin8 = the input is the 8-bit frame
const BOX = HEAD + `
uniform sampler2D img; uniform ivec2 size, dir; uniform int r, lin8;
void main() {
  ivec2 px = ivec2(gl_FragCoord.xy); vec3 acc = vec3(0.0);
  for (int k = -r; k <= r; k++) { vec3 c = texelFetch(img, clamp(px + dir * k, ivec2(0), size - 1), 0).rgb; acc += lin8 == 1 ? S2L(c) : c; }
  o = vec4(acc / float(2 * r + 1), 1.0);
}`;
const SOFTMIX = HEAD + `
uniform sampler2D img, bl; uniform float k;
void main() { ivec2 px = ivec2(gl_FragCoord.xy); vec3 l = S2L(texelFetch(img, px, 0).rgb); o = vec4(L2S(l + (texelFetch(bl, px, 0).rgb - l) * k), 1.0); }`;
// 光暈: the blurred bright light (grid), screened in red-orange
const HAL = HEAD + `
uniform sampler2D img, g; uniform ivec2 gs; uniform int f; uniform float k;
void main() {
  ivec2 px = ivec2(gl_FragCoord.xy); vec4 s = texelFetch(img, px, 0);
  vec2 q = clamp((vec2(px) + 0.5) / float(f) - 0.5, vec2(0.0), vec2(gs - 1));
  ivec2 a = ivec2(floor(q)), b = min(gs - 1, a + 1); vec2 t = q - vec2(a);
  float gl = mix(mix(texelFetch(g, a, 0).r, texelFetch(g, ivec2(b.x, a.y), 0).r, t.x), mix(texelFetch(g, ivec2(a.x, b.y), 0).r, texelFetch(g, b, 0).r, t.x), t.y);
  if (gl < 1e-4) { o = s; return; }
  vec3 add = min(vec3(1.0), k * gl * vec3(1.0, 0.3, 0.1));
  o = vec4(L2S(1.0 - (1.0 - S2L(s.rgb)) * (1.0 - add)), 1.0);
}`;
// 膠片顆粒: the grain cells (L, Cr, Cb) bilinear, strongest in the mid-tones, added in 8-bit steps
const GRAIN = HEAD + `
uniform sampler2D img, g; uniform float cell, amp;
void main() {
  ivec2 px = ivec2(gl_FragCoord.xy); vec3 v8 = round(texelFetch(img, px, 0).rgb * 255.0);
  vec2 q = vec2(px) / cell; ivec2 a = ivec2(floor(q)); vec2 t = q - vec2(a);
  vec4 c00 = texelFetch(g, a, 0), c01 = texelFetch(g, a + ivec2(1, 0), 0), c10 = texelFetch(g, a + ivec2(0, 1), 0);
  float nL = (c00.r * (1.0 - t.x) + c01.r * t.x) * (1.0 - t.y) + (c10.r * (1.0 - t.x) + texelFetch(g, a + ivec2(1, 1), 0).r * t.x) * t.y;
  float v = dot(v8, LUM) / 255.0, m = amp * (0.35 + 2.6 * v * (1.0 - v));
  o = vec4(clamp(v8 + m * vec3(nL + c00.g * 0.25, nL, nL + c00.b * 0.25), 0.0, 255.0) / 255.0, 1.0);
}`;

export function gpuEffects() {
  const canvas = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(1, 1) : (typeof document !== 'undefined' ? document.createElement('canvas') : null);
  if (!canvas) return null;
  const gl = canvas.getContext('webgl2', { antialias: false, depth: false, stencil: false, premultipliedAlpha: false, preserveDrawingBuffer: false, powerPreference: 'high-performance' });
  if (!gl || !gl.getExtension('EXT_color_buffer_float')) return null; // (float render targets: the grids, 柔化's blur)
  const compile = (type, text) => { const s = gl.createShader(type); gl.shaderSource(s, text); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
  const vs = compile(gl.VERTEX_SHADER, VS);
  const P = {};
  try {
    for (const [k, src] of Object.entries({ DOWN, FRINGE, FX, BOX, SOFTMIX, HAL, GRAIN })) {
      const pr = gl.createProgram(); gl.attachShader(pr, vs); gl.attachShader(pr, compile(gl.FRAGMENT_SHADER, src)); gl.bindAttribLocation(pr, 0, 'p'); gl.linkProgram(pr);
      if (!gl.getProgramParameter(pr, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(pr));
      const loc = new Map(); P[k] = { pr, u: (n) => { if (!loc.has(n)) loc.set(n, gl.getUniformLocation(pr, n)); return loc.get(n); } };
    }
  } catch (err) { return null; }
  const R = { lost: false, info: { renderer: '' } };
  try { const ext = gl.getExtension('WEBGL_debug_renderer_info'); R.info.renderer = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER); } catch { /* (hidden by the browser) */ }
  if (canvas.addEventListener) canvas.addEventListener('webglcontextlost', (e) => { e.preventDefault(); R.lost = true; });
  const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1); gl.pixelStorei(gl.PACK_ALIGNMENT, 1);
  // textures with framebuffers, (re)allocated when the size changes; kind: '8' RGBA8, 'h' RGBA16F, 'f' RGBA32F
  const FMT = { 8: [gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE], h: [gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT], f: [gl.RGBA32F, gl.RGBA, gl.FLOAT] };
  const targets = new Map();
  const target = (name, kind, w, h) => {
    let t = targets.get(name);
    if (!t) { t = { tex: gl.createTexture(), fb: gl.createFramebuffer(), w: 0, h: 0, kind }; targets.set(name, t); }
    if (t.w !== w || t.h !== h) {
      gl.bindTexture(gl.TEXTURE_2D, t.tex);
      for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.NEAREST], [gl.TEXTURE_MAG_FILTER, gl.NEAREST], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
      const [ifmt, fmt, type] = FMT[kind]; gl.texImage2D(gl.TEXTURE_2D, 0, ifmt, w, h, 0, fmt, type, null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, t.fb); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t.tex, 0);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error(`GPU effects: ${kind} target not renderable`);
      t.w = w; t.h = h;
    }
    return t;
  };
  const upload = (name, kind, w, h, data) => { const t = target(name, kind, w, h); gl.bindTexture(gl.TEXTURE_2D, t.tex); const [, fmt, type] = FMT[kind]; gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, w, h, fmt, type, data); return t; };
  const rgba = (src3, n, ch = 3) => { const a = new Float32Array(n * 4); for (let i = 0; i < n; i++) for (let c = 0; c < ch; c++) a[4 * i + c] = src3[ch * i + c]; return a; };
  // one pass: program k, textures {uniform: target}, scalar uniforms, into target `to`
  const pass = (k, tex, uni, to) => {
    const p = P[k]; gl.useProgram(p.pr);
    let unit = 0;
    for (const [n, t] of Object.entries(tex)) { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, t.tex); gl.uniform1i(p.u(n), unit++); }
    for (const [n, v] of Object.entries(uni)) {
      const l = p.u(n); if (!l) continue;
      if (Array.isArray(v)) { if (v.int) (v.length === 2 ? gl.uniform2i(l, v[0], v[1]) : null); else if (v.length === 3) gl.uniform3f(l, v[0], v[1], v[2]); else gl.uniform2f(l, v[0], v[1]); }
      else if (typeof v === 'object') v.i !== undefined ? gl.uniform1i(l, v.i) : null;
      else gl.uniform1f(l, v);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, to.fb); gl.viewport(0, 0, to.w, to.h);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    return to;
  };
  const I = (v) => ({ i: v }), I2 = (a, b) => Object.assign([a, b], { int: true });
  // a grid of the frame (mode 0 glow, 1 soft copy, 2 bloom, 3 halation) averaged on the card, read back, blurred here
  const readGrid = (src, w, h, f, sw, sh, mode) => {
    const t = pass('DOWN', { img: src }, { size: I2(w, h), f: I(f), mode: I(mode), k: I(fringeStep(w, h)) }, target(`down${sw}x${sh}`, 'f', sw, sh));
    const a = new Float32Array(sw * sh * 4); gl.bindFramebuffer(gl.FRAMEBUFFER, t.fb); gl.readPixels(0, 0, sw, sh, gl.RGBA, gl.FLOAT, a);
    return a;
  };
  const ping = (w, h) => [target('A', '8', w, h), target('B', '8', w, h)];

  /** in place on RGBA, as applyEffects(img, w, h, fx, hazeMap) followed by applyFilm(img, w, h, fx, frame) */
  R.apply = (img, w, h, fx, hazeMap = null, frame = 0) => {
    if (R.lost || gl.isContextLost()) { R.lost = true; throw new Error('GPU context lost'); }
    const plan = effectsPlan(fx, !!hazeMap), film = hasFilm(fx), fr = fringePlan(fx);
    if (!plan && !film && !fr) return img;
    const [A, B] = ping(w, h); let cur = A, other = B;
    gl.bindTexture(gl.TEXTURE_2D, A.tex); gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, img instanceof Uint8Array ? img : new Uint8Array(img.buffer, img.byteOffset, img.length));
    const swap = () => { const t = cur; cur = other; other = t; };
    if (fr) { // 色散 first (a lens effect), as in the worker's processor path
      const { f, sw, sh } = gridDims(w, h), a = readGrid(cur, w, h, f, sw, sh, 4), g = new Float32Array(sw * sh * 2);
      for (let q = 0; q < sw * sh; q++) { g[2 * q] = a[4 * q]; g[2 * q + 1] = a[4 * q + 1]; }
      const W = fringeWeights(fr, { g, f, sw, sh }), wt = upload('fringe', 'f', sw, sh, rgba(W, sw * sh, 1));
      pass('FRINGE', { img: cur, wt }, { size: I2(w, h), gs: I2(sw, sh), f: I(f), ms: fringeMaxShift(w, h) }, other); swap();
    }
    if (plan) {
      const { f, sw, sh } = gridDims(w, h), n = sw * sh;
      const grid = (mode, radius, name) => { const a = readGrid(cur, w, h, f, sw, sh, mode), s3 = new Float32Array(n * 3); for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) s3[3 * i + c] = a[4 * i + c]; blurGrid(s3, sw, sh, radius); upload(name, 'f', sw, sh, rgba(s3, n)); return { small: s3, sw, sh, f }; };
      const G = plan.gridG ? grid(0, plan.gridG, 'gG') : null, H = plan.gridH ? grid(1, plan.gridH, 'gH') : null, Bm = plan.gridB ? grid(2, plan.gridB, 'gB') : null;
      const { veil, bk } = Bm ? veilOf(plan, Bm) : { veil: [0, 0, 0], bk: 1 };
      const dummy = targets.get('gG') || targets.get('gH') || target('one', 'f', 1, 1);
      const hz = plan.depth || plan.where ? upload('hz', 'f', hazeMap.w, hazeMap.h, rgba(hazeMap.data, hazeMap.w * hazeMap.h, 1)) : dummy;
      pass('FX', { img: cur, gG: G ? targets.get('gG') : dummy, gH: H ? targets.get('gH') : dummy, gB: Bm ? targets.get('gB') : dummy, hz }, {
        size: I2(w, h), gs: I2(sw, sh), f: I(f), hasG: I(G ? 1 : 0), hasH: I(H ? 1 : 0), hasHz: I(plan.depth || plan.where ? 1 : 0),
        glow: plan.glow, haze: plan.haze, kSoft: plan.kSoft, kBloom: plan.kBloom, kVeil: plan.kVeil, bk, vig: plan.vig, v0: 0.75 - 0.5 * plan.vigSize, depth: plan.depth, veil, tm: plan.tm, hasAir: plan.air ? 1 : 0, air: plan.air || [1, 1, 1], where: plan.where || 0,
      }, other); swap();
    }
    if (film) {
      const c01 = (v) => Math.min(1, Math.max(0, +v || 0));
      const soft = c01(fx.filmSoft), hal = c01(fx.halation), grain = c01(fx.filmGrain), s = w / 1280;
      if (soft > 0) { // two separable box blurs (as film.js blurChannels twice) in half-float, then the mix
        const r = Math.max(1, Math.round((1 + 1.5 * soft) * s)), F1 = target('F1', 'h', w, h), F2 = target('F2', 'h', w, h);
        pass('BOX', { img: cur }, { size: I2(w, h), dir: I2(1, 0), r: I(r), lin8: I(1) }, F1);
        pass('BOX', { img: F1 }, { size: I2(w, h), dir: I2(0, 1), r: I(r), lin8: I(0) }, F2);
        pass('BOX', { img: F2 }, { size: I2(w, h), dir: I2(1, 0), r: I(r), lin8: I(0) }, F1);
        pass('BOX', { img: F1 }, { size: I2(w, h), dir: I2(0, 1), r: I(r), lin8: I(0) }, F2);
        pass('SOFTMIX', { img: cur, bl: F2 }, { k: 0.85 * soft }, other); swap();
      }
      if (hal > 0) {
        const f = Math.max(1, Math.round(Math.max(w, h) / 640)), sw = Math.ceil(w / f), sh = Math.ceil(h / f);
        const a = readGrid(cur, w, h, f, sw, sh, 3), g = new Float32Array(sw * sh); for (let i = 0; i < g.length; i++) g[i] = a[4 * i];
        const rr = Math.max(1, Math.round(0.006 * Math.max(sw, sh))); for (let p = 0; p < 3; p++) blurChannels(g, sw, sh, 1, rr);
        const gt = upload('hal', 'f', sw, sh, rgba(g, sw * sh, 1));
        pass('HAL', { img: cur, g: gt }, { gs: I2(sw, sh), f: I(f), k: 1.6 * hal }, other); swap();
      }
      if (grain > 0) {
        const { cell, gw, gh, L, C } = grainGrid(w, h, c01(fx.filmGrainSize ?? 0.5), frame), a = new Float32Array(gw * gh * 4);
        for (let q = 0; q < gw * gh; q++) { a[4 * q] = L[q]; a[4 * q + 1] = C[2 * q]; a[4 * q + 2] = C[2 * q + 1]; }
        const gt = upload('grain', 'f', gw, gh, a);
        pass('GRAIN', { img: cur, g: gt }, { cell, amp: 20 * grain }, other); swap();
      }
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, cur.fb);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, img instanceof Uint8Array ? img : new Uint8Array(img.buffer, img.byteOffset, img.length));
    return img;
  };
  return R;
}
