// GPU (WebGL2) version of renderFrame's per-pixel grade: the shaper, the 3D LUT (trilinear), the region layers (skin
// protection / local grade: a second LUT mixed in by a mask) and the local exposure of the Lighting Intent. The same
// formulas as the CPU path in render.js — the result matches it to within rounding (test/e2e/gpu.cjs) — so a frame can
// be graded on the graphics card instead of the processor: the work per frame no longer grows with the pixel count in
// the same way (4K is 9× the pixels of 720p). Works on the page and in a worker (OffscreenCanvas). Spatial effects
// (haze, softness, grain, blur) are not here yet: they still run on the processor after this step.
//
// gpuRenderer() → renderer or null (no WebGL2 / float textures). renderer.render(src, w, h, lut, ev, layers) → RGBA
// Uint8ClampedArray, the same arguments as renderFrame. renderer.lost: true after the browser took the GPU context back
// (memory pressure, tab in the background): the caller falls back to renderFrame.

const VS = `#version 300 es
in vec2 p; out vec2 uv;
void main() { uv = p * 0.5 + 0.5; gl_Position = vec4(p, 0.0, 1.0); }`;

const MAX_LAYERS = 3;
const FS = `#version 300 es
precision highp float; precision highp int; precision highp sampler3D;
in vec2 uv; out vec4 o;
uniform sampler2D src;       // RGBA8 frame (texel (x, y) = pixel (x, y): the frame is uploaded top row first)
uniform highp sampler2D shaper; // 256×1 R32F: 8-bit level -> fractional grid coordinate (identity when no shaper)
uniform sampler3D lut;       // N³ RGBA32F, texel (b, g, r)
uniform int N, hasLut, nLayers, hasEv;
uniform sampler3D lutL[${MAX_LAYERS}];
uniform highp sampler2D maskL[${MAX_LAYERS}];
uniform float kL[${MAX_LAYERS}];
uniform highp sampler2D ev;  // gain grid (2^stops), R32F
uniform ivec2 size;
const float KNEE = 0.8;
float rollOff(float y) { if (y <= KNEE) return y; float t = y - KNEE, r = 1.0 - KNEE; return KNEE + r * t / (t + r); }
float s2l(float c) { return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4); }
float l2s(float c) { c = clamp(c, 0.0, 1.0); return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1.0 / 2.4) - 0.055; }
vec3 tri(sampler3D L, ivec3 b0, vec3 d) { // trilinear, as render.js (b fastest, then g, then r)
  vec3 c000 = texelFetch(L, ivec3(b0.z, b0.y, b0.x), 0).rgb, c001 = texelFetch(L, ivec3(b0.z + 1, b0.y, b0.x), 0).rgb;
  vec3 c010 = texelFetch(L, ivec3(b0.z, b0.y + 1, b0.x), 0).rgb, c011 = texelFetch(L, ivec3(b0.z + 1, b0.y + 1, b0.x), 0).rgb;
  vec3 c100 = texelFetch(L, ivec3(b0.z, b0.y, b0.x + 1), 0).rgb, c101 = texelFetch(L, ivec3(b0.z + 1, b0.y, b0.x + 1), 0).rgb;
  vec3 c110 = texelFetch(L, ivec3(b0.z, b0.y + 1, b0.x + 1), 0).rgb, c111 = texelFetch(L, ivec3(b0.z + 1, b0.y + 1, b0.x + 1), 0).rgb;
  vec3 c00 = c000 + (c001 - c000) * d.z, c01 = c010 + (c011 - c010) * d.z, c10 = c100 + (c101 - c100) * d.z, c11 = c110 + (c111 - c110) * d.z;
  vec3 c0 = c00 + (c01 - c00) * d.y, c1 = c10 + (c11 - c10) * d.y;
  return c0 + (c1 - c0) * d.x;
}
float maskAt(int i, ivec2 px) { // nearest cell, as render.js (round((x + 0.5)·mw/w − 0.5))
  ivec2 ms = i == 0 ? textureSize(maskL[0], 0) : i == 1 ? textureSize(maskL[1], 0) : textureSize(maskL[2], 0);
  ivec2 c = clamp(ivec2(floor((vec2(px) + 0.5) * vec2(ms) / vec2(size) - 0.5 + 0.5)), ivec2(0), ms - 1);
  return i == 0 ? texelFetch(maskL[0], c, 0).r : i == 1 ? texelFetch(maskL[1], c, 0).r : texelFetch(maskL[2], c, 0).r;
}
void main() {
  ivec2 px = ivec2(gl_FragCoord.xy); // (framebuffer row 0 = frame row 0: the frame was uploaded top row first and is read back the same way)
  vec4 s = texelFetch(src, px, 0);
  ivec3 lv = ivec3(round(s.rgb * 255.0));
  vec3 col = s.rgb;
  if (hasLut == 1) {
    int n1 = N - 1;
    vec3 f = vec3(texelFetch(shaper, ivec2(lv.r, 0), 0).r, texelFetch(shaper, ivec2(lv.g, 0), 0).r, texelFetch(shaper, ivec2(lv.b, 0), 0).r);
    ivec3 b0 = min(ivec3(n1 - 1), ivec3(f));
    vec3 d = f - vec3(b0);
    col = tri(lut, b0, d);
    for (int i = 0; i < ${MAX_LAYERS}; i++) {
      if (i >= nLayers) break;
      float m = maskAt(i, px) * kL[i];
      if (m > 0.004) { vec3 c2 = i == 0 ? tri(lutL[0], b0, d) : i == 1 ? tri(lutL[1], b0, d) : tri(lutL[2], b0, d); col += (c2 - col) * m; }
    }
  }
  if (hasEv == 1) { // local exposure: bilinear gain, scaled through luminance with the highlight roll-off (render.js expose)
    ivec2 gs = textureSize(ev, 0);
    vec2 g = clamp((vec2(px) + 0.5) * vec2(gs) / vec2(size) - 0.5, vec2(0.0), vec2(gs - 1));
    ivec2 g0 = ivec2(floor(g)), g1 = min(gs - 1, g0 + 1); vec2 fg = g - vec2(g0);
    float a0 = mix(texelFetch(ev, ivec2(g0.x, g0.y), 0).r, texelFetch(ev, ivec2(g1.x, g0.y), 0).r, fg.x);
    float a1 = mix(texelFetch(ev, ivec2(g0.x, g1.y), 0).r, texelFetch(ev, ivec2(g1.x, g1.y), 0).r, fg.x);
    float k = mix(a0, a1, fg.y);
    if (k != 1.0) {
      vec3 lin = vec3(s2l(col.r), s2l(col.g), s2l(col.b));
      float Y = dot(lin, vec3(0.2126, 0.7152, 0.0722));
      if (Y > 1e-6) {
        float sc = (k > 1.0 ? rollOff(Y * k) : Y * k) / Y;
        lin *= sc;
        float mx = max(lin.r, max(lin.g, lin.b));
        if (mx > 1.0) { float Yn = Y * sc, t = (mx - 1.0) / max(mx - Yn, 1e-6); lin += (vec3(Yn) - lin) * t; }
      }
      col = vec3(l2s(lin.r), l2s(lin.g), l2s(lin.b));
    }
  }
  o = vec4(clamp(col, 0.0, 1.0), 1.0);
}`;

export function gpuRenderer() {
  const canvas = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(1, 1) : (typeof document !== 'undefined' ? document.createElement('canvas') : null);
  if (!canvas) return null;
  const gl = canvas.getContext('webgl2', { antialias: false, depth: false, stencil: false, premultipliedAlpha: false, preserveDrawingBuffer: false, powerPreference: 'high-performance' });
  if (!gl) return null;
  const compile = (type, text) => { const s = gl.createShader(type); gl.shaderSource(s, text); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
  let prog;
  try {
    prog = gl.createProgram(); gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  } catch (err) { return null; }
  const R = { lost: false, info: { renderer: '' } };
  try { const ext = gl.getExtension('WEBGL_debug_renderer_info'); R.info.renderer = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER); } catch { /* (hidden by the browser) */ }
  if (canvas.addEventListener) canvas.addEventListener('webglcontextlost', (e) => { e.preventDefault(); R.lost = true; });
  gl.useProgram(prog);
  const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, 'p'); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  const U = (n) => gl.getUniformLocation(prog, n);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1); gl.pixelStorei(gl.PACK_ALIGNMENT, 1);
  const tex = () => { const t = gl.createTexture(); return t; };
  const params = (target) => { for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.NEAREST], [gl.TEXTURE_MAG_FILTER, gl.NEAREST], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(target, k, v); if (target === gl.TEXTURE_3D) gl.texParameteri(target, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE); };
  // texture units: 0 src, 1 shaper, 2 lut, 3..5 layer luts, 6..8 masks, 9 ev
  const T = { src: tex(), shaper: tex(), lut: tex(), lutL: [tex(), tex(), tex()], maskL: [tex(), tex(), tex()], ev: tex() };
  gl.uniform1i(U('src'), 0); gl.uniform1i(U('shaper'), 1); gl.uniform1i(U('lut'), 2);
  gl.uniform1iv(U('lutL'), [3, 4, 5]); gl.uniform1iv(U('maskL'), [6, 7, 8]); gl.uniform1i(U('ev'), 9);
  const bind = (unit, target, t) => { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(target, t); };
  // every sampler must have a valid texture of its type even when unused (a 1-texel placeholder)
  for (const [u, t] of [[3, T.lutL[0]], [4, T.lutL[1]], [5, T.lutL[2]], [2, T.lut]]) { bind(u, gl.TEXTURE_3D, t); params(gl.TEXTURE_3D); gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA32F, 2, 2, 2, 0, gl.RGBA, gl.FLOAT, new Float32Array(32)); }
  for (const [u, t] of [[6, T.maskL[0]], [7, T.maskL[1]], [8, T.maskL[2]], [9, T.ev], [1, T.shaper]]) { bind(u, gl.TEXTURE_2D, t); params(gl.TEXTURE_2D); gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, 1, 1, 0, gl.RED, gl.FLOAT, new Float32Array(1)); }
  bind(0, gl.TEXTURE_2D, T.src); params(gl.TEXTURE_2D);
  const lutCache = new WeakMap(); // lut object -> RGBA32F data (uploaded when it changes)
  const lutRGBA = (lut) => { let d = lutCache.get(lut); if (d) return d; const n = lut.size ** 3; d = new Float32Array(n * 4); for (let i = 0; i < n; i++) { d[4 * i] = lut.data[3 * i]; d[4 * i + 1] = lut.data[3 * i + 1]; d[4 * i + 2] = lut.data[3 * i + 2]; d[4 * i + 3] = 1; } lutCache.set(lut, d); return d; };
  const uploaded = new Map(); // texture unit -> the object last uploaded there
  const upLut = (unit, t, lut) => { bind(unit, gl.TEXTURE_3D, t); if (uploaded.get(unit) === lut) return; gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA32F, lut.size, lut.size, lut.size, 0, gl.RGBA, gl.FLOAT, lutRGBA(lut)); uploaded.set(unit, lut); };
  const up2D = (unit, t, data, w, h) => { bind(unit, gl.TEXTURE_2D, t); if (uploaded.get(unit) === data) return; gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, w, h, 0, gl.RED, gl.FLOAT, data instanceof Float32Array ? data : Float32Array.from(data)); uploaded.set(unit, data); };
  let fbW = 0, fbH = 0;
  /** same arguments as render.js renderFrame (without fx): src RGBA, lut {size, data, index?} | null, ev {data, w, h} stops | null */
  R.render = (src, w, h, lut, ev, layers = null, out = new Uint8ClampedArray(w * h * 4)) => {
    if (R.lost || gl.isContextLost()) { R.lost = true; throw new Error('GPU context lost'); }
    if (fbW !== w || fbH !== h) { canvas.width = w; canvas.height = h; fbW = w; fbH = h; gl.viewport(0, 0, w, h); }
    bind(0, gl.TEXTURE_2D, T.src);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, src instanceof Uint8Array ? src : new Uint8Array(src.buffer, src.byteOffset, src.length));
    gl.uniform2i(U('size'), w, h);
    if (lut) {
      gl.uniform1i(U('hasLut'), 1); gl.uniform1i(U('N'), lut.size);
      upLut(2, T.lut, lut);
      const sh = lut.index ? (lut.index instanceof Float32Array ? lut.index : Float32Array.from(lut.index)) : (lutRGBA.ident || (lutRGBA.ident = new Map())).get(lut.size) || (() => { const a = new Float32Array(256); for (let v = 0; v < 256; v++) a[v] = v / 255 * (lut.size - 1); lutRGBA.ident.set(lut.size, a); return a; })();
      up2D(1, T.shaper, sh, 256, 1);
      const sameIndex = (a, b2) => (!a && !b2) || (a && b2 && a.length === b2.length && a.every((v, i) => v === b2[i]));
      const LY = (layers ? (Array.isArray(layers) ? layers : [layers]) : []).filter((l) => l && l.lut && l.lut.size === lut.size && sameIndex(l.lut.index, lut.index));
      if (LY.length > MAX_LAYERS) throw new Error(`more than ${MAX_LAYERS} region layers`);
      gl.uniform1i(U('nLayers'), LY.length);
      const ks = [0, 0, 0];
      LY.forEach((ly, i) => { upLut(3 + i, T.lutL[i], ly.lut); up2D(6 + i, T.maskL[i], ly.mask.data, ly.mask.w, ly.mask.h); ks[i] = ly.k; });
      gl.uniform1fv(U('kL'), ks);
    } else { gl.uniform1i(U('hasLut'), 0); gl.uniform1i(U('nLayers'), 0); }
    if (ev) { const g = new Float32Array(ev.data.length); for (let i = 0; i < g.length; i++) g[i] = Math.pow(2, ev.data[i]); uploaded.delete(9); up2D(9, T.ev, g, ev.w, ev.h); gl.uniform1i(U('hasEv'), 1); } else gl.uniform1i(U('hasEv'), 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(out.buffer, out.byteOffset, out.length));
    return out;
  };
  return R;
}
