// Distil a 3D LUT from a (video frame, AI-graded version of the same frame) pair — the browser port of the
// reference-grade skill's lut_from_pair.py. The AI image is only a target: the video is still rendered as
// original frames through one LUT, so no generated content reaches the output.
//   * alignment: integer shift search on colour-independent edge maps (AI tools often move a few pixels),
//   * pairs compared on blurred copies, so small AI detail changes do not dominate,
//   * smooth 17³ lattice: data term + Laplacian² smoothness + tiny ridge on the offset (out − in), solved by
//     conjugate gradients; IRLS (Huber-like) drops pixels the AI changed beyond a colour mapping.

function luma(rgba, w, h) {
  const g = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) g[i] = 0.2126 * rgba[4 * i] + 0.7152 * rgba[4 * i + 1] + 0.0722 * rgba[4 * i + 2];
  return g;
}

function edgeMap(rgba, w, h) {
  const g = luma(rgba, w, h), e = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x;
    e[i] = Math.hypot(g[i + 1] - g[i - 1], g[i + w] - g[i - w]);
  }
  return e;
}

/** Normalised correlation of two edge maps, b shifted by (dx, dy), over the overlap. */
function corr(a, b, w, h, dx, dy) {
  let n = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
  const m = 2;
  for (let y = m + Math.max(0, -dy); y < h - m - Math.max(0, dy); y++) {
    for (let x = m + Math.max(0, -dx); x < w - m - Math.max(0, dx); x++) {
      const p = a[y * w + x], q = b[(y + dy) * w + x + dx];
      n++; sa += p; sb += q; saa += p * p; sbb += q * q; sab += p * q;
    }
  }
  const va = saa - sa * sa / n, vb = sbb - sb * sb / n;
  return (sab - sa * sb / n) / Math.sqrt(Math.max(va * vb, 1e-12));
}

/** Best integer shift of the AI image relative to the frame. score ≥ 0.8 = same composition. */
export function alignShift(src, ai, w, h, maxShift = 4) {
  const a = edgeMap(src, w, h), b = edgeMap(ai, w, h);
  let best = { dx: 0, dy: 0, score: -1 };
  for (let dy = -maxShift; dy <= maxShift; dy++) for (let dx = -maxShift; dx <= maxShift; dx++) {
    const s = corr(a, b, w, h, dx, dy);
    if (s > best.score) best = { dx, dy, score: s };
  }
  return best;
}

function blur(rgba, w, h, passes = 2) { // separable box blur, 3 channels as floats 0..1
  let cur = new Float32Array(w * h * 3);
  for (let i = 0; i < w * h; i++) for (let c = 0; c < 3; c++) cur[3 * i + c] = rgba[4 * i + c] / 255;
  const tmp = new Float32Array(cur.length);
  for (let p = 0; p < passes; p++) {
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < 3; c++) {
      const l = Math.max(0, x - 1), r = Math.min(w - 1, x + 1);
      tmp[3 * (y * w + x) + c] = (cur[3 * (y * w + l) + c] + cur[3 * (y * w + x) + c] + cur[3 * (y * w + r) + c]) / 3;
    }
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < 3; c++) {
      const u = Math.max(0, y - 1), d = Math.min(h - 1, y + 1);
      cur[3 * (y * w + x) + c] = (tmp[3 * (u * w + x) + c] + tmp[3 * (y * w + x) + c] + tmp[3 * (d * w + x) + c]) / 3;
    }
  }
  return cur;
}

/** Trilinear weights of n points (x: Float32Array 3n in 0..1): 8 indices + 8 weights per point. */
function trilinear(x, n, N) {
  const idx = new Int32Array(8 * n), wt = new Float32Array(8 * n), n1 = N - 1;
  for (let i = 0; i < n; i++) {
    const f = [0, 1, 2].map((c) => Math.min(1, Math.max(0, x[3 * i + c])) * n1);
    const i0 = f.map((v) => Math.min(n1 - 1, Math.floor(v))), d = f.map((v, c) => v - i0[c]);
    let k = 0;
    for (let dr = 0; dr < 2; dr++) for (let dg = 0; dg < 2; dg++) for (let db = 0; db < 2; db++) {
      idx[8 * i + k] = ((i0[0] + dr) * N + (i0[1] + dg)) * N + (i0[2] + db);
      wt[8 * i + k] = (dr ? d[0] : 1 - d[0]) * (dg ? d[1] : 1 - d[1]) * (db ? d[2] : 1 - d[2]);
      k++;
    }
  }
  return { idx, wt };
}

function laplacian(x, N, out) { // Neumann boundary: sum over existing neighbours of (x_j − x_i); symmetric
  for (let r = 0; r < N; r++) for (let g = 0; g < N; g++) for (let b = 0; b < N; b++) {
    const i = (r * N + g) * N + b; let s = 0;
    if (r > 0) s += x[i - N * N] - x[i]; if (r < N - 1) s += x[i + N * N] - x[i];
    if (g > 0) s += x[i - N] - x[i]; if (g < N - 1) s += x[i + N] - x[i];
    if (b > 0) s += x[i - 1] - x[i]; if (b < N - 1) s += x[i + 1] - x[i];
    out[i] = s;
  }
  return out;
}

/**
 * src, ai: RGBA Uint8ClampedArray, same w×h (the AI image already resized to the frame).
 * Returns {lut: {size, data}, shift, meanAbs (0..255, on the blurred pair), outlierShare}.
 */
export function fitPairLUT(src, ai, w, h, { N = 17, lam = 0.3, ridge = 1e-3, irls = 4, cgIters = 150, shift = null } = {}) {
  const s = shift || alignShift(src, ai, w, h);
  const A = blur(src, w, h), B = blur(ai, w, h);
  // overlap after the shift, keeping a 2 px border away from blur edge effects
  const xs = [], ys = [];
  for (let y = 2 + Math.max(0, -s.dy); y < h - 2 - Math.max(0, s.dy); y++) {
    for (let x = 2 + Math.max(0, -s.dx); x < w - 2 - Math.max(0, s.dx); x++) {
      const i = y * w + x, j = (y + s.dy) * w + x + s.dx;
      xs.push(A[3 * i], A[3 * i + 1], A[3 * i + 2]); ys.push(B[3 * j], B[3 * j + 1], B[3 * j + 2]);
    }
  }
  const n = xs.length / 3, X = Float32Array.from(xs), Y = Float32Array.from(ys), V = N * N * N;
  const { idx, wt } = trilinear(X, n, N);
  const mu = lam * n / V, rho = ridge * n / V;
  const wIr = new Float32Array(n).fill(1);
  const off = [new Float64Array(V), new Float64Array(V), new Float64Array(V)];
  const t1 = new Float64Array(V), t2 = new Float64Array(V);
  const applyM = (x, out) => { // (AᵀWA + μ L² + ρ I) x
    out.fill(0);
    for (let i = 0; i < n; i++) {
      let v = 0; for (let k = 0; k < 8; k++) v += wt[8 * i + k] * x[idx[8 * i + k]];
      v *= wIr[i];
      for (let k = 0; k < 8; k++) out[idx[8 * i + k]] += wt[8 * i + k] * v;
    }
    laplacian(x, N, t1); laplacian(t1, N, t2);
    for (let j = 0; j < V; j++) out[j] += mu * t2[j] + rho * x[j];
    return out;
  };
  const res = new Float32Array(n);
  for (let it = 0; it < irls; it++) {
    for (let c = 0; c < 3; c++) { // CG warm-started from the previous IRLS round
      const rhs = new Float64Array(V);
      for (let i = 0; i < n; i++) { const v = wIr[i] * (Y[3 * i + c] - X[3 * i + c]); for (let k = 0; k < 8; k++) rhs[idx[8 * i + k]] += wt[8 * i + k] * v; }
      const x = off[c], r = new Float64Array(V), Ap = new Float64Array(V);
      applyM(x, Ap); for (let j = 0; j < V; j++) r[j] = rhs[j] - Ap[j];
      const p = Float64Array.from(r); let rr = r.reduce((a, v) => a + v * v, 0);
      const rr0 = rr;
      for (let k = 0; k < cgIters && rr > 1e-14 * rr0; k++) {
        applyM(p, Ap);
        let pAp = 0; for (let j = 0; j < V; j++) pAp += p[j] * Ap[j];
        const alpha = rr / pAp;
        let rrN = 0; for (let j = 0; j < V; j++) { x[j] += alpha * p[j]; r[j] -= alpha * Ap[j]; rrN += r[j] * r[j]; }
        const beta = rrN / rr; rr = rrN;
        for (let j = 0; j < V; j++) p[j] = r[j] + beta * p[j];
      }
    }
    for (let i = 0; i < n; i++) {
      let e2 = 0;
      for (let c = 0; c < 3; c++) { let v = X[3 * i + c]; for (let k = 0; k < 8; k++) v += wt[8 * i + k] * off[c][idx[8 * i + k]]; e2 += (v - Y[3 * i + c]) ** 2; }
      res[i] = Math.sqrt(e2);
    }
    const med = Float32Array.from(res).sort()[n >> 1], scale = 1.4826 * med + 1e-6;
    for (let i = 0; i < n; i++) wIr[i] = Math.min(1, 2.5 * scale / Math.max(res[i], 1e-9));
  }
  const data = new Float32Array(V * 3);
  for (let r = 0; r < N; r++) for (let g = 0; g < N; g++) for (let b = 0; b < N; b++) {
    const j = (r * N + g) * N + b, base = [r, g, b].map((v) => v / (N - 1));
    for (let c = 0; c < 3; c++) data[3 * j + c] = Math.min(1, Math.max(0, base[c] + off[c][j]));
  }
  let abs = 0, out = 0;
  for (let i = 0; i < n; i++) { abs += res[i]; if (wIr[i] < 1) out++; }
  return { lut: { size: N, data }, shift: s, meanAbs: (abs / n) / Math.sqrt(3) * 255, outlierShare: out / n, samples: n };
}
