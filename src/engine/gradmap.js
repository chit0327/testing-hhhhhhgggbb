// 漸層映射 (gradient map): the reference's colour at each lightness, as a gradient dark → bright. Applied in colour
// mode (primary.js gmap): a pixel keeps its lightness and its colour moves towards the gradient's colour at that
// lightness — like painting values first (black and white) and glazing one harmonious gradient over them.
export const GMAP_N = 17; // nodes at OKLab L = 0, 1/16, … 1

/** lab: OKLab triples (a, b unscaled). Returns {a, b} arrays of GMAP_N colours, smooth, defined everywhere. */
export function extractGradient(lab) {
  const N = GMAP_N, sa = new Float64Array(N), sb = new Float64Array(N), sw = new Float64Array(N);
  for (let i = 0; i < lab.length; i += 3) { // triangular weights onto the two nearest nodes
    const x = Math.min(1, Math.max(0, lab[i])) * (N - 1), k = Math.min(N - 2, Math.floor(x)), t = x - k;
    sa[k] += (1 - t) * lab[i + 1]; sb[k] += (1 - t) * lab[i + 2]; sw[k] += 1 - t;
    sa[k + 1] += t * lab[i + 1]; sb[k + 1] += t * lab[i + 2]; sw[k + 1] += t;
  }
  const tot = sw.reduce((s, v) => s + v, 0), min = Math.max(1e-9, tot * 0.004); // nodes with almost no pixels: from neighbours
  let a = Array.from(sa, (v, k) => (sw[k] >= min ? v / sw[k] : NaN)), b = Array.from(sb, (v, k) => (sw[k] >= min ? v / sw[k] : NaN));
  const fill = (arr) => { const ok = arr.map((v) => !Number.isNaN(v)); if (!ok.some(Boolean)) return arr.map(() => 0);
    return arr.map((v, k) => { if (ok[k]) return v; let lo = k, hi = k; while (lo >= 0 && !ok[lo]) lo--; while (hi < arr.length && !ok[hi]) hi++;
      if (lo < 0) return arr[hi]; if (hi >= arr.length) return arr[lo]; const t = (k - lo) / (hi - lo); return arr[lo] + t * (arr[hi] - arr[lo]); }); };
  const smooth = (arr) => arr.map((v, k) => (arr[Math.max(0, k - 1)] + 2 * v + arr[Math.min(arr.length - 1, k + 1)]) / 4);
  a = smooth(fill(a)); b = smooth(fill(b));
  return { a: a.map((v) => +v.toFixed(4)), b: b.map((v) => +v.toFixed(4)) };
}
/** the gradient's colour at lightness L → out [a, b] */
export function gradientAt(g, L, out = [0, 0]) {
  const N = g.a.length, x = Math.min(1, Math.max(0, L)) * (N - 1), k = Math.min(N - 2, Math.floor(x)), t = x - k;
  out[0] = g.a[k] + t * (g.a[k + 1] - g.a[k]); out[1] = g.b[k] + t * (g.b[k + 1] - g.b[k]);
  return out;
}
