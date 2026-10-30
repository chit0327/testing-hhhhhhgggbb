// AI segmentation in its own thread during processing, so it runs at the same time as decoding (main thread)
// and grading (render worker). A CLASSIC worker on purpose: MediaPipe's loader uses importScripts, which module
// workers do not have; the segmenter code itself is the same module as on the main thread (segment.js),
// loaded with import().
let mod = null;
self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      mod = await import('./segment.js');
      await mod.initSegmenter();
      self.postMessage({ type: 'ready', id: m.id });
    } else if (m.type === 'seg') {
      const r = mod.segment(new ImageData(new Uint8ClampedArray(m.data), m.w, m.h), m.mw, m.mh);
      self.postMessage({ type: 'seg', id: m.id, r }, Object.values(r).map((g) => g.buffer));
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: m.id, message: err && err.message ? err.message : String(err) });
  }
};
