// Sequential frame source for processing: the MP4 is demuxed once (mp4box.js) and decoded front to back with
// WebCodecs VideoDecoder, instead of seeking the <video> element for every frame (each seek waits for a
// decode from the previous keyframe and a presentation: tens of ms per frame). Frame selection matches
// grabFrame: for a time t the picture shown at t, i.e. the last frame whose timestamp is ≤ t, on the media
// timeline that starts at the first presented frame (what <video>.currentTime counts from).
// openFrameStream returns null when the file / codec / browser cannot be used this way (the caller falls back
// to seeking): no WebCodecs, not MP4 / MOV, a rotated clip (the <video> element applies the rotation, the
// decoder does not), or a codec the decoder does not support.

const LOOKAHEAD = 6; // decoded frames kept ahead of the one asked for (bounded memory: VideoFrames hold GPU / decoder buffers)

async function demux(bytes) {
  const { createFile, MP4BoxBuffer, DataStream } = await import('../vendor/mp4box/mp4box.all.mjs');
  const file = createFile(), samples = [];
  let info = null, failed = null, track = null;
  file.onError = (e) => { failed = String(e); };
  file.onSamples = (id, user, list) => { for (const s of list) samples.push(s); };
  file.onReady = (i) => {
    info = i; track = i.videoTracks && i.videoTracks[0];
    if (track) { file.setExtractionOptions(track.id, null, { nbSamples: track.nb_samples }); file.start(); }
  };
  file.appendBuffer(MP4BoxBuffer.fromArrayBuffer(bytes, 0));
  file.flush();
  if (failed || !info || !track || !samples.length) return null;
  // decoder description: the codec configuration box (avcC / hvcC / vpcC / av1C) without its 8-byte header
  let description;
  for (const e of file.getTrackById(track.id).mdia.minf.stbl.stsd.entries) {
    const box = e.avcC || e.hvcC || e.vpcC || e.av1C;
    if (box) { const s = new DataStream(undefined, 0, DataStream.BIG_ENDIAN); box.write(s); description = new Uint8Array(s.buffer, 8); break; }
  }
  const m = track.matrix; // [a b u c d v x y w] (16.16 fixed): a rotation shows as b / c ≠ 0 or a / d < 0
  const rotated = m && (m[1] !== 0 || m[3] !== 0 || m[0] < 0 || m[4] < 0);
  return { track, samples, description, rotated };
}

export async function openFrameStream(bytes) {
  if (typeof VideoDecoder === 'undefined' || typeof EncodedVideoChunk === 'undefined') return null;
  let d;
  try { d = await demux(bytes); } catch { return null; }
  if (!d || d.rotated) return null;
  const { track, samples, description } = d;
  const config = { codec: track.codec, codedWidth: track.video.width, codedHeight: track.video.height, ...(description ? { description } : {}), optimizeForLatency: false };
  try { const s = await VideoDecoder.isConfigSupported(config); if (!s.supported) return null; } catch { return null; }
  const origin = Math.min(...samples.map((s) => s.cts)) / track.timescale; // media time 0 = first presented frame
  const ready = []; // decoded frames, presentation order
  let waiting = null, error = null, fed = 0, flushed = false, done = false; // done: flush finished, every frame delivered
  const decoder = new VideoDecoder({
    output: (f) => { ready.push(f); if (waiting) { const w = waiting; waiting = null; w(); } },
    error: (e) => { error = e; if (waiting) { const w = waiting; waiting = null; w(); } },
  });
  decoder.configure(config);
  const feed = () => { // keep the decoder busy, but only a few frames ahead
    while (fed < samples.length && ready.length + decoder.decodeQueueSize < LOOKAHEAD + 4) {
      const s = samples[fed++];
      decoder.decode(new EncodedVideoChunk({ type: s.is_sync ? 'key' : 'delta', timestamp: Math.round(((s.cts / s.timescale) - origin) * 1e6), duration: Math.round((s.duration / s.timescale) * 1e6), data: s.data }));
    }
    if (fed >= samples.length && !flushed) { flushed = true; decoder.flush().then(() => { done = true; if (waiting) { const w = waiting; waiting = null; w(); } }, (e) => { error = error || e; done = true; if (waiting) { const w = waiting; waiting = null; w(); } }); }
  };
  // (the decode queue being empty does not mean every frame is out: the decoder holds the last few for reordering until the flush ends)
  const more = () => new Promise((res) => { waiting = res; feed(); if (done) setTimeout(res, 0); });
  let last = null; // the frame currently "on screen" (kept until a later one is needed)
  return {
    codec: track.codec,
    /** the picture shown at media time t (seconds), drawn at w×h into canvas → ImageData. Times must not decrease. */
    async frameAt(t, w, h, canvas) {
      const us = t * 1e6;
      for (;;) {
        if (error) throw error;
        while (ready.length && ready[0].timestamp <= us) { if (last) last.close(); last = ready.shift(); }
        if (ready.length || done) break; // the next frame is later than t (or none left)
        await more();
      }
      feed();
      if (!last) throw new Error('解碼器沒有輸出畫面');
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(last, 0, 0, w, h);
      return ctx.getImageData(0, 0, w, h);
    },
    close() { if (last) last.close(); for (const f of ready) f.close(); ready.length = 0; if (decoder.state !== 'closed') decoder.close(); },
  };
}
