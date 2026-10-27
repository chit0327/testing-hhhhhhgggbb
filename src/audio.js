// The source's audio track, carried into the export unchanged (no decode / re-encode, so no quality loss):
// the MP4 is demuxed with mp4box.js and its AAC samples are written as they are next to the graded video.
// Only AAC in MP4 / MOV is carried (what the muxer supports); other cases return a reason instead.
const DECODER_CONFIG = 4, DEC_SPECIFIC_INFO = 5; // MPEG-4 descriptor tags

/**
 * bytes: ArrayBuffer of the source file. Returns {track, samples} or {skipped: reason}.
 * track: {codec, sampleRate, numberOfChannels, description}; samples: [{data, timestamp, duration}] in µs.
 */
export async function extractAudio(bytes) {
  const { createFile, MP4BoxBuffer } = await import('../vendor/mp4box/mp4box.all.mjs');
  const file = createFile(), samples = [];
  let info = null, failed = null;
  file.onError = (e) => { failed = String(e); };
  file.onSamples = (id, user, list) => {
    for (const s of list) samples.push({ data: new Uint8Array(s.data), timestamp: Math.round((s.cts * 1e6) / s.timescale), duration: Math.round((s.duration * 1e6) / s.timescale) });
  };
  file.onReady = (i) => { // samples must be requested here: flush() releases the parsed buffers
    info = i;
    const a = i.audioTracks && i.audioTracks[0];
    if (a && /^mp4a\.40/.test(a.codec)) { file.setExtractionOptions(a.id, null, { nbSamples: a.nb_samples }); file.start(); }
  };
  file.appendBuffer(MP4BoxBuffer.fromArrayBuffer(bytes, 0));
  file.flush();
  if (failed) return { skipped: `無法讀取影片檔的音訊（${failed}）` };
  if (!info) return { skipped: '無法讀取影片檔的結構（不是 MP4／MOV？）' };
  const t = info.audioTracks && info.audioTracks[0];
  if (!t) return { skipped: '原片沒有音軌' };
  if (!/^mp4a\.40/.test(t.codec)) return { skipped: `原片音訊格式 ${t.codec} 不是 AAC，暫不支援保留` };
  const entry = file.getTrackById(t.id).mdia.minf.stbl.stsd.entries[0];
  const esds = entry.esds || (entry.wave && entry.wave.esds);
  const dcd = esds && esds.esd && esds.esd.findDescriptor(DECODER_CONFIG);
  const dsi = dcd && dcd.findDescriptor(DEC_SPECIFIC_INFO);
  if (!dsi || !dsi.data) return { skipped: '原片的 AAC 音訊缺少解碼設定' };
  if (!samples.length) return { skipped: '原片音軌沒有樣本' };
  return {
    track: { codec: t.codec, sampleRate: t.audio.sample_rate, numberOfChannels: t.audio.channel_count, description: new Uint8Array(dsi.data) },
    samples,
  };
}

/** Write the samples into an mp4-muxer Muxer that was created with a matching `audio` option. */
export function muxAudio(muxer, audio, maxDurationUs = Infinity) {
  let first = true, n = 0;
  for (const s of audio.samples) {
    if (s.timestamp >= maxDurationUs) break; // keep audio within the video's length
    const meta = first ? { decoderConfig: { codec: audio.track.codec, sampleRate: audio.track.sampleRate, numberOfChannels: audio.track.numberOfChannels, description: audio.track.description } } : undefined;
    muxer.addAudioChunkRaw(s.data, 'key', s.timestamp, s.duration, meta);
    first = false; n++;
  }
  return n;
}
