// The source's audio track, carried into the export unchanged (no decode / re-encode, so no quality loss):
// the MP4's index is read with mp4box.js (mp4index.js) and its AAC samples are written as they are next to the graded video.
// Only AAC in MP4 / MOV is carried (what the muxer supports); other cases return a reason instead.
import { sampleReader } from './mp4index.js';

const DECODER_CONFIG = 4, DEC_SPECIFIC_INFO = 5; // MPEG-4 descriptor tags

/**
 * blob: the source File; src: its parsed index ({file, info} from openMp4, null when it could not be read).
 * Returns {track, samples} or {skipped: reason}.
 * track: {codec, sampleRate, numberOfChannels, description}; samples: [{data, timestamp, duration}] in µs.
 */
export async function extractAudio(blob, src) {
  if (!src) return { skipped: '無法讀取影片檔的結構（不是 MP4／MOV？）' };
  const { file, info } = src;
  const t = info.audioTracks && info.audioTracks[0];
  if (!t) return { skipped: '原片沒有音軌' };
  if (!/^mp4a\.40/.test(t.codec)) return { skipped: `原片音訊格式 ${t.codec} 不是 AAC，暫不支援保留` };
  const entry = file.getTrackById(t.id).mdia.minf.stbl.stsd.entries[0];
  const esds = entry.esds || (entry.wave && entry.wave.esds);
  const dcd = esds && esds.esd && esds.esd.findDescriptor(DECODER_CONFIG);
  const dsi = dcd && dcd.findDescriptor(DEC_SPECIFIC_INFO);
  if (!dsi || !dsi.data) return { skipped: '原片的 AAC 音訊缺少解碼設定' };
  const recs = file.getTrackById(t.id).samples || [];
  if (!recs.length) return { skipped: '原片音軌沒有樣本' };
  const read = sampleReader(blob, recs), samples = []; // (AAC is small: ~1 MB a minute)
  for (let k = 0; k < recs.length; k++) { const s = recs[k]; samples.push({ data: new Uint8Array(await read(k)), timestamp: Math.round((s.cts * 1e6) / s.timescale), duration: Math.round((s.duration * 1e6) / s.timescale) }); }
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
