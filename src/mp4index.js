// Reading an MP4 / MOV without loading the whole file into memory. Phone clips are often 0.5–2 GB; holding the file
// (and, before, a second copy of every video sample) next to the AI models and the frames in flight ran the tab
// out of memory. Here only the index (moov box) is parsed, reading the file piece by piece with Blob.slice and
// skipping the media data (mp4box tells where the next box starts); sample data is read later, a window of a few
// MB at a time, when it is needed.

const PARSE_CHUNK = 1 << 20; // 1 MB per read while looking for the index
const WINDOW = 4 << 20; // sample data read in windows of up to 4 MB

/**
 * Parse the file's index. blob: File / Blob. Returns {file (mp4box ISOFile), info} or null (not an MP4 / MOV,
 * or no index found).
 */
export async function openMp4(blob) {
  const { createFile, MP4BoxBuffer } = await import('../vendor/mp4box/mp4box.all.mjs');
  const file = createFile();
  let info = null, failed = false;
  file.onError = () => { failed = true; };
  file.onReady = (i) => { info = i; };
  let pos = 0, reads = 0;
  while (!info && !failed && pos < blob.size && reads < 4096) {
    const ab = await blob.slice(pos, pos + PARSE_CHUNK).arrayBuffer();
    reads++;
    const end = pos + ab.byteLength;
    const next = file.appendBuffer(MP4BoxBuffer.fromArrayBuffer(ab, pos), end >= blob.size);
    // next: where the parser wants to continue (past a media-data box it skipped, or the end of what it has)
    pos = next > pos ? next : end;
  }
  if (!info || failed) return null;
  return { file, info };
}

/**
 * Sample data on demand. samples: mp4box sample records ({offset, size}, in the order they will be asked for).
 * get(k) → Promise<Uint8Array> of sample k. Reads a window covering the next samples whose data follow on in the
 * file (up to WINDOW bytes), so a clip is read in a few hundred reads, and keeps only the current window.
 */
export function sampleReader(blob, samples) {
  let win = null, winStart = 0, winEnd = 0;
  return async function get(k) {
    const s = samples[k];
    if (!win || s.offset < winStart || s.offset + s.size > winEnd) {
      let end = s.offset + s.size;
      for (let j = k + 1; j < samples.length; j++) { // extend over the following samples while they stay close and in order
        const q = samples[j];
        if (q.offset < s.offset || q.offset > end + (1 << 20) || q.offset + q.size - s.offset > WINDOW) break; // (small gaps: interleaved audio)
        end = Math.max(end, q.offset + q.size);
      }
      winStart = s.offset; winEnd = end;
      win = await blob.slice(winStart, winEnd).arrayBuffer();
    }
    return new Uint8Array(win, s.offset - winStart, s.size);
  };
}
