// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

// Shared by the Worker and the local Node harness.
// The test blob is a fixed pseudo-random pattern of prime length repeated to the requested
// size, so any (offset, length) read is a few memcpys and costs the host almost nothing.
const PERIOD = 1048573;
let pattern;
export function getPattern() {
  if (pattern) return pattern;
  pattern = new Uint8Array(PERIOD);
  let x = 0x9e3779b9 | 0;
  for (let i = 0; i < PERIOD; i++) {
    x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
    pattern[i] = x & 0xff;
  }
  return pattern;
}
export function fill(dest, offset) {
  const p = getPattern();
  let pos = offset % PERIOD;
  let done = 0;
  while (done < dest.length) {
    const n = Math.min(PERIOD - pos, dest.length - done);
    dest.set(p.subarray(pos, pos + n), done);
    done += n;
    pos = 0;
  }
}
const STATS = ["symbol_size", "batch_columns", "batch_bytes", "row_buffer_bytes",
  "systematic_batches", "full_passes", "bytes_read", "read_calls", "row_encodes",
  "column_encodes", "row_encoding_ms", "column_encoding_ms", "hashing_ms"];
const b64url = (bytes) => {
  let s = "";
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s).replaceAll("+", "-").replaceAll("/", "_").replaceAll("=", "");
};
function fillMemory(w, ptr, len) {
  for (let off = 0; off < len; off += 1 << 24) {
    const n = Math.min(1 << 24, len - off);
    fill(new Uint8Array(w.memory.buffer, ptr + off, n), off);
  }
}
// mode: "stream" (ours, budget in MiB), "memory" (ours, in-memory API), "stock" (walrus-core).
export function run(module, mode, len, budgetMib) {
  const instance = new WebAssembly.Instance(module, { env: { now_ms: () => 0 } });
  const w = instance.exports;
  const startMemory = w.memory.buffer.byteLength;
  let id, stats = null;
  if (mode === "stock") {
    const blob = w.blob_alloc(len);
    fillMemory(w, blob, len);
    const ptr = w.blob_id(blob, len, 1000);
    if (ptr === 0) throw new Error("blob_id failed");
    id = b64url(new Uint8Array(w.memory.buffer, ptr, 32));
  } else if (mode === "memory") {
    const blob = w.blob_alloc(len);
    fillMemory(w, blob, len);
    const ptr = w.blob_metadata(blob, len, 1000);
    if (ptr === 0) throw new Error("blob_metadata failed");
    id = b64url(new Uint8Array(w.memory.buffer, ptr, 32));
  } else {
    const h = w.enc_new(len, 1000, budgetMib * 1048576, 0);
    if (h === 0) throw new Error("enc_new failed");
    while (w.enc_next(h)) {
      fill(new Uint8Array(w.memory.buffer, w.enc_dest(h), w.enc_req_len(h)), w.enc_req_offset(h));
      if (!w.enc_complete(h)) throw new Error("enc_complete failed");
    }
    const ptr = w.enc_finish(h);
    if (ptr === 0) throw new Error("enc_finish failed");
    const out = new Uint8Array(w.memory.buffer, ptr, 32 + 64000 + 8 * STATS.length).slice();
    id = b64url(out.subarray(0, 32));
    const values = new Float64Array(out.slice(32 + 64000).buffer);
    stats = Object.fromEntries(STATS.slice(0, 10).map((k, i) => [k, values[i]]));
  }
  return { blob_id: id, start_wasm_memory: startMemory, wasm_memory: w.memory.buffer.byteLength, stats };
}
