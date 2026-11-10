// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

// Usage: node|bun run.mjs <module.wasm> <file> [budget_mib=64] [n_shards=1000]
//          [--async] [--no-timing] [--scalar] [--memory] [--selftest]
//
// Streams <file> through the pull protocol of walrus-blob-encoder-wasm, reading each request
// into wasm memory (synchronously, or with --async through a promise per read), and prints the
// blob ID and a JSON line of counters and times. --memory instead copies the whole file into
// wasm memory and uses the in-memory encoder. --selftest runs the module's BLAKE2b self-test
// (build with --features selftest) and exits non-zero on a mismatch.
import fs from "node:fs";

const args = process.argv.slice(2);
const flags = new Set(args.filter((a) => a.startsWith("--")));
const [wasmPath, filePath, budgetMib = "64", nShardsArg = "1000"] = args.filter(
  (a) => !a.startsWith("--"),
);
const nShards = Number(nShardsArg);
const STATS = ["symbol_size", "batch_columns", "batch_bytes", "row_buffer_bytes",
  "systematic_batches", "full_passes", "bytes_read", "read_calls", "row_encodes",
  "column_encodes", "row_encoding_ms", "column_encoding_ms", "hashing_ms"];

const cpu0 = process.cpuUsage();
const t0 = performance.now();
const { instance } = await WebAssembly.instantiate(fs.readFileSync(wasmPath), {
  env: { now_ms: () => performance.now() },
});
const w = instance.exports;
const runtime = typeof Bun !== "undefined" ? `bun ${Bun.version}` : `node ${process.version}`;

if (flags.has("--selftest")) {
  const failures = w.selftest_blake2b();
  console.log(JSON.stringify({ runtime, selftest_blake2b_failures: failures, backend: w.backend() }));
  process.exit(failures === 0 ? 0 : 1);
}

const fd = fs.openSync(filePath, "r");
const len = fs.fstatSync(fd).size;
let maxMem = w.memory.buffer.byteLength;
let ioMs = 0;
let reads = 0;
let ptr;
const tStart = performance.now();
if (flags.has("--memory")) {
  const blob = w.blob_alloc(len);
  const t = performance.now();
  for (let off = 0; off < len; off += 1 << 24) {
    const n = Math.min(1 << 24, len - off);
    fs.readSync(fd, new Uint8Array(w.memory.buffer, blob + off, n), 0, n, off);
  }
  ioMs += performance.now() - t;
  maxMem = Math.max(maxMem, w.memory.buffer.byteLength);
  ptr = w.blob_metadata(blob, len, nShards);
  if (ptr === 0) throw new Error("blob_metadata failed");
} else {
  const fh = flags.has("--async") ? await fs.promises.open(filePath, "r") : null;
  let flagBits = flags.has("--no-timing") ? 0 : 1;
  if (flags.has("--scalar")) flagBits |= 2;
  const h = w.enc_new(len, nShards, Number(budgetMib) * 1048576, flagBits);
  if (h === 0) throw new Error("enc_new failed");
  while (w.enc_next(h)) {
    const offset = w.enc_req_offset(h);
    const n = w.enc_req_len(h);
    const dest = new Uint8Array(w.memory.buffer, w.enc_dest(h), n);
    const t = performance.now();
    if (fh) {
      const { bytesRead } = await fh.read(dest, 0, n, offset);
      if (bytesRead !== n) throw new Error("short read");
    } else if (fs.readSync(fd, dest, 0, n, offset) !== n) {
      throw new Error("short read");
    }
    ioMs += performance.now() - t;
    if (!w.enc_complete(h)) throw new Error("enc_complete failed");
    reads++;
  }
  maxMem = Math.max(maxMem, w.memory.buffer.byteLength);
  ptr = w.enc_finish(h);
  if (ptr === 0) throw new Error("enc_finish failed");
}
const wall = performance.now() - tStart;
maxMem = Math.max(maxMem, w.memory.buffer.byteLength);
const out = new Uint8Array(w.memory.buffer, ptr, 32 + 64 * nShards + 8 * STATS.length).slice();
const values = new Float64Array(out.slice(32 + 64 * nShards).buffer);
const stats = Object.fromEntries(STATS.map((k, i) => [k, Math.round(values[i] * 10) / 10]));
const cpu = process.cpuUsage(cpu0);
console.log("blob_id", Buffer.from(out.slice(0, 32)).toString("base64url"));
console.log(JSON.stringify({
  runtime, len, budget_mib: Number(budgetMib), mode: flags.has("--memory") ? "memory"
    : flags.has("--async") ? "async" : "sync", ...stats, io_ms: Math.round(ioMs),
  host_reads: reads, wall_ms: Math.round(wall), total_ms: Math.round(performance.now() - t0),
  cpu_ms: Math.round((cpu.user + cpu.system) / 1000), max_wasm_memory: maxMem,
  rss: process.memoryUsage().rss,
}));
