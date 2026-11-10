// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

// node gen.mjs <len> <path>   writes the test blob of the given length
import fs from "node:fs";
import { fill } from "./src/bench.mjs";
const [len, path] = [Number(process.argv[2]), process.argv[3]];
const fd = fs.openSync(path, "w");
const buf = new Uint8Array(1 << 24);
for (let off = 0; off < len; off += buf.length) {
  const n = Math.min(buf.length, len - off);
  fill(buf.subarray(0, n), off);
  fs.writeSync(fd, buf, 0, n);
}
