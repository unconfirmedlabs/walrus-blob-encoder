// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

// node local.mjs <impl> <mode> <len> [budgetMib]   (same code path as the Worker, under Node)
import fs from "node:fs";
import { run } from "./src/bench.mjs";
const [impl, mode, len, budget = "32"] = process.argv.slice(2);
const file = { "ours-simd128": "ours_simd128", "ours-relaxed": "ours_relaxed", stock: "stock", "stock-simd128": "stock_simd128" }[impl];
const module = new WebAssembly.Module(fs.readFileSync(new URL(`./wasm/${file}.wasm`, import.meta.url)));
const t0 = performance.now();
const r = run(module, mode, Number(len), Number(budget));
console.log(JSON.stringify({ impl, mode, len: Number(len), budget: Number(budget), elapsed_ms: Math.round(performance.now() - t0), ...r, stats: undefined }));
