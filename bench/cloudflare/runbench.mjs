// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

// node runbench.mjs <url> <key-file> <wrangler> <plan.json>  > results.jsonl
// Runs each plan entry against the deployed Worker, sequentially, and joins the response with
// the cpuTime / wallTime / outcome Cloudflare reports for that request on the live tail.
import fs from "node:fs";
import { spawn } from "node:child_process";
const [url, keyFile, wrangler, planFile] = process.argv.slice(2);
const key = fs.readFileSync(keyFile, "utf8").trim();
const plan = JSON.parse(fs.readFileSync(planFile, "utf8"));
const events = new Map();
const tail = spawn(wrangler, ["tail", "walrus-encoder-bench", "--format", "json"], { stdio: ["ignore", "pipe", "ignore"] });
let buf = "";
tail.stdout.on("data", (chunk) => {
  buf += chunk;
  for (;;) {
    // wrangler prints one pretty-printed JSON object after another
    const end = buf.indexOf("\n}\n");
    if (end < 0) break;
    const text = buf.slice(0, end + 2);
    buf = buf.slice(end + 3);
    try {
      const e = JSON.parse(text);
      const rid = new URL(e.event.request.url).searchParams.get("rid");
      if (rid) events.set(rid, { cpu_ms: e.cpuTime, wall_ms: e.wallTime, outcome: e.outcome });
    } catch {}
  }
});
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
await sleep(12000);
let n = 0;
for (const { impl, mode, len, budget = 32, repeat = 1 } of plan) {
  for (let i = 0; i < repeat; i++) {
    const rid = `${Date.now()}-${n++}`;
    const t0 = performance.now();
    let status, body = {};
    try {
      const res = await fetch(`${url}/run?key=${key}&rid=${rid}&impl=${impl}&mode=${mode}&len=${len}&budget=${budget}`);
      status = res.status;
      const text = await res.text();
      try { body = JSON.parse(text); } catch { body = { error: text.slice(0, 120).replace(/\s+/g, " ") }; }
    } catch (error) { status = 0; body = { error: String(error) }; }
    const client_ms = Math.round(performance.now() - t0);
    for (let w = 0; w < 40 && !events.has(rid); w++) await sleep(500);
    const row = { impl, mode, len, budget, status, client_ms, ...(events.get(rid) ?? {}),
      blob_id: body.blob_id, wasm_memory: body.wasm_memory, error: body.error,
      passes: body.stats?.full_passes, bytes_read: body.stats?.bytes_read, read_calls: body.stats?.read_calls };
    console.log(JSON.stringify(row));
    await sleep(1500);
  }
}
tail.kill();
process.exit(0);
