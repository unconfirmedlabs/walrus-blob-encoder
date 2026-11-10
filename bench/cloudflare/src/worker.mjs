// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

import oursSimd from "../wasm/ours_simd128.wasm";
import oursRelaxed from "../wasm/ours_relaxed.wasm";
import stock from "../wasm/stock.wasm";
import stockSimd from "../wasm/stock_simd128.wasm";
import { run } from "./bench.mjs";

const MODULES = { "ours-simd128": oursSimd, "ours-relaxed": oursRelaxed, stock, "stock-simd128": stockSimd };

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname !== "/run" || url.searchParams.get("key") !== env.BENCH_KEY) {
      return new Response("not found", { status: 404 });
    }
    const impl = url.searchParams.get("impl");
    const mode = url.searchParams.get("mode");
    const len = Number(url.searchParams.get("len"));
    const budget = Number(url.searchParams.get("budget") ?? 32);
    const module = MODULES[impl];
    if (!module || !["stream", "memory", "stock"].includes(mode) || !(len >= 0)) {
      return new Response("bad request", { status: 400 });
    }
    // The Workers clock only advances across I/O, so yield to a timer on each side of the run.
    await scheduler.wait(1);
    const t0 = Date.now();
    let result;
    try {
      result = run(module, mode, len, budget);
    } catch (error) {
      return Response.json({ impl, mode, len, budget, error: String(error) }, { status: 500 });
    }
    await scheduler.wait(1);
    const elapsed = Date.now() - t0;
    return Response.json({ impl, mode, len, budget, elapsed_ms: elapsed, ...result });
  },
};
