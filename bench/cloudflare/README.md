# Cloudflare Workers benchmark

A Worker that computes Walrus blob IDs with four WebAssembly modules, to compare this crate's
streaming encoder against stock `walrus-core` on Cloudflare itself:

| Module | What |
|---|---|
| `ours_simd128.wasm` | `walrus-blob-encoder-wasm`, `+simd128` |
| `ours_relaxed.wasm` | `walrus-blob-encoder-wasm`, `+simd128,+relaxed-simd` |
| `stock.wasm` | `walrus-core` 1.61.0 (`compute_blob_id`), release build, no target features |
| `stock_simd128.wasm` | the same with `+simd128` |

The blob is a fixed pseudo-random pattern of prime length repeated to the requested size
(`src/bench.mjs`), so a read costs the host a few copies and no storage is involved. The clock
inside a Worker does not advance during computation, so `runbench.mjs` takes each request's CPU
time from the live tail (`wrangler tail`).

## Build and run

```sh
# from the repository root
RUSTFLAGS="-C target-feature=+simd128" cargo +stable build -p walrus-blob-encoder-wasm --profile wasm \
  --target wasm32-unknown-unknown --target-dir target/wasm-simd128
RUSTFLAGS="-C target-feature=+simd128,+relaxed-simd" cargo +stable build -p walrus-blob-encoder-wasm --profile wasm \
  --target wasm32-unknown-unknown --target-dir target/wasm-relaxed
mkdir -p bench/cloudflare/wasm
cp target/wasm-simd128/wasm32-unknown-unknown/wasm/walrus_blob_encoder_wasm.wasm bench/cloudflare/wasm/ours_simd128.wasm
cp target/wasm-relaxed/wasm32-unknown-unknown/wasm/walrus_blob_encoder_wasm.wasm bench/cloudflare/wasm/ours_relaxed.wasm

cd bench/cloudflare/stock-wasm
RUSTFLAGS='--cfg getrandom_backend="custom"' cargo +stable build --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/walrus_stock_wasm.wasm ../wasm/stock.wasm
RUSTFLAGS='--cfg getrandom_backend="custom" -C target-feature=+simd128' cargo +stable build --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/walrus_stock_wasm.wasm ../wasm/stock_simd128.wasm

cd ..
node local.mjs ours-relaxed stream 5000000 32          # the same code path under Node
openssl rand -hex 16 > .bench-key
wrangler deploy --var BENCH_KEY:$(cat .bench-key)       # /run needs ?key=
node runbench.mjs https://<worker>.workers.dev .bench-key wrangler plan.json > results.jsonl
```

`stock-wasm` registers an entropy source that always fails: `walrus-core` links `getrandom`
but computing a blob ID never calls it, and `wasm32-unknown-unknown` has none.

## Results (2026-11-10)

CPU time as Cloudflare reports it per request, best of 2–3 runs, with the WebAssembly memory
the module grew to. Every blob ID equals native `walrus-core`'s. Run-to-run variation on
Cloudflare is large (identical requests differed by up to 50%).

| Blob | Stock | Stock, simd128 | Ours, in memory | Ours, streaming (32 MiB budget) |
|---|---|---|---|---|
| 1 MB | 1.88 s / 43 MiB | 1.54 s / 43 MiB | 0.61 s / 4.5 MiB | 0.59 s / 4.5 MiB |
| 5 MB | 2.07 s / 49 MiB | 1.65 s / 49 MiB | 0.59 s / 10 MiB | 0.62 s / 10 MiB |
| 10 MB | 1.93 s / 56 MiB | 1.60 s / 56 MiB | 0.62 s / 17 MiB | 0.63 s / 17 MiB |
| 20 MB | 3.57 s / 70 MiB | 2.34 s / 70 MiB | 0.67 s / 31 MiB | 0.65 s / 31 MiB |
| 40 MB | 5.13 s / 99 MiB | 3.02 s / 99 MiB | 0.96 s / 60 MiB | 0.92 s / 22 MiB |
| 60 MB | 7.41 s / 128 MiB | 5.51 s / 128 MiB | 1.19 s / 89 MiB | 1.15 s / 32 MiB |
| 100 MB | – | – | 1.60 s / 146 MiB | 1.66 s / 35 MiB |

Streaming, `+simd128`:

| Blob | Budget | CPU time | Memory | Passes |
|---|---|---|---|---|
| 100 MB | 8 MiB | 3.85 s | 11 MiB | 7 |
| 100 MB | 32 MiB | 1.66 s | 35 MiB | 2 |
| 259 MB | 32 MiB | 4.09 s | 36 MiB | 4 |
| 1 GB | 16 MiB | 47.4 s | 26 MiB | 31 |
| 1 GB | 32 MiB | 27.7 s | 41 MiB | 16 |
| 1 GB | 64 MiB | 20.6 s | 70 MiB | 8 |
| 1 GB | 84 MiB | 17.7 s | 90 MiB | 6 |

`+relaxed-simd` against `+simd128`, alternating on the same requests: faster in all four pairs
at 100 MB (1.42–1.68 s against 1.65–1.90 s) and all three at 1 GB with a 64 MiB budget
(14.8–19.3 s against 21.3–23.4 s). About 0.6 s is a floor at any size: the 1000 × 1000 symbol
grid is always hashed in full (about 3 million BLAKE2b compressions).
