# Benchmarks

## Iteration 4: the streaming engine (local)

AMD EPYC 7313P (Zen 3, 16 cores / 32 threads, AVX2, no AVX-512), 62 GiB RAM, a shared host:
differences under about 3% are noise. Native builds use Rust 1.94.1 with `-C target-cpu=native`
(256-bit `Wide` Reed-Solomon, AVX2 BLAKE2b) or `-C target-feature=+ssse3,+sse4.1` with the
`sse41` backend (128-bit everything); WebAssembly builds use Rust 1.99.1 (stable; the pinned
toolchain has no wasm32 target installed here) with `+simd128`, or `+simd128,+relaxed-simd`,
under Node v24.22.1 and Bun 1.4.4. Every value is the best of three runs
(`scripts/measure.py`), and every run's blob ID was checked against walrus-core d6c6d3c. Inputs
are `/dev/urandom` bytes of 5000000, 100000000, 259209074 (the size of the 259 MB file of the
earlier iterations) and 1000000000 bytes, read from the page cache. The AVX-512 engine is not
benchmarked: this CPU has no AVX-512.

Columns: wall and CPU (user + system) time of the whole process; maximum RSS of the process
(for Node and Bun, the runtime's own 50 to 70 MB included); the peak WebAssembly linear memory;
the module's own phase times (row and column Reed-Solomon, hashing) and the time spent in reads;
full passes over the blob; bytes requested and read calls. Streaming reads the file with
positioned reads, so its RSS excludes the blob; the in-memory mode reads the whole file first.

```sh
BUILD=native scripts/bench_native.sh stream:32 stream:64 stream:84 stream:256 memory:1 memory:4 memory:16 -- <files>
BUILD=ssse3 scripts/bench_native.sh stream:32 stream:64 stream:84 stream:256 memory:1 -- <files>
scripts/bench_wasm.sh node bun -- 32 64 84 256 -- <files>
MODE=async scripts/bench_wasm.sh node bun -- 32 -- <files>
FEATURES=relaxed scripts/bench_wasm.sh node bun -- 32 64 256 -- <files>
```

### In-memory: the encoder of 88a9fed against the streaming engine

The in-memory entry points now run the streaming engine's expansion (ITERATIONS.md, iteration
4). Same machine and inputs, `target-cpu=native` for every build. "88a9fed" is that revision's
in-memory encoder with reed-solomon-simd 3.1.0 (its AVX2 engine); "88a9fed code, new kernels"
is the same encoder body on this branch's Reed-Solomon crate, measured before it was deleted;
"streaming engine in memory" is `BlobEncoder::metadata` now. The streaming columns read the
file with a budget for a single pass (2048 MiB), with 256 MiB (two passes for 1 GB) and with
32 MiB. Each cell: wall time, CPU time, maximum RSS.

| Blob | Threads | 88a9fed in-memory | 88a9fed code, new kernels | streaming engine in memory (final) | streaming, one-pass budget | streaming, W=256 MiB | streaming, W=32 MiB |
|---|---|---|---|---|---|---|---|
| 5 MB | 1 | 0.23 s, 0.23 CPU, 48 MiB | 0.22 s, 0.22 CPU, 40 MiB | 0.19 s, 0.19 CPU, 11 MiB | 0.20 s, 0.19 CPU, 11 MiB | 0.20 s, 0.20 CPU, 11 MiB | 0.19 s, 0.18 CPU, 11 MiB |
| 5 MB | 4 | 0.09 s, 0.26 CPU, 50 MiB | 0.08 s, 0.28 CPU, 42 MiB | 0.07 s, 0.22 CPU, 20 MiB | - | - | - |
| 5 MB | 16 | 0.06 s, 0.40 CPU, 60 MiB | 0.05 s, 0.52 CPU, 52 MiB | 0.04 s, 0.34 CPU, 25 MiB | - | - | - |
| 100 MB | 1 | 0.59 s, 0.59 CPU, 185 MiB | 0.57 s, 0.57 CPU, 177 MiB | 0.55 s, 0.55 CPU, 147 MiB | 0.55 s, 0.54 CPU, 148 MiB | 0.55 s, 0.54 CPU, 148 MiB | 0.55 s, 0.55 CPU, 36 MiB |
| 100 MB | 4 | 0.23 s, 0.61 CPU, 186 MiB | 0.23 s, 0.67 CPU, 179 MiB | 0.22 s, 0.62 CPU, 159 MiB | - | - | - |
| 100 MB | 16 | 0.14 s, 0.82 CPU, 192 MiB | 0.16 s, 1.14 CPU, 189 MiB | 0.14 s, 0.99 CPU, 169 MiB | - | - | - |
| 259 MB | 1 | 1.21 s, 1.20 CPU, 413 MiB | 1.21 s, 1.20 CPU, 406 MiB | 1.17 s, 1.17 CPU, 376 MiB | 1.15 s, 1.14 CPU, 376 MiB | 1.13 s, 1.13 CPU, 253 MiB | 1.34 s, 1.33 CPU, 37 MiB |
| 259 MB | 4 | 0.46 s, 1.25 CPU, 417 MiB | 0.46 s, 1.27 CPU, 410 MiB | 0.46 s, 1.26 CPU, 390 MiB | - | - | - |
| 259 MB | 16 | 0.28 s, 1.49 CPU, 430 MiB | 0.29 s, 1.79 CPU, 428 MiB | 0.29 s, 1.75 CPU, 409 MiB | - | - | - |
| 1 GB | 1 | 4.03 s, 4.02 CPU, 1478 MiB | 4.03 s, 4.02 CPU, 1470 MiB | 4.00 s, 4.00 CPU, 1440 MiB | 3.97 s, 3.96 CPU, 1438 MiB | 3.85 s, 3.84 CPU, 250 MiB | 9.14 s, 9.13 CPU, 42 MiB |
| 1 GB | 4 | 1.56 s, 4.11 CPU, 1491 MiB | 1.59 s, 4.24 CPU, 1484 MiB | 1.56 s, 4.10 CPU, 1468 MiB | - | - | - |
| 1 GB | 16 | 0.99 s, 4.96 CPU, 1543 MiB | 1.01 s, 5.23 CPU, 1542 MiB | 1.00 s, 5.06 CPU, 1526 MiB | - | - | - |

### Against the prototype

The prototype the streaming engine was ported from (RustCrypto BLAKE2b one message at a time,
1000 incremental row trees, unaligned batches, two Reed-Solomon encoders, the fork's 128-bit
`Wide` engine with its 8 MiB table), rebuilt and re-measured on the same machine, next to this
branch (`simd128`, and `simd128` + `relaxed-simd`, under Node; native with `target-cpu=native`).
Each cell: wall time, peak linear memory (WebAssembly) or maximum RSS (native).

| Blob, budget | Prototype (Node) | This branch, `simd128` (Node) | This branch, `+relaxed-simd` (Node) | Prototype, native | This branch, native |
|---|---|---|---|---|---|
| 5 MB, 32 MiB | 0.72 s, 17.7 MiB | 0.52 s, 10.2 MiB | 0.49 s, 10.2 MiB | - | 0.19 s, 11.2 MiB |
| 100 MB, 32 MiB | 1.82 s, 43.2 MiB (3 passes) | 1.36 s, 35.2 MiB (2 passes) | 1.23 s, 35.2 MiB | - | 0.55 s, 36.5 MiB |
| 1 GB, 32 MiB | 24.22 s, 53.1 MiB | 19.81 s, 41.2 MiB | 16.52 s, 41.2 MiB | - | 9.16 s, 42.5 MiB |
| 1 GB, 64 MiB | 16.91 s, 84.6 MiB | 13.68 s, 69.9 MiB | 11.69 s, 69.9 MiB | 11.63 s, 85.5 MiB | 6.06 s, 71.0 MiB |
| 1 GB, 256 MiB | 11.72 s, 276.3 MiB (3 passes) | 9.17 s, 248.6 MiB (2 passes) | 8.15 s, 248.6 MiB | - | 3.82 s, 249.8 MiB |

### Native, AVX2 (`-C target-cpu=native`)

| Blob | Mode | Wall s | CPU s | Max RSS MiB | Row RS s | Column RS s | Hash s | I/O s | Passes | Bytes read | Reads |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 MB | stream W=32 MiB | 0.19 | 0.18 | 11.2 | 0.01 | 0.03 | 0.15 | 0.00 | 1 | 5000000 | 313 |
| 5 MB | stream W=64 MiB | 0.19 | 0.19 | 11.2 | 0.01 | 0.03 | 0.15 | 0.00 | 1 | 5000000 | 313 |
| 5 MB | stream W=84 MiB | 0.19 | 0.18 | 11.2 | 0.01 | 0.03 | 0.15 | 0.00 | 1 | 5000000 | 313 |
| 5 MB | stream W=256 MiB | 0.19 | 0.19 | 11.2 | 0.01 | 0.03 | 0.15 | 0.00 | 1 | 5000000 | 313 |
| 5 MB | memory x1 | 0.19 | 0.19 | 11.0 | - | - | - | - | - | - | - |
| 5 MB | memory x4 | 0.07 | 0.22 | 20.5 | - | - | - | - | - | - | - |
| 5 MB | memory x16 | 0.04 | 0.34 | 25.2 | - | - | - | - | - | - | - |
| 100 MB | stream W=32 MiB | 0.55 | 0.54 | 36.5 | 0.08 | 0.13 | 0.28 | 0.06 | 2 | 300000000 | 1668 |
| 100 MB | stream W=64 MiB | 0.51 | 0.50 | 52.2 | 0.04 | 0.13 | 0.28 | 0.05 | 1 | 200000000 | 1001 |
| 100 MB | stream W=84 MiB | 0.51 | 0.51 | 52.2 | 0.05 | 0.13 | 0.29 | 0.05 | 1 | 200000000 | 1001 |
| 100 MB | stream W=256 MiB | 0.54 | 0.53 | 147.5 | 0.07 | 0.13 | 0.28 | 0.06 | 1 | 100000000 | 334 |
| 100 MB | memory x1 | 0.55 | 0.55 | 147.2 | - | - | - | - | - | - | - |
| 100 MB | memory x4 | 0.22 | 0.62 | 158.8 | - | - | - | - | - | - | - |
| 100 MB | memory x16 | 0.14 | 0.99 | 168.9 | - | - | - | - | - | - | - |
| 259 MB | stream W=32 MiB | 1.31 | 1.30 | 36.8 | 0.33 | 0.29 | 0.53 | 0.16 | 4 | 1296045370 | 4007 |
| 259 MB | stream W=64 MiB | 1.12 | 1.12 | 67.5 | 0.18 | 0.29 | 0.53 | 0.12 | 2 | 777627222 | 2004 |
| 259 MB | stream W=84 MiB | 1.14 | 1.14 | 88.2 | 0.18 | 0.29 | 0.53 | 0.13 | 2 | 777627222 | 1670 |
| 259 MB | stream W=256 MiB | 1.12 | 1.11 | 253.0 | 0.11 | 0.29 | 0.53 | 0.17 | 1 | 518418148 | 668 |
| 259 MB | memory x1 | 1.17 | 1.17 | 376.0 | - | - | - | - | - | - | - |
| 259 MB | memory x4 | 0.46 | 1.26 | 390.1 | - | - | - | - | - | - | - |
| 259 MB | memory x16 | 0.29 | 1.75 | 408.8 | - | - | - | - | - | - | - |
| 1 GB | stream W=32 MiB | 9.16 | 9.16 | 42.5 | 4.53 | 1.09 | 1.61 | 1.94 | 16 | 17000000000 | 15695 |
| 1 GB | stream W=64 MiB | 6.06 | 6.05 | 71.0 | 2.31 | 1.09 | 1.60 | 1.05 | 8 | 9000000000 | 8015 |
| 1 GB | stream W=84 MiB | 5.28 | 5.27 | 91.0 | 1.73 | 1.09 | 1.61 | 0.83 | 6 | 7000000000 | 6011 |
| 1 GB | stream W=256 MiB | 3.82 | 3.82 | 249.8 | 0.65 | 1.09 | 1.60 | 0.46 | 2 | 3000000000 | 2004 |
| 1 GB | memory x1 | 4.00 | 4.00 | 1440.5 | - | - | - | - | - | - | - |
| 1 GB | memory x4 | 1.56 | 4.10 | 1467.7 | - | - | - | - | - | - | - |
| 1 GB | memory x16 | 1.00 | 5.06 | 1526.0 | - | - | - | - | - | - | - |

### Native, SSSE3 and SSE4.1 only

| Blob | Mode | Wall s | CPU s | Max RSS MiB | Row RS s | Column RS s | Hash s | I/O s | Passes | Bytes read | Reads |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 MB | stream W=32 MiB | 0.33 | 0.32 | 11.2 | 0.02 | 0.05 | 0.26 | 0.00 | 1 | 5000000 | 313 |
| 5 MB | stream W=64 MiB | 0.33 | 0.32 | 11.2 | 0.02 | 0.05 | 0.26 | 0.00 | 1 | 5000000 | 313 |
| 5 MB | stream W=84 MiB | 0.33 | 0.32 | 11.2 | 0.02 | 0.05 | 0.26 | 0.00 | 1 | 5000000 | 313 |
| 5 MB | stream W=256 MiB | 0.33 | 0.33 | 11.2 | 0.02 | 0.05 | 0.26 | 0.00 | 1 | 5000000 | 313 |
| 5 MB | memory x1 | 0.33 | 0.32 | 10.8 | - | - | - | - | - | - | - |
| 100 MB | stream W=32 MiB | 0.95 | 0.94 | 36.2 | 0.14 | 0.24 | 0.51 | 0.06 | 2 | 300000000 | 1668 |
| 100 MB | stream W=64 MiB | 0.88 | 0.88 | 52.0 | 0.08 | 0.24 | 0.51 | 0.05 | 1 | 200000000 | 1001 |
| 100 MB | stream W=84 MiB | 0.89 | 0.88 | 52.2 | 0.08 | 0.25 | 0.51 | 0.05 | 1 | 200000000 | 1001 |
| 100 MB | stream W=256 MiB | 0.91 | 0.91 | 147.2 | 0.10 | 0.24 | 0.50 | 0.06 | 1 | 100000000 | 334 |
| 100 MB | memory x1 | 0.93 | 0.92 | 147.2 | - | - | - | - | - | - | - |
| 259 MB | stream W=32 MiB | 2.28 | 2.28 | 36.5 | 0.59 | 0.55 | 0.98 | 0.17 | 4 | 1296045370 | 4007 |
| 259 MB | stream W=64 MiB | 1.97 | 1.96 | 67.2 | 0.32 | 0.55 | 0.97 | 0.12 | 2 | 777627222 | 2004 |
| 259 MB | stream W=84 MiB | 1.97 | 1.97 | 88.0 | 0.32 | 0.55 | 0.97 | 0.13 | 2 | 777627222 | 1670 |
| 259 MB | stream W=256 MiB | 1.89 | 1.88 | 252.8 | 0.18 | 0.54 | 0.97 | 0.18 | 1 | 518418148 | 668 |
| 259 MB | memory x1 | 1.95 | 1.94 | 376.0 | - | - | - | - | - | - | - |
| 1 GB | stream W=32 MiB | 14.97 | 14.97 | 42.0 | 8.03 | 2.02 | 3.00 | 1.92 | 16 | 17000000000 | 15695 |
| 1 GB | stream W=64 MiB | 10.19 | 10.18 | 70.8 | 4.12 | 2.01 | 3.00 | 1.05 | 8 | 9000000000 | 8015 |
| 1 GB | stream W=84 MiB | 8.95 | 8.94 | 90.8 | 3.10 | 2.01 | 3.01 | 0.82 | 6 | 7000000000 | 6011 |
| 1 GB | stream W=256 MiB | 6.64 | 6.63 | 249.5 | 1.15 | 2.02 | 2.99 | 0.45 | 2 | 3000000000 | 2004 |
| 1 GB | memory x1 | 6.63 | 6.63 | 1440.2 | - | - | - | - | - | - | - |

### WebAssembly, `simd128`

| Blob | Runtime | Reads | W MiB | Wall s | CPU s | Max RSS MiB | Peak wasm memory MiB | Row RS s | Column RS s | Hash s | I/O s | Passes | Bytes read | Reads |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 MB | node simd128 | sync | 32 | 0.52 | 0.57 | 67.8 | 10.2 | 0.03 | 0.07 | 0.38 | 0.00 | 1 | 5000000 | 313 |
| 5 MB | node simd128 | sync | 64 | 0.51 | 0.55 | 67.6 | 10.2 | 0.03 | 0.07 | 0.38 | 0.00 | 1 | 5000000 | 313 |
| 5 MB | node simd128 | sync | 84 | 0.51 | 0.57 | 67.4 | 10.2 | 0.03 | 0.07 | 0.37 | 0.00 | 1 | 5000000 | 313 |
| 5 MB | node simd128 | sync | 256 | 0.52 | 0.57 | 68.1 | 10.2 | 0.03 | 0.07 | 0.38 | 0.00 | 1 | 5000000 | 313 |
| 5 MB | bun simd128 | sync | 32 | 0.61 | 0.69 | 52.5 | 10.2 | 0.02 | 0.06 | 0.50 | 0.00 | 1 | 5000000 | 313 |
| 5 MB | bun simd128 | sync | 64 | 0.61 | 0.70 | 53.5 | 10.2 | 0.02 | 0.06 | 0.50 | 0.00 | 1 | 5000000 | 313 |
| 5 MB | bun simd128 | sync | 84 | 0.61 | 0.70 | 52.5 | 10.2 | 0.03 | 0.06 | 0.50 | 0.00 | 1 | 5000000 | 313 |
| 5 MB | bun simd128 | sync | 256 | 0.61 | 0.70 | 52.2 | 10.2 | 0.03 | 0.06 | 0.50 | 0.00 | 1 | 5000000 | 313 |
| 100 MB | node simd128 | sync | 32 | 1.36 | 1.41 | 93.0 | 35.2 | 0.19 | 0.33 | 0.73 | 0.04 | 2 | 300000000 | 1668 |
| 100 MB | node simd128 | sync | 64 | 1.26 | 1.33 | 108.9 | 51.2 | 0.11 | 0.33 | 0.73 | 0.03 | 1 | 200000000 | 1001 |
| 100 MB | node simd128 | sync | 84 | 1.28 | 1.34 | 109.0 | 51.2 | 0.11 | 0.34 | 0.73 | 0.03 | 1 | 200000000 | 1001 |
| 100 MB | node simd128 | sync | 256 | 1.30 | 1.38 | 204.0 | 146.3 | 0.11 | 0.32 | 0.72 | 0.02 | 1 | 100000000 | 334 |
| 100 MB | bun simd128 | sync | 32 | 1.57 | 1.66 | 76.8 | 35.2 | 0.19 | 0.32 | 0.96 | 0.04 | 2 | 300000000 | 1668 |
| 100 MB | bun simd128 | sync | 64 | 1.48 | 1.61 | 93.2 | 51.2 | 0.11 | 0.32 | 0.97 | 0.03 | 1 | 200000000 | 1001 |
| 100 MB | bun simd128 | sync | 84 | 1.48 | 1.60 | 92.8 | 51.2 | 0.11 | 0.32 | 0.96 | 0.03 | 1 | 200000000 | 1001 |
| 100 MB | bun simd128 | sync | 256 | 1.53 | 1.64 | 185.2 | 146.3 | 0.12 | 0.32 | 0.96 | 0.02 | 1 | 100000000 | 334 |
| 259 MB | node simd128 | sync | 32 | 3.18 | 3.24 | 95.6 | 35.6 | 0.81 | 0.76 | 1.39 | 0.16 | 4 | 1296045370 | 4007 |
| 259 MB | node simd128 | sync | 64 | 2.75 | 2.84 | 123.3 | 66.4 | 0.44 | 0.74 | 1.39 | 0.10 | 2 | 777627222 | 2004 |
| 259 MB | node simd128 | sync | 84 | 2.77 | 2.86 | 144.9 | 87.1 | 0.44 | 0.74 | 1.40 | 0.10 | 2 | 777627222 | 1670 |
| 259 MB | node simd128 | sync | 256 | 2.65 | 2.73 | 310.0 | 251.8 | 0.24 | 0.75 | 1.40 | 0.06 | 1 | 518418148 | 668 |
| 259 MB | bun simd128 | sync | 32 | 3.61 | 3.73 | 78.0 | 35.6 | 0.80 | 0.73 | 1.88 | 0.15 | 4 | 1296045370 | 4007 |
| 259 MB | bun simd128 | sync | 64 | 3.21 | 3.33 | 109.0 | 66.4 | 0.44 | 0.74 | 1.87 | 0.10 | 2 | 777627222 | 2004 |
| 259 MB | bun simd128 | sync | 84 | 3.22 | 3.32 | 129.2 | 87.1 | 0.44 | 0.73 | 1.87 | 0.09 | 2 | 777627222 | 1670 |
| 259 MB | bun simd128 | sync | 256 | 3.10 | 3.21 | 292.0 | 251.8 | 0.25 | 0.73 | 1.88 | 0.06 | 1 | 518418148 | 668 |
| 1 GB | node simd128 | sync | 32 | 19.81 | 19.88 | 102.3 | 41.2 | 10.79 | 2.73 | 4.30 | 1.92 | 16 | 17000000000 | 15695 |
| 1 GB | node simd128 | sync | 64 | 13.68 | 13.76 | 129.8 | 69.9 | 5.52 | 2.75 | 4.30 | 1.02 | 8 | 9000000000 | 8015 |
| 1 GB | node simd128 | sync | 84 | 12.31 | 12.39 | 150.3 | 89.9 | 4.33 | 2.72 | 4.34 | 0.81 | 6 | 7000000000 | 6011 |
| 1 GB | node simd128 | sync | 256 | 9.17 | 9.25 | 306.1 | 248.6 | 1.62 | 2.72 | 4.29 | 0.34 | 2 | 3000000000 | 2004 |
| 1 GB | bun simd128 | sync | 32 | 21.54 | 21.70 | 81.9 | 41.2 | 11.11 | 2.69 | 5.79 | 1.90 | 16 | 17000000000 | 15695 |
| 1 GB | bun simd128 | sync | 64 | 15.30 | 15.48 | 110.1 | 69.9 | 5.70 | 2.70 | 5.80 | 1.03 | 8 | 9000000000 | 8015 |
| 1 GB | bun simd128 | sync | 84 | 13.63 | 13.76 | 129.2 | 89.9 | 4.30 | 2.69 | 5.78 | 0.78 | 6 | 7000000000 | 6011 |
| 1 GB | bun simd128 | sync | 256 | 10.58 | 10.74 | 287.5 | 248.6 | 1.60 | 2.69 | 5.78 | 0.34 | 2 | 3000000000 | 2004 |
| 5 MB | node simd128 | memory | memory | 0.50 | 0.55 | 66.3 | 10.2 | - | - | - | 0.00 | - | - | 0 |
| 5 MB | bun simd128 | memory | memory | 0.61 | 0.69 | 51.0 | 10.2 | - | - | - | 0.00 | - | - | 0 |
| 100 MB | node simd128 | memory | memory | 1.31 | 1.39 | 203.1 | 146.4 | - | - | - | 0.02 | - | - | 0 |
| 100 MB | bun simd128 | memory | memory | 1.53 | 1.67 | 187.8 | 146.4 | - | - | - | 0.02 | - | - | 0 |
| 100 MB | node simd128 | async | 32 | 1.45 | 1.50 | 94.1 | 35.2 | 0.20 | 0.33 | 0.72 | 0.13 | 2 | 300000000 | 1668 |
| 100 MB | bun simd128 | async | 32 | 1.61 | 1.76 | 80.2 | 35.2 | 0.19 | 0.32 | 0.96 | 0.09 | 2 | 300000000 | 1668 |
| 1 GB | node simd128 | async | 32 | 21.47 | 21.59 | 101.0 | 41.2 | 11.14 | 2.80 | 4.32 | 3.14 | 16 | 17000000000 | 15695 |
| 1 GB | bun simd128 | async | 32 | 23.04 | 23.40 | 86.7 | 41.2 | 11.23 | 2.72 | 5.81 | 3.22 | 16 | 17000000000 | 15695 |

### WebAssembly, `simd128` and `relaxed-simd`

| Blob | Runtime | Reads | W MiB | Wall s | CPU s | Max RSS MiB | Peak wasm memory MiB | Row RS s | Column RS s | Hash s | I/O s | Passes | Bytes read | Reads |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 MB | node relaxed | sync | 32 | 0.49 | 0.53 | 67.0 | 10.2 | 0.02 | 0.05 | 0.37 | 0.00 | 1 | 5000000 | 313 |
| 5 MB | node relaxed | sync | 64 | 0.49 | 0.53 | 67.5 | 10.2 | 0.02 | 0.06 | 0.38 | 0.00 | 1 | 5000000 | 313 |
| 5 MB | node relaxed | sync | 256 | 0.49 | 0.54 | 67.0 | 10.2 | 0.02 | 0.06 | 0.37 | 0.00 | 1 | 5000000 | 313 |
| 5 MB | bun relaxed | sync | 32 | 0.60 | 0.69 | 52.5 | 10.2 | 0.02 | 0.06 | 0.50 | 0.00 | 1 | 5000000 | 313 |
| 5 MB | bun relaxed | sync | 64 | 0.60 | 0.70 | 52.5 | 10.2 | 0.02 | 0.05 | 0.50 | 0.00 | 1 | 5000000 | 313 |
| 5 MB | bun relaxed | sync | 256 | 0.60 | 0.70 | 51.8 | 10.2 | 0.02 | 0.06 | 0.50 | 0.00 | 1 | 5000000 | 313 |
| 100 MB | node relaxed | sync | 32 | 1.23 | 1.29 | 93.0 | 35.2 | 0.15 | 0.25 | 0.72 | 0.04 | 2 | 300000000 | 1668 |
| 100 MB | node relaxed | sync | 64 | 1.15 | 1.21 | 109.3 | 51.2 | 0.09 | 0.24 | 0.72 | 0.03 | 1 | 200000000 | 1001 |
| 100 MB | node relaxed | sync | 256 | 1.19 | 1.26 | 203.0 | 146.3 | 0.09 | 0.24 | 0.72 | 0.02 | 1 | 100000000 | 334 |
| 100 MB | bun relaxed | sync | 32 | 1.52 | 1.63 | 77.2 | 35.2 | 0.17 | 0.29 | 0.97 | 0.04 | 2 | 300000000 | 1668 |
| 100 MB | bun relaxed | sync | 64 | 1.44 | 1.58 | 93.8 | 51.2 | 0.10 | 0.30 | 0.97 | 0.03 | 1 | 200000000 | 1001 |
| 100 MB | bun relaxed | sync | 256 | 1.48 | 1.58 | 186.2 | 146.3 | 0.10 | 0.29 | 0.96 | 0.02 | 1 | 100000000 | 334 |
| 1 GB | node relaxed | sync | 32 | 16.52 | 16.59 | 101.4 | 41.2 | 8.20 | 2.01 | 4.32 | 1.92 | 16 | 17000000000 | 15695 |
| 1 GB | node relaxed | sync | 64 | 11.69 | 11.77 | 129.7 | 69.9 | 4.28 | 2.02 | 4.32 | 0.99 | 8 | 9000000000 | 8015 |
| 1 GB | node relaxed | sync | 256 | 8.15 | 8.20 | 305.6 | 248.6 | 1.21 | 2.04 | 4.35 | 0.35 | 2 | 3000000000 | 2004 |
| 1 GB | bun relaxed | sync | 32 | 20.32 | 20.48 | 81.7 | 41.2 | 10.04 | 2.43 | 5.85 | 1.94 | 16 | 17000000000 | 15695 |
| 1 GB | bun relaxed | sync | 64 | 14.56 | 14.67 | 110.3 | 69.9 | 5.16 | 2.47 | 5.86 | 1.00 | 8 | 9000000000 | 8015 |
| 1 GB | bun relaxed | sync | 256 | 10.21 | 10.32 | 288.8 | 248.6 | 1.46 | 2.44 | 5.80 | 0.35 | 2 | 3000000000 | 2004 |

### Input SHA-256

- 5000000: `ca8ecfe72606c9a838010fdcfd32c7dae9ed61f8a50bc61d523c0f1b5579b2e4`
- 100000000: `e60ff646f26648f05252aebc9acdefa9b8918d060986dc361c054f44c1f68ab9`
- 100000001: `f30c88328e63b5060383973970b6f2aa8ed26616d5a1da350882e1ed87e79c67`
- 259209074: `4e10af9b45dda83f76d53299b9bfe4268b7cdd00d0cf4cf1ab856ff588548998`
- 1000000000: `94ac5b456408f4f285f24ecd9bfb6cfc18878112f77ef727a8bd80d9944c81a9`

## Iterations 0 to 3: the in-memory encoder of 88a9fed (AWS)

These tables measure the in-memory encoder that iteration 4 replaced, with
reed-solomon-simd 3.1.0, on different hardware than iteration 4.


Measurements use isolated AWS c6a.2xlarge instances (AMD EPYC 7R13,
8 vCPUs / 4 physical cores, 16 GiB RAM), Rust 1.94.1, release defaults, CPUs 2–5.
These CPUs are four separate physical cores. Times are seconds. The user authorized
this instance after shared-devbox variability made local comparisons unreliable.
Each accepted change is compared with a fresh baseline on the same instance.
AWS reclaimed two Spot instances; iteration 2 uses an approved On-Demand replacement.
Local exploratory results are below.
Each cell lists 0.7 / 17 / 70 / 259 MB (710603 / 17066416 / 69741071 / 259209074 bytes).
Take the minimum per configuration across two runs, each reporting best of five.
Inputs are high-entropy `/dev/urandom` bytes; all IDs are checked against the pinned walrus-core.
Sub-2% deltas are not treated as demonstrated wins.

```sh
CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=4 taskset -c 2-5 cargo run --release --locked --example bench -- --ref --backend all --threads 1,4 --repeat 5 bench/files/*
```

| Iteration | walrus-core ×1 | avx2 ×1 | avx2 ×4 | sse41 ×1 | sse41 ×4 | scalar ×1 | scalar ×4 |
|---|---|---|---|---|---|---|---|
| 0 | 0.495 / 0.524 / 0.862 / 2.053 | 0.200 / 0.233 / 0.400 / 1.041 | 0.053 / 0.062 / 0.103 / 0.280 | 0.310 / 0.339 / 0.572 / 1.525 | 0.080 / 0.088 / 0.149 / 0.394 | 0.883 / 1.288 / 2.764 / 9.364 | 0.224 / 0.326 / 0.699 / 2.373 |
| 1 | 0.494 / 0.523 / 0.861 / 2.050 | 0.199 / 0.233 / 0.398 / 1.091 | 0.053 / 0.062 / 0.104 / 0.283 | 0.308 / 0.340 / 0.580 / 1.515 | 0.080 / 0.088 / 0.150 / 0.387 | 0.882 / 1.283 / 2.773 / 9.365 | 0.225 / 0.326 / 0.704 / 2.376 |
| 2 | 0.499 / 0.525 / 0.860 / 2.099 | 0.199 / 0.233 / 0.404 / 1.045 | 0.053 / 0.061 / 0.105 / 0.282 | 0.310 / 0.338 / 0.572 / 1.500 | 0.080 / 0.088 / 0.151 / 0.393 | 0.884 / 1.284 / 2.760 / 9.371 | 0.224 / 0.325 / 0.699 / 2.371 |
| 3 | 0.496 / 0.524 / 0.876 / 2.059 | 0.199 / 0.233 / 0.414 / 1.047 | 0.053 / 0.061 / 0.109 / 0.285 | 0.310 / 0.339 / 0.596 / 1.533 | 0.080 / 0.089 / 0.155 / 0.399 | 0.884 / 1.296 / 2.781 / 9.377 | 0.225 / 0.326 / 0.705 / 2.375 |

Iteration 0 was measured on the first Spot instance; iteration 1 on its replacement
following an AWS interruption. Iteration 1 retains no library changes (the rebuilt
benchmark was byte-identical to the saved baseline), so cross-instance wall-time
variation is not an encoder performance delta. Iteration 1's 259 MB peak RSS is 427296 KiB (AVX2, four threads).
After the second Spot interruption, the unchanged iteration 1 code was remeasured
on the On-Demand instance; that fresh baseline is the denominator for iteration 2.

Iteration 2 also retains no encoder changes and has the same executable hash as
the baseline. Its wall-time differences are measurement variation, not accepted
speedups or slowdowns. Peak RSS: 427440 KiB (baseline binary: 427448 KiB).

Iteration 3 also retains no production-code changes. Its executable is byte-identical
to the original baseline; the observed wall-time variation is not an encoder delta.
Final 259 MB peak RSS: 427268 KiB. All runtime experiments were reverted.

### On-Demand comparison baseline

Same c6a.2xlarge instance used for iteration 2, unchanged iteration 1 code.
This is the denominator for the rejected experiments in `ITERATIONS.md`.

| Measurement | walrus-core ×1 | avx2 ×1 | avx2 ×4 | sse41 ×1 | sse41 ×4 | scalar ×1 | scalar ×4 |
|---|---|---|---|---|---|---|---|
| Before iteration 2 | 0.496 / 0.524 / 0.865 / 2.052 | 0.201 / 0.232 / 0.389 / 1.054 | 0.053 / 0.062 / 0.105 / 0.284 | 0.309 / 0.342 / 0.581 / 1.528 | 0.080 / 0.088 / 0.150 / 0.393 | 0.885 / 1.287 / 2.763 / 9.374 | 0.225 / 0.328 / 0.705 / 2.375 |

### Input SHA-256 (iterations 0 to 3)

- 710603: `0a4e79a43a56b528f85419a9026b91882ea3a4ab26a89a49467343136c6a9305`
- 17066416: `850553a71b1c9585713d2f09c64deac71d0b2a73949f6c886a723bb3e17819f4`
- 69741071: `da8a582739196267def5dc5e33a7dea972ba9f48920c6d0701420088be01c11e`
- 259209074: `814dadf5aba0baac2b3df9ea85c83c793977b412856dda264e2383b229370c0f`

### Original shared-devbox baseline

AMD EPYC 7313P, same inputs and commands. Recorded before cloud authorization;
not used as the denominator for EC2 improvements.

| Iteration | walrus-core ×1 | avx2 ×1 | avx2 ×4 | sse41 ×1 | sse41 ×4 | scalar ×1 | scalar ×4 |
|---|---|---|---|---|---|---|---|
| Local | 0.472 / 0.505 / 0.833 / 1.961 | 0.200 / 0.230 / 0.381 / 1.006 | 0.072 / 0.084 / 0.135 / 0.277 | 0.296 / 0.325 / 0.552 / 1.459 | 0.103 / 0.113 / 0.164 / 0.382 | 0.847 / 1.237 / 2.642 / 9.008 | 0.289 / 0.395 / 0.864 / 2.289 |
