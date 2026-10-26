# Optimization iterations

## Baseline (iteration 0)

Base: `cb149bd` (v0.1.0). Read the complete issue (no comments), README, src, tests,
and annotated tag before benchmarking. The GitHub release object does not exist;
the annotated tag records the validation history. Generated the four prescribed
high-entropy inputs and ran the prescribed benchmark twice, best of five per run,
with every ID matching walrus-core. See BENCHMARKS.md for per-cell minima and input
hashes. Inputs, raw logs, and rejected patches are locally git-ignored under bench/.

The user subsequently authorized an isolated EC2 Spot instance, temporary bootstrap
packages, a $5 budget, and an eight-hour deadline. The same inputs were transferred
to a c6a.2xlarge (AMD EPYC 7R13, 8 vCPUs / 4 physical cores, 16 GiB RAM). CPUs 2–5
are four distinct physical cores. The complete baseline was repeated there.
The first instance was reclaimed after about 25 minutes; its experiment results
had already been copied locally. A replacement of the same type uses the original
deadline, with regular local checkpoints. No other repositories were modified.

## Iteration 1 — memory traffic and scheduling

### Hypotheses and evidence

1. Reduce Merkle scratch from sixteen to eight trees per task to fit more of the
   working set in a core's 512 KiB L2 cache while retaining full SIMD lanes.
2. Initialize the 32 MB leaf buffer in parallel to distribute allocation/first-touch
   work measured on the shared host.
3. Pack independent short-symbol columns into the otherwise padded positions of a
   64-byte Reed-Solomon work block. A four-byte symbol otherwise uses only two of
   the 32 GF(2^16) coordinates. Preserve each symbol's low/high byte halves while
   packing and unpacking; use the existing engines rather than a new RS engine.

`perf stat -e cycles` was denied on the devbox (`perf_event_paranoid=4`). Temporary
feature-gated timers measured aggregate task wall times and phase wall times.
Encoder times include source copying; `gather_copy` overlaps those totals and must
not be added to them. Allocation timings cover initial buffers and encoder
construction, not every allocator call. The instrumentation was removed after the
experiments, leaving no new library feature or public API.

Warm single-thread AVX2, 0.7 MB: total 207.9 ms, row encoding 9.8 ms,
column encoding 28.0 ms, leaves 48.6 ms, inner hashes 97.8 ms, explicit
allocation 15.2 ms, gather/copy 7.1 ms, Merkle gather 3.4 ms.
259 MB: total 1019.6 ms, row encoding 101.6 ms, column encoding 284.1 ms,
leaves 450.7 ms, inner hashes 98.9 ms, gather/copy 111.0 ms.
Scalar 259 MB spends about 7.4 s in row/column encoding out of 9.0 s total.
Four-thread branch lifetimes are nearly equal on the large input. They are not
worker-idle measurements: Rayon work stealing can make their lifetimes overlap.
The evidence therefore does not justify changing the barrier.

On EC2, hardware cycle counters are unavailable, but `perf record -F 199 -e cpu-clock
--call-graph dwarf` works. The initial small-blob AVX2 sample had 127 samples, about
67% in the BLAKE2b kernel and 10% in the AVX2 RS FFT; this is coarse corroboration,
not a precise phase decomposition.

### Experiments and negative results

- Eight trees alone, shared devbox, best of three: small AVX2 0.204/0.063 s at 1/4
  threads; large AVX2 1.056/0.309 s versus baseline 1.006/0.277 s. Reverted because
  the observed large-input regression exceeded 2%; shared-host noise limits causal
  interpretation.
- Parallel leaf-buffer initialization, shared devbox: small SSE4.1 0.563/0.186 s
  and large AVX2 1.668/0.393 s. Reverted; these noisy observations are not reliable
  estimates of an isolated implementation effect.
- Packed columns passed a focused comparison against independently encoded NoSimd
  columns for every even symbol size 2–32, shard counts 4/7/31/1000, and full and
  partial batches. The prescribed EC2 benchmark was run twice and all IDs matched
  walrus-core. The full correctness gates were not run on this rejected candidate.
  Its complete performance results are below. It improved the small input but
  exceeded the 2% regression limit on several larger SIMD cases.
- An adjacent baseline/candidate comparison (best of seven) cleared the large AVX2
  concern but retained a 17 MB SSE4.1 slowdown: 0.345 versus 0.355 s at one thread.
  Making disabled timers expand directly to their body, removing instrumentation
  entirely, and combining packing with eight-tree Merkle batches did not clear all
  regressions. The packed-column implementation and its dedicated test were reverted.

Candidate seconds, with percentage change versus the isolated baseline; negative
percentages are faster. Sizes are 0.7 / 17 / 70 / 259 MB.

| Backend | Threads | Candidate seconds | Change |
|---|---|---|---|
| avx2 | 1 | 0.178 / 0.230 / 0.406 / 1.082 | -11.0% / -1.3% / +1.5% / +3.9% |
| avx2 | 4 | 0.048 / 0.061 / 0.107 / 0.278 | -9.4% / -1.6% / +3.9% / -0.7% |
| sse41 | 1 | 0.288 / 0.350 / 0.587 / 1.492 | -7.1% / +3.2% / +2.6% / -2.2% |
| sse41 | 4 | 0.075 / 0.091 / 0.153 / 0.391 | -6.2% / +3.4% / +2.7% / -0.8% |
| scalar | 1 | 0.578 / 1.284 / 2.762 / 9.331 | -34.5% / -0.3% / -0.1% / -0.4% |
| scalar | 4 | 0.148 / 0.325 / 0.696 / 2.364 | -33.9% / -0.3% / -0.4% / -0.4% |

### Retained change and memory

The only executable-source change is an opt-in serial test mode:
`WALRUS_BLOB_ENCODER_SERIAL_TESTS=1`. It creates one dedicated pool at a time and
runs every backend and 1/2/3/8-thread configuration. Combine with
`RUST_TEST_THREADS=1`, `RAYON_NUM_THREADS=8`, and `CARGO_BUILD_JOBS=8` to bound
concurrency. Default CI behavior and every correctness assertion remain intact.
README documents the mode.

There is no retained encoder optimization. After reverting the experiments, the
release benchmark executable was byte-identical to the saved baseline:
SHA-256 `352f85bde55ce87a7944f26ddb03b0f211f1d148230003124fda27f0939f0997`.
The accepted library performance delta is therefore zero; fresh wall times on the
replacement instance also serve as the comparison point for iteration 2.

259209074-byte input, fresh process, AVX2, four threads, `/usr/bin/time -v`:
shared-devbox baseline 426732 KiB; isolated baseline 427500 KiB; rejected packed
candidate 427252 KiB. The candidate's 248 KiB difference is negligible and is not
claimed as a memory improvement. Final replacement-instance RSS is 427296 KiB (about 417 MiB).

### Correctness gates

Passed on the replacement instance:

- Full `cargo test --release --locked -- --include-ignored` suites with the backend
  unset, avx2, sse41, and scalar; without default features; and with v3 target flags
  both with and without default features. All metadata, CLI vectors, high-entropy,
  determinism, hash-reference, and RS-reference checks remain intact.
- v4+GFNI all-targets build, using the host-tooling separation described below.
- Fmt, clippy with `-D warnings` with/without default features, and rustdoc with
  warnings denied.
- Scalar Miri on `nightly-2026-10-11`, exactly CI's scope: eight passed, two existing
  slow-under-Miri tests ignored. Those tests passed in the full release suite.
- cargo-deny 0.20.2: advisories, bans, licenses, and sources all passed.

The optional real-file test was not configured; the high-entropy suite was run.
GitHub CI, including the ARM runner, is pending.

The native v4 command failed in the **num-traits host build script** with SIGILL,
not in encoder code: Cargo applied the unsupported target features to executable
host tooling. Use an explicit target to keep host tooling at its baseline while
compiling every target artifact with v4+GFNI:

```sh
RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+gfni" \
  cargo build --release --locked --all-targets --target x86_64-unknown-linux-gnu
```

### Needs AVX-512 validation

None: no retained encoder, AVX-512, or GFNI changes.

## Iteration 2

Pending iteration 1 merge.

## Iteration 3

Pending iteration 2 merge.
