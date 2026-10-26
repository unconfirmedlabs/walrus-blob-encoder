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
replacement instance are recorded in row 1. Iteration 2 was rebaselined again after
the second Spot interruption.

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
All 18 GitHub CI checks passed, including the ARM/NEON runner, before PR #2 was merged.
The AVX-512 runtime job reported that the runner lacked AVX-512; the local v4+GFNI
compile gate passed as described below.

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

## Iteration 2 — kernels

Base: `1df04e6`, after PR #2 merged with green CI. The second Spot instance was
also reclaimed. The user approved an On-Demand c6a.2xlarge at $0.306/hour under the
original $5 total budget and shutdown deadline. Both the accepted baseline and
candidate are remeasured on that same On-Demand instance with the same input bytes.

### Hypothesis and profile evidence

Fresh software `perf` sampling at 499 Hz on the accepted baseline, 0.7 and 259 MB,
one and four threads, two repeats per case: AVX2 spends about 51.85% of sampled
CPU time in its BLAKE2b kernel and 27.84% in RS FFT/IFFT; SSE4.1 spends about 65.97%
in its BLAKE2b kernel. Scalar spends about 77.64% in NoSimd FFT/IFFT and 19.91% in
hashing. Hardware cycle counters are unavailable. The phase decomposition and
allocation/barrier limitations from iteration 1 still apply to this identical
encoder implementation.

### Rejected: 64-byte message specialization

Merkle inner nodes hash 64 bytes plus a one-byte domain prefix. The resulting
65-byte block occupies nine words; the last seven words are always zero. A
length-guarded specialization made those zeros visible to the compiler in scalar,
AVX2, SSE4.1, NEON, and AVX-512. Other lengths retained the generic path.

All local correctness gates passed, including Miri and the v4+GFNI compile. Existing
RustCrypto tests cover 63/64/65-byte boundaries, prefixes 0/1/255, scattered unaligned
messages, counts through 33, full Merkle levels, and ragged SIMD groups.

However, the prescribed two best-of-five runs failed the performance gate. Deltas
below are elapsed time (negative is faster), ordered 0.7 / 17 / 70 / 259 MB:

| Backend / threads | Rejected specialization delta |
|---|---|
| avx2 ×1 | -4.0% / -3.9% / +3.1% / +2.8% |
| avx2 ×4 | -3.8% / -4.8% / +1.0% / -0.4% |
| sse41 ×1 | -1.0% / -1.5% / +0.3% / -0.4% |
| sse41 ×4 | -1.2% / -1.1% / +4.0% / +1.3% |
| scalar ×1 | -1.2% / -1.2% / +0.1% / +0.1% |
| scalar ×4 | -1.8% / -1.8% / -0.7% / +0.0% |

AVX2 ×1 regressed 3.1% at 70 MB and 2.8% at 259 MB; SSE4.1 ×4 regressed 4.0% at
70 MB. The early focused trial's small-input win did not generalize. The entire
specialization was reverted. Peak RSS was 427308 KiB; no memory improvement.

### Rejected: reuse rotation masks

AVX2 and SSE4.1 deliberately obscure the 16-bit byte-shuffle mask with `black_box`
to avoid LLVM rewriting one shuffle into two instructions. That operation occurred
once per compression block. Constructing both masks once per equal-length batch
may remove repeated mask setup without duplicating the compression kernels. The
round function, mask values, loads, padding, and counters remain unchanged. NEON
and AVX-512 use rotate instructions rather than these masks; scalar is unchanged.

The ten focused hash-reference tests passed. A best-of-three screening run produced
the following deltas against the fresh baseline (same size order as above):

| Backend / threads | Rejected rotation-mask trial delta |
|---|---|
| avx2 ×1 | +0.0% / +0.0% / -0.5% / +1.1% |
| avx2 ×4 | +0.0% / -1.6% / -3.8% / -0.7% |
| sse41 ×1 | +1.6% / +1.2% / +0.3% / +0.9% |
| sse41 ×4 | +1.2% / +5.7% / +1.3% / +1.5% |
| scalar ×1 | +0.5% / +0.1% / +0.1% / +0.2% |
| scalar ×4 | +0.0% / -0.3% / -0.7% / +0.1% |

SSE4.1 ×4 regressed 5.7% at 17 MB; most AVX2 cases changed less than 2%.
The isolated 70 MB/four-thread AVX2 improvement did not justify retaining the
cross-backend regression. Both backend edits were reverted before final gates.
NEON already uses native shift/insert rotations, so this mask-specific change does
not apply there. No replacement RS engine was introduced: the scalar profile is
RS-bound, but the short-column packing experiment in iteration 1 already failed
the cross-backend regression gate. A new engine was not justified by evidence.

### Retained result

No encoder changes; accepted performance delta is zero on every backend and thread
count. The final release benchmark is byte-identical to the baseline, SHA-256
`352f85bde55ce87a7944f26ddb03b0f211f1d148230003124fda27f0939f0997`.

All final local gates passed on the reverted source: full release/ignored tests
with backend unset, AVX2, SSE4.1, and scalar; static baseline; v3 with/without
default features; v4+GFNI compile with explicit target; fmt; clippy with warnings
denied for both feature configurations; rustdoc; scalar Miri (eight passed, two
existing slow tests excluded as in CI); cargo-deny. All 18 CI checks passed, including actual ARM/NEON tests, before PR #3 was
merged. Final timings are row 2 of `BENCHMARKS.md`. Observed wall-time deltas for
this unchanged executable, in the same size order:

| Backend / threads | Final timing variation (not an encoder delta) |
|---|---|
| avx2 ×1 | -1.0% / +0.4% / +3.9% / -0.9% |
| avx2 ×4 | +0.0% / -1.6% / +0.0% / -0.7% |
| sse41 ×1 | +0.3% / -1.2% / -1.5% / -1.8% |
| sse41 ×4 | +0.0% / +0.0% / +0.7% / +0.0% |
| scalar ×1 | -0.1% / -0.2% / -0.1% / -0.0% |
| scalar ×4 | -0.4% / -0.9% / -0.9% / -0.2% |

The unchanged binary itself varied +3.9% in the 70 MB AVX2 ×1 case. This limits
causal attribution of small measured regressions; neither experiment cleared the
prescribed cutoff, so neither is retained. No improvement is inferred from the
other fluctuations. Peak RSS for 259 MB, AVX2 ×4: 427440 KiB versus 427448 KiB for
the saved baseline binary, an immaterial 8 KiB difference.

### Needs AVX-512 validation

None: both kernel experiments were reverted, including the AVX-512 specialization
and AVX2 remainder-path changes.

## Iteration 3 — minimalism and efficiency

Base: `2eb6138`, after PR #3 merged with green CI. The same On-Demand instance and
input bytes are used; no additional cloud resource was created.

### Profile and memory evidence

Fresh 499 Hz software `perf` sampling: AVX2 BLAKE2b 52.51%, RS FFT/IFFT 28.05%;
SSE4.1 BLAKE2b 65.87%; scalar RS FFT/IFFT 77.68%, hashing 19.78%. A separate,
uncommitted diagnostic copy refreshed the phase timers with the original 16-tree
batches and no rejected packing or kernel changes. These are warm second-call
milliseconds, not acceptance timings:

| AVX2 input / threads | Total wall | Row RS | Column RS | Gather/copy | Leaf hash | Inner hash | Allocation |
|---|---|---|---|---|---|---|---|
| 0.7 MB ×1 | 211.0 | 10.5 | 29.0 | 7.0 | 53.7 | 108.1 | 1.3 |
| 0.7 MB ×4 | 55.5 | 10.4 | 31.0 | 7.2 | 54.0 | 108.8 | 1.4 |
| 259 MB ×1 | 1110.0 | 119.4 | 330.1 | 107.8 | 474.9 | 108.3 | 1.5 |
| 259 MB ×4 | 291.1 | 115.6 | 330.2 | 113.0 | 475.4 | 109.2 | 6.2 |

Task timers accumulate across workers; row/column RS include their input copies,
so these columns overlap and must not be summed. Merkle gathering separately took
3.2–4.0 ms. Allocation timing includes the instrumented constructors/zero filling,
not every allocation in the process. Instrumentation changes code generation and
adds overhead; only the uninstrumented matrix decides acceptance.

At four threads, row/systematic branch spans were 10.9/17.3 ms for small AVX2 and
181.8/181.8 ms for large AVX2. Large SSE4.1 showed 89.6/239.2 ms, while scalar was
1691.5/1691.5 ms. Rayon can steal work while a branch waits; branch-span differences
are not idle-thread measurements and do not establish an exploitable barrier stall.

For the 259209074-byte input, symbols are 1164 bytes. The input, 129462408-byte row
recovery buffer, and 32000000-byte leaf matrix alone occupy 401.18 MiB. The tail,
RS workspace/tables, code, and runtime account for the remainder of roughly 417 MiB
RSS. Recovery is explicitly freed before Merkle scratch is allocated. The earlier
scratch-batch experiment failed the timing gate, and deleting the lazily allocated
GFNI table would not save memory on this AVX2 host.

### API, dependencies, and binary size

The public surface consists of the encoder and shard constant, backend selection,
ID representation/parsing, metadata and encoding types, and their errors. Hashing,
RS engines, layout, and scratch types are already private. No public removal was
justified by the consumer requirements or a measured benefit.

`cargo +nightly-2026-10-11 udeps --all-targets --locked` (cargo-udeps 0.1.61) found
no unused dependencies. Normal dependencies remain rayon and reed-solomon-simd;
blake2, criterion, proptest, and walrus-core are used only for development.
BLAKE2b shares its IV/permutation constants. The AVX-512 shuffle engine reuses the
dependency's tables; its separate GFNI affine table is a different representation,
initialized once on demand, not a redundant copy to delete.

A temporary minimal consumer reads a file, calls `BlobEncoder::mainnet().blob_id`,
and prints the ID. It uses the same lockfile versions and release fat LTO/one
codegen unit. cargo-bloat 0.12.1 attributes approximately 60.5% of default `.text`
to std, 14.9% to this crate, 10.5% to reed-solomon-simd, and 9.9% to rayon/core.
Attribution is approximate, particularly with LTO. The benchmark example includes
walrus-core, so its file size is not used as a consumer size measurement.

Before this change, the consumer's normal release ELF was 692744 bytes with default
dispatch and 670960 bytes with v3/static selection. The existing static configuration
was already 3.1% smaller; that is not an improvement made by this iteration.
cargo-bloat creates a debug-bearing inspection artifact; these ELF sizes were taken
after a normal release rebuild, with symbols retained. Both consumers printed the
known empty-blob ID.

### Rejected: remove the unavailable encoder variant

Symbol inspection found four AVX-512 RS FFT/IFFT functions in the static v3 binary,
although its constructors can never select that engine. The first candidate gated
the module, encoder variant, and encode arm on runtime detection or static
AVX-512F/BW support, preserving all code and tests in unit-test builds.

The static consumer shrank from 670960 to 657640 bytes (13320 bytes, 2.0%); the
default consumer remained 692744 bytes. Focused RS tests passed, but the paired
two-run best-of-five static v3 matrix failed the performance gate. Elapsed-time
deltas, 0.7 / 17 / 70 / 259 MB:

| Backend / threads | Rejected variant-pruning delta |
|---|---|
| avx2 ×1 | +1.5% / +0.9% / -0.8% / +1.2% |
| avx2 ×4 | +1.9% / +0.0% / +2.9% / +1.4% |
| sse41 ×1 | +0.0% / -1.4% / +1.7% / -0.8% |
| sse41 ×4 | +1.2% / -1.1% / -0.6% / -0.7% |
| scalar ×1 | -0.5% / -0.3% / -0.2% / -0.0% |
| scalar ×4 | -0.5% / -0.4% / +0.4% / +0.1% |

The 70 MB AVX2 ×4 case measured 2.9% slower. Reverted this version.

### Rejected: preserve the encoder layout

Retain the existing enum and constructors, but do not instantiate the AVX-512
encoding call in static builds lacking AVX-512F/BW. Constructors cannot create
that variant in such builds; its encode arm reports an unreachable state. Unit-test
builds retain the original path and every existing kernel/table test. This may
remove the unreachable functions without changing the enum layout. No kernel
arithmetic, unsafe block, table, or public API changes.

The static consumer was 657952 bytes, 13008 bytes (1.9%) smaller than baseline.
Focused RS tests passed, but the two-run best-of-five matrix again failed the gate:

| Backend / threads | Rejected encode-arm pruning delta |
|---|---|
| avx2 ×1 | +0.0% / +2.2% / -0.5% / +0.2% |
| avx2 ×4 | +1.9% / +1.6% / +2.9% / +0.4% |
| sse41 ×1 | +0.0% / -1.7% / +1.7% / -0.5% |
| sse41 ×4 | +0.0% / -1.1% / -0.6% / -1.2% |
| scalar ×1 | -0.1% / -0.4% / -0.2% / -0.3% |
| scalar ×4 | -0.5% / +0.0% / +0.4% / -0.1% |

AVX2 ×1 at 17 MB measured 2.2% slower and AVX2 ×4 at 70 MB measured 2.9% slower.
Preserving the enum layout did not clear the measured regression. This version was
also reverted. The four unused kernels remain; no binary-size reduction is claimed.

### Retained result and gates

No production-code, API, dependency, or table changes. README refreshes the benchmark
table, uses precise memory units, and corrects the dispatch heading to say runtime
detection is enabled by default. The accepted encoder and binary-size deltas are
zero for every backend and thread count. The final benchmark executable is
byte-identical to the original baseline:
`352f85bde55ce87a7944f26ddb03b0f211f1d148230003124fda27f0939f0997`.
Row 3 in `BENCHMARKS.md` and the README record the final two best-of-five runs.
Observed variation versus row 2, in the same size order:

| Backend / threads | Final timing variation (not an encoder delta) |
|---|---|
| avx2 ×1 | +0.0% / +0.0% / +2.5% / +0.2% |
| avx2 ×4 | +0.0% / +0.0% / +3.8% / +1.1% |
| sse41 ×1 | +0.0% / +0.3% / +4.2% / +2.2% |
| sse41 ×4 | +0.0% / +1.1% / +2.6% / +1.5% |
| scalar ×1 | +0.0% / +0.9% / +0.8% / +0.1% |
| scalar ×4 | +0.4% / +0.3% / +0.9% / +0.2% |

The unchanged executable varied by up to 4.2% in these encoder measurements. This
limits causal attribution of small differences; the cutoff was applied
conservatively to the experimental changes, and none is retained. Peak RSS is
427268 KiB (417.3 MiB), versus 427440 KiB in iteration 2; this small fluctuation
is not a memory improvement.

All 15 final local gates passed: full release/ignored suites for the unset backend,
AVX2, SSE4.1, scalar, static baseline, and v3 with/without default features; v4+GFNI
compilation both with and without default features using an explicit target; fmt;
clippy with warnings denied for both feature configurations; rustdoc; scalar Miri
(eight passed, two existing slow tests excluded as in CI); cargo-deny. ARM/NEON CI
must also be green before merge. No fixture or correctness assertion was weakened.


### Needs AVX-512 validation

None: both static-pruning experiments were reverted. No AVX-512 or GFNI changes
remain. The final compile matrix includes v4+GFNI with and without default features.
