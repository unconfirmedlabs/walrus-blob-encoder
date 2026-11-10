# Changelog

## 0.x.x - UNRELEASED
- Documentation improvements.

## 0.1.0 - 2022-01-04
- First public version.

## 2.0.0 - 2023-11-16
- First release as reed-solomon-simd.
- SSSE3 and AVX2 engines for x84(-64).

## 2.1.0 - 2023-12-25
- Neon engine for AArch64.
- Make trait Engine 'object safe'

## 2.2.0 - 2024-02-12
- Remove `fwht()` from `trait Engine` as this opens up for better compiler optimizations.
- Let the compiler generate target specific code for the `eval_poly()` function, as this improves decoding throughput.

## 2.2.1 - 2024-02-21
- Faster Walsh-Hadamard transform (used in decoding).

## 2.2.2 - 2024-04-22
- Make DefaultEngine Send + Sync.
- Slightly faster `eval_poly()` (used in decoding).

## 3.0.0 - 2024-10-07
- Require shard length to be divisible by 2 instead of 64. Note regarding compatibility between versions: Shards of a length divisible by 64 are compatible across versions.
- Improved types for internal data structures. This raises the Minimum Supported Rust Version (MSRV) to 1.80.
- Faster `NoSimd` engine.
- Removed dependencies: bytemuck and once\_cell.

## 3.0.1 - 2024-11-23
- AVX2: Up to 20% higher throughput in encoding and up to 10% faster decoding.

## 3.1.0 - 2025-10-14
- `no_std` support when new `std` feature is disabled (`std` feature is enabled by default).
- Require Rust 1.82.0 or newer
- Implement trait `ExactSizeIterator` for `EncoderResult` and `DecoderResult` iterators.

## UNRELEASED (misofm fork)
- `Wide` engine: portable 128-bit SIMD through the `wide` crate (SSSE3,
  WebAssembly `simd128`, NEON), selected at compile time.
- `DefaultEngine` no longer detects CPU features at runtime: it is `Wide` when
  the target has SSSE3, `simd128` or NEON enabled, otherwise `NoSimd`. The
  `cpufeatures` dependency is removed.
- Removed the `Avx2`, `Ssse3`, `Neon` and `Naive` engines. The crate has no
  `unsafe` code. The known-answer roundtrip tests that ran on `Naive` now run
  on `Wide`.
- Removed `examples/quick-comparison.rs` and the four crates it compared
  against.
