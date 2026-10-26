# walrus-blob-encoder

Computes Walrus blob IDs and full blob metadata (the primary and secondary Merkle root of every
sliver pair, the unencoded length and the encoding type) without materializing the slivers.
The output is byte-identical to walrus-core at rev
[`c377852712d30a23031c04d07dc13bf5e20a92de`](https://github.com/MystenLabs/walrus/tree/c377852712d30a23031c04d07dc13bf5e20a92de)
(walrus-core 1.50.0, the code in the mainnet `walrus` CLI 1.58.1) for encoding type RS2
(`DEFAULT_ENCODING`). Mainnet uses 1000 shards; every shard count walrus-core can encode,
4 to 49,155, is supported. The crate does not encode or decode slivers.

## Usage

The crate is not on crates.io; depend on it by Git revision:

```toml
[dependencies]
walrus-blob-encoder = { git = "https://github.com/unconfirmedlabs/walrus-blob-encoder", rev = "<commit>" }
```

```rust
use std::{error::Error, num::NonZeroU16, sync::Arc};

use walrus_blob_encoder::{Backend, BlobEncoder};

fn main() -> Result<(), Box<dyn Error>> {
    let blob = std::fs::read("track.flac")?;

    // Mainnet: 1000 shards, the fastest BLAKE2b backend of this CPU, the current rayon pool.
    let encoder = BlobEncoder::mainnet();
    let id = encoder.blob_id(&blob)?;
    println!("{id}"); // URL-safe base64 without padding, as printed by `walrus blob-id`
    let bytes: [u8; 32] = id.into(); // the on-chain u256 blob ID, little endian

    // Full metadata: one primary and one secondary hash per shard.
    let metadata = encoder.metadata(&blob)?;
    assert_eq!(metadata.blob_id(), id);
    assert_eq!(metadata.unencoded_length(), blob.len() as u64);
    let pair = metadata.sliver_pairs()[0];
    println!("{:?} {:?}", pair.primary_hash, pair.secondary_hash);

    // A dedicated thread pool, a forced backend and another shard count.
    let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(4).build()?);
    let encoder = BlobEncoder::new(NonZeroU16::new(100).unwrap())?
        .with_backend(Backend::Scalar)?
        .with_thread_pool(pool);
    println!("{} {bytes:?}", encoder.blob_id(&blob)?);
    Ok(())
}
```

## How it works

For `n` shards and `f = (n - 1) / 3`, the blob is a message matrix of `n - 2f` rows and
`n - f` columns of symbols (334 x 667 for mainnet), expanded by Reed-Solomon to `n x n`.
The computation runs on rayon in four phases:

- The rows are encoded into their `f` recovery symbols while, concurrently, the systematic
  columns, which only need the blob, are encoded and their leaves hashed.
- After that barrier, the `f` non-systematic columns (333 for mainnet) are encoded from the row
  recovery symbols and their leaves hashed.
- The Merkle roots of all rows (primary slivers) and columns (secondary slivers) are computed,
  16 trees per task and level by level, so that every batch of inner-node hashes fills the SIMD
  lanes.
- The sliver-pair tree and the blob ID are computed.

The input is never copied into a matrix: symbols are slices of the caller's blob, and only the
rows that reach past its end are copied into a small zero-padded tail. The hasher reads full
blocks straight from those slices.

## Backends and dispatch

BLAKE2b runs as a multi-buffer hasher that hashes several equal-length messages at once, one
per SIMD lane:

| backend | architecture | messages at a time |
|---|---|---|
| `avx512` | x86-64, AVX-512 F and BW, AVX2 | 8 |
| `avx2` | x86-64, AVX2 | 4 |
| `sse41` | x86-64, SSE4.1 | 2 |
| `neon` | AArch64 | 2 |
| `scalar` | any | 1 |

### Dispatch policy: static first, runtime detection enabled by default

A backend whose target features are enabled at compile time (`-C target-cpu` or
`-C target-feature`) is always available and is called without any CPU detection. The
`runtime-detection` cargo feature, on by default, additionally compiles the faster backends and
picks them at runtime when the CPU has them. Without the feature the build contains no CPU
detection at all, and the backend is fixed when the crate is compiled.

BLAKE2b backend by build target:

| target | static backend | with `runtime-detection` (default) | `default-features = false` |
|---|---|---|---|
| `x86-64` (baseline, SSE2) | scalar | avx512, avx2 or sse41 if detected, else scalar | scalar |
| `x86-64-v2` | sse41 | avx512 or avx2 if detected, else sse41 | sse41 |
| `x86-64-v3` | avx2 | avx512 if detected, else avx2 | avx2 |
| `x86-64-v4` | avx512 | avx512, no detection runs | avx512 |
| `x86-64-v4` + `gfni` | avx512 | avx512, no detection runs | avx512, recommended for AVX-512 hosts |
| AArch64 | neon | neon, no detection runs | neon |
| other | scalar | scalar | scalar |

`Backend::is_available` is true for the backends enabled at compile time and, with the
feature, for those the running CPU supports. `Backend::detect` returns the fastest available
backend: with the feature it is detected once per process (unless the fastest backend is
already enabled at compile time); without it, it is a compile-time constant.
`BlobEncoder::with_backend` forces a backend and fails with `Error::UnsupportedBackend` if it
is not available.

Each SIMD kernel is a `#[target_feature]` function. Stable Rust still requires `unsafe` to call
one when the features are only enabled for the whole build (error E0133), so every backend has
exactly one `unsafe` call into its kernel. In a static build it is justified by the
compile-time configuration, and no detection code exists in the binary; with detection, by
the CPU check that made the backend available.

The feature is on by default because a plain `x86-64` build without it can only use the scalar
backend and the portable Reed-Solomon engine, several times slower (BLAKE2b alone is about 5x
slower than AVX-512 and 3x slower than AVX2), and most builds of a library target that
baseline. Builds for a known CPU, such as an enclave image, should set
the CPU level and turn detection off, so that the code that runs is the code that was
compiled:

```toml
[dependencies]
walrus-blob-encoder = { git = "...", rev = "<commit>", default-features = false }
```

```sh
RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+gfni" cargo build --release
```

GFNI is not part of `x86-64-v4` but is used by the AVX-512 Reed-Solomon engine, hence the extra
target feature. Such a binary needs a CPU with every compiled feature (here AVX-512 F, BW, CD,
DQ and VL plus GFNI, as on Ice Lake and later Xeons and on Zen 4) and stops with an illegal
instruction on others.

Reed-Solomon encoding runs through [reed-solomon-simd](https://crates.io/crates/reed-solomon-simd)
`=3.1.0` with a pluggable engine, and engine selection follows the same policy:

- `avx512`: this crate's AVX-512 engine (`src/reed_solomon/avx512.rs`, adapted from
  reed-solomon-simd's AVX2 engine), with GFNI affine multiplies when GFNI is enabled at compile
  time or detected, and AVX-512BW `vpshufb` multiplies otherwise (Skylake-SP, Cascade Lake). On
  Ice Lake it is 1.8x faster than the AVX2 engine with GFNI and 1.2x without, for a 259 MB blob.
- other backends: reed-solomon-simd's AVX2, SSSE3 or NEON engine when its features are enabled
  at compile time; otherwise, with `runtime-detection`, its `DefaultEngine`, which detects one
  at runtime; otherwise its portable `NoSimd` engine.
- `scalar`: always the portable `NoSimd` engine.

Every engine produces identical recovery symbols; the AVX-512 kernels are tested against
`NoSimd` for every parameter set the encoder uses and every field element.

`Backend::Scalar` therefore gives a path without explicit SIMD code.

The output is identical for every backend and thread count.

The `WALRUS_BLOB_ENCODER_BACKEND` environment variable is read only by the test suite: set to a
backend name, it restricts the tests to that backend and fails them if the CPU lacks it. The
library itself reads no environment variables.

## Memory

Besides the caller's blob, computing the metadata for `n` shards with symbol size `s`
allocates:

- the row recovery symbols, `(n - 2f) * f * s` bytes, about half the blob, freed before the
  Merkle phase;
- the leaf hashes of the expanded matrix, `32 * n^2` bytes: a fixed 32 MB for 1000 shards;
- the zero-padded tail rows, at most `2 * (n - 2f) * (n - f) + (n - f) * s` bytes;
- per thread, up to two Reed-Solomon encoders of about `n * s` bytes each and about
  `3 * 32 * 16 * n` bytes of Merkle scratch;
- once per process, for the GFNI Reed-Solomon kernel, a 2 MiB table of multiplication
  matrices.

Measured peak RSS for the 259209074-byte high-entropy input with AVX2 and four threads:
427268 KiB (417.3 MiB), including the caller's blob, about 1.69 times its size.
The input, row recovery symbols, and leaf hashes alone occupy 401.2 MiB. See
[ITERATIONS.md](ITERATIONS.md) for allocation accounting and rejected experiments.

## Benchmarks

Seconds per blob ID for 1000 shards, from [BENCHMARKS.md](BENCHMARKS.md), iteration 3.
AWS c6a.2xlarge (AMD EPYC 7R13), Rust 1.94.1, release defaults, CPUs 2–5 on four
separate physical cores. Inputs are high-entropy generated bytes. Each value is the
better of two runs, each reporting best of five; walrus-core runs on one thread.

| Blob | walrus-core ×1 | AVX2 ×1 | AVX2 ×4 | SSE4.1 ×1 | SSE4.1 ×4 | Scalar ×1 | Scalar ×4 |
|---|---|---|---|---|---|---|---|
| 0.7 MB | 0.496 | 0.199 | 0.053 | 0.310 | 0.080 | 0.884 | 0.225 |
| 17 MB | 0.524 | 0.233 | 0.061 | 0.339 | 0.089 | 1.296 | 0.326 |
| 70 MB | 0.876 | 0.414 | 0.109 | 0.596 | 0.155 | 2.781 | 0.705 |
| 259 MB | 2.059 | 1.047 | 0.285 | 1.533 | 0.399 | 9.377 | 2.375 |

The scalar backend also uses portable Reed–Solomon. AVX-512/GFNI and NEON timings
were not measured in this run; ARM correctness is checked in CI. Earlier Ice Lake
measurements remain in the [v0.1.0 README](https://github.com/unconfirmedlabs/walrus-blob-encoder/blob/v0.1.0/README.md#benchmarks).

Run this command twice with the same inputs, checking each ID against walrus-core:

```sh
CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=4 taskset -c 2-5 \
  cargo run --release --locked --example bench -- --ref --backend all --threads 1,4 --repeat 5 <files...>
```

The criterion benchmarks run the same comparison on generated inputs of the same sizes; a regex
selects a subset of the `blob_id/<size>/<implementation>/x<threads>` IDs:

```sh
cargo bench --bench encoder
cargo bench --bench encoder -- 'blob_id/17MB/(walrus-core|avx2)'
```

## Compatibility guards

- Differential tests against walrus-core at the pinned rev, a dev-dependency only. Each case
  compares the full metadata (blob ID, encoding type, length, shard count and every primary
  and secondary hash) for every available backend, rotating through 1, 2, 3 and 8 threads:
  edge lengths, lengths at and next to every symbol-size step up to 10 bytes and sampled steps
  up to 258 bytes (1002 with `--ignored`) for 1000 shards, every step up to symbol
  size 512 and a sample across the whole range up to 65,534 for 4 and 7 shards, random lengths up to 4 MiB, 14 other
  shard counts from 4 to 2000, the maximum blob size and the rejection of one byte more on
  small committees, the maximum blob size and symbol size up to 49,155 shards, and the blob ID
  text form, parsing included.
- Golden vectors from the `walrus` CLI 1.58.1: 32 generated inputs in
  [`tests/vectors/cli-1000.txt`](tests/vectors/cli-1000.txt); high-entropy SplitMix64 inputs of
  1 KB, 10 KB, 100 KB, 1 MB, 10 MB and 100 MB in
  [`tests/vectors/cli-1000-high-entropy.txt`](tests/vectors/cli-1000-high-entropy.txt), each
  also checked against walrus-core's full metadata with every backend and thread count; and
  four real files checked when `WALRUS_BLOB_ENCODER_FILES` names a directory holding them.
- The supported shard counts, exactly 4 to 49,155, are checked against walrus-core's encoder
  construction for 1 to 2100, 49,000 to 49,300 and a sample up to 65,535, and the blob IDs
  for 1 to 40 shards.
- The BLAKE2b backends are tested against RustCrypto's `blake2` over block-boundary lengths,
  batch sizes, symbol sizes and overlapping messages, plus a proptest; the Merkle trees against
  a direct construction; the Reed-Solomon engines against each other.
- Miri runs the BLAKE2b tests on the scalar backend.
- CI runs fmt, clippy and rustdoc with warnings denied on x86-64 and AArch64, with and without
  default features; the tests with each backend forced where the runner's CPU has it (x86-64:
  avx512, avx2, sse41, scalar; AArch64: neon, scalar) and with detection; the tests of
  static builds (`x86-64-v4` where the runner has AVX-512, `x86-64-v3` and the baseline without
  default features); Miri; and cargo-deny for advisories, licenses, duplicate versions and
  sources.

On a shared host, serialize the test configurations while preserving every backend and
1/2/3/8-thread case:

```sh
CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8 RUST_TEST_THREADS=1 WALRUS_BLOB_ENCODER_SERIAL_TESTS=1 \
  cargo test --release --locked -- --include-ignored
```

The serial mode creates only one dedicated pool at a time. Leave
`WALRUS_BLOB_ENCODER_BACKEND` unset for the default run, or set it to an available backend
name to run the same suite with that backend forced.

## License

Apache-2.0, see [LICENSE](LICENSE). The dependencies are rayon (MIT OR Apache-2.0) and
reed-solomon-simd (MIT AND BSD-3-Clause). The crate re-derives walrus-core's algorithm and
contains no walrus-core code; walrus-core is used only by the tests and benchmarks.
