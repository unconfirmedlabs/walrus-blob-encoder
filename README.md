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

The backend is chosen at runtime by CPU feature detection, never by compile-time target
features, so a build with `-C target-cpu=x86-64` still uses AVX-512 where the CPU has it.
`Backend::detect` returns the fastest available backend and `BlobEncoder::with_backend` forces
one, failing with `Error::UnsupportedBackend` if the CPU lacks it.

Reed-Solomon encoding uses [reed-solomon-simd](https://crates.io/crates/reed-solomon-simd)
`=3.1.0`, which detects its own AVX2, SSSE3 or NEON engine at runtime, independently of the
BLAKE2b backend. `Backend::Scalar` is the exception: it also selects the portable Reed-Solomon
engine, giving a path without explicit SIMD code.

<!-- RS-AVX512 -->

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
  `3 * 32 * 16 * n` bytes of Merkle scratch.

Measured peak RSS with 4 threads, including the blob: 47 MB for a 0.7 MB blob and 427 MB for a
259 MB blob (walrus-core: 426 MB). For a 256 MiB blob the peak is about 1.65 times the blob
size.

## Benchmarks

Seconds per blob ID for 1000 shards, best of 3, with the speedup over walrus-core, which is
single-threaded. The blobs are FLAC files of 0.7, 17, 70 and 259 MB.

AMD EPYC 7313P, `taskset -c 0-3`:

| blob | walrus-core | avx2, 1 thread | avx2, 4 threads | sse41, 4 threads | scalar, 4 threads |
|---|---|---|---|---|---|
| 0.7 MB | 0.469 | 0.190 (2.5x) | 0.051 (9.2x) | 0.077 (6.1x) | 0.217 (2.2x) |
| 17 MB | 0.500 | 0.225 (2.2x) | 0.059 (8.5x) | 0.085 (5.9x) | 0.310 (1.6x) |
| 70 MB | 0.797 | 0.396 (2.0x) | 0.106 (7.5x) | 0.142 (5.6x) | 0.679 (1.2x) |
| 259 MB | 1.946 | 1.035 (1.9x) | 0.273 (7.1x) | 0.373 (5.2x) | 2.276 (0.9x) |

Intel Xeon Platinum 8375C (Ice Lake), AWS c6i.2xlarge limited to 4 vCPUs (2 cores with
hyperthreads):

| blob | walrus-core | avx512, 1 thread | avx512, 4 threads | avx2, 1 thread | avx2, 4 threads |
|---|---|---|---|---|---|
| 0.7 MB | 0.643 | 0.150 (4.3x) | 0.072 (8.9x) | 0.226 (2.8x) | 0.106 (6.1x) |
| 17 MB | 0.675 | 0.179 (3.8x) | 0.083 (8.1x) | 0.264 (2.6x) | 0.118 (5.7x) |
| 70 MB | 1.090 | 0.320 (3.4x) | 0.155 (7.0x) | 0.447 (2.4x) | 0.212 (5.1x) |
| 259 MB | 2.581 | 0.907 (2.8x) | 0.524 (4.9x) | 1.231 (2.1x) | 0.570 (4.5x) |

The scalar backend is slow because it also uses the portable Reed-Solomon engine; it exists as
the reference path.

To reproduce with your own files, checking every ID against walrus-core:

```sh
cargo run --release --example bench -- --ref --backend all --threads 1,4 --repeat 3 <files...>
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
  [`tests/vectors/cli-1000.txt`](tests/vectors/cli-1000.txt), and four real files checked when
  `WALRUS_BLOB_ENCODER_FILES` names a directory holding them.
- The supported shard counts, exactly 4 to 49,155, are checked against walrus-core's encoder
  construction for 1 to 2100, 49,000 to 49,300 and a sample up to 65,535, and the blob IDs
  for 1 to 40 shards.
- The BLAKE2b backends are tested against RustCrypto's `blake2` over block-boundary lengths,
  batch sizes, symbol sizes and overlapping messages, plus a proptest; the Merkle trees against
  a direct construction; the Reed-Solomon engines against each other.
- Miri runs the BLAKE2b tests on the scalar backend.
- CI runs fmt, clippy and rustdoc with warnings denied on x86-64 and AArch64, the tests with
  each backend forced where the runner's CPU has it (x86-64: avx512, avx2, sse41, scalar;
  AArch64: neon, scalar) and with detection, Miri, and cargo-deny for advisories, licenses,
  duplicate versions and sources.

## License

Apache-2.0, see [LICENSE](LICENSE). The dependencies are rayon (MIT OR Apache-2.0) and
reed-solomon-simd (MIT AND BSD-3-Clause). The crate re-derives walrus-core's algorithm and
contains no walrus-core code; walrus-core is used only by the tests and benchmarks.
