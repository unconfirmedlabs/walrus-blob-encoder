# walrus-blob-encoder

Computes Walrus blob IDs and full blob metadata (the primary and secondary Merkle root of every
sliver pair, the unencoded length and the encoding type) without materializing the slivers,
either from a blob in memory or streamed from a seekable source in a bounded amount of memory.
The output is byte-identical to walrus-core at rev
[`d6c6d3c59ab0cd9e5abd7be96c7133d056de4c84`](https://github.com/MystenLabs/walrus/tree/d6c6d3c59ab0cd9e5abd7be96c7133d056de4c84)
(walrus-core 1.61.0; earlier, rev `c377852`, walrus-core 1.50.0, the code in the mainnet `walrus`
CLI 1.58.1) for encoding type RS2 (`DEFAULT_ENCODING`). Mainnet uses 1000 shards; every shard
count walrus-core can encode, 4 to 49,155, is supported. The crate does not encode or decode
slivers.

The primary target is WebAssembly on Cloudflare Workers (128 MB of memory, no threads): a raw
`extern "C"` module streams a blob of any size through a pull protocol whose reads the host can
serve asynchronously, for example from R2. Browsers and native servers use the same engine.

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

Cargo features, both on by default:

- `rayon`: the in-memory encoder runs on a rayon pool (`with_thread_pool`, or the current pool).
  Without it, for example in WebAssembly, everything runs on the calling thread and rayon is not
  a dependency.
- `runtime-detection`: faster BLAKE2b backends and the AVX-512 Reed-Solomon engine are compiled
  in and picked at runtime (see [Backends and dispatch](#backends-and-dispatch)).

### Streaming

`BlobEncoder::stream(len, budget)` returns a `StreamEncoder`, a pull state machine over reads
of the blob: it asks for a read, the caller fills its destination (from a file, an HTTP range
request, an R2 object...) and completes it, which runs the computation up to the next read.
The column batches it holds take at most `budget` bytes (at least one column, `334 * s` bytes
for mainnet and symbol size `s`); a larger budget means fewer passes over the blob.

```rust
use walrus_blob_encoder::{BlobEncoder, SeekSource};

fn id_of_file(path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let len = file.metadata()?.len();
    let encoder = BlobEncoder::mainnet();

    // Driven by a `Source` (`&[u8]`, or any `Read + Seek` through `SeekSource`)...
    let metadata = encoder.metadata_from_source(&mut SeekSource::new(file, len), 32 << 20)?;

    // ...or by hand, for asynchronous reads.
    let blob = std::fs::read(path)?;
    let mut stream = encoder.stream(len, 32 << 20)?;
    while let Some(read) = stream.next_read() {
        let offset = read.offset as usize;
        stream.dest().copy_from_slice(&blob[offset..offset + read.len]);
        stream.complete_read();
    }
    assert_eq!(stream.finish(), metadata);
    Ok(metadata.blob_id().to_string())
}
```

### WebAssembly

`crates/walrus-blob-encoder-wasm` exports the pull protocol without wasm-bindgen (see its crate
documentation for the ABI); `crates/walrus-blob-encoder-wasm/js/run.mjs` drives it from Node or
Bun with synchronous or asynchronous reads:

```sh
RUSTFLAGS="-C target-feature=+simd128" cargo build -p walrus-blob-encoder-wasm --profile wasm \
  --target wasm32-unknown-unknown
node crates/walrus-blob-encoder-wasm/js/run.mjs \
  target/wasm32-unknown-unknown/wasm/walrus_blob_encoder_wasm.wasm <file> <budget MiB> [--async]
```

The `wasm` profile is the release profile (fat LTO, one codegen unit) with `panic = "abort"`.
Build with `+simd128`: BLAKE2b then uses its two-lane `simd128` backend and Reed-Solomon its
`Wide` engine; without it both are scalar and several times slower. `+relaxed-simd` (Node and
Bun accept it) also lets the Reed-Solomon table lookups use `i8x16.relaxed_swizzle`.

### Command line

`crates/walrus-blob-encoder-cli` builds `walrus-blob-id`, which streams files with positioned
reads under a budget (`--budget MIB`, default 64) or reads them into memory (`--memory`,
`--threads N`), and prints counters and phase times with `--stats`.

## How it works

For `n` shards and `f = (n - 1) / 3`, the blob is a message matrix of `n - 2f` rows and
`n - f` columns of symbols (334 x 667 for mainnet), expanded by Reed-Solomon to `n x n`:
each row to `n` symbols (`f` recovery symbols), each column to `n` symbols. Primary sliver `i`
is row `i` of the expanded matrix, secondary sliver `i` column `n - 1 - i`; each sliver hash is
the Merkle root of its symbols' leaf hashes.

One engine computes this column by column (`src/expansion.rs`): each column is encoded from its
`n - 2f` source symbols, its `n` leaves are hashed in one multi-buffer batch, its root is built
at once, and its leaves are appended to the `n` row trees. The row trees all have the same
shape and receive their leaves in column order, so they grow together (`merkle::Forest`): each
completed pair of nodes, on any level, is hashed for every row in one batch, and only one
pending pair per row and level is kept. There is no leaf matrix. The systematic columns (below
`n - f`) need only the blob; the parity columns need the rows' recovery symbols.

- **In memory** (`BlobEncoder::metadata`): one pass. Systematic columns are read from the
  caller's blob in place (only rows reaching past its end are copied into a zero-padded tail),
  every row is encoded once into its recovery symbols, and the parity columns are read from
  those. With more than one rayon thread, rows are encoded in parallel and columns in groups of
  256 whose leaves are then appended to the row trees in parallel by groups of 64 rows.
- **Streaming** (`BlobEncoder::stream`): the columns are walked in batches of `k` columns,
  `k = budget / ((n - 2f) * s)`. A systematic batch is filled by one strided read per row. A
  parity batch needs every row's recovery symbols, so it is a pass over the whole blob that
  encodes each row and keeps the batch's slice of them; the Reed-Solomon encoder computes only
  that slice's FFT groups. Systematic batches stop at column `n - f`, so parity batches start
  exactly there, and the `f` parity columns are split into the fewest passes the budget allows,
  balanced: `P = ceil(f / k)` passes of `ceil(f / P)` columns. A budget for all `n` columns is a
  single sequential pass. Reads are completed by the caller, so they can be asynchronous.

One Reed-Solomon encoder and work area serves both rows and columns. The hasher reads full
blocks straight from the symbol slices.

## Backends and dispatch

BLAKE2b runs as a multi-buffer hasher that hashes several equal-length messages at once, one
per SIMD lane:

| backend | architecture | messages at a time |
|---|---|---|
| `avx512` | x86-64, AVX-512 F and BW, AVX2 | 8 |
| `avx2` | x86-64, AVX2 | 4 |
| `sse41` | x86-64, SSE4.1 | 2 |
| `neon` | AArch64 | 2 |
| `simd128` | WebAssembly built with `simd128` (static only) | 2 |
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

Reed-Solomon encoding runs through the vendored fork of
[reed-solomon-simd](https://crates.io/crates/reed-solomon-simd) 3.1.0 in
`crates/reed-solomon-simd`, with a pluggable engine:

- `avx512`: its AVX-512 engine (`crates/reed-solomon-simd/src/engine/engine_avx512.rs`, adapted
  from reed-solomon-simd's AVX2 engine), with GFNI affine multiplies when GFNI is enabled at
  compile time or detected, and AVX-512BW `vpshufb` multiplies otherwise (Skylake-SP, Cascade
  Lake). On Ice Lake it was 1.8x faster than the AVX2 engine with GFNI and 1.2x without, for a
  259 MB blob. It is detected at runtime with `runtime-detection`, like the BLAKE2b backends.
- other backends: the fork's `Wide` engine, written with the `wide` crate, when the target
  enables SSSE3, WebAssembly `simd128` or NEON at compile time, with 256-bit vectors when it
  enables AVX2; otherwise the portable `NoSimd` engine. `Wide` has no runtime detection.
- `scalar`: always the portable `NoSimd` engine.

The `Wide` and `NoSimd` engines of the encoder build the multiplication tables of the skew
factors its two encoders read (positions below 1536 for 1000 shards, at most 384 KiB) instead
of the global 8 MiB tables; the AVX-512 engine still uses those (and a 2 MiB GFNI table).

**Build for the CPU.** Since `Wide` is chosen at compile time, a plain `x86-64` build (SSE2
only) encodes with `NoSimd`, which is about 10x slower than `Wide` per encode; the release of
88a9fed instead detected reed-solomon-simd's AVX2 engine at runtime. Native consumers should
build with `-C target-cpu=native`, `-C target-cpu=x86-64-v3` (AVX2) or at least
`-C target-feature=+ssse3`, and WebAssembly builds with `+simd128`.

Every engine produces identical recovery symbols: `Wide` (128- and 256-bit, full and compact
tables, ranged FFTs) and the AVX-512 kernels are tested against `NoSimd` for every parameter set
the encoder uses and every even shard size from 2 to 130 plus larger samples, the AVX-512
kernels also for every field element.

`Backend::Scalar` therefore gives a path without explicit SIMD code.

The output is identical for every backend and thread count.

The `WALRUS_BLOB_ENCODER_BACKEND` environment variable is read only by the test suite: set to a
backend name, it restricts the tests to that backend and fails them if the CPU lacks it. The
library itself reads no environment variables.

## Memory

Computing the metadata of an in-memory blob for `n` shards with symbol size `s` allocates,
besides the caller's blob:

- the row recovery symbols, `(n - 2f) * f * s` bytes, about half the blob;
- the zero-padded tail rows, at most `2 * (n - 2f) * (n - f) + (n - f) * s` bytes;
- the row trees, about `64 * n * (log2(n) + 2)` bytes (0.8 MB for 1000 shards);
- per thread, one Reed-Solomon encoder with about `2 * next_power_of_two(f) * s` bytes of work
  space (symbols rounded up to 64 bytes) and at most 384 KiB of multiplication tables;
- with more than one thread, the leaf hashes of 256 columns at a time, `8192 * n` bytes
  (8 MB for 1000 shards);
- with the GFNI kernel, once per process, a 2 MiB table of multiplication matrices.

The streaming encoder holds no blob: its batch buffer (at most the budget), one row of
`(n - f) * s` bytes when parity columns take more than one pass, one Reed-Solomon encoder and
the row trees. `StreamStats` reports the buffer sizes. A 1 GB blob streams with a peak RSS of
43 MB natively at a 32 MiB budget; in WebAssembly the module's peak linear memory stays within
the budget plus about 10 MB (41 MiB at 32 MiB, 90 MiB at 84 MiB for 1 GB).

Peak RSS of the in-memory encoder for the 259209074-byte high-entropy input, AVX2, one thread:
385280 KiB, including the 253134 KiB blob (88a9fed: 423228 KiB). See
[BENCHMARKS.md](BENCHMARKS.md).

## Benchmarks

Seconds per blob ID for 1000 shards on an AMD EPYC 7313P (Zen 3, no AVX-512), from
[BENCHMARKS.md](BENCHMARKS.md), iteration 4: best of three, every ID checked against
walrus-core, high-entropy inputs, one thread unless noted. Native: `-C target-cpu=native`
(AVX2). WebAssembly: `+simd128` under Node 24 (in parentheses, with `+relaxed-simd`). Streaming
reads the file with positioned reads; "in memory" includes reading the whole file.

| Blob | Native in memory ×1 | Native in memory ×16 | Native, W=32 MiB | Native, W=256 MiB | Node, W=32 MiB | Node, W=84 MiB | Node, W=256 MiB |
|---|---|---|---|---|---|---|---|
| 5 MB | 0.19 | 0.04 | 0.19 | 0.19 | 0.52 (0.49) | 0.51 | 0.52 (0.49) |
| 100 MB | 0.55 | 0.14 | 0.55 | 0.54 | 1.36 (1.23) | 1.28 | 1.30 (1.19) |
| 259 MB | 1.17 | 0.29 | 1.31 | 1.12 | 3.18 | 2.77 | 2.65 |
| 1 GB | 4.00 | 1.00 | 9.16 | 3.82 | 19.81 (16.52) | 12.31 | 9.17 (8.15) |

Peak WebAssembly linear memory for a 1 GB blob: 41 MiB at a 32 MiB budget, 90 MiB at 84 MiB,
249 MiB at 256 MiB; 10 MiB for a 5 MB blob at any budget. Against the in-memory encoder of
88a9fed on the same machine, the in-memory entry points are as fast or faster single-threaded
(0.19 / 0.55 / 1.17 / 4.00 s against 0.23 / 0.59 / 1.21 / 4.03 s) with 32 to 38 MB less RSS,
and within noise with 4 and 16 threads. AVX-512 and NEON were not measured.

```sh
BUILD=native scripts/bench_native.sh stream:32 stream:256 memory:1 memory:16 -- <files...>
scripts/bench_wasm.sh node bun -- 32 84 256 -- <files...>
```

The `bench` example and the criterion benchmarks compare the in-memory encoder with
walrus-core per backend and thread count; a regex selects a subset of the
`blob_id/<size>/<implementation>/x<threads>` IDs:

```sh
cargo run --release --locked -p walrus-blob-encoder --example bench -- --ref --backend all --threads 1,4 --repeat 5 <files...>
cargo bench -p walrus-blob-encoder --bench encoder -- 'blob_id/17MB/(walrus-core|avx2)'
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
  [`tests/vectors/cli-1000.txt`](crates/walrus-blob-encoder/tests/vectors/cli-1000.txt); high-entropy SplitMix64 inputs of
  1 KB, 10 KB, 100 KB, 1 MB, 10 MB and 100 MB in
  [`tests/vectors/cli-1000-high-entropy.txt`](crates/walrus-blob-encoder/tests/vectors/cli-1000-high-entropy.txt), each
  also checked against walrus-core's full metadata with every backend and thread count; and
  four real files checked when `WALRUS_BLOB_ENCODER_FILES` names a directory holding them.
- The streaming encoder against walrus-core (blob ID, metadata root and all 2000 sliver hashes)
  and against the in-memory encoder for 294 budgets: lengths 1, 1000 and 222,777 to 222,779 with
  batch edges at 1, 2, 7, 332 to 334, 666 to 668, 999 and 1000 columns; 5 MB and 5 MiB; 100 MB
  + 1 at 8 to 256 MiB; and 4, 7, 10, 13 and 100 shards at six lengths each. The read sequence
  (offsets in bounds, counts, bytes, passes) is checked, and an empty blob needs no reads.
- The supported shard counts, exactly 4 to 49,155, are checked against walrus-core's encoder
  construction for 1 to 2100, 49,000 to 49,300 and a sample up to 65,535, and the blob IDs
  for 1 to 40 shards.
- The BLAKE2b backends are tested against RustCrypto's `blake2` over block-boundary lengths,
  batch sizes, symbol sizes and overlapping messages, plus a proptest; the WebAssembly module
  runs the same comparison for its `simd128` and scalar backends (`selftest_blake2b`); the
  Merkle trees and row forest against a direct construction; the Reed-Solomon engines against
  each other, and the vendored crate's own suite (CI: without SIMD, with SSSE3 and AVX2).
- `scripts/crosscheck.sh` compares walrus-core with the native CLI (streamed at 1 and 32 MiB,
  in memory) and the WebAssembly module under Node and Bun (synchronous and asynchronous
  reads).
- Miri runs the BLAKE2b tests on the scalar backend.
- CI runs fmt, clippy and rustdoc with warnings denied on x86-64 and AArch64, with and without
  default features; the tests with each backend forced where the runner's CPU has it (x86-64:
  avx512, avx2, sse41, scalar; AArch64: neon, scalar) and with detection; the tests of
  static builds (`x86-64-v4` where the runner has AVX-512, `x86-64-v3` and the baseline without
  default features, and with SSSE3 only); the Reed-Solomon crate's tests; the WebAssembly
  module's BLAKE2b self-test and its blob IDs against walrus-core under Node; Miri; and
  cargo-deny for advisories, licenses, duplicate versions and sources.

On a shared host, serialize the test configurations while preserving every backend and
1/2/3/8-thread case:

```sh
CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8 RUST_TEST_THREADS=1 WALRUS_BLOB_ENCODER_SERIAL_TESTS=1 \
  cargo test --release --locked -p walrus-blob-encoder -- --include-ignored
```

The serial mode creates only one dedicated pool at a time. Leave
`WALRUS_BLOB_ENCODER_BACKEND` unset for the default run, or set it to an available backend
name to run the same suite with that backend forced.

## Workspace

| crate | path | contents |
|---|---|---|
| `walrus-blob-encoder` | `crates/walrus-blob-encoder` | the library: in-memory and streaming encoders |
| `reed-solomon-simd` | `crates/reed-solomon-simd` | the Reed-Solomon fork, with every engine |
| `walrus-blob-encoder-wasm` | `crates/walrus-blob-encoder-wasm` | raw-ABI WebAssembly module, JS harness |
| `walrus-blob-encoder-cli` | `crates/walrus-blob-encoder-cli` | the `walrus-blob-id` command |

`crates/reed-solomon-simd` came in with its history through `git subtree add` and depends on
nothing else in the workspace, so `git subtree split --prefix crates/reed-solomon-simd`
extracts it again. The encoder depends on it by path.

## License

Apache-2.0, see [LICENSE](LICENSE). The dependencies are rayon (MIT OR Apache-2.0, optional,
the default `rayon` feature) and the vendored reed-solomon-simd (MIT AND BSD-3-Clause, and
Apache-2.0 for its AVX-512 engine; see [LICENSE-THIRD-PARTY](LICENSE-THIRD-PARTY)), which
depends on `wide`, `bytemuck`, `safe_arch` (Zlib OR Apache-2.0 OR MIT), `fixedbitset` and
`once_cell` (MIT OR Apache-2.0). The crate re-derives walrus-core's algorithm and contains no
walrus-core code; walrus-core is used only by the tests, examples and benchmarks.
