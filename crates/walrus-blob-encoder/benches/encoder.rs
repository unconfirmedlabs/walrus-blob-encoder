// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Blob-ID throughput per backend and thread count, against walrus-core.
//!
//! Benchmark IDs are `blob_id/<size>/<implementation>/x<threads>`, for example
//! `blob_id/17MB/avx2/x4`; pass a regex to select a subset:
//!
//! ```text
//! cargo bench --bench encoder -- 'blob_id/(0.7|17)MB/avx2'
//! ```

use std::{
    hint::black_box,
    num::NonZeroU16,
    sync::{Arc, OnceLock},
    time::Duration,
};

use criterion::{BenchmarkId, Criterion, SamplingMode, Throughput};
use rayon::{ThreadPool, ThreadPoolBuilder};
use walrus_blob_encoder::{Backend, BlobEncoder, MAINNET_N_SHARDS};
use walrus_core::{
    DEFAULT_ENCODING,
    encoding::{EncodingConfig, EncodingFactory as _},
};

/// The sizes of the real benchmark files, labelled as in the README.
const SIZES: [(&str, usize); 4] = [
    ("0.7MB", 710_603),
    ("17MB", 17_066_416),
    ("70MB", 69_741_071),
    ("259MB", 259_209_074),
];

const THREAD_COUNTS: [usize; 2] = [1, 4];

/// Deterministic pseudo-random bytes from xorshift64.
fn generate(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut blob = Vec::with_capacity(len.next_multiple_of(8));
    while blob.len() < len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        blob.extend_from_slice(&state.to_le_bytes());
    }
    blob.truncate(len);
    blob
}

fn pool(threads: usize) -> Arc<ThreadPool> {
    Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool"),
    )
}

fn blob_id(c: &mut Criterion) {
    let pools = THREAD_COUNTS.map(|threads| (threads, pool(threads)));
    let backends = Backend::ALL
        .iter()
        .copied()
        .filter(|backend| backend.is_available());
    let encoders: Vec<(Backend, usize, BlobEncoder)> = backends
        .flat_map(|backend| {
            let encoder = BlobEncoder::mainnet()
                .with_backend(backend)
                .expect("available backend");
            pools.iter().map(move |(threads, pool)| {
                (
                    backend,
                    *threads,
                    encoder.clone().with_thread_pool(pool.clone()),
                )
            })
        })
        .collect();
    let config =
        EncodingConfig::new(NonZeroU16::from(MAINNET_N_SHARDS)).get_for_type(DEFAULT_ENCODING);
    let single_thread = &pools[0].1;

    for (label, len) in SIZES {
        // Generated on first use, so that filtered-out sizes cost nothing.
        let blob = OnceLock::new();
        let blob = || -> &[u8] { blob.get_or_init(|| generate(len, len as u64)) };

        let mut group = c.benchmark_group(format!("blob_id/{label}"));
        group
            .throughput(Throughput::Bytes(len as u64))
            .sampling_mode(SamplingMode::Flat)
            .sample_size(10)
            .warm_up_time(Duration::from_secs(1))
            .measurement_time(Duration::from_secs(3));

        group.bench_function(BenchmarkId::new("walrus-core", "x1"), |b| {
            let blob = blob();
            single_thread.install(|| b.iter(|| config.compute_blob_id(black_box(blob)).unwrap()));
        });
        for (backend, threads, encoder) in &encoders {
            let id = BenchmarkId::new(backend.name(), format!("x{threads}"));
            group.bench_function(id, |b| {
                let blob = blob();
                b.iter(|| encoder.blob_id(black_box(blob)).unwrap());
            });
        }
        group.finish();
    }
}

fn main() {
    let mut c = Criterion::default().configure_from_args();
    blob_id(&mut c);
    c.final_summary();
}
