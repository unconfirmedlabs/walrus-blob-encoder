// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::{
    num::NonZeroU16,
    sync::{Arc, OnceLock},
};

use rayon::{ThreadPool, ThreadPoolBuilder};
use walrus_blob_encoder::{Backend, BlobEncoder};

/// Thread counts every input is checked with, each on a dedicated pool.
const THREAD_COUNTS: [usize; 4] = [1, 2, 3, 8];

/// The backends under test: all available ones, or exactly `$WALRUS_BLOB_ENCODER_BACKEND`.
fn backends() -> Vec<Backend> {
    match std::env::var("WALRUS_BLOB_ENCODER_BACKEND") {
        Ok(name) => {
            let backend = *Backend::ALL
                .iter()
                .find(|backend| backend.name() == name)
                .unwrap_or_else(|| panic!("unknown backend {name}"));
            assert!(backend.is_available(), "backend {backend} is not available");
            vec![backend]
        }
        Err(_) => Backend::ALL
            .iter()
            .copied()
            .filter(|backend| backend.is_available())
            .collect(),
    }
}

fn pools() -> &'static [Arc<ThreadPool>] {
    static POOLS: OnceLock<Vec<Arc<ThreadPool>>> = OnceLock::new();
    POOLS.get_or_init(|| {
        THREAD_COUNTS
            .iter()
            .map(|&n| Arc::new(ThreadPoolBuilder::new().num_threads(n).build().unwrap()))
            .collect()
    })
}

fn encoder(n_shards: u16, backend: Backend, pool: &Arc<ThreadPool>) -> (String, BlobEncoder) {
    let encoder = BlobEncoder::new(NonZeroU16::new(n_shards).unwrap())
        .unwrap()
        .with_backend(backend)
        .unwrap()
        .with_thread_pool(pool.clone());
    (
        format!("{backend} x{}", pool.current_num_threads()),
        encoder,
    )
}

fn run_concurrently(
    encoders: Vec<(String, BlobEncoder)>,
    check: impl Fn(&str, &BlobEncoder) + Sync,
) {
    std::thread::scope(|scope| {
        for (label, encoder) in &encoders {
            let check = &check;
            scope.spawn(move || check(label, encoder));
        }
    });
}

/// Runs `check` concurrently with an encoder for every backend under test. Successive `case`
/// numbers rotate through the thread counts, so every test covers all of them.
pub fn for_each_backend(n_shards: u16, case: usize, check: impl Fn(&str, &BlobEncoder) + Sync) {
    let pool = &pools()[case % THREAD_COUNTS.len()];
    let encoders = backends()
        .into_iter()
        .map(|backend| encoder(n_shards, backend, pool))
        .collect();
    run_concurrently(encoders, check);
}

/// Runs `check` concurrently with an encoder for every backend under test and thread count.
pub fn for_each_backend_and_pool(n_shards: u16, check: impl Fn(&str, &BlobEncoder) + Sync) {
    let encoders = backends()
        .into_iter()
        .flat_map(|backend| {
            pools()
                .iter()
                .map(move |pool| encoder(n_shards, backend, pool))
        })
        .collect();
    run_concurrently(encoders, check);
}

/// Deterministic test input; the generator of `tests/vectors/cli-1000.txt`.
pub fn generate(len: usize, seed: u64) -> Vec<u8> {
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
