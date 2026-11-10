// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroU16;
#[cfg(feature = "rayon")]
use std::sync::{Arc, OnceLock};

#[cfg(feature = "rayon")]
use rayon::{ThreadPool, ThreadPoolBuilder};
use walrus_blob_encoder::{Backend, BlobEncoder};

/// Thread counts every input is checked with, each on a dedicated pool.
#[cfg(feature = "rayon")]
const THREAD_COUNTS: [usize; 4] = [1, 2, 3, 8];
/// Without rayon every encoder runs on the calling thread.
#[cfg(not(feature = "rayon"))]
const THREAD_COUNTS: [usize; 1] = [1];

/// A thread pool of a given size, or just its size without rayon.
#[cfg(feature = "rayon")]
type Pool = Arc<ThreadPool>;
#[cfg(not(feature = "rayon"))]
type Pool = usize;

#[cfg(feature = "rayon")]
fn pool(threads: usize) -> Pool {
    Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap(),
    )
}

#[cfg(not(feature = "rayon"))]
fn pool(threads: usize) -> Pool {
    threads
}

#[cfg(feature = "rayon")]
fn threads(pool: &Pool) -> usize {
    pool.current_num_threads()
}

#[cfg(not(feature = "rayon"))]
fn threads(pool: &Pool) -> usize {
    *pool
}

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

#[cfg(feature = "rayon")]
fn pools() -> &'static [Pool] {
    static POOLS: OnceLock<Vec<Pool>> = OnceLock::new();
    POOLS.get_or_init(|| THREAD_COUNTS.iter().map(|&n| pool(n)).collect())
}

#[cfg(not(feature = "rayon"))]
fn pools() -> &'static [Pool] {
    &THREAD_COUNTS
}

fn encoder(n_shards: u16, backend: Backend, pool: &Pool) -> (String, BlobEncoder) {
    let encoder = BlobEncoder::new(NonZeroU16::new(n_shards).unwrap())
        .unwrap()
        .with_backend(backend)
        .unwrap();
    #[cfg(feature = "rayon")]
    let encoder = encoder.with_thread_pool(pool.clone());
    (format!("{backend} x{}", threads(pool)), encoder)
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

/// Uses one pool at a time on resource-constrained hosts, preserving every test case.
fn run_serial(n_shards: u16, threads: &[usize], check: &impl Fn(&str, &BlobEncoder)) -> bool {
    if std::env::var_os("WALRUS_BLOB_ENCODER_SERIAL_TESTS").is_none() {
        return false;
    }
    for &threads in threads {
        let pool = pool(threads);
        for backend in backends() {
            let (label, encoder) = encoder(n_shards, backend, &pool);
            check(&label, &encoder);
        }
    }
    true
}

/// Runs `check` concurrently with an encoder for every backend under test. Successive `case`
/// numbers rotate through the thread counts, so every test covers all of them.
pub fn for_each_backend(n_shards: u16, case: usize, check: impl Fn(&str, &BlobEncoder) + Sync) {
    if run_serial(
        n_shards,
        &[THREAD_COUNTS[case % THREAD_COUNTS.len()]],
        &check,
    ) {
        return;
    }
    let pool = &pools()[case % THREAD_COUNTS.len()];
    let encoders = backends()
        .into_iter()
        .map(|backend| encoder(n_shards, backend, pool))
        .collect();
    run_concurrently(encoders, check);
}

/// Runs `check` concurrently with an encoder for every backend under test and thread count.
pub fn for_each_backend_and_pool(n_shards: u16, check: impl Fn(&str, &BlobEncoder) + Sync) {
    if run_serial(n_shards, &THREAD_COUNTS, &check) {
        return;
    }
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
