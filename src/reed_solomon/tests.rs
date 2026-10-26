// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use reed_solomon_simd::engine::NoSimd;
use reed_solomon_simd::rate::{DefaultRateEncoder, RateEncoder};

use super::{Encoder, Engine, SimdEngine};

/// SplitMix64, for reproducible random test data.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }

    fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next() as u8).collect()
    }
}

fn recovery<E: reed_solomon_simd::engine::Engine>(
    engine: E,
    originals: &[Vec<u8>],
    recovery: usize,
) -> Vec<Vec<u8>> {
    let shard_bytes = originals[0].len();
    let mut encoder =
        DefaultRateEncoder::new(originals.len(), recovery, shard_bytes, engine, None).unwrap();
    for shard in originals {
        encoder.add_original_shard(shard).unwrap();
    }
    let result = encoder.encode().unwrap();
    result.recovery_iter().map(<[u8]>::to_vec).collect()
}

/// Parameter sets `(original, recovery)`: those of the crate for `n` shards, and `(n - 2f, 2f)`.
fn parameter_sets() -> Vec<(usize, usize)> {
    [
        4, 5, 7, 10, 13, 31, 64, 100, 101, 333, 999, 1000, 1001, 2000,
    ]
    .into_iter()
    .flat_map(|n| {
        let f = (n - 1) / 3;
        [(n - f, f), (n - 2 * f, 2 * f)]
    })
    .collect()
}

fn shard_sizes(rng: &mut Rng) -> Vec<usize> {
    let mut sizes = vec![2, 4, 6, 62, 64, 66, 126, 128, 130, 314, 1164, 1206, 4096];
    sizes.extend((0..3).map(|_| 2 + 2 * rng.below(300)));
    sizes
}

#[test]
fn engines_agree() {
    let shards: Vec<Vec<u8>> = (0..667u16)
        .map(|i| (0..64u16).map(|j| (i * 31 + j * 7) as u8).collect())
        .collect();
    let encode = |engine| {
        let mut encoder = Encoder::new(engine, 667, 333, 64);
        let result = encoder.encode(shards.iter().map(Vec::as_slice));
        result
            .recovery_iter()
            .map(<[u8]>::to_vec)
            .collect::<Vec<_>>()
    };
    let portable = encode(Engine::Portable);
    assert_eq!(portable.len(), 333);
    assert_eq!(portable, encode(Engine::Simd));
    #[cfg(target_arch = "x86_64")]
    assert_eq!(portable, encode(Engine::Avx512));
}

#[test]
fn simd_engine_matches_portable() {
    let mut rng = Rng(1);
    for (original, recovery_count) in parameter_sets() {
        for shard_bytes in shard_sizes(&mut rng) {
            let originals: Vec<_> = (0..original).map(|_| rng.bytes(shard_bytes)).collect();
            assert_eq!(
                recovery(SimdEngine::new(), &originals, recovery_count),
                recovery(NoSimd::new(), &originals, recovery_count),
                "({original}, {recovery_count}) with {shard_bytes}-byte shards"
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use reed_solomon_simd::engine::{Engine as _, ShardsRefMut};

    use super::*;
    use crate::reed_solomon::avx512::Avx512;

    /// The AVX-512 kernels the CPU supports. With `WALRUS_BLOB_ENCODER_BACKEND=avx512` the
    /// `vpshufb` kernel is required.
    fn engines() -> Vec<(&'static str, Avx512)> {
        let required = std::env::var("WALRUS_BLOB_ENCODER_BACKEND").is_ok_and(|b| b == "avx512");
        let engines: Vec<_> = [("shuffle", Avx512::shuffle()), ("gfni", Avx512::gfni())]
            .into_iter()
            .filter_map(|(name, engine)| {
                if engine.is_none() {
                    eprintln!("note: the CPU lacks the {name} AVX-512 kernel; not tested");
                }
                Some(name).zip(engine)
            })
            .collect();
        assert!(
            !required || engines.iter().any(|(name, _)| *name == "shuffle"),
            "WALRUS_BLOB_ENCODER_BACKEND=avx512: the AVX-512 engine is not available on this CPU"
        );
        engines
    }

    #[test]
    fn recovery_matches_portable() {
        let engines = engines();
        if engines.is_empty() {
            return;
        }
        let mut rng = Rng(2);
        for (original, recovery_count) in parameter_sets() {
            for shard_bytes in shard_sizes(&mut rng) {
                let originals: Vec<_> = (0..original).map(|_| rng.bytes(shard_bytes)).collect();
                let expected = recovery(NoSimd::new(), &originals, recovery_count);
                for &(name, engine) in &engines {
                    assert!(
                        recovery(engine, &originals, recovery_count) == expected,
                        "{name}: ({original}, {recovery_count}) with {shard_bytes}-byte shards"
                    );
                }
            }
        }
    }

    /// Random shards of `count` shards of `chunks` 64-byte chunks.
    fn random_shards(rng: &mut Rng, count: usize, chunks: usize) -> Vec<[u8; 64]> {
        (0..count * chunks)
            .map(|_| rng.bytes(64).try_into().unwrap())
            .collect()
    }

    #[test]
    fn transforms_match_portable() {
        let engines = engines();
        let reference = NoSimd::new();
        let mut rng = Rng(3);
        for log_size in 0..=10 {
            let size = 1 << log_size;
            for _ in 0..8 {
                let chunks = 1 + rng.below(3);
                let pos = rng.below(4);
                let truncated_size = 1 + rng.below(size);
                let skew_delta = size * (1 + rng.below(60_000 / size));
                let input = random_shards(&mut rng, pos + size, chunks);
                let log_m = rng.next() as u16;
                for &(name, engine) in &engines {
                    let run = |transform: &dyn Fn(&mut ShardsRefMut)| {
                        let mut data = input.clone();
                        transform(&mut ShardsRefMut::new(pos + size, chunks, &mut data));
                        data
                    };
                    let case = format!(
                        "{name}: size {size}, pos {pos}, truncated {truncated_size}, skew \
                         {skew_delta}, {chunks} chunks"
                    );
                    assert!(
                        run(&|d| engine.fft(d, pos, size, truncated_size, skew_delta))
                            == run(&|d| reference.fft(d, pos, size, truncated_size, skew_delta)),
                        "fft, {case}"
                    );
                    assert!(
                        run(&|d| engine.ifft(d, pos, size, truncated_size, skew_delta))
                            == run(&|d| reference.ifft(d, pos, size, truncated_size, skew_delta)),
                        "ifft, {case}"
                    );
                    let mut x = input.clone();
                    let mut y = input.clone();
                    engine.mul(&mut x, log_m);
                    reference.mul(&mut y, log_m);
                    assert!(x == y, "mul by {log_m}, {case}");
                }
            }
        }
    }

    #[test]
    fn mul_matches_portable_for_every_factor() {
        let engines = engines();
        let reference = NoSimd::new();
        let input = random_shards(&mut Rng(4), 1, 1);
        for log_m in 0..=u16::MAX {
            let mut expected = input.clone();
            reference.mul(&mut expected, log_m);
            for &(name, engine) in &engines {
                let mut x = input.clone();
                engine.mul(&mut x, log_m);
                assert!(x == expected, "{name}: mul by {log_m}");
            }
        }
    }
}
