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
