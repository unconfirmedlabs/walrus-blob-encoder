// Copyright (c) Unconfirmed Labs, LLC
// SPDX-License-Identifier: Apache-2.0

//! Reed-Solomon encoders with a selectable `reed-solomon-simd` engine.

use reed_solomon_simd::{
    EncoderResult, ReedSolomonEncoder,
    engine::{DefaultEngine, NoSimd},
    rate::{DefaultRateEncoder, RateEncoder},
};

/// Whether `reed-solomon-simd` can encode `original` source into `recovery` recovery shards.
pub(crate) fn supports(original: usize, recovery: usize) -> bool {
    ReedSolomonEncoder::supports(original, recovery)
}

/// Engine choice: portable code only, or the best SIMD engine `reed-solomon-simd` detects at
/// runtime. All engines produce identical recovery shards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Engine {
    Portable,
    Detected,
}

pub(crate) enum Encoder {
    Portable(DefaultRateEncoder<NoSimd>),
    Detected(DefaultRateEncoder<DefaultEngine>),
}

impl Encoder {
    /// # Panics
    ///
    /// Panics if the parameters are not supported; callers validate them via [`supports`] and use
    /// even, non-zero shard sizes.
    pub(crate) fn new(
        engine: Engine,
        original: usize,
        recovery: usize,
        shard_bytes: usize,
    ) -> Self {
        const VALID: &str = "Reed-Solomon parameters are validated by the shard layout";
        match engine {
            Engine::Portable => Encoder::Portable(
                DefaultRateEncoder::new(original, recovery, shard_bytes, NoSimd::new(), None)
                    .expect(VALID),
            ),
            Engine::Detected => Encoder::Detected(
                DefaultRateEncoder::new(
                    original,
                    recovery,
                    shard_bytes,
                    DefaultEngine::new(),
                    None,
                )
                .expect(VALID),
            ),
        }
    }

    /// Encodes exactly `original` shards of `shard_bytes` bytes each.
    pub(crate) fn encode<'s, 'a>(
        &'s mut self,
        originals: impl IntoIterator<Item = &'a [u8]>,
    ) -> EncoderResult<'s> {
        const SHAPE: &str = "shard count and size match the encoder";
        match self {
            Encoder::Portable(encoder) => {
                originals
                    .into_iter()
                    .for_each(|shard| encoder.add_original_shard(shard).expect(SHAPE));
                encoder.encode().expect(SHAPE)
            }
            Encoder::Detected(encoder) => {
                originals
                    .into_iter()
                    .for_each(|shard| encoder.add_original_shard(shard).expect(SHAPE));
                encoder.encode().expect(SHAPE)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engines_agree() {
        let shards: Vec<Vec<u8>> = (0..667u16)
            .map(|i| (0..64u16).map(|j| (i * 31 + j * 7) as u8).collect())
            .collect();
        let mut portable = Encoder::new(Engine::Portable, 667, 333, 64);
        let mut detected = Encoder::new(Engine::Detected, 667, 333, 64);
        let a: Vec<Vec<u8>> = portable
            .encode(shards.iter().map(Vec::as_slice))
            .recovery_iter()
            .map(<[u8]>::to_vec)
            .collect();
        let b: Vec<Vec<u8>> = detected
            .encode(shards.iter().map(Vec::as_slice))
            .recovery_iter()
            .map(<[u8]>::to_vec)
            .collect();
        assert_eq!(a.len(), 333);
        assert_eq!(a, b);
    }
}
