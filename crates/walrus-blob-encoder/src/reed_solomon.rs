// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Reed-Solomon encoders with a selectable `reed-solomon-simd` engine.

use std::ops::Range;

#[cfg(test)]
mod tests;

use reed_solomon_simd::{
    EncoderResult, ReedSolomonEncoder,
    engine::{self, NoSimd},
    rate::{DefaultRateEncoder, RateEncoder},
};

/// Whether `reed-solomon-simd` can encode `original` source into `recovery` recovery shards.
pub(crate) fn supports(original: usize, recovery: usize) -> bool {
    ReedSolomonEncoder::supports(original, recovery)
}

/// The SIMD engine, chosen at compile time as `reed-solomon-simd`'s `DefaultEngine` is:
/// [`engine::Wide`] when the target enables SSSE3, WebAssembly `simd128` or NEON (with 256-bit
/// vectors when it also enables AVX2), otherwise [`NoSimd`].
#[cfg(any(
    target_feature = "ssse3",
    target_feature = "simd128",
    all(target_arch = "aarch64", target_feature = "neon"),
))]
type SimdEngine = engine::Wide;
#[cfg(not(any(
    target_feature = "ssse3",
    target_feature = "simd128",
    all(target_arch = "aarch64", target_feature = "neon"),
)))]
type SimdEngine = NoSimd;

/// One past the largest skew-table position the encoders for `shapes` (original and recovery
/// counts) read: the portable and SIMD engines build multiplication tables for these positions
/// only, about 128 bytes each, instead of a global 8 MiB table.
pub(crate) fn skew_limit(shapes: &[(usize, usize)]) -> usize {
    shapes
        .iter()
        .map(|&(original, recovery)| engine::utils::encoder_skew_limit(original, recovery))
        .max()
        .unwrap_or(0)
}

/// Engine choice: portable code only, the compile-time SIMD engine, or the AVX-512 engine where
/// available. All engines produce identical recovery shards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Engine {
    Portable,
    Simd,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

pub(crate) enum Encoder {
    Portable(DefaultRateEncoder<NoSimd>),
    Simd(DefaultRateEncoder<SimdEngine>),
    #[cfg(target_arch = "x86_64")]
    Avx512(DefaultRateEncoder<engine::Avx512>),
}

impl Encoder {
    /// An encoder for `engine`, falling back to [`Engine::Simd`] if the AVX-512 engine is
    /// unavailable. The portable and SIMD engines get multiplication tables for the skew
    /// positions below `skew_limit` (see [`skew_limit`]), enough for every shape it is reset to
    /// within that limit.
    ///
    /// # Panics
    ///
    /// Panics if the parameters are not supported; callers validate them via [`supports`] and use
    /// even, non-zero shard sizes.
    pub(crate) fn new(
        engine: Engine,
        original: usize,
        recovery: usize,
        shard_bytes: usize,
        skew_limit: usize,
    ) -> Self {
        let shape = (original, recovery, shard_bytes);
        match engine {
            Engine::Portable => {
                Encoder::Portable(rate_encoder(shape, NoSimd::with_skew_limit(skew_limit)))
            }
            #[cfg(target_arch = "x86_64")]
            Engine::Avx512 => match engine::Avx512::new() {
                Some(engine) => Encoder::Avx512(rate_encoder(shape, engine)),
                None => Self::new(Engine::Simd, original, recovery, shard_bytes, skew_limit),
            },
            Engine::Simd => {
                Encoder::Simd(rate_encoder(shape, SimdEngine::with_skew_limit(skew_limit)))
            }
        }
    }

    /// Makes the encoder encode `original` source into `recovery` recovery shards of
    /// `shard_bytes` bytes, keeping its work space where the size allows.
    pub(crate) fn reset(&mut self, original: usize, recovery: usize, shard_bytes: usize) {
        const SHAPE: &str = "Reed-Solomon parameters are validated by the shard layout";
        match self {
            Encoder::Portable(encoder) => encoder.reset(original, recovery, shard_bytes),
            Encoder::Simd(encoder) => encoder.reset(original, recovery, shard_bytes),
            #[cfg(target_arch = "x86_64")]
            Encoder::Avx512(encoder) => encoder.reset(original, recovery, shard_bytes),
        }
        .expect(SHAPE);
    }

    /// Encodes exactly `original` shards of `shard_bytes` bytes each.
    pub(crate) fn encode<'s, 'a>(
        &'s mut self,
        originals: impl IntoIterator<Item = &'a [u8]>,
    ) -> EncoderResult<'s> {
        self.encode_range(originals, 0..usize::MAX)
    }

    /// [`Self::encode`] where only the recovery shards `recovery` need to be valid: the
    /// transforms skip what only the others need.
    pub(crate) fn encode_range<'s, 'a>(
        &'s mut self,
        originals: impl IntoIterator<Item = &'a [u8]>,
        recovery: Range<usize>,
    ) -> EncoderResult<'s> {
        match self {
            Encoder::Portable(encoder) => encode(encoder, originals, recovery),
            Encoder::Simd(encoder) => encode(encoder, originals, recovery),
            #[cfg(target_arch = "x86_64")]
            Encoder::Avx512(encoder) => encode(encoder, originals, recovery),
        }
    }
}

/// A rate encoder for `(original, recovery, shard_bytes)`.
fn rate_encoder<E: engine::Engine>(
    (original, recovery, shard_bytes): (usize, usize, usize),
    engine: E,
) -> DefaultRateEncoder<E> {
    DefaultRateEncoder::new(original, recovery, shard_bytes, engine, None)
        .expect("Reed-Solomon parameters are validated by the shard layout")
}

fn encode<'s, 'a, E: engine::Engine>(
    encoder: &'s mut DefaultRateEncoder<E>,
    originals: impl IntoIterator<Item = &'a [u8]>,
    recovery: Range<usize>,
) -> EncoderResult<'s> {
    const SHAPE: &str = "shard count and size match the encoder";
    originals
        .into_iter()
        .for_each(|shard| encoder.add_original_shard(shard).expect(SHAPE));
    encoder.encode_range(recovery).expect(SHAPE)
}
