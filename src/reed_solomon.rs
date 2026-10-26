// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Reed-Solomon encoders with a selectable `reed-solomon-simd` engine.

#[cfg(target_arch = "x86_64")]
mod avx512;
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

/// The best `reed-solomon-simd` engine: the one the target statically supports, or, with the
/// `runtime-detection` feature and below AVX2, the one detected at runtime.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2"
))]
type SimdEngine = engine::Avx2;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(target_feature = "avx2"),
    feature = "runtime-detection"
))]
type SimdEngine = engine::DefaultEngine;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(target_feature = "avx2"),
    not(feature = "runtime-detection"),
    target_feature = "ssse3"
))]
type SimdEngine = engine::Ssse3;
#[cfg(target_arch = "aarch64")]
type SimdEngine = engine::Neon;
#[cfg(not(any(
    target_arch = "aarch64",
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        any(
            target_feature = "avx2",
            target_feature = "ssse3",
            feature = "runtime-detection"
        )
    )
)))]
type SimdEngine = NoSimd;

/// Engine choice: portable code only, the best `reed-solomon-simd` engine, or this crate's
/// AVX-512 engine where available. All engines produce identical recovery shards.
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
    Avx512(DefaultRateEncoder<avx512::Avx512>),
}

impl Encoder {
    /// An encoder for `engine`, falling back to [`Engine::Simd`] if the AVX-512 engine is
    /// unavailable.
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
    ) -> Self {
        let shape = (original, recovery, shard_bytes);
        match engine {
            Engine::Portable => Encoder::Portable(rate_encoder(shape, NoSimd::new())),
            #[cfg(target_arch = "x86_64")]
            Engine::Avx512 => match avx512::Avx512::new() {
                Some(engine) => Encoder::Avx512(rate_encoder(shape, engine)),
                None => Self::new(Engine::Simd, original, recovery, shard_bytes),
            },
            Engine::Simd => Encoder::Simd(rate_encoder(shape, SimdEngine::new())),
        }
    }

    /// Encodes exactly `original` shards of `shard_bytes` bytes each.
    pub(crate) fn encode<'s, 'a>(
        &'s mut self,
        originals: impl IntoIterator<Item = &'a [u8]>,
    ) -> EncoderResult<'s> {
        match self {
            Encoder::Portable(encoder) => encode(encoder, originals),
            Encoder::Simd(encoder) => encode(encoder, originals),
            #[cfg(target_arch = "x86_64")]
            Encoder::Avx512(encoder) => encode(encoder, originals),
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
) -> EncoderResult<'s> {
    const SHAPE: &str = "shard count and size match the encoder";
    originals
        .into_iter()
        .for_each(|shard| encoder.add_original_shard(shard).expect(SHAPE));
    encoder.encode().expect(SHAPE)
}
