// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Reed-Solomon encoders with a selectable `reed-solomon-simd` engine.

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

/// The `reed-solomon-simd` SIMD engine, chosen at compile time: [`engine::Wide`] when the target
/// enables SSSE3, WebAssembly `simd128` or NEON, otherwise [`NoSimd`].
type SimdEngine = engine::DefaultEngine;

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
            Engine::Avx512 => match engine::Avx512::new() {
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
