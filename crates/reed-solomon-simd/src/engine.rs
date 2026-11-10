//! Low-level building blocks for Reed-Solomon encoding/decoding.
//!
//! **This is an advanced module which is not needed for [simple usage] or [basic usage].**
//!
//! This module is relevant if you want to
//! - use [`rate`] module and need an [`Engine`] to use with it.
//! - create your own [`Engine`].
//! - understand/benchmark/test at low level.
//!
//! # Engines
//!
//! An [`Engine`] is an implementation of basic low-level algorithms
//! needed for Reed-Solomon encoding/decoding.
//!
//! - [`NoSimd`]
//!     - Basic optimized engine without SIMD so that it works on all CPUs.
//! - [`Wide`]
//!     - Optimized engine using portable 128-bit SIMD: SSSE3 on x86(-64),
//!       `simd128` on WebAssembly, Neon on `AArch64`.
//! - [`Avx512`] (x86-64 only)
//!     - Engine using AVX-512F and AVX-512BW, with GFNI where available.
//!       Constructed only if the CPU supports it: statically enabled target
//!       features, or with the `runtime-detection` feature, detected at runtime.
//!       Never chosen by [`DefaultEngine`].
//! - [`DefaultEngine`]
//!     - Default engine which is used when no specific engine is given.
//!     - [`Wide`] when the build enables one of those target features,
//!       otherwise [`NoSimd`]. Chosen at compile time.
//!
//! [simple usage]: crate#simple-usage
//! [basic usage]: crate#basic-usage
//! [`ReedSolomonEncoder`]: crate::ReedSolomonEncoder
//! [`ReedSolomonDecoder`]: crate::ReedSolomonDecoder
//! [`rate`]: crate::rate

pub(crate) use self::shards::Shards;
pub(crate) use utils::{fft_skew_end, formal_derivative, ifft_skew_end, xor_within};

pub use self::{
    engine_default::DefaultEngine, engine_nosimd::NoSimd, engine_wide::Wide, shards::ShardsRefMut,
};

#[cfg(target_arch = "x86_64")]
pub use self::engine_avx512::Avx512;

#[cfg(target_arch = "x86_64")]
mod engine_avx512;
mod engine_default;
mod engine_nosimd;
mod engine_wide;

mod fwht;
mod shards;

pub mod tables;
pub mod utils;

// ======================================================================
// CONST - PUBLIC

/// Size of Galois field element [`GfElement`] in bits.
pub const GF_BITS: usize = 16;

/// Galois field order, i.e. number of elements.
pub const GF_ORDER: usize = 65536;

/// `GF_ORDER - 1`
pub const GF_MODULUS: GfElement = 65535;

/// Galois field polynomial.
pub const GF_POLYNOMIAL: usize = 0x1002D;

/// TODO
pub const CANTOR_BASIS: [GfElement; GF_BITS] = [
    0x0001, 0xACCA, 0x3C0E, 0x163E, 0xC582, 0xED2E, 0x914C, 0x4012, 0x6C98, 0x10D8, 0x6A72, 0xB900,
    0xFDB8, 0xFB34, 0xFF38, 0x991E,
];

// ======================================================================
// TYPE ALIASES - PUBLIC

/// Galois field element.
pub type GfElement = u16;

// ======================================================================
// Engine - PUBLIC

/// Trait for compute-intensive low-level algorithms needed
/// for Reed-Solomon encoding/decoding.
///
/// This is the trait you would implement to provide SIMD support
/// for a CPU architecture not already provided.
pub trait Engine {
    // ============================================================
    // REQUIRED

    /// In-place decimation-in-time FFT (fast Fourier transform).
    ///
    /// - FFT is done on chunk `data[pos .. pos + size]`
    /// - `size` must be `2^n`
    /// - Before function call `data[pos .. pos + size]` must be valid.
    /// - After function call
    ///     - `data[pos .. pos + truncated_size]`
    ///       contains valid FFT result.
    ///     - `data[pos + truncated_size .. pos + size]`
    ///       contains valid FFT result if this contained
    ///       only `0u8`:s and garbage otherwise.
    fn fft(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    );

    /// In-place decimation-in-time IFFT (inverse fast Fourier transform).
    ///
    /// - IFFT is done on chunk `data[pos .. pos + size]`
    /// - `size` must be `2^n`
    /// - Before function call `data[pos .. pos + size]` must be valid.
    /// - After function call
    ///     - `data[pos .. pos + truncated_size]`
    ///       contains valid IFFT result.
    ///     - `data[pos + truncated_size .. pos + size]`
    ///       contains valid IFFT result if this contained
    ///       only `0u8`:s and garbage otherwise.
    fn ifft(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    );

    /// `x[] *= log_m`
    fn mul(&self, x: &mut [[u8; 64]], log_m: GfElement);

    // ============================================================
    // PROVIDED

    /// Evaluate polynomial.
    fn eval_poly(erasures: &mut [GfElement; GF_ORDER], truncated_size: usize)
    where
        Self: Sized,
    {
        utils::eval_poly(erasures, truncated_size);
    }
}

// ======================================================================
// TESTS

// Engines are tested indirectly via roundtrip tests of HighRate and LowRate.
