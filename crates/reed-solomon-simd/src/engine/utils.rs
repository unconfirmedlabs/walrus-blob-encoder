//! A collection of utility functions and helpers to facilitate the implementation of the [`Engine`] trait.
//!
//! [`Engine`]: crate::engine::Engine

use crate::engine::{fwht, tables, Engine, GfElement, ShardsRefMut, GF_BITS, GF_MODULUS, GF_ORDER};
use crate::rate::use_high_rate;
use core::iter::zip;

// ======================================================================
// FUNCTIONS - PUBLIC

/// Evaluate Polynomial using Fast Walsh-Hadamard Transform (FWHT).
///
/// This function is designed to be inlined into an Engine's
/// implementation of `eval_poly`.
#[inline(always)]
pub fn eval_poly(erasures: &mut [GfElement; GF_ORDER], truncated_size: usize) {
    let log_walsh = tables::get_log_walsh();

    fwht::fwht(erasures, truncated_size);

    for (e, factor) in zip(erasures.iter_mut(), log_walsh.iter()) {
        let product = u32::from(*e) * u32::from(*factor);
        *e = add_mod(product as GfElement, (product >> GF_BITS) as GfElement);
    }

    fwht::fwht(erasures, GF_ORDER);
}

/// One past the largest skew-table position that the encoders of [`rate`]
/// read for `original_count` original and `recovery_count` recovery shards,
/// or [`GF_MODULUS`] (the table length) if the counts are unsupported.
///
/// The transforms of a work area of `work_count` shards in chunks of
/// `chunk_size` shards read positions below `work_count + chunk_size`.
///
/// [`rate`]: crate::rate
/// [`GF_MODULUS`]: crate::engine::GF_MODULUS
pub fn encoder_skew_limit(original_count: usize, recovery_count: usize) -> usize {
    let table_len = usize::from(GF_MODULUS);
    let (chunk_size, work_count) = match use_high_rate(original_count, recovery_count) {
        Ok(true) => {
            let chunk_size = recovery_count.next_power_of_two();
            (chunk_size, original_count.next_multiple_of(chunk_size))
        }
        Ok(false) => {
            let chunk_size = original_count.next_power_of_two();
            (chunk_size, recovery_count.next_multiple_of(chunk_size))
        }
        Err(_) => return table_len,
    };
    (work_count + chunk_size).min(table_len)
}

/// `x[] ^= y[]`
#[inline(always)]
pub fn xor(xs: &mut [[u8; 64]], ys: &[[u8; 64]]) {
    debug_assert_eq!(xs.len(), ys.len());

    for (x_chunk, y_chunk) in zip(xs.iter_mut(), ys.iter()) {
        for (x, y) in zip(x_chunk.iter_mut(), y_chunk.iter()) {
            *x ^= y;
        }
    }
}

/// `data[x .. x + count] ^= data[y .. y + count]`
///
/// Ranges must not overlap.
#[inline(always)]
pub fn xor_within(data: &mut ShardsRefMut, x: usize, y: usize, count: usize) {
    let (xs, ys) = data.flat2_mut(x, y, count);
    xor(xs, ys);
}

// ======================================================================
// FUNCTIONS - CRATE - Galois field operations

/// Some kind of addition.
#[inline(always)]
pub(crate) fn add_mod(x: GfElement, y: GfElement) -> GfElement {
    let sum = u32::from(x) + u32::from(y);
    (sum + (sum >> GF_BITS)) as GfElement
}

/// Some kind of subtraction.
#[inline(always)]
pub(crate) fn sub_mod(x: GfElement, y: GfElement) -> GfElement {
    let dif = u32::from(x).wrapping_sub(u32::from(y));
    dif.wrapping_add(dif >> GF_BITS) as GfElement
}

// ======================================================================
// FUNCTIONS - CRATE

/// FFT with `skew_delta = pos + size`.
#[inline(always)]
pub(crate) fn fft_skew_end(
    engine: &impl Engine,
    data: &mut ShardsRefMut,
    pos: usize,
    size: usize,
    truncated_size: usize,
) {
    engine.fft(data, pos, size, truncated_size, pos + size);
}

/// IFFT with `skew_delta = pos + size`.
#[inline(always)]
pub(crate) fn ifft_skew_end(
    engine: &impl Engine,
    data: &mut ShardsRefMut,
    pos: usize,
    size: usize,
    truncated_size: usize,
) {
    engine.ifft(data, pos, size, truncated_size, pos + size);
}

// Formal derivative.
pub(crate) fn formal_derivative(data: &mut ShardsRefMut) {
    for i in 1..data.len() {
        let width: usize = 1 << i.trailing_zeros();
        xor_within(data, i - width, i, width);
    }
}
