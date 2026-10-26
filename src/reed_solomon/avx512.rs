// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Adapted and modified from `src/engine/engine_avx2.rs` of reed-solomon-simd 3.1.0, which is
// based on Leopard-RS. The original notices follow; the full license texts are in
// LICENSE-THIRD-PARTY.
//
// Copyright (c) 2023 Anders Trier Olesen
// Copyright (c) 2022 Markus Laire
// Licensed under the MIT License.
//
// Copyright (c) 2017 Christopher A. Taylor. All rights reserved.
// Licensed under the BSD 3-Clause License.

//! AVX-512 `reed-solomon-simd` engine.
//!
//! A 64-byte chunk holds 32 GF(2^16) elements as 32 low bytes followed by 32 high bytes, so one
//! zmm register holds a whole chunk. Multiplying by a constant combines products of the chunk and
//! of its half-swapped copy: either nibble table lookups (`vpshufb`) or, with GFNI, 8x8 bit
//! matrix products (`vgf2p8affineqb`). The FFT and IFFT apply both butterfly layers of a radix-4
//! step to a chunk while it is in registers.

#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m512i, _mm_loadu_si128, _mm512_and_si512, _mm512_broadcast_i32x4,
    _mm512_gf2p8affine_epi64_epi8, _mm512_loadu_si512, _mm512_mask_broadcast_i32x4,
    _mm512_set_epi64, _mm512_set1_epi8, _mm512_setzero_si512, _mm512_shuffle_epi8,
    _mm512_shuffle_i64x2, _mm512_srli_epi64, _mm512_storeu_si512, _mm512_ternarylogic_epi64,
    _mm512_xor_si512,
};
use std::iter::zip;
use std::sync::OnceLock;

use reed_solomon_simd::engine::{
    Engine, GF_MODULUS, GF_ORDER, GfElement, ShardsRefMut,
    tables::{self, Mul128, Multiply128lutT, Skew},
};

type Chunk = [u8; 64];

/// An [`Engine`] for CPUs with AVX-512F and AVX-512BW, using GFNI when available. Only
/// constructible through constructors that check its kernel's features, statically or at runtime,
/// so a value proves that the kernel can run.
#[derive(Clone, Copy)]
pub(crate) struct Avx512 {
    kernel: KernelChoice,
    skew: &'static Skew,
}

#[derive(Clone, Copy)]
enum KernelChoice {
    Shuffle(Shuffle),
    Gfni(Gfni),
}

/// Whether the CPU has all `features`: statically enabled for the target, or, with the
/// `runtime-detection` feature, detected at runtime.
macro_rules! available {
    ($($feature:tt),+) => {
        $((cfg!(target_feature = $feature) || detected!($feature)))&&+
    };
}

#[cfg(feature = "runtime-detection")]
macro_rules! detected {
    ($feature:tt) => {
        std::is_x86_feature_detected!($feature)
    };
}

#[cfg(not(feature = "runtime-detection"))]
macro_rules! detected {
    ($feature:tt) => {
        false
    };
}

/// Calls a kernel entry point. Where `$static` shows the kernel's features are statically enabled,
/// the entry point is an ordinary function; otherwise it is a `#[target_feature]` function,
/// called on the strength of the kernel value passed to it.
macro_rules! enter {
    ($static:meta, $call:expr) => {{
        #[cfg($static)]
        {
            $call
        }
        #[cfg(not($static))]
        {
            // SAFETY: a kernel value exists only if the CPU supports its features.
            unsafe { $call }
        }
    }};
}

impl Avx512 {
    /// The engine with the fastest kernel available, or `None` without AVX-512F and AVX-512BW.
    pub(crate) fn new() -> Option<Self> {
        Self::gfni().or_else(Self::shuffle)
    }

    /// The engine with the `vpshufb` kernel, or `None` if its features are unavailable.
    pub(crate) fn shuffle() -> Option<Self> {
        available!("avx512f", "avx512bw")
            .then(|| Self::with(KernelChoice::Shuffle(Shuffle(tables::get_mul128()))))
    }

    /// The engine with the GFNI kernel, or `None` if its features are unavailable.
    pub(crate) fn gfni() -> Option<Self> {
        available!("avx512f", "avx512bw", "gfni")
            .then(|| Self::with(KernelChoice::Gfni(Gfni(affine_table()))))
    }

    fn with(kernel: KernelChoice) -> Self {
        Avx512 {
            kernel,
            skew: tables::get_skew(),
        }
    }
}

impl Engine for Avx512 {
    fn fft(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        let args = (self.skew, pos, size, truncated_size, skew_delta);
        match self.kernel {
            KernelChoice::Shuffle(kernel) => enter!(
                all(target_feature = "avx512f", target_feature = "avx512bw"),
                fft_shuffle(kernel, data, args)
            ),
            KernelChoice::Gfni(kernel) => enter!(
                all(
                    target_feature = "avx512f",
                    target_feature = "avx512bw",
                    target_feature = "gfni"
                ),
                fft_gfni(kernel, data, args)
            ),
        }
    }

    fn ifft(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        let args = (self.skew, pos, size, truncated_size, skew_delta);
        match self.kernel {
            KernelChoice::Shuffle(kernel) => enter!(
                all(target_feature = "avx512f", target_feature = "avx512bw"),
                ifft_shuffle(kernel, data, args)
            ),
            KernelChoice::Gfni(kernel) => enter!(
                all(
                    target_feature = "avx512f",
                    target_feature = "avx512bw",
                    target_feature = "gfni"
                ),
                ifft_gfni(kernel, data, args)
            ),
        }
    }

    fn mul(&self, x: &mut [Chunk], log_m: GfElement) {
        match self.kernel {
            KernelChoice::Shuffle(kernel) => enter!(
                all(target_feature = "avx512f", target_feature = "avx512bw"),
                mul_shuffle(kernel, x, log_m)
            ),
            KernelChoice::Gfni(kernel) => enter!(
                all(
                    target_feature = "avx512f",
                    target_feature = "avx512bw",
                    target_feature = "gfni"
                ),
                mul_gfni(kernel, x, log_m)
            ),
        }
    }
}

/// The skew table and the `pos, size, truncated_size, skew_delta` arguments of an FFT or IFFT.
type Args = (&'static Skew, usize, usize, usize, usize);

#[cfg_attr(
    not(all(target_feature = "avx512f", target_feature = "avx512bw")),
    target_feature(enable = "avx512f,avx512bw")
)]
fn fft_shuffle(kernel: Shuffle, data: &mut ShardsRefMut, args: Args) {
    fft(kernel, data, args);
}

#[cfg_attr(
    not(all(target_feature = "avx512f", target_feature = "avx512bw")),
    target_feature(enable = "avx512f,avx512bw")
)]
fn ifft_shuffle(kernel: Shuffle, data: &mut ShardsRefMut, args: Args) {
    ifft(kernel, data, args);
}

#[cfg_attr(
    not(all(target_feature = "avx512f", target_feature = "avx512bw")),
    target_feature(enable = "avx512f,avx512bw")
)]
fn mul_shuffle(kernel: Shuffle, x: &mut [Chunk], log_m: GfElement) {
    mul(kernel, x, log_m);
}

#[cfg_attr(
    not(all(
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "gfni"
    )),
    target_feature(enable = "avx512f,avx512bw,gfni")
)]
fn fft_gfni(kernel: Gfni, data: &mut ShardsRefMut, args: Args) {
    fft(kernel, data, args);
}

#[cfg_attr(
    not(all(
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "gfni"
    )),
    target_feature(enable = "avx512f,avx512bw,gfni")
)]
fn ifft_gfni(kernel: Gfni, data: &mut ShardsRefMut, args: Args) {
    ifft(kernel, data, args);
}

#[cfg_attr(
    not(all(
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "gfni"
    )),
    target_feature(enable = "avx512f,avx512bw,gfni")
)]
fn mul_gfni(kernel: Gfni, x: &mut [Chunk], log_m: GfElement) {
    mul(kernel, x, log_m);
}

/// Multiplication of a chunk by a constant.
///
/// # Safety
///
/// Implementors may only be constructed once the CPU is known to support AVX-512F and AVX-512BW
/// and the implementor's own instructions; the provided methods rely on the former.
unsafe trait Kernel: Copy {
    type Lut: Copy;

    /// The table for multiplying by `log_m`.
    fn lut(self, log_m: GfElement) -> Self::Lut;

    /// The table for multiplying by zero.
    fn zero(self) -> Self::Lut;

    /// `chunk * lut`.
    fn mul(self, lut: Self::Lut, chunk: __m512i) -> __m512i;

    /// The table for a butterfly with factor `log_m`, where `GF_MODULUS` stands for zero.
    #[inline(always)]
    fn butterfly_lut(self, log_m: GfElement) -> Self::Lut {
        if log_m == GF_MODULUS {
            self.zero()
        } else {
            self.lut(log_m)
        }
    }

    #[inline(always)]
    fn load(self, chunk: &Chunk) -> __m512i {
        // SAFETY: the chunk is 64 readable bytes, and the trait contract gives AVX-512F.
        unsafe { _mm512_loadu_si512(chunk.as_ptr().cast()) }
    }

    #[inline(always)]
    fn store(self, chunk: &mut Chunk, value: __m512i) {
        // SAFETY: the chunk is 64 writable bytes, and the trait contract gives AVX-512F.
        unsafe { _mm512_storeu_si512(chunk.as_mut_ptr().cast(), value) }
    }

    /// `x ^ y * lut`.
    #[inline(always)]
    fn mul_add(self, x: __m512i, y: __m512i, lut: Self::Lut) -> __m512i {
        // SAFETY: the trait contract gives AVX-512F.
        unsafe { _mm512_xor_si512(x, self.mul(lut, y)) }
    }

    /// The chunk with its 256-bit halves swapped, so its high bytes face the low bytes.
    #[inline(always)]
    fn swap_halves(self, chunk: __m512i) -> __m512i {
        // SAFETY: the trait contract gives AVX-512F.
        unsafe { _mm512_shuffle_i64x2::<0b01_00_11_10>(chunk, chunk) }
    }

    /// `x ^ y`.
    #[inline(always)]
    fn xor(self, x: __m512i, y: __m512i) -> __m512i {
        // SAFETY: the trait contract gives AVX-512F.
        unsafe { _mm512_xor_si512(x, y) }
    }

    /// The FFT butterfly `x ^= y * lut; y ^= x`.
    #[inline(always)]
    fn fft_butterfly(self, x: &mut __m512i, y: &mut __m512i, lut: Self::Lut) {
        *x = self.mul_add(*x, *y, lut);
        *y = self.xor(*y, *x);
    }

    /// The IFFT butterfly `y ^= x; x ^= y * lut`.
    #[inline(always)]
    fn ifft_butterfly(self, x: &mut __m512i, y: &mut __m512i, lut: Self::Lut) {
        *y = self.xor(*y, *x);
        *x = self.mul_add(*x, *y, lut);
    }
}

/// Nibble table kernel: the low half of the product takes the low-byte tables of the chunk's
/// nibbles, the high half the high-byte tables, each 128-bit table broadcast to its half.
#[derive(Clone, Copy)]
struct Shuffle(&'static Mul128);

// SAFETY: `Shuffle` is only constructed in `Avx512::shuffle` once AVX-512F and AVX-512BW are
// known to be available, and needs nothing else.
unsafe impl Kernel for Shuffle {
    type Lut = [__m512i; 4];

    #[inline(always)]
    fn lut(self, log_m: GfElement) -> Self::Lut {
        let Multiply128lutT { lo, hi } = &self.0[usize::from(log_m)];
        let pair = |low: &u128, high: &u128| {
            // SAFETY: both tables are 16 readable bytes, and the kernel implies AVX-512F.
            unsafe {
                let low = _mm512_broadcast_i32x4(_mm_loadu_si128(core::ptr::from_ref(low).cast()));
                let high = _mm_loadu_si128(core::ptr::from_ref(high).cast());
                _mm512_mask_broadcast_i32x4(low, 0xff00, high)
            }
        };
        [
            pair(&lo[0], &hi[2]),
            pair(&lo[1], &hi[3]),
            pair(&lo[2], &hi[0]),
            pair(&lo[3], &hi[1]),
        ]
    }

    #[inline(always)]
    fn zero(self) -> Self::Lut {
        // SAFETY: the kernel implies AVX-512F.
        [unsafe { _mm512_setzero_si512() }; 4]
    }

    #[inline(always)]
    fn mul(self, lut: Self::Lut, chunk: __m512i) -> __m512i {
        let swapped = self.swap_halves(chunk);
        // SAFETY: the kernel implies AVX-512F and AVX-512BW.
        unsafe {
            let mask = _mm512_set1_epi8(0x0f);
            let low = |v| _mm512_and_si512(v, mask);
            let high = |v| _mm512_and_si512(_mm512_srli_epi64::<4>(v), mask);
            let a = _mm512_shuffle_epi8(lut[0], low(chunk));
            let b = _mm512_shuffle_epi8(lut[1], high(chunk));
            let c = _mm512_shuffle_epi8(lut[2], low(swapped));
            let d = _mm512_shuffle_epi8(lut[3], high(swapped));
            _mm512_xor_si512(_mm512_ternarylogic_epi64::<0x96>(a, b, c), d)
        }
    }
}

/// GFNI kernel: the product is the chunk times the diagonal blocks of the 16x16 bit matrix of the
/// multiplication, plus the half-swapped chunk times the off-diagonal blocks.
#[derive(Clone, Copy)]
struct Gfni(&'static AffineTable);

/// Per factor, the 8x8 bit matrices `[low -> low, high -> high, high -> low, low -> high]` in
/// `vgf2p8affineqb` form.
type AffineTable = [[u64; 4]; GF_ORDER];

// SAFETY: `Gfni` is only constructed in `Avx512::gfni` once AVX-512F, AVX-512BW and GFNI are
// known to be available.
unsafe impl Kernel for Gfni {
    type Lut = [__m512i; 2];

    #[inline(always)]
    fn lut(self, log_m: GfElement) -> Self::Lut {
        let [ll, hh, hl, lh] = self.0[usize::from(log_m)].map(|m| m as i64);
        // SAFETY: the kernel implies AVX-512F.
        unsafe {
            [
                _mm512_set_epi64(hh, hh, hh, hh, ll, ll, ll, ll),
                _mm512_set_epi64(lh, lh, lh, lh, hl, hl, hl, hl),
            ]
        }
    }

    #[inline(always)]
    fn zero(self) -> Self::Lut {
        // SAFETY: the kernel implies AVX-512F.
        [unsafe { _mm512_setzero_si512() }; 2]
    }

    #[inline(always)]
    fn mul(self, lut: Self::Lut, chunk: __m512i) -> __m512i {
        let swapped = self.swap_halves(chunk);
        // SAFETY: the kernel implies AVX-512F, AVX-512BW and GFNI.
        unsafe {
            _mm512_xor_si512(
                _mm512_gf2p8affine_epi64_epi8::<0>(chunk, lut[0]),
                _mm512_gf2p8affine_epi64_epi8::<0>(swapped, lut[1]),
            )
        }
    }
}

fn affine_table() -> &'static AffineTable {
    static TABLE: OnceLock<Box<AffineTable>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let tables::ExpLog { exp, log } = tables::get_exp_log();
        let mut table: Box<AffineTable> = vec![[0; 4]; GF_ORDER]
            .into_boxed_slice()
            .try_into()
            .expect("GF_ORDER entries");
        for (log_m, matrices) in zip(0.., table.iter_mut()) {
            let products = core::array::from_fn(|j| tables::mul(1 << j, log_m, exp, log));
            *matrices = affine_matrices(&products);
        }
        table
    })
}

/// The matrices of the multiplication that maps input bit `j` to `products[j]`.
fn affine_matrices(products: &[GfElement; 16]) -> [u64; 4] {
    // Byte `j` of the transpose's input is the output byte of input bit `j`; the transpose puts
    // output bit `i` in byte `i`, and `vgf2p8affineqb` wants it in byte `7 - i`.
    let matrix = |input: usize, output: usize| {
        let columns = (0..8).fold(0u64, |columns, j| {
            columns | u64::from(products[8 * input + j] >> (8 * output) & 0xff) << (8 * j)
        });
        transpose(columns).swap_bytes()
    };
    [matrix(0, 0), matrix(1, 1), matrix(1, 0), matrix(0, 1)]
}

/// Transposes the 8x8 bit matrix whose row `r` is byte `r`.
fn transpose(mut x: u64) -> u64 {
    for (shift, mask) in [
        (7, 0x00aa_00aa_00aa_00aa),
        (14, 0x0000_cccc_0000_cccc),
        (28, 0x0000_0000_f0f0_f0f0),
    ] {
        let t = (x ^ (x >> shift)) & mask;
        x ^= t ^ (t << shift);
    }
    x
}

#[inline(always)]
fn mul<K: Kernel>(kernel: K, x: &mut [Chunk], log_m: GfElement) {
    let lut = kernel.lut(log_m);
    for chunk in x {
        kernel.store(chunk, kernel.mul(lut, kernel.load(chunk)));
    }
}

#[inline(always)]
fn fft<K: Kernel>(kernel: K, data: &mut ShardsRefMut, args: Args) {
    let (skew, pos, size, truncated_size, skew_delta) = args;
    let mut dist4 = size;
    let mut dist = size >> 2;
    while dist != 0 {
        for r in (0..truncated_size).step_by(dist4) {
            let base = r + dist + skew_delta - 1;
            let l01 = kernel.butterfly_lut(skew[base]);
            let l02 = kernel.butterfly_lut(skew[base + dist]);
            let l23 = kernel.butterfly_lut(skew[base + dist * 2]);
            for i in r..r + dist {
                let (s0, s1, s2, s3) = data.dist4_mut(pos + i, dist);
                for (((c0, c1), c2), c3) in zip(zip(zip(s0, s1), s2), s3) {
                    let [mut x0, mut x1, mut x2, mut x3] = [
                        kernel.load(c0),
                        kernel.load(c1),
                        kernel.load(c2),
                        kernel.load(c3),
                    ];
                    kernel.fft_butterfly(&mut x0, &mut x2, l02);
                    kernel.fft_butterfly(&mut x1, &mut x3, l02);
                    kernel.fft_butterfly(&mut x0, &mut x1, l01);
                    kernel.fft_butterfly(&mut x2, &mut x3, l23);
                    for (c, x) in zip([c0, c1, c2, c3], [x0, x1, x2, x3]) {
                        kernel.store(c, x);
                    }
                }
            }
        }
        dist4 = dist;
        dist >>= 2;
    }

    if dist4 == 2 {
        for r in (0..truncated_size).step_by(2) {
            let lut = kernel.butterfly_lut(skew[r + skew_delta]);
            let (x, y) = data.dist2_mut(pos + r, 1);
            butterflies(kernel, x, y, lut, K::fft_butterfly);
        }
    }
}

#[inline(always)]
fn ifft<K: Kernel>(kernel: K, data: &mut ShardsRefMut, args: Args) {
    let (skew, pos, size, truncated_size, skew_delta) = args;
    let mut dist = 1;
    let mut dist4 = 4;
    while dist4 <= size {
        for r in (0..truncated_size).step_by(dist4) {
            let base = r + dist + skew_delta - 1;
            let l01 = kernel.butterfly_lut(skew[base]);
            let l02 = kernel.butterfly_lut(skew[base + dist]);
            let l23 = kernel.butterfly_lut(skew[base + dist * 2]);
            for i in r..r + dist {
                let (s0, s1, s2, s3) = data.dist4_mut(pos + i, dist);
                for (((c0, c1), c2), c3) in zip(zip(zip(s0, s1), s2), s3) {
                    let [mut x0, mut x1, mut x2, mut x3] = [
                        kernel.load(c0),
                        kernel.load(c1),
                        kernel.load(c2),
                        kernel.load(c3),
                    ];
                    kernel.ifft_butterfly(&mut x0, &mut x1, l01);
                    kernel.ifft_butterfly(&mut x2, &mut x3, l23);
                    kernel.ifft_butterfly(&mut x0, &mut x2, l02);
                    kernel.ifft_butterfly(&mut x1, &mut x3, l02);
                    for (c, x) in zip([c0, c1, c2, c3], [x0, x1, x2, x3]) {
                        kernel.store(c, x);
                    }
                }
            }
        }
        dist = dist4;
        dist4 <<= 2;
    }

    if dist < size {
        let lut = kernel.butterfly_lut(skew[dist + skew_delta - 1]);
        for i in 0..dist {
            let (x, y) = data.dist2_mut(pos + i, dist);
            butterflies(kernel, x, y, lut, K::ifft_butterfly);
        }
    }
}

/// Applies `butterfly` to each pair of chunks of the shards `x` and `y`.
#[inline(always)]
fn butterflies<K: Kernel>(
    kernel: K,
    x: &mut [Chunk],
    y: &mut [Chunk],
    lut: K::Lut,
    butterfly: impl Fn(K, &mut __m512i, &mut __m512i, K::Lut),
) {
    for (cx, cy) in zip(x, y) {
        let (mut vx, mut vy) = (kernel.load(cx), kernel.load(cy));
        butterfly(kernel, &mut vx, &mut vy, lut);
        kernel.store(cx, vx);
        kernel.store(cy, vy);
    }
}
