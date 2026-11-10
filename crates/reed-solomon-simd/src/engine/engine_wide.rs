use core::iter::zip;

use wide::{bytemuck::cast, u16x8, u8x16};

use crate::engine::{
    tables::{self, Mul128, Multiply128lutT, Skew},
    utils, Engine, GfElement, ShardsRefMut, GF_MODULUS, GF_ORDER,
};

// ======================================================================
// Wide - PUBLIC

/// Optimized [`Engine`] using portable 128-bit SIMD from the [`wide`] crate.
///
/// [`Wide`] follows the same algorithm as [`NoSimd`] and the same structure
/// as the SSSE3 engine. The instruction set is chosen at compile time from
/// the enabled target features: SSSE3 on x86(-64), `simd128` on WebAssembly,
/// NEON on `AArch64`. Without one of these it still produces correct output,
/// through a scalar fallback that is slower than [`NoSimd`].
///
/// [`NoSimd`]: crate::engine::NoSimd
#[derive(Clone, Copy)]
pub struct Wide {
    mul128: &'static Mul128,
    skew: &'static Skew,
}

impl Wide {
    /// Creates new [`Wide`], initializing all [tables]
    /// needed for encoding or decoding.
    ///
    /// Currently only difference between encoding/decoding is
    /// [`LogWalsh`] (128 kiB) which is only needed for decoding.
    ///
    /// [`LogWalsh`]: crate::engine::tables::LogWalsh
    pub fn new() -> Self {
        let mul128 = tables::get_mul128();
        let skew = tables::get_skew();

        Self { mul128, skew }
    }
}

impl Engine for Wide {
    fn fft(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        self.fft_private(data, pos, size, truncated_size, skew_delta);
    }

    fn ifft(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        self.ifft_private(data, pos, size, truncated_size, skew_delta);
    }

    fn mul(&self, x: &mut [[u8; 64]], log_m: GfElement) {
        let lut = &self.mul128[log_m as usize];
        for chunk in x.iter_mut() {
            let [x0_lo, x1_lo, x0_hi, x1_hi] = load(chunk);
            let (prod0_lo, prod0_hi) = Self::mul_128(x0_lo, x0_hi, lut);
            let (prod1_lo, prod1_hi) = Self::mul_128(x1_lo, x1_hi, lut);
            store(chunk, [prod0_lo, prod1_lo, prod0_hi, prod1_hi]);
        }
    }

    fn eval_poly(erasures: &mut [GfElement; GF_ORDER], truncated_size: usize) {
        utils::eval_poly(erasures, truncated_size);
    }
}

// ======================================================================
// Wide - IMPL Default

impl Default for Wide {
    fn default() -> Self {
        Self::new()
    }
}

// ======================================================================
// Wide - PRIVATE
//
//

// Each 64-byte chunk holds 32 field elements as four 16-byte vectors:
// low bytes of elements 0..16, low bytes of 16..32, then the two high halves.
#[inline(always)]
fn load(chunk: &[u8; 64]) -> [u8x16; 4] {
    cast(*chunk)
}

#[inline(always)]
fn store(chunk: &mut [u8; 64], vectors: [u8x16; 4]) {
    *chunk = cast(vectors);
}

// The high nibble of every byte, in the low nibble position.
#[inline(always)]
fn high_nibbles(value: u8x16, mask: u8x16) -> u8x16 {
    let shifted: u16x8 = cast::<u8x16, u16x8>(value) >> 4_u32;
    cast::<u16x8, u8x16>(shifted) & mask
}

impl Wide {
    // Implementation of LEO_MUL_128
    #[inline(always)]
    fn mul_128(value_lo: u8x16, value_hi: u8x16, lut: &Multiply128lutT) -> (u8x16, u8x16) {
        let table = |entry: u128| -> u8x16 { cast(entry) };
        let mask = u8x16::splat(0x0f);

        // Every index is a nibble, so every shuffle index is in range.
        let data_0 = value_lo & mask;
        let mut prod_lo = table(lut.lo[0]).shuffle(data_0);
        let mut prod_hi = table(lut.hi[0]).shuffle(data_0);

        let data_1 = high_nibbles(value_lo, mask);
        prod_lo ^= table(lut.lo[1]).shuffle(data_1);
        prod_hi ^= table(lut.hi[1]).shuffle(data_1);

        let data_0 = value_hi & mask;
        prod_lo ^= table(lut.lo[2]).shuffle(data_0);
        prod_hi ^= table(lut.hi[2]).shuffle(data_0);

        let data_1 = high_nibbles(value_hi, mask);
        prod_lo ^= table(lut.lo[3]).shuffle(data_1);
        prod_hi ^= table(lut.hi[3]).shuffle(data_1);

        (prod_lo, prod_hi)
    }

    //// {x_lo, x_hi} ^= {y_lo, y_hi} * log_m
    // Implementation of LEO_MULADD_128
    #[inline(always)]
    fn muladd_128(
        x_lo: u8x16,
        x_hi: u8x16,
        y_lo: u8x16,
        y_hi: u8x16,
        lut: &Multiply128lutT,
    ) -> (u8x16, u8x16) {
        let (prod_lo, prod_hi) = Self::mul_128(y_lo, y_hi, lut);
        (x_lo ^ prod_lo, x_hi ^ prod_hi)
    }
}

// ======================================================================
// Wide - PRIVATE - FFT (fast Fourier transform)

impl Wide {
    // Implementation of LEO_FFTB_128
    #[inline(always)]
    fn fftb_128(&self, x: &mut [u8; 64], y: &mut [u8; 64], log_m: GfElement) {
        let lut = &self.mul128[log_m as usize];
        let [mut x0_lo, mut x1_lo, mut x0_hi, mut x1_hi] = load(x);
        let [mut y0_lo, mut y1_lo, mut y0_hi, mut y1_hi] = load(y);

        (x0_lo, x0_hi) = Self::muladd_128(x0_lo, x0_hi, y0_lo, y0_hi, lut);
        (x1_lo, x1_hi) = Self::muladd_128(x1_lo, x1_hi, y1_lo, y1_hi, lut);
        store(x, [x0_lo, x1_lo, x0_hi, x1_hi]);

        y0_lo ^= x0_lo;
        y1_lo ^= x1_lo;
        y0_hi ^= x0_hi;
        y1_hi ^= x1_hi;
        store(y, [y0_lo, y1_lo, y0_hi, y1_hi]);
    }

    // Partial butterfly, caller must do `GF_MODULUS` check with `xor`.
    #[inline(always)]
    fn fft_butterfly_partial(&self, x: &mut [[u8; 64]], y: &mut [[u8; 64]], log_m: GfElement) {
        for (x_chunk, y_chunk) in zip(x.iter_mut(), y.iter_mut()) {
            self.fftb_128(x_chunk, y_chunk, log_m);
        }
    }

    #[inline(always)]
    fn fft_butterfly_two_layers(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        dist: usize,
        log_m01: GfElement,
        log_m23: GfElement,
        log_m02: GfElement,
    ) {
        let (s0, s1, s2, s3) = data.dist4_mut(pos, dist);

        // FIRST LAYER

        if log_m02 == GF_MODULUS {
            utils::xor(s2, s0);
            utils::xor(s3, s1);
        } else {
            self.fft_butterfly_partial(s0, s2, log_m02);
            self.fft_butterfly_partial(s1, s3, log_m02);
        }

        // SECOND LAYER

        if log_m01 == GF_MODULUS {
            utils::xor(s1, s0);
        } else {
            self.fft_butterfly_partial(s0, s1, log_m01);
        }

        if log_m23 == GF_MODULUS {
            utils::xor(s3, s2);
        } else {
            self.fft_butterfly_partial(s2, s3, log_m23);
        }
    }

    #[inline(always)]
    fn fft_private(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        // TWO LAYERS AT TIME

        let mut dist4 = size;
        let mut dist = size >> 2;
        while dist != 0 {
            let mut r = 0;
            while r < truncated_size {
                let base = r + dist + skew_delta - 1;

                let log_m01 = self.skew[base];
                let log_m02 = self.skew[base + dist];
                let log_m23 = self.skew[base + dist * 2];

                for i in r..r + dist {
                    self.fft_butterfly_two_layers(data, pos + i, dist, log_m01, log_m23, log_m02);
                }

                r += dist4;
            }
            dist4 = dist;
            dist >>= 2;
        }

        // FINAL ODD LAYER

        if dist4 == 2 {
            let mut r = 0;
            while r < truncated_size {
                let log_m = self.skew[r + skew_delta];

                let (x, y) = data.dist2_mut(pos + r, 1);

                if log_m == GF_MODULUS {
                    utils::xor(y, x);
                } else {
                    self.fft_butterfly_partial(x, y, log_m);
                }

                r += 2;
            }
        }
    }
}

// ======================================================================
// Wide - PRIVATE - IFFT (inverse fast Fourier transform)

impl Wide {
    // Implementation of LEO_IFFTB_128
    #[inline(always)]
    fn ifftb_128(&self, x: &mut [u8; 64], y: &mut [u8; 64], log_m: GfElement) {
        let lut = &self.mul128[log_m as usize];
        let [mut x0_lo, mut x1_lo, mut x0_hi, mut x1_hi] = load(x);
        let [mut y0_lo, mut y1_lo, mut y0_hi, mut y1_hi] = load(y);

        y0_lo ^= x0_lo;
        y1_lo ^= x1_lo;
        y0_hi ^= x0_hi;
        y1_hi ^= x1_hi;
        store(y, [y0_lo, y1_lo, y0_hi, y1_hi]);

        (x0_lo, x0_hi) = Self::muladd_128(x0_lo, x0_hi, y0_lo, y0_hi, lut);
        (x1_lo, x1_hi) = Self::muladd_128(x1_lo, x1_hi, y1_lo, y1_hi, lut);
        store(x, [x0_lo, x1_lo, x0_hi, x1_hi]);
    }

    #[inline(always)]
    fn ifft_butterfly_partial(&self, x: &mut [[u8; 64]], y: &mut [[u8; 64]], log_m: GfElement) {
        for (x_chunk, y_chunk) in zip(x.iter_mut(), y.iter_mut()) {
            self.ifftb_128(x_chunk, y_chunk, log_m);
        }
    }

    #[inline(always)]
    fn ifft_butterfly_two_layers(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        dist: usize,
        log_m01: GfElement,
        log_m23: GfElement,
        log_m02: GfElement,
    ) {
        let (s0, s1, s2, s3) = data.dist4_mut(pos, dist);

        // FIRST LAYER

        if log_m01 == GF_MODULUS {
            utils::xor(s1, s0);
        } else {
            self.ifft_butterfly_partial(s0, s1, log_m01);
        }

        if log_m23 == GF_MODULUS {
            utils::xor(s3, s2);
        } else {
            self.ifft_butterfly_partial(s2, s3, log_m23);
        }

        // SECOND LAYER

        if log_m02 == GF_MODULUS {
            utils::xor(s2, s0);
            utils::xor(s3, s1);
        } else {
            self.ifft_butterfly_partial(s0, s2, log_m02);
            self.ifft_butterfly_partial(s1, s3, log_m02);
        }
    }

    #[inline(always)]
    fn ifft_private(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        // TWO LAYERS AT TIME

        let mut dist = 1;
        let mut dist4 = 4;
        while dist4 <= size {
            let mut r = 0;
            while r < truncated_size {
                let base = r + dist + skew_delta - 1;

                let log_m01 = self.skew[base];
                let log_m02 = self.skew[base + dist];
                let log_m23 = self.skew[base + dist * 2];

                for i in r..r + dist {
                    self.ifft_butterfly_two_layers(data, pos + i, dist, log_m01, log_m23, log_m02);
                }

                r += dist4;
            }
            dist = dist4;
            dist4 <<= 2;
        }

        // FINAL ODD LAYER

        if dist < size {
            let log_m = self.skew[dist + skew_delta - 1];
            if log_m == GF_MODULUS {
                utils::xor_within(data, pos + dist, pos, dist);
            } else {
                let (mut a, mut b) = data.split_at_mut(pos + dist);
                for i in 0..dist {
                    self.ifft_butterfly_partial(
                        &mut a[pos + i], // data[pos + i]
                        &mut b[i],       // data[pos + i + dist]
                        log_m,
                    );
                }
            }
        }
    }
}

// ======================================================================
// TESTS

// Engines are tested indirectly via roundtrip tests of HighRate and LowRate.
