#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};
use core::iter::zip;

#[cfg(target_feature = "avx2")]
use wide::{u16x16, u8x32};
#[cfg(not(target_feature = "avx2"))]
use wide::{u16x8, u8x16};

use wide::bytemuck::cast;

use crate::engine::{
    tables::{self, Mul128, Multiply128lutT, Skew},
    utils, Engine, GfElement, ShardsRefMut, GF_MODULUS, GF_ORDER,
};

// ======================================================================
// Wide - PUBLIC

/// Optimized [`Engine`] using portable SIMD from the [`wide`] crate.
///
/// [`Wide`] follows the same algorithm as [`NoSimd`] and the same structure
/// as the SSSE3 engine. The instruction set is chosen at compile time from
/// the enabled target features: 256-bit AVX2 vectors when AVX2 is enabled,
/// otherwise 128-bit vectors on SSSE3 (x86(-64)), `simd128` (WebAssembly) or
/// NEON (`AArch64`). Without one of these it still produces correct output,
/// through a scalar fallback that is slower than [`NoSimd`].
///
/// [`Wide::new`] uses the global 8 MiB [`Mul128`] table.
/// [`Wide::with_skew_limit`] instead builds the lookup tables of the first
/// skew factors only, which is all an encoder needs (see
/// [`Wide::for_encoder`]), and falls back to the global table only if a
/// transform reaches past them.
///
/// [`NoSimd`]: crate::engine::NoSimd
#[derive(Clone)]
pub struct Wide {
    skew: &'static Skew,
    /// The table of `skew[i]`, for every position `i < by_position.len()`.
    by_position: Box<[Multiply128lutT]>,
    /// The global table, for factors beyond `by_position` and for [`Engine::mul`].
    full: Option<&'static Mul128>,
    #[cfg(test)]
    strict: bool,
}

impl Wide {
    /// Creates new [`Wide`], initializing all [tables]
    /// needed for encoding or decoding.
    ///
    /// Currently only difference between encoding/decoding is
    /// [`LogWalsh`] (128 kiB) which is only needed for decoding.
    ///
    /// [tables]: crate::engine::tables
    /// [`LogWalsh`]: crate::engine::tables::LogWalsh
    pub fn new() -> Self {
        Self {
            skew: tables::get_skew(),
            by_position: Box::new([]),
            full: Some(tables::get_mul128()),
            #[cfg(test)]
            strict: false,
        }
    }

    /// Creates new [`Wide`] with lookup tables for the skew factors at
    /// positions `0..limit` only, 128 bytes each, instead of the 8 MiB
    /// [`Mul128`] table.
    ///
    /// FFT and IFFT steps whose factor lies at or beyond `limit`, and
    /// [`Engine::mul`] (used in decoding only), compute or look up their
    /// tables elsewhere: `mul` builds its table on each call, transforms use
    /// the global [`Mul128`] table, initializing it on first use.
    pub fn with_skew_limit(limit: usize) -> Self {
        let skew = tables::get_skew();
        let tables::ExpLog { exp, log } = tables::get_exp_log();
        let limit = limit.min(skew.len());
        let by_position: Vec<Multiply128lutT> = skew[..limit]
            .iter()
            .map(|&log_m| {
                if log_m == GF_MODULUS {
                    // Never looked up: a factor of zero is a plain `xor`.
                    Multiply128lutT {
                        lo: [0; 4],
                        hi: [0; 4],
                    }
                } else {
                    tables::mul128_lut(log_m, exp, log)
                }
            })
            .collect();
        Self {
            skew,
            by_position: by_position.into_boxed_slice(),
            full: None,
            #[cfg(test)]
            strict: false,
        }
    }

    /// Creates new [`Wide`] whose tables cover every factor the encoders of
    /// [`rate`] use for `original_count` original and `recovery_count`
    /// recovery shards; see [`utils::encoder_skew_limit`].
    ///
    /// [`rate`]: crate::rate
    pub fn for_encoder(original_count: usize, recovery_count: usize) -> Self {
        Self::with_skew_limit(utils::encoder_skew_limit(original_count, recovery_count))
    }

    /// The table of the skew factor at `position`, or `None` if the factor
    /// is zero (`GF_MODULUS`).
    #[inline(always)]
    fn factor(&self, position: usize) -> Option<Lut> {
        let log_m = self.skew[position];
        if log_m == GF_MODULUS {
            None
        } else if let Some(lut) = self.by_position.get(position) {
            Some(Lut::new(lut))
        } else {
            #[cfg(test)]
            assert!(!self.strict, "skew position {position} is past the tables");
            let full = self.full.unwrap_or_else(tables::get_mul128);
            Some(Lut::new(&full[usize::from(log_m)]))
        }
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
        let lut = if let Some(full) = self.full {
            Lut::new(&full[usize::from(log_m)])
        } else {
            let tables::ExpLog { exp, log } = tables::get_exp_log();
            Lut::new(&tables::mul128_lut(log_m, exp, log))
        };
        for chunk in x.iter_mut() {
            store(chunk, lut.mul(load(chunk)));
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
// Wide - PRIVATE - vectors
//
// Each 64-byte chunk holds 32 field elements: the 32 low bytes, then the 32
// high bytes. With AVX2 a chunk is two 256-bit vectors `[low, high]`; with
// 128-bit vectors it is `[low 0..16, low 16..32, high 0..16, high 16..32]`.

#[cfg(target_feature = "avx2")]
type Vector = u8x32;
#[cfg(not(target_feature = "avx2"))]
type Vector = u8x16;

#[cfg(target_feature = "avx2")]
type Chunk = [Vector; 2];
#[cfg(not(target_feature = "avx2"))]
type Chunk = [Vector; 4];

#[inline(always)]
fn load(chunk: &[u8; 64]) -> Chunk {
    cast(*chunk)
}

#[inline(always)]
fn store(chunk: &mut [u8; 64], vectors: Chunk) {
    *chunk = cast(vectors);
}

#[inline(always)]
fn xor(x: Chunk, y: Chunk) -> Chunk {
    core::array::from_fn(|i| x[i] ^ y[i])
}

/// A 16-entry table of a `Multiply128lutT`, in every 128-bit lane.
#[cfg(target_feature = "avx2")]
#[inline(always)]
fn table(entry: u128) -> Vector {
    cast([entry, entry])
}

#[cfg(not(target_feature = "avx2"))]
#[inline(always)]
fn table(entry: u128) -> Vector {
    cast(entry)
}

/// Looks up every byte of `index` in `table`, per 128-bit lane. Every index
/// is a nibble, so the relaxed lookup is exact.
#[cfg(target_feature = "avx2")]
#[inline(always)]
fn lookup(table: Vector, index: Vector) -> Vector {
    table.swizzle_half_relaxed(index)
}

#[cfg(not(target_feature = "avx2"))]
#[inline(always)]
fn lookup(table: Vector, index: Vector) -> Vector {
    table.shuffle(index)
}

/// The high nibble of every byte, in the low nibble position.
#[cfg(target_feature = "avx2")]
#[inline(always)]
fn high_nibbles(value: Vector, mask: Vector) -> Vector {
    cast::<u16x16, Vector>(cast::<Vector, u16x16>(value) >> 4_u32) & mask
}

#[cfg(not(target_feature = "avx2"))]
#[inline(always)]
fn high_nibbles(value: Vector, mask: Vector) -> Vector {
    cast::<u16x8, Vector>(cast::<Vector, u16x8>(value) >> 4_u32) & mask
}

/// The tables of one factor, broadcast to vectors once per butterfly group.
#[derive(Clone, Copy)]
struct Lut {
    lo: [Vector; 4],
    hi: [Vector; 4],
}

impl Lut {
    #[inline(always)]
    fn new(lut: &Multiply128lutT) -> Self {
        Self {
            lo: lut.lo.map(table),
            hi: lut.hi.map(table),
        }
    }

    // Implementation of LEO_MUL_128 (LEO_MUL_256 with AVX2) on a vector of
    // low bytes and the vector of the matching high bytes.
    #[inline(always)]
    fn mul_vectors(&self, value_lo: Vector, value_hi: Vector) -> (Vector, Vector) {
        let mask = Vector::splat(0x0f);

        let data_0 = value_lo & mask;
        let mut prod_lo = lookup(self.lo[0], data_0);
        let mut prod_hi = lookup(self.hi[0], data_0);

        let data_1 = high_nibbles(value_lo, mask);
        prod_lo ^= lookup(self.lo[1], data_1);
        prod_hi ^= lookup(self.hi[1], data_1);

        let data_0 = value_hi & mask;
        prod_lo ^= lookup(self.lo[2], data_0);
        prod_hi ^= lookup(self.hi[2], data_0);

        let data_1 = high_nibbles(value_hi, mask);
        prod_lo ^= lookup(self.lo[3], data_1);
        prod_hi ^= lookup(self.hi[3], data_1);

        (prod_lo, prod_hi)
    }

    /// `chunk * log_m`
    #[cfg(target_feature = "avx2")]
    #[inline(always)]
    fn mul(&self, [lo, hi]: Chunk) -> Chunk {
        let (prod_lo, prod_hi) = self.mul_vectors(lo, hi);
        [prod_lo, prod_hi]
    }

    /// `chunk * log_m`
    #[cfg(not(target_feature = "avx2"))]
    #[inline(always)]
    fn mul(&self, [lo0, lo1, hi0, hi1]: Chunk) -> Chunk {
        let (prod0_lo, prod0_hi) = self.mul_vectors(lo0, hi0);
        let (prod1_lo, prod1_hi) = self.mul_vectors(lo1, hi1);
        [prod0_lo, prod1_lo, prod0_hi, prod1_hi]
    }
}

// ======================================================================
// Wide - PRIVATE - FFT (fast Fourier transform)

impl Wide {
    // Implementation of LEO_FFTB_128: `x ^= y * log_m; y ^= x`.
    #[inline(always)]
    fn fft_butterfly_partial(x: &mut [[u8; 64]], y: &mut [[u8; 64]], lut: &Lut) {
        for (x_chunk, y_chunk) in zip(x.iter_mut(), y.iter_mut()) {
            let y_old = load(y_chunk);
            let x_new = xor(load(x_chunk), lut.mul(y_old));
            store(x_chunk, x_new);
            store(y_chunk, xor(y_old, x_new));
        }
    }

    #[inline(always)]
    fn fft_butterfly(x: &mut [[u8; 64]], y: &mut [[u8; 64]], lut: Option<&Lut>) {
        match lut {
            None => utils::xor(y, x),
            Some(lut) => Self::fft_butterfly_partial(x, y, lut),
        }
    }

    #[inline(always)]
    fn fft_butterfly_two_layers(
        data: &mut ShardsRefMut,
        pos: usize,
        dist: usize,
        [lut01, lut23, lut02]: [Option<&Lut>; 3],
    ) {
        let (s0, s1, s2, s3) = data.dist4_mut(pos, dist);

        // FIRST LAYER

        Self::fft_butterfly(s0, s2, lut02);
        Self::fft_butterfly(s1, s3, lut02);

        // SECOND LAYER

        Self::fft_butterfly(s0, s1, lut01);
        Self::fft_butterfly(s2, s3, lut23);
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

                let lut01 = self.factor(base);
                let lut02 = self.factor(base + dist);
                let lut23 = self.factor(base + dist * 2);
                let luts = [lut01.as_ref(), lut23.as_ref(), lut02.as_ref()];

                for i in r..r + dist {
                    Self::fft_butterfly_two_layers(data, pos + i, dist, luts);
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
                let lut = self.factor(r + skew_delta);
                let (x, y) = data.dist2_mut(pos + r, 1);
                Self::fft_butterfly(x, y, lut.as_ref());
                r += 2;
            }
        }
    }
}

// ======================================================================
// Wide - PRIVATE - IFFT (inverse fast Fourier transform)

impl Wide {
    // Implementation of LEO_IFFTB_128: `y ^= x; x ^= y * log_m`.
    #[inline(always)]
    fn ifft_butterfly_partial(x: &mut [[u8; 64]], y: &mut [[u8; 64]], lut: &Lut) {
        for (x_chunk, y_chunk) in zip(x.iter_mut(), y.iter_mut()) {
            let x_old = load(x_chunk);
            let y_new = xor(load(y_chunk), x_old);
            store(y_chunk, y_new);
            store(x_chunk, xor(x_old, lut.mul(y_new)));
        }
    }

    #[inline(always)]
    fn ifft_butterfly(x: &mut [[u8; 64]], y: &mut [[u8; 64]], lut: Option<&Lut>) {
        match lut {
            None => utils::xor(y, x),
            Some(lut) => Self::ifft_butterfly_partial(x, y, lut),
        }
    }

    #[inline(always)]
    fn ifft_butterfly_two_layers(
        data: &mut ShardsRefMut,
        pos: usize,
        dist: usize,
        [lut01, lut23, lut02]: [Option<&Lut>; 3],
    ) {
        let (s0, s1, s2, s3) = data.dist4_mut(pos, dist);

        // FIRST LAYER

        Self::ifft_butterfly(s0, s1, lut01);
        Self::ifft_butterfly(s2, s3, lut23);

        // SECOND LAYER

        Self::ifft_butterfly(s0, s2, lut02);
        Self::ifft_butterfly(s1, s3, lut02);
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

                let lut01 = self.factor(base);
                let lut02 = self.factor(base + dist);
                let lut23 = self.factor(base + dist * 2);
                let luts = [lut01.as_ref(), lut23.as_ref(), lut02.as_ref()];

                for i in r..r + dist {
                    Self::ifft_butterfly_two_layers(data, pos + i, dist, luts);
                }

                r += dist4;
            }
            dist = dist4;
            dist4 <<= 2;
        }

        // FINAL ODD LAYER

        if dist < size {
            match self.factor(dist + skew_delta - 1) {
                None => utils::xor_within(data, pos + dist, pos, dist),
                Some(lut) => {
                    let (mut a, mut b) = data.split_at_mut(pos + dist);
                    for i in 0..dist {
                        Self::ifft_butterfly_partial(
                            &mut a[pos + i], // data[pos + i]
                            &mut b[i],       // data[pos + i + dist]
                            &lut,
                        );
                    }
                }
            }
        }
    }
}

// ======================================================================
// TESTS

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::NoSimd,
        rate::{DefaultRateEncoder, RateEncoder},
    };
    #[cfg(not(feature = "std"))]
    use alloc::{format, vec};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    /// An engine that panics if a transform reaches past its tables.
    fn strict(limit: usize) -> Wide {
        Wide {
            strict: true,
            ..Wide::with_skew_limit(limit)
        }
    }

    fn encode<E: Engine>(engine: E, original: &[Vec<u8>], recovery_count: usize) -> Vec<Vec<u8>> {
        let shard_bytes = original[0].len();
        let mut encoder =
            DefaultRateEncoder::new(original.len(), recovery_count, shard_bytes, engine, None)
                .unwrap();
        for shard in original {
            encoder.add_original_shard(shard).unwrap();
        }
        let result = encoder.encode().unwrap();
        result.recovery_iter().map(<[u8]>::to_vec).collect()
    }

    /// `(n - f, f)` and `(n - 2f, 2f)` for Walrus committees of `n` shards.
    fn walrus_shapes() -> Vec<(usize, usize)> {
        [4, 5, 7, 10, 13, 31, 100, 334, 1000, 1001, 2000]
            .into_iter()
            .flat_map(|n: usize| {
                let f = (n - 1) / 3;
                [(n - f, f), (n - 2 * f, 2 * f)]
            })
            .collect()
    }

    #[test]
    fn compact_tables_cover_the_encoders() {
        let mut rng = ChaCha8Rng::from_seed([1; 32]);
        for (original_count, recovery_count) in walrus_shapes() {
            let limit = utils::encoder_skew_limit(original_count, recovery_count);
            for shard_bytes in [2, 64, 66, 130] {
                let original: Vec<Vec<u8>> = (0..original_count)
                    .map(|_| (0..shard_bytes).map(|_| rng.random()).collect())
                    .collect();
                let expected = encode(NoSimd::new(), &original, recovery_count);
                let case = format!("({original_count}, {recovery_count}) x {shard_bytes}");
                assert_eq!(
                    encode(strict(limit), &original, recovery_count),
                    expected,
                    "{case}"
                );
                assert_eq!(encode(Wide::new(), &original, recovery_count), expected);
                // Tables shorter than needed still give the same result through the global
                // table.
                let short = Wide::with_skew_limit(limit / 3);
                assert_eq!(encode(short, &original, recovery_count), expected, "{case}");
            }
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn mainnet_encoders_reach_skew_position_1407() {
        let original: Vec<Vec<u8>> = (0..1000u16).map(|i| vec![i as u8; 2]).collect();
        for (original_count, recovery_count) in [(667, 333), (334, 666)] {
            encode(strict(1408), &original[..original_count], recovery_count);
            assert_eq!(
                utils::encoder_skew_limit(original_count, recovery_count),
                1536
            );
        }
        let result = std::panic::catch_unwind(|| encode(strict(1407), &original[..334], 666));
        assert!(result.is_err(), "(334, 666) uses skew position 1407");
    }

    #[test]
    fn mul() {
        let nosimd = NoSimd::default();
        let mut rng = ChaCha8Rng::from_seed([0; 32]);

        for shard_chunks in 0..6 {
            let mut data = vec![[0; 64]; shard_chunks];
            rng.fill(data.as_flattened_mut());
            let log_m = rng.random();
            let mut expected = data.clone();
            nosimd.mul(&mut expected, log_m);
            for wide in [Wide::new(), Wide::with_skew_limit(0)] {
                let mut got = data.clone();
                wide.mul(&mut got, log_m);
                assert_eq!(got, expected);
            }
        }
    }
}
