#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};
use core::iter::zip;

use crate::engine::{
    tables::{self, Mul16, Skew},
    utils, Engine, GfElement, ShardsRefMut, GF_MODULUS,
};

/// The [`Mul16`] entry of one factor.
type Lut = [[GfElement; 16]; 4];

// ======================================================================
// NoSimd - PUBLIC

/// Optimized [`Engine`] without SIMD.
///
/// [`NoSimd`] is a basic optimized engine which works on all CPUs.
///
/// [`NoSimd::new`] uses the global 8 MiB [`Mul16`] table;
/// [`NoSimd::with_skew_limit`] and [`NoSimd::for_encoder`] build only the
/// tables of the first skew factors, as [`Wide::with_skew_limit`] does.
///
/// [`Wide::with_skew_limit`]: crate::engine::Wide::with_skew_limit
#[derive(Clone)]
pub struct NoSimd {
    skew: &'static Skew,
    /// The table of `skew[i]`, for every position `i < by_position.len()`.
    by_position: Box<[Lut]>,
    /// The global table, for factors beyond `by_position` and for [`Engine::mul`].
    full: Option<&'static Mul16>,
}

impl NoSimd {
    /// Creates new [`NoSimd`], initializing all [tables]
    /// needed for encoding or decoding.
    ///
    /// Currently only difference between encoding/decoding is
    /// [`LogWalsh`] (128 kiB) which is only needed for decoding.
    ///
    /// [`LogWalsh`]: crate::engine::tables::LogWalsh
    pub fn new() -> Self {
        Self {
            skew: tables::get_skew(),
            by_position: Box::new([]),
            full: Some(tables::get_mul16()),
        }
    }

    /// Creates new [`NoSimd`] with lookup tables for the skew factors at
    /// positions `0..limit` only, 128 bytes each, instead of the 8 MiB
    /// [`Mul16`] table. Factors beyond them use the global table,
    /// initializing it on first use; [`Engine::mul`] builds its table per call.
    pub fn with_skew_limit(limit: usize) -> Self {
        let skew = tables::get_skew();
        let tables::ExpLog { exp, log } = tables::get_exp_log();
        let limit = limit.min(skew.len());
        let by_position: Vec<Lut> = skew[..limit]
            .iter()
            .map(|&log_m| {
                if log_m == GF_MODULUS {
                    [[0; 16]; 4]
                } else {
                    tables::mul16_lut(log_m, exp, log)
                }
            })
            .collect();
        Self {
            skew,
            by_position: by_position.into_boxed_slice(),
            full: None,
        }
    }

    /// Creates new [`NoSimd`] whose tables cover every factor the encoders of
    /// [`rate`] use for these shard counts; see [`utils::encoder_skew_limit`].
    ///
    /// [`rate`]: crate::rate
    pub fn for_encoder(original_count: usize, recovery_count: usize) -> Self {
        Self::with_skew_limit(utils::encoder_skew_limit(original_count, recovery_count))
    }

    /// The table of the skew factor at `position`, or `None` if the factor
    /// is zero (`GF_MODULUS`).
    #[inline(always)]
    fn factor(&self, position: usize) -> Option<&Lut> {
        let log_m = self.skew[position];
        if log_m == GF_MODULUS {
            None
        } else if let Some(lut) = self.by_position.get(position) {
            Some(lut)
        } else {
            let full = self.full.unwrap_or_else(tables::get_mul16);
            Some(&full[usize::from(log_m)])
        }
    }
}

impl Engine for NoSimd {
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

    fn fft_from(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
        first: usize,
    ) {
        self.fft_ranged(data, pos, size, first..truncated_size, skew_delta);
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
        let owned;
        let lut = if let Some(full) = self.full {
            &full[usize::from(log_m)]
        } else {
            let tables::ExpLog { exp, log } = tables::get_exp_log();
            owned = tables::mul16_lut(log_m, exp, log);
            &owned
        };

        for x_chunk in x.iter_mut() {
            let (x_lo, x_hi) = x_chunk.split_at_mut(32);

            for i in 0..32 {
                let lo = x_lo[i];
                let hi = x_hi[i];
                let prod = lut[0][usize::from(lo & 15)]
                    ^ lut[1][usize::from(lo >> 4)]
                    ^ lut[2][usize::from(hi & 15)]
                    ^ lut[3][usize::from(hi >> 4)];
                x_lo[i] = prod as u8;
                x_hi[i] = (prod >> 8) as u8;
            }
        }
    }
}

// ======================================================================
// NoSimd - IMPL Default

impl Default for NoSimd {
    fn default() -> Self {
        Self::new()
    }
}

// ======================================================================
// NoSimd - PRIVATE

impl NoSimd {
    /// `x[] ^= y[] * log_m`
    fn mul_add(x: &mut [[u8; 64]], y: &[[u8; 64]], lut: &Lut) {
        for (x_chunk, y_chunk) in zip(x.iter_mut(), y.iter()) {
            let (x_lo, x_hi) = x_chunk.split_at_mut(32);
            let (y_lo, y_hi) = y_chunk.split_at(32);

            for i in 0..32 {
                let lo = y_lo[i];
                let hi = y_hi[i];
                let prod = lut[0][usize::from(lo & 15)]
                    ^ lut[1][usize::from(lo >> 4)]
                    ^ lut[2][usize::from(hi & 15)]
                    ^ lut[3][usize::from(hi >> 4)];
                x_lo[i] ^= prod as u8;
                x_hi[i] ^= (prod >> 8) as u8;
            }
        }
    }
}

// ======================================================================
// NoSimd - PRIVATE - FFT (fast Fourier transform)

impl NoSimd {
    // Partial butterfly, caller must do `GF_MODULUS` check with `xor`.
    #[inline(always)]
    fn fft_butterfly_partial(x: &mut [[u8; 64]], y: &mut [[u8; 64]], lut: &Lut) {
        Self::mul_add(x, y, lut);
        utils::xor(y, x);
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
        self.fft_ranged(data, pos, size, 0..truncated_size, skew_delta);
    }

    /// The FFT, valid on `data[pos + range.start .. pos + range.end]`: at
    /// every layer, only the groups that overlap `range` are computed.
    #[inline(always)]
    fn fft_ranged(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        range: core::ops::Range<usize>,
        skew_delta: usize,
    ) {
        let truncated_size = range.end;

        // TWO LAYERS AT TIME

        let mut dist4 = size;
        let mut dist = size >> 2;
        while dist != 0 {
            let mut r = range.start / dist4 * dist4;
            while r < truncated_size {
                let base = r + dist + skew_delta - 1;

                let luts = [
                    self.factor(base),
                    self.factor(base + dist * 2),
                    self.factor(base + dist),
                ];

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
            let mut r = range.start / 2 * 2;
            while r < truncated_size {
                let lut = self.factor(r + skew_delta);
                let (x, y) = data.dist2_mut(pos + r, 1);
                Self::fft_butterfly(x, y, lut);

                r += 2;
            }
        }
    }
}

// ======================================================================
// NoSimd - PRIVATE - IFFT (inverse fast Fourier transform)

impl NoSimd {
    // Partial butterfly, caller must do `GF_MODULUS` check with `xor`.
    #[inline(always)]
    fn ifft_butterfly_partial(x: &mut [[u8; 64]], y: &mut [[u8; 64]], lut: &Lut) {
        utils::xor(y, x);
        Self::mul_add(x, y, lut);
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

                let luts = [
                    self.factor(base),
                    self.factor(base + dist * 2),
                    self.factor(base + dist),
                ];

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
                            lut,
                        );
                    }
                }
            }
        }
    }
}

// ======================================================================
// TESTS

// Engines are tested indirectly via roundtrip tests of HighRate and LowRate.

#[cfg(test)]
mod tests {
    use crate::engine::{Engine, NoSimd, Wide};

    #[cfg(not(feature = "std"))]
    use alloc::vec;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn mul() {
        let wide = Wide::default();
        let nosimd = NoSimd::default();

        let mut rng = ChaCha8Rng::from_seed([0; 32]);

        for shard_chunks in 0..6 {
            let mut data_nosimd = vec![[0; 64]; shard_chunks];
            rng.fill(data_nosimd.as_flattened_mut());
            let mut data_wide = data_nosimd.clone();

            let log_m = rng.random();

            nosimd.mul(&mut data_nosimd, log_m);
            wide.mul(&mut data_wide, log_m);

            assert_eq!(data_nosimd, data_wide);

            let mut data_compact = data_wide.clone();
            nosimd.mul(&mut data_wide, log_m);
            NoSimd::with_skew_limit(0).mul(&mut data_compact, log_m);
            assert_eq!(data_compact, data_wide);
        }
    }

    #[test]
    fn compact_tables_match_full_tables() {
        use crate::rate::{DefaultRateEncoder, RateEncoder};
        #[cfg(not(feature = "std"))]
        use alloc::vec::Vec;

        fn encode(engine: NoSimd, original: &[Vec<u8>], recovery_count: usize) -> Vec<Vec<u8>> {
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

        let mut rng = ChaCha8Rng::from_seed([2; 32]);
        for n in [4, 7, 13, 100, 1000] {
            let f = (n - 1) / 3;
            for (original_count, recovery_count) in [(n - f, f), (n - 2 * f, 2 * f)] {
                let original: Vec<Vec<u8>> = (0..original_count)
                    .map(|_| (0..66).map(|_| rng.random()).collect())
                    .collect();
                let compact = NoSimd::for_encoder(original_count, recovery_count);
                let short = NoSimd::with_skew_limit(7);
                let expected = encode(NoSimd::new(), &original, recovery_count);
                assert_eq!(encode(compact, &original, recovery_count), expected);
                assert_eq!(encode(short, &original, recovery_count), expected);
            }
        }
    }
}
