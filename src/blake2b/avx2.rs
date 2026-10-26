// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! AVX2 BLAKE2b, four messages at a time: vector `i` holds state or message word `i` of every
//! lane, so the rounds need no shuffles.

#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m256i, _mm256_add_epi64, _mm256_loadu_si256, _mm256_or_si256, _mm256_permute2x128_si256,
    _mm256_set1_epi64x, _mm256_setr_epi8, _mm256_shuffle_epi8, _mm256_shuffle_epi32,
    _mm256_srli_epi64, _mm256_storeu_si256, _mm256_unpackhi_epi64, _mm256_unpacklo_epi64,
    _mm256_xor_si256,
};

use core::hint::black_box;

use super::{Backend, Block, DIGEST_LEN, Digest, H0, Hasher, IV, Lanes, SIGMA, for_each_group};

pub(super) const LANES: usize = 4;

pub(super) fn hash_batch(hasher: Hasher, prefix: u8, messages: &[&[u8]], out: &mut [Digest]) {
    assert_eq!(hasher.backend(), Backend::Avx2);
    // SAFETY: a `Hasher` only holds available backends, so AVX2 is enabled at compile time or
    // was detected on the running CPU.
    unsafe { kernel(prefix, messages, out) }
}

/// `messages` must be non-empty, of equal lengths and as many as `out`.
#[target_feature(enable = "avx2")]
pub(super) fn kernel(prefix: u8, messages: &[&[u8]], out: &mut [Digest]) {
    let mut lanes = Lanes::<LANES>::new(prefix, messages[0].len());
    for_each_group(messages, out, |group| hash_group(&mut lanes, group));
}

#[target_feature(enable = "avx2")]
fn hash_group(lanes: &mut Lanes<LANES>, messages: &[&[u8]; LANES]) -> [Digest; LANES] {
    let mut h = H0.map(|word| splat(word));
    let geometry = lanes.geometry;
    for k in 0..geometry.blocks {
        let (counter, last) = (geometry.counter(k), geometry.is_last(k));
        let m = load(lanes.block(messages, k));
        compress(&mut h, &m, counter, last);
    }
    transpose([h[0], h[1], h[2], h[3]]).map(|row| {
        let mut digest = [0; DIGEST_LEN];
        // SAFETY: `digest` is 32 writable bytes, the size of one vector; the store is unaligned.
        unsafe { _mm256_storeu_si256(digest.as_mut_ptr().cast(), row) };
        digest
    })
}

#[target_feature(enable = "avx2")]
fn splat(word: u64) -> __m256i {
    _mm256_set1_epi64x(word as i64)
}

/// Transposes the message words of four blocks into `m[i]` = word `i` of every lane.
#[target_feature(enable = "avx2")]
fn load(blocks: [&Block; LANES]) -> [__m256i; 16] {
    let mut m = [splat(0); 16];
    for (j, words) in m.chunks_exact_mut(4).enumerate() {
        let rows = blocks.map(|block| {
            let row: &[u8; 32] = block[32 * j..32 * j + 32].try_into().expect("32 bytes");
            // SAFETY: `row` is 32 readable bytes, the size of one vector; the load is unaligned.
            unsafe { _mm256_loadu_si256(row.as_ptr().cast()) }
        });
        words.copy_from_slice(&transpose(rows));
    }
    m
}

/// Transposes a 4 x 4 matrix of words, one row per vector.
#[target_feature(enable = "avx2")]
fn transpose(rows: [__m256i; 4]) -> [__m256i; 4] {
    let t0 = _mm256_unpacklo_epi64(rows[0], rows[1]);
    let t1 = _mm256_unpackhi_epi64(rows[0], rows[1]);
    let t2 = _mm256_unpacklo_epi64(rows[2], rows[3]);
    let t3 = _mm256_unpackhi_epi64(rows[2], rows[3]);
    [
        _mm256_permute2x128_si256::<0x20>(t0, t2),
        _mm256_permute2x128_si256::<0x20>(t1, t3),
        _mm256_permute2x128_si256::<0x31>(t0, t2),
        _mm256_permute2x128_si256::<0x31>(t1, t3),
    ]
}

#[target_feature(enable = "avx2")]
fn compress(h: &mut [__m256i; 8], m: &[__m256i; 16], counter: u64, last: bool) {
    let r = Rotations::new();
    let mut v = [splat(0); 16];
    v[..8].copy_from_slice(h);
    for (v, iv) in v[8..].iter_mut().zip(IV) {
        *v = splat(iv);
    }
    v[12] = splat(IV[4] ^ counter);
    if last {
        v[14] = splat(!IV[6]);
    }
    round::<0>(&mut v, m, r);
    round::<1>(&mut v, m, r);
    round::<2>(&mut v, m, r);
    round::<3>(&mut v, m, r);
    round::<4>(&mut v, m, r);
    round::<5>(&mut v, m, r);
    round::<6>(&mut v, m, r);
    round::<7>(&mut v, m, r);
    round::<8>(&mut v, m, r);
    round::<9>(&mut v, m, r);
    round::<10>(&mut v, m, r);
    round::<11>(&mut v, m, r);
    for i in 0..8 {
        h[i] = _mm256_xor_si256(h[i], _mm256_xor_si256(v[i], v[i + 8]));
    }
}

#[target_feature(enable = "avx2")]
fn round<const R: usize>(v: &mut [__m256i; 16], m: &[__m256i; 16], r: Rotations) {
    let s = &SIGMA[R % 10];
    g(v, [0, 4, 8, 12], m[s[0]], m[s[1]], r);
    g(v, [1, 5, 9, 13], m[s[2]], m[s[3]], r);
    g(v, [2, 6, 10, 14], m[s[4]], m[s[5]], r);
    g(v, [3, 7, 11, 15], m[s[6]], m[s[7]], r);
    g(v, [0, 5, 10, 15], m[s[8]], m[s[9]], r);
    g(v, [1, 6, 11, 12], m[s[10]], m[s[11]], r);
    g(v, [2, 7, 8, 13], m[s[12]], m[s[13]], r);
    g(v, [3, 4, 9, 14], m[s[14]], m[s[15]], r);
}

#[target_feature(enable = "avx2")]
fn g(v: &mut [__m256i; 16], [a, b, c, d]: [usize; 4], x: __m256i, y: __m256i, r: Rotations) {
    v[a] = _mm256_add_epi64(_mm256_add_epi64(v[a], v[b]), x);
    v[d] = _mm256_shuffle_epi32::<0b10_11_00_01>(_mm256_xor_si256(v[d], v[a]));
    v[c] = _mm256_add_epi64(v[c], v[d]);
    v[b] = _mm256_shuffle_epi8(_mm256_xor_si256(v[b], v[c]), r.by24);
    v[a] = _mm256_add_epi64(_mm256_add_epi64(v[a], v[b]), y);
    v[d] = _mm256_shuffle_epi8(_mm256_xor_si256(v[d], v[a]), r.by16);
    v[c] = _mm256_add_epi64(v[c], v[d]);
    let x = _mm256_xor_si256(v[b], v[c]);
    v[b] = _mm256_or_si256(_mm256_add_epi64(x, x), _mm256_srli_epi64::<63>(x));
}

/// Byte shuffles rotating every word right by 24 and by 16 bits.
#[derive(Clone, Copy)]
struct Rotations {
    by24: __m256i,
    by16: __m256i,
}

impl Rotations {
    /// The 16-bit mask goes through `black_box`: given a constant, LLVM rewrites that `vpshufb` as
    /// `vpshuflw` + `vpshufhw`, two shuffle-port uops instead of one.
    #[target_feature(enable = "avx2")]
    fn new() -> Self {
        let by24 = _mm256_setr_epi8(
            3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10, 3, 4, 5, 6, 7, 0, 1, 2, 11, 12,
            13, 14, 15, 8, 9, 10,
        );
        let by16 = _mm256_setr_epi8(
            2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9, 2, 3, 4, 5, 6, 7, 0, 1, 10, 11,
            12, 13, 14, 15, 8, 9,
        );
        Rotations {
            by24,
            by16: black_box(by16),
        }
    }
}
