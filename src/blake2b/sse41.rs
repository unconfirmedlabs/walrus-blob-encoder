// Copyright (c) Unconfirmed Labs, LLC
// SPDX-License-Identifier: Apache-2.0

//! SSE4.1 BLAKE2b, two messages at a time: vector `i` holds state or message word `i` of every
//! lane, so the rounds need no shuffles.

#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m128i, _mm_add_epi64, _mm_loadu_si128, _mm_or_si128, _mm_set1_epi64x, _mm_setr_epi8,
    _mm_shuffle_epi8, _mm_shuffle_epi32, _mm_srli_epi64, _mm_storeu_si128, _mm_unpackhi_epi64,
    _mm_unpacklo_epi64, _mm_xor_si128,
};

use core::hint::black_box;

use super::{Backend, Block, DIGEST_LEN, Digest, H0, Hasher, IV, Lanes, SIGMA, for_each_group};

const LANES: usize = 2;

pub(super) fn hash_batch(hasher: Hasher, prefix: u8, messages: &[&[u8]], out: &mut [Digest]) {
    assert_eq!(hasher.backend(), Backend::Sse41);
    // SAFETY: a `Hasher` only holds available backends, and this one requires SSE4.1.
    unsafe { hash_batch_unchecked(prefix, messages, out) }
}

/// # Safety
///
/// The CPU must support SSE4.1. `messages` must be non-empty, of equal lengths and as many as `out`.
#[target_feature(enable = "sse4.1")]
unsafe fn hash_batch_unchecked(prefix: u8, messages: &[&[u8]], out: &mut [Digest]) {
    let mut lanes = Lanes::<LANES>::new(prefix, messages[0].len());
    for_each_group(messages, out, |group| hash_group(&mut lanes, group));
}

#[target_feature(enable = "sse4.1")]
fn hash_group(lanes: &mut Lanes<LANES>, messages: &[&[u8]; LANES]) -> [Digest; LANES] {
    let mut h = H0.map(|word| splat(word));
    let geometry = lanes.geometry;
    for k in 0..geometry.blocks {
        let (counter, last) = (geometry.counter(k), geometry.is_last(k));
        let m = load(lanes.block(messages, k));
        compress(&mut h, &m, counter, last);
    }
    let halves = [
        [
            _mm_unpacklo_epi64(h[0], h[1]),
            _mm_unpacklo_epi64(h[2], h[3]),
        ],
        [
            _mm_unpackhi_epi64(h[0], h[1]),
            _mm_unpackhi_epi64(h[2], h[3]),
        ],
    ];
    halves.map(|halves| {
        let mut digest = [0; DIGEST_LEN];
        for (bytes, half) in digest.chunks_exact_mut(16).zip(halves) {
            // SAFETY: `bytes` is 16 writable bytes, the size of one vector; the store is
            // unaligned.
            unsafe { _mm_storeu_si128(bytes.as_mut_ptr().cast(), half) };
        }
        digest
    })
}

#[target_feature(enable = "sse4.1")]
fn splat(word: u64) -> __m128i {
    _mm_set1_epi64x(word as i64)
}

/// Transposes the message words of two blocks into `m[i]` = word `i` of both lanes.
#[target_feature(enable = "sse4.1")]
fn load(blocks: [&Block; LANES]) -> [__m128i; 16] {
    let mut m = [splat(0); 16];
    for (j, words) in m.chunks_exact_mut(2).enumerate() {
        let [r0, r1] = blocks.map(|block| {
            let row: &[u8; 16] = block[16 * j..16 * j + 16].try_into().expect("16 bytes");
            // SAFETY: `row` is 16 readable bytes, the size of one vector; the load is unaligned.
            unsafe { _mm_loadu_si128(row.as_ptr().cast()) }
        });
        words[0] = _mm_unpacklo_epi64(r0, r1);
        words[1] = _mm_unpackhi_epi64(r0, r1);
    }
    m
}

#[target_feature(enable = "sse4.1")]
fn compress(h: &mut [__m128i; 8], m: &[__m128i; 16], counter: u64, last: bool) {
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
        h[i] = _mm_xor_si128(h[i], _mm_xor_si128(v[i], v[i + 8]));
    }
}

#[target_feature(enable = "sse4.1")]
fn round<const R: usize>(v: &mut [__m128i; 16], m: &[__m128i; 16], r: Rotations) {
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

#[target_feature(enable = "sse4.1")]
fn g(v: &mut [__m128i; 16], [a, b, c, d]: [usize; 4], x: __m128i, y: __m128i, r: Rotations) {
    v[a] = _mm_add_epi64(_mm_add_epi64(v[a], v[b]), x);
    v[d] = _mm_shuffle_epi32::<0b10_11_00_01>(_mm_xor_si128(v[d], v[a]));
    v[c] = _mm_add_epi64(v[c], v[d]);
    v[b] = _mm_shuffle_epi8(_mm_xor_si128(v[b], v[c]), r.by24);
    v[a] = _mm_add_epi64(_mm_add_epi64(v[a], v[b]), y);
    v[d] = _mm_shuffle_epi8(_mm_xor_si128(v[d], v[a]), r.by16);
    v[c] = _mm_add_epi64(v[c], v[d]);
    let x = _mm_xor_si128(v[b], v[c]);
    v[b] = _mm_or_si128(_mm_add_epi64(x, x), _mm_srli_epi64::<63>(x));
}

/// Byte shuffles rotating every word right by 24 and by 16 bits.
#[derive(Clone, Copy)]
struct Rotations {
    by24: __m128i,
    by16: __m128i,
}

impl Rotations {
    /// The 16-bit mask goes through `black_box`: given a constant, LLVM rewrites that `pshufb` as
    /// `pshuflw` + `pshufhw`, two shuffle-port uops instead of one.
    #[target_feature(enable = "sse4.1")]
    fn new() -> Self {
        let by24 = _mm_setr_epi8(3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10);
        let by16 = _mm_setr_epi8(2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9);
        Rotations {
            by24,
            by16: black_box(by16),
        }
    }
}
