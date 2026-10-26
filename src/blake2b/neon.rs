// Copyright (c) Unconfirmed Labs, LLC
// SPDX-License-Identifier: Apache-2.0

//! NEON BLAKE2b, two messages at a time: vector `i` holds state or message word `i` of both
//! lanes, so the rounds need no shuffles.

#![allow(unsafe_code)]

use core::arch::aarch64::{
    uint64x2_t, vaddq_u64, vdupq_n_u64, veorq_u64, vld1q_u8, vreinterpretq_u32_u64,
    vreinterpretq_u64_u8, vreinterpretq_u64_u32, vrev64q_u32, vshlq_n_u64, vsriq_n_u64, vst1q_u64,
    vzip1q_u64, vzip2q_u64,
};

use super::{Backend, Block, Digest, H0, Hasher, IV, Lanes, SIGMA, digest, for_each_group};

const LANES: usize = 2;

pub(super) fn hash_batch(hasher: Hasher, prefix: u8, messages: &[&[u8]], out: &mut [Digest]) {
    assert_eq!(hasher.backend(), Backend::Neon);
    // SAFETY: a `Hasher` only holds available backends, and this one requires NEON.
    unsafe { hash_batch_unchecked(prefix, messages, out) }
}

/// # Safety
///
/// The CPU must support NEON. `messages` must be non-empty, of equal lengths and as many as `out`.
#[target_feature(enable = "neon")]
unsafe fn hash_batch_unchecked(prefix: u8, messages: &[&[u8]], out: &mut [Digest]) {
    let mut lanes = Lanes::<LANES>::new(prefix, messages[0].len());
    for_each_group(messages, out, |group| hash_group(&mut lanes, group));
}

#[target_feature(enable = "neon")]
fn hash_group(lanes: &mut Lanes<LANES>, messages: &[&[u8]; LANES]) -> [Digest; LANES] {
    let mut h = H0.map(|word| vdupq_n_u64(word));
    let geometry = lanes.geometry;
    for k in 0..geometry.blocks {
        let (counter, last) = (geometry.counter(k), geometry.is_last(k));
        let m = load(lanes.block(messages, k));
        compress(&mut h, &m, counter, last);
    }
    let words = [h[0], h[1], h[2], h[3]].map(|v| {
        let mut lanes = [0u64; LANES];
        // SAFETY: `lanes` is two writable words, the size of one vector.
        unsafe { vst1q_u64(lanes.as_mut_ptr(), v) };
        lanes
    });
    core::array::from_fn(|l| digest(words.map(|w| w[l])))
}

/// Transposes the message words of two blocks into `m[i]` = word `i` of both lanes.
#[target_feature(enable = "neon")]
fn load([block0, block1]: [&Block; LANES]) -> [uint64x2_t; 16] {
    let mut m = [vdupq_n_u64(0); 16];
    for (j, words) in m.chunks_exact_mut(2).enumerate() {
        let [r0, r1] = [block0, block1].map(|block| {
            let row: &[u8; 16] = block[16 * j..16 * j + 16].try_into().expect("16 bytes");
            // SAFETY: `row` is 16 readable bytes, the size of one vector, and `vld1q_u8` needs
            // no alignment beyond that of `u8`.
            vreinterpretq_u64_u8(unsafe { vld1q_u8(row.as_ptr()) })
        });
        words[0] = vzip1q_u64(r0, r1);
        words[1] = vzip2q_u64(r0, r1);
    }
    m
}

#[target_feature(enable = "neon")]
fn compress(h: &mut [uint64x2_t; 8], m: &[uint64x2_t; 16], counter: u64, last: bool) {
    let mut v = [vdupq_n_u64(0); 16];
    v[..8].copy_from_slice(h);
    for (v, iv) in v[8..].iter_mut().zip(IV) {
        *v = vdupq_n_u64(iv);
    }
    v[12] = vdupq_n_u64(IV[4] ^ counter);
    if last {
        v[14] = vdupq_n_u64(!IV[6]);
    }
    round::<0>(&mut v, m);
    round::<1>(&mut v, m);
    round::<2>(&mut v, m);
    round::<3>(&mut v, m);
    round::<4>(&mut v, m);
    round::<5>(&mut v, m);
    round::<6>(&mut v, m);
    round::<7>(&mut v, m);
    round::<8>(&mut v, m);
    round::<9>(&mut v, m);
    round::<10>(&mut v, m);
    round::<11>(&mut v, m);
    for i in 0..8 {
        h[i] = veorq_u64(h[i], veorq_u64(v[i], v[i + 8]));
    }
}

#[target_feature(enable = "neon")]
fn round<const R: usize>(v: &mut [uint64x2_t; 16], m: &[uint64x2_t; 16]) {
    let s = &SIGMA[R % 10];
    g(v, [0, 4, 8, 12], m[s[0]], m[s[1]]);
    g(v, [1, 5, 9, 13], m[s[2]], m[s[3]]);
    g(v, [2, 6, 10, 14], m[s[4]], m[s[5]]);
    g(v, [3, 7, 11, 15], m[s[6]], m[s[7]]);
    g(v, [0, 5, 10, 15], m[s[8]], m[s[9]]);
    g(v, [1, 6, 11, 12], m[s[10]], m[s[11]]);
    g(v, [2, 7, 8, 13], m[s[12]], m[s[13]]);
    g(v, [3, 4, 9, 14], m[s[14]], m[s[15]]);
}

#[target_feature(enable = "neon")]
fn g(v: &mut [uint64x2_t; 16], [a, b, c, d]: [usize; 4], x: uint64x2_t, y: uint64x2_t) {
    v[a] = vaddq_u64(vaddq_u64(v[a], v[b]), x);
    v[d] = rotr32(veorq_u64(v[d], v[a]));
    v[c] = vaddq_u64(v[c], v[d]);
    v[b] = rotr::<24, 40>(veorq_u64(v[b], v[c]));
    v[a] = vaddq_u64(vaddq_u64(v[a], v[b]), y);
    v[d] = rotr::<16, 48>(veorq_u64(v[d], v[a]));
    v[c] = vaddq_u64(v[c], v[d]);
    v[b] = rotr::<63, 1>(veorq_u64(v[b], v[c]));
}

#[target_feature(enable = "neon")]
fn rotr32(x: uint64x2_t) -> uint64x2_t {
    vreinterpretq_u64_u32(vrev64q_u32(vreinterpretq_u32_u64(x)))
}

/// Rotates right by `N`; `L` must be `64 - N`.
#[target_feature(enable = "neon")]
fn rotr<const N: i32, const L: i32>(x: uint64x2_t) -> uint64x2_t {
    vsriq_n_u64::<N>(vshlq_n_u64::<L>(x), x)
}
