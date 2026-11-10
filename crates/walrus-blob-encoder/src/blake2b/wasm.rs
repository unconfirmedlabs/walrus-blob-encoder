// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! WebAssembly `simd128` BLAKE2b, two messages at a time: vector `i` holds state or message word
//! `i` of both lanes, so the rounds need no shuffles between words. Selected statically: a
//! module built with `simd128` cannot be instantiated by an engine without it.
//!
//! The twelve rounds are a loop rather than unrolled: JavaScriptCore (Bun, Safari) runs the
//! unrolled function about three times slower, while V8 runs both at the same speed.

#![allow(unsafe_code)]

use core::arch::wasm32::{
    i8x16_shuffle, i32x4_shuffle, i64x2_add, i64x2_shuffle, u64x2_extract_lane, u64x2_shr,
    u64x2_splat, v128, v128_load, v128_or, v128_xor,
};

use super::{Block, Digest, H0, IV, Lanes, SIGMA, digest, for_each_group};

const LANES: usize = 2;

pub(super) fn hash_batch(prefix: u8, messages: &[&[u8]], out: &mut [Digest]) {
    let mut lanes = Lanes::<LANES>::new(prefix, messages[0].len());
    for_each_group(messages, out, |group| hash_group(&mut lanes, group));
}

#[inline(always)]
fn hash_group(lanes: &mut Lanes<LANES>, messages: &[&[u8]; LANES]) -> [Digest; LANES] {
    let mut h = H0.map(|word| u64x2_splat(word));
    let geometry = lanes.geometry;
    for k in 0..geometry.blocks {
        let (counter, last) = (geometry.counter(k), geometry.is_last(k));
        let m = load(lanes.block(messages, k));
        compress(&mut h, &m, counter, last);
    }
    [
        digest([0, 1, 2, 3].map(|i| u64x2_extract_lane::<0>(h[i]))),
        digest([0, 1, 2, 3].map(|i| u64x2_extract_lane::<1>(h[i]))),
    ]
}

/// Transposes the message words of two blocks into `m[i]` = word `i` of both lanes.
#[inline(always)]
fn load(blocks: [&Block; LANES]) -> [v128; 16] {
    let mut m = [u64x2_splat(0); 16];
    for (j, words) in m.chunks_exact_mut(2).enumerate() {
        let [r0, r1] = blocks.map(|block| {
            let row: &[u8; 16] = block[16 * j..16 * j + 16].try_into().expect("16 bytes");
            // SAFETY: `row` is 16 readable bytes, the size of one vector; `v128_load` has no
            // alignment requirement.
            unsafe { v128_load(row.as_ptr().cast()) }
        });
        words[0] = i64x2_shuffle::<0, 2>(r0, r1);
        words[1] = i64x2_shuffle::<1, 3>(r0, r1);
    }
    m
}

#[inline(always)]
fn compress(h: &mut [v128; 8], m: &[v128; 16], counter: u64, last: bool) {
    let mut v = [u64x2_splat(0); 16];
    v[..8].copy_from_slice(h);
    for (v, iv) in v[8..].iter_mut().zip(IV) {
        *v = u64x2_splat(iv);
    }
    v[12] = u64x2_splat(IV[4] ^ counter);
    if last {
        v[14] = u64x2_splat(!IV[6]);
    }
    for r in 0..12 {
        round(&mut v, m, &SIGMA[r % 10]);
    }
    for i in 0..8 {
        h[i] = v128_xor(h[i], v128_xor(v[i], v[i + 8]));
    }
}

#[inline(always)]
fn round(v: &mut [v128; 16], m: &[v128; 16], s: &[usize; 16]) {
    g(v, [0, 4, 8, 12], m[s[0]], m[s[1]]);
    g(v, [1, 5, 9, 13], m[s[2]], m[s[3]]);
    g(v, [2, 6, 10, 14], m[s[4]], m[s[5]]);
    g(v, [3, 7, 11, 15], m[s[6]], m[s[7]]);
    g(v, [0, 5, 10, 15], m[s[8]], m[s[9]]);
    g(v, [1, 6, 11, 12], m[s[10]], m[s[11]]);
    g(v, [2, 7, 8, 13], m[s[12]], m[s[13]]);
    g(v, [3, 4, 9, 14], m[s[14]], m[s[15]]);
}

#[inline(always)]
fn g(v: &mut [v128; 16], [a, b, c, d]: [usize; 4], x: v128, y: v128) {
    v[a] = i64x2_add(i64x2_add(v[a], v[b]), x);
    let t = v128_xor(v[d], v[a]);
    v[d] = i32x4_shuffle::<1, 0, 3, 2>(t, t);
    v[c] = i64x2_add(v[c], v[d]);
    let t = v128_xor(v[b], v[c]);
    v[b] = i8x16_shuffle::<3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10>(t, t);
    v[a] = i64x2_add(i64x2_add(v[a], v[b]), y);
    let t = v128_xor(v[d], v[a]);
    v[d] = i8x16_shuffle::<2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9>(t, t);
    v[c] = i64x2_add(v[c], v[d]);
    let x = v128_xor(v[b], v[c]);
    v[b] = v128_or(i64x2_add(x, x), u64x2_shr(x, 63));
}
