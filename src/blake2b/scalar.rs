// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Portable BLAKE2b, one message at a time.

use super::{Block, Digest, H0, IV, Lanes, SIGMA, digest, for_each_group};

pub(super) fn hash_batch(prefix: u8, messages: &[&[u8]], out: &mut [Digest]) {
    let mut lanes = Lanes::<1>::new(prefix, messages[0].len());
    for_each_group(messages, out, |&[message]| [hash(&mut lanes, message)]);
}

fn hash(lanes: &mut Lanes<1>, message: &[u8]) -> Digest {
    let mut h = H0;
    let geometry = lanes.geometry;
    for k in 0..geometry.blocks {
        let (counter, last) = (geometry.counter(k), geometry.is_last(k));
        let [block] = lanes.block(&[message], k);
        compress(&mut h, block, counter, last);
    }
    digest([h[0], h[1], h[2], h[3]])
}

fn compress(h: &mut [u64; 8], block: &Block, counter: u64, last: bool) {
    let m: [u64; 16] = core::array::from_fn(|i| {
        u64::from_le_bytes(block[8 * i..8 * i + 8].try_into().expect("8 bytes"))
    });
    let mut v = [0; 16];
    v[..8].copy_from_slice(h);
    v[8..].copy_from_slice(&IV);
    v[12] ^= counter;
    if last {
        v[14] = !v[14];
    }
    round::<0>(&mut v, &m);
    round::<1>(&mut v, &m);
    round::<2>(&mut v, &m);
    round::<3>(&mut v, &m);
    round::<4>(&mut v, &m);
    round::<5>(&mut v, &m);
    round::<6>(&mut v, &m);
    round::<7>(&mut v, &m);
    round::<8>(&mut v, &m);
    round::<9>(&mut v, &m);
    round::<10>(&mut v, &m);
    round::<11>(&mut v, &m);
    for (i, word) in h.iter_mut().enumerate() {
        *word ^= v[i] ^ v[i + 8];
    }
}

#[inline(always)]
fn round<const R: usize>(v: &mut [u64; 16], m: &[u64; 16]) {
    let s = &SIGMA[R % 10];
    g(v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
    g(v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
    g(v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
    g(v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
    g(v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
    g(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
    g(v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
    g(v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
}

#[inline(always)]
fn g(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}
