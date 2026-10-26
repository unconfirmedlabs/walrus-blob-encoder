// Copyright (c) Unconfirmed Labs, LLC
// SPDX-License-Identifier: Apache-2.0

//! AVX-512 BLAKE2b, eight messages at a time: vector `i` holds state or message word `i` of every
//! lane, so the rounds need no shuffles.

#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m512i, _mm512_add_epi64, _mm512_alignr_epi64, _mm512_maskz_loadu_epi8, _mm512_or_si512,
    _mm512_ror_epi64, _mm512_set1_epi64, _mm512_shuffle_i64x2, _mm512_slli_epi64,
    _mm512_srli_epi64, _mm512_storeu_si512, _mm512_unpackhi_epi64, _mm512_unpacklo_epi64,
    _mm512_xor_si512,
};

use super::{
    BLOCK_LEN, Backend, DIGEST_LEN, Digest, Geometry, H0, Hasher, IV, SIGMA, avx2, for_each_group,
};

const LANES: usize = 8;

pub(super) fn hash_batch(hasher: Hasher, prefix: u8, messages: &[&[u8]], out: &mut [Digest]) {
    assert_eq!(hasher.backend(), Backend::Avx512);
    // SAFETY: a `Hasher` only holds available backends, and AVX-512 availability includes
    // AVX-512F, AVX-512BW and AVX2.
    unsafe { hash_batch_unchecked(prefix, messages, out) }
}

/// Hashes groups of eight; a remainder of at most four goes to the AVX2 kernel, which is cheaper
/// than a padded group of eight.
///
/// # Safety
///
/// The CPU must support AVX-512F, AVX-512BW and AVX2. `messages` must be non-empty,
/// of equal lengths and as many as `out`.
#[target_feature(enable = "avx2,avx512f,avx512bw")]
unsafe fn hash_batch_unchecked(prefix: u8, messages: &[&[u8]], out: &mut [Digest]) {
    let remainder = messages.len() % LANES;
    let split = if remainder <= avx2::LANES {
        messages.len() - remainder
    } else {
        messages.len()
    };
    let (messages, rest) = messages.split_at(split);
    let (out, out_rest) = out.split_at_mut(split);
    if let Some(first) = messages.first() {
        let geometry = Geometry::new(first.len());
        let prefix = splat(u64::from(prefix) << 56);
        for_each_group(messages, out, |group| hash_group(geometry, prefix, group));
    }
    if !rest.is_empty() {
        // SAFETY: the caller guarantees AVX2; `rest` is non-empty, of equal lengths and as many
        // as `out_rest`.
        unsafe { avx2::hash_batch_unchecked(prefix, rest, out_rest) };
    }
}

/// `prefix` holds the prefix byte in the top byte of every word.
#[target_feature(enable = "avx512f,avx512bw")]
fn hash_group(geometry: Geometry, prefix: __m512i, messages: &[&[u8]; LANES]) -> [Digest; LANES] {
    let mut h = H0.map(|word| splat(word));
    for k in 0..geometry.blocks {
        let m = load(geometry, k, prefix, messages);
        compress(&mut h, &m, geometry.counter(k), geometry.is_last(k));
    }
    // a[0] holds words 0 and 1 of lanes 0, 2, 4, 6, a[1] of lanes 1, 3, 5, 7; a[2] and a[3]
    // words 2 and 3. Each output vector then holds the digests of two adjacent lanes.
    let a = [
        _mm512_unpacklo_epi64(h[0], h[1]),
        _mm512_unpackhi_epi64(h[0], h[1]),
        _mm512_unpacklo_epi64(h[2], h[3]),
        _mm512_unpackhi_epi64(h[2], h[3]),
    ];
    let low = |x, y| _mm512_shuffle_i64x2::<0b01_00_01_00>(x, y);
    let high = |x, y| _mm512_shuffle_i64x2::<0b11_10_11_10>(x, y);
    let (p0, q0) = (low(a[0], a[2]), low(a[1], a[3]));
    let (p1, q1) = (high(a[0], a[2]), high(a[1], a[3]));
    let rows = [even(p0, q0), odd(p0, q0), even(p1, q1), odd(p1, q1)];
    let mut digests = [[0; DIGEST_LEN]; LANES];
    for (pair, row) in digests.as_flattened_mut().chunks_exact_mut(64).zip(rows) {
        // SAFETY: `pair` is 64 writable bytes, the size of one vector; the store is unaligned.
        unsafe { _mm512_storeu_si512(pair.as_mut_ptr().cast(), row) };
    }
    digests
}

/// Words 0 and 2 of every 128-bit pair of `x`, then of `y`.
#[target_feature(enable = "avx512f")]
fn even(x: __m512i, y: __m512i) -> __m512i {
    _mm512_shuffle_i64x2::<0b10_00_10_00>(x, y)
}

/// Words 1 and 3 of every 128-bit pair of `x`, then of `y`.
#[target_feature(enable = "avx512f")]
fn odd(x: __m512i, y: __m512i) -> __m512i {
    _mm512_shuffle_i64x2::<0b11_01_11_01>(x, y)
}

#[target_feature(enable = "avx512f")]
fn splat(word: u64) -> __m512i {
    _mm512_set1_epi64(word as i64)
}

/// Where every lane's 64-byte half `half` of block `k` comes from.
#[derive(Clone, Copy)]
enum Half {
    /// `count` message bytes from offset `from`, zero padded.
    Bytes { from: usize, count: usize },
    /// The prefix, then the first `count` message bytes, zero padded.
    Prefixed { count: usize },
}

impl Half {
    fn new(geometry: Geometry, k: usize, half: usize) -> Self {
        let start = BLOCK_LEN * k + 64 * half;
        match start.checked_sub(1) {
            None => Half::Prefixed {
                count: geometry.len.min(63),
            },
            Some(from) => Half::Bytes {
                from: from.min(geometry.len),
                count: (geometry.len + 1).saturating_sub(start).min(64),
            },
        }
    }
}

/// Loads block `k` of every lane straight from the messages with byte-masked loads, so partial
/// blocks need no staging copy, and transposes it into `m[i]` = word `i` of every lane.
#[target_feature(enable = "avx512f,avx512bw")]
fn load(geometry: Geometry, k: usize, prefix: __m512i, messages: &[&[u8]; LANES]) -> [__m512i; 16] {
    let mut m = [splat(0); 16];
    for (half, words) in m.chunks_exact_mut(8).enumerate() {
        let rows = match Half::new(geometry, k, half) {
            Half::Bytes { from, count } => {
                messages.map(|message| load_bytes(&message[from..from + count]))
            }
            Half::Prefixed { count } => messages.map(|message| {
                let bytes = load_bytes(&message[..count]);
                let previous = _mm512_alignr_epi64::<7>(bytes, prefix);
                _mm512_or_si512(
                    _mm512_slli_epi64::<8>(bytes),
                    _mm512_srli_epi64::<56>(previous),
                )
            }),
        };
        // a[2i] holds words 0, 2, 4, 6 of lanes 2i and 2i + 1, a[2i + 1] words 1, 3, 5, 7.
        let a = [
            _mm512_unpacklo_epi64(rows[0], rows[1]),
            _mm512_unpackhi_epi64(rows[0], rows[1]),
            _mm512_unpacklo_epi64(rows[2], rows[3]),
            _mm512_unpackhi_epi64(rows[2], rows[3]),
            _mm512_unpacklo_epi64(rows[4], rows[5]),
            _mm512_unpackhi_epi64(rows[4], rows[5]),
            _mm512_unpacklo_epi64(rows[6], rows[7]),
            _mm512_unpackhi_epi64(rows[6], rows[7]),
        ];
        // b[0] holds words 0 and 4 of lanes 0..4, b[1] words 2 and 6, b[2] words 1 and 5,
        // b[3] words 3 and 7; b[4..8] the same for lanes 4..8.
        let b = [
            even(a[0], a[2]),
            odd(a[0], a[2]),
            even(a[1], a[3]),
            odd(a[1], a[3]),
            even(a[4], a[6]),
            odd(a[4], a[6]),
            even(a[5], a[7]),
            odd(a[5], a[7]),
        ];
        words[0] = even(b[0], b[4]);
        words[4] = odd(b[0], b[4]);
        words[2] = even(b[1], b[5]);
        words[6] = odd(b[1], b[5]);
        words[1] = even(b[2], b[6]);
        words[5] = odd(b[2], b[6]);
        words[3] = even(b[3], b[7]);
        words[7] = odd(b[3], b[7]);
    }
    m
}

/// Loads at most 64 bytes, zero padded.
#[target_feature(enable = "avx512f,avx512bw")]
fn load_bytes(bytes: &[u8]) -> __m512i {
    let mask = u64::MAX.checked_shr(64 - bytes.len() as u32).unwrap_or(0);
    // SAFETY: the mask selects the first `bytes.len()` (at most 64) bytes, which are readable;
    // masked-off bytes are not accessed. The load is unaligned.
    unsafe { _mm512_maskz_loadu_epi8(mask, bytes.as_ptr().cast()) }
}

#[target_feature(enable = "avx512f")]
fn compress(h: &mut [__m512i; 8], m: &[__m512i; 16], counter: u64, last: bool) {
    let mut v = [splat(0); 16];
    v[..8].copy_from_slice(h);
    for (v, iv) in v[8..].iter_mut().zip(IV) {
        *v = splat(iv);
    }
    v[12] = splat(IV[4] ^ counter);
    if last {
        v[14] = splat(!IV[6]);
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
        h[i] = _mm512_xor_si512(h[i], _mm512_xor_si512(v[i], v[i + 8]));
    }
}

#[target_feature(enable = "avx512f")]
fn round<const R: usize>(v: &mut [__m512i; 16], m: &[__m512i; 16]) {
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

#[target_feature(enable = "avx512f")]
fn g(v: &mut [__m512i; 16], [a, b, c, d]: [usize; 4], x: __m512i, y: __m512i) {
    v[a] = _mm512_add_epi64(_mm512_add_epi64(v[a], v[b]), x);
    v[d] = _mm512_ror_epi64::<32>(_mm512_xor_si512(v[d], v[a]));
    v[c] = _mm512_add_epi64(v[c], v[d]);
    v[b] = _mm512_ror_epi64::<24>(_mm512_xor_si512(v[b], v[c]));
    v[a] = _mm512_add_epi64(_mm512_add_epi64(v[a], v[b]), y);
    v[d] = _mm512_ror_epi64::<16>(_mm512_xor_si512(v[d], v[a]));
    v[c] = _mm512_add_epi64(v[c], v[d]);
    v[b] = _mm512_ror_epi64::<63>(_mm512_xor_si512(v[b], v[c]));
}
