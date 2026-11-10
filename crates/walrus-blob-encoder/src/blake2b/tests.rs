// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use blake2::Blake2b;
use blake2::Digest as _;
use blake2::digest::consts::U32;
use proptest::prelude::*;

use super::{Backend, Digest, Hasher, test_backends};

fn reference(prefix: u8, message: &[u8]) -> Digest {
    Blake2b::<U32>::new()
        .chain_update([prefix])
        .chain_update(message)
        .finalize()
        .into()
}

/// Deterministic pseudo-random bytes.
fn bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect()
}

/// Checks every test backend against the reference on `messages`.
fn check(prefix: u8, messages: &[&[u8]]) {
    let expected: Vec<Digest> = messages.iter().map(|m| reference(prefix, m)).collect();
    for backend in test_backends() {
        let hasher = Hasher::new(backend).expect("test backends are available");
        let mut out = vec![[0xaa; 32]; messages.len()];
        hasher.hash_batch(prefix, messages, &mut out);
        for (i, (got, want)) in out.iter().zip(&expected).enumerate() {
            assert_eq!(
                got,
                want,
                "{backend}: prefix {prefix:#04x}, {} messages of {} bytes, message {i}",
                messages.len(),
                messages[0].len(),
            );
        }
    }
}

/// `count` distinct messages of `len` bytes at scattered, unaligned offsets of one buffer.
fn scattered(buffer: &[u8], count: usize, len: usize, seed: u64) -> Vec<&[u8]> {
    let span = buffer.len() - len;
    (0..count)
        .map(|i| {
            let offset = (seed as usize)
                .wrapping_add(i.wrapping_mul(7919))
                .wrapping_mul(2_654_435_761)
                % (span + 1);
            &buffer[offset..offset + len]
        })
        .collect()
}

fn lengths() -> Vec<usize> {
    let mut lengths = vec![
        0, 1, 2, 4, 40, 63, 64, 65, 126, 127, 128, 129, 255, 256, 257,
    ];
    let blocks = if cfg!(miri) { 3 } else { 10 };
    for k in 2..=blocks {
        lengths.extend([128 * k - 1, 128 * k, 128 * k + 1]);
    }
    if !cfg!(miri) {
        lengths.extend([1164, 1206]);
    }
    lengths
}

#[test]
fn lengths_and_counts() {
    let counts: &[usize] = if cfg!(miri) {
        &[1, 9]
    } else {
        &[1, 2, 3, 4, 5, 7, 8, 9, 17]
    };
    let prefixes: &[u8] = if cfg!(miri) {
        &[0x01]
    } else {
        &[0x00, 0x01, 0xff]
    };
    let buffer = bytes(if cfg!(miri) { 4_000 } else { 40_000 }, 1);
    for len in lengths() {
        for &count in counts {
            for &prefix in prefixes {
                check(prefix, &scattered(&buffer, count, len, len as u64));
            }
        }
    }
}

#[test]
fn every_count_up_to_33() {
    let max = if cfg!(miri) { 9 } else { 33 };
    let buffer = bytes(10_000, 2);
    for len in [0, 4, 64, 200] {
        for count in 0..=max {
            check(0x01, &scattered(&buffer, count, len, count as u64));
        }
    }
}

#[cfg_attr(miri, ignore = "slow under Miri")]
#[test]
fn symbol_sizes() {
    let buffer = bytes(200_000, 3);
    for len in (2..=2400).step_by(46).chain([65_532, 65_534]) {
        check(0x00, &scattered(&buffer, 11, len, len as u64));
    }
}

#[cfg_attr(miri, ignore = "slow under Miri")]
#[test]
fn full_column_and_level() {
    let buffer = bytes(1_300_000, 4);
    let column: Vec<&[u8]> = buffer.chunks_exact(1164).take(1000).collect();
    check(0x00, &column);
    let level: Vec<&[u8]> = buffer.chunks_exact(64).take(1000).collect();
    check(0x01, &level);
}

#[test]
fn overlapping_and_identical_messages() {
    let buffer = bytes(600, 5);
    let overlapping: Vec<&[u8]> = (0..20).map(|i| &buffer[i..i + 300]).collect();
    check(0x00, &overlapping);
    check(0x01, &[&buffer[..129]; 13]);
}

#[test]
#[should_panic(expected = "one output per message")]
fn output_count_mismatch_panics() {
    let mut out = [[0; 32]; 1];
    Hasher::new(Backend::Scalar)
        .expect("scalar is available")
        .hash_batch(0, &[b"a", b"b"], &mut out);
}

#[test]
#[should_panic(expected = "equal lengths")]
fn unequal_lengths_panic() {
    let mut out = [[0; 32]; 2];
    Hasher::new(Backend::Scalar)
        .expect("scalar is available")
        .hash_batch(0, &[b"a", b"bc"], &mut out);
}

#[test]
fn backend_selection() {
    assert!(Backend::detect().is_available());
    assert!(Backend::Scalar.is_available());
    for &backend in Backend::ALL {
        let hasher = Hasher::new(backend);
        assert_eq!(hasher.is_some(), backend.is_available(), "{backend}");
        if let Some(hasher) = hasher {
            assert_eq!(hasher.backend(), backend);
        }
        assert_eq!(backend.name().parse::<Backend>(), Ok(backend));
        assert_eq!(backend.to_string(), backend.name());
    }
    assert!("AVX2".parse::<Backend>().is_err());
    assert!("".parse::<Backend>().is_err());
    #[cfg(not(target_arch = "x86_64"))]
    for backend in [Backend::Avx2, Backend::Avx512, Backend::Sse41] {
        assert!(!backend.is_available(), "{backend}");
    }
    #[cfg(not(target_arch = "aarch64"))]
    assert!(!Backend::Neon.is_available());
}

#[test]
fn compile_time_backends() {
    for &backend in Backend::ALL {
        if backend.is_enabled_statically() {
            assert!(backend.is_available(), "{backend}");
        }
        #[cfg(not(feature = "runtime-detection"))]
        assert_eq!(
            backend.is_available(),
            backend.is_enabled_statically(),
            "{backend}"
        );
    }
    let fastest_static = Backend::ALL
        .iter()
        .copied()
        .find(|backend| backend.is_enabled_statically())
        .expect("scalar is always enabled");
    if !cfg!(feature = "runtime-detection") || fastest_static == Backend::ALL[0] {
        assert_eq!(Backend::detect(), fastest_static);
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: if cfg!(miri) { 4 } else { 256 },
        ..ProptestConfig::default()
    })]

    #[test]
    fn random_batches(
        prefix: u8,
        count in 0usize..=40,
        len in 0usize..=700,
        seed: u64,
    ) {
        let buffer = bytes(len + 1000, seed);
        check(prefix, &scattered(&buffer, count, len, seed));
    }
}
