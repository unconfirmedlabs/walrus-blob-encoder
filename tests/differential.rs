// Copyright (c) Unconfirmed Labs, LLC
// SPDX-License-Identifier: Apache-2.0

//! Differential tests against walrus-core.

mod common;

use std::num::NonZeroU16;

use common::{for_each_backend, for_each_backend_and_pool, generate};
use walrus_blob_encoder::{BlobEncoder, BlobId, BlobMetadata, Error};
use walrus_core::{
    DEFAULT_ENCODING,
    encoding::{EncodingConfig, EncodingConfigEnum, EncodingFactory as _},
    metadata::{BlobMetadataApi as _, VerifiedBlobMetadataWithId},
};

/// Source symbols of the message matrix for 1000 shards (334 rows, 667 columns).
const SYMBOLS: usize = 334 * 667;

fn walrus_config(n_shards: u16) -> EncodingConfigEnum {
    EncodingConfig::new(NonZeroU16::new(n_shards).unwrap()).get_for_type(DEFAULT_ENCODING)
}

fn assert_same(ours: &BlobMetadata, theirs: &VerifiedBlobMetadataWithId, context: &str) {
    assert_eq!(ours.blob_id().as_bytes(), &theirs.blob_id().0, "{context}");
    let metadata = theirs.metadata();
    assert_eq!(
        u8::from(ours.encoding_type()),
        u8::from(metadata.encoding_type()),
        "{context}"
    );
    assert_eq!(
        ours.unencoded_length(),
        metadata.unencoded_length(),
        "{context}"
    );
    assert_eq!(ours.n_shards(), theirs.n_shards(), "{context}");
    let hashes = metadata.hashes();
    assert_eq!(ours.sliver_pairs().len(), hashes.len(), "{context}");
    for (i, (ours, theirs)) in ours.sliver_pairs().iter().zip(hashes).enumerate() {
        assert_eq!(
            ours.primary_hash,
            theirs.primary_hash.bytes(),
            "{context} pair {i}"
        );
        assert_eq!(
            ours.secondary_hash,
            theirs.secondary_hash.bytes(),
            "{context} pair {i}"
        );
    }
}

/// Checks the full metadata of `blob` with every backend, on the thread count `case` selects.
fn check(n_shards: u16, case: usize, blob: &[u8]) {
    let expected = walrus_config(n_shards).compute_metadata(blob).unwrap();
    for_each_backend(n_shards, case, |label, encoder| {
        let context = format!("n_shards={n_shards} len={} {label}", blob.len());
        assert_same(&encoder.metadata(blob).unwrap(), &expected, &context);
    });
}

fn check_lengths(n_shards: u16, lengths: impl IntoIterator<Item = usize>) {
    for (case, len) in lengths.into_iter().enumerate() {
        check(n_shards, case, &generate(len, len as u64));
    }
}

#[test]
fn edge_lengths() {
    check_lengths(1000, [0, 1, 2, 3, 127, 128, 129, 1000, 4096, 65_535]);
}

#[test]
fn compute_blob_id_matches() {
    let config = walrus_config(1000);
    for (case, len) in [0, 1, 4096].into_iter().enumerate() {
        let blob = generate(len, 7);
        let expected = config.compute_blob_id(&blob).unwrap();
        for_each_backend(1000, case, |label, encoder| {
            assert_eq!(
                encoder.blob_id(&blob).unwrap().as_bytes(),
                &expected.0,
                "{label}"
            );
        });
    }
}

fn boundaries(ks: impl IntoIterator<Item = usize>) -> impl Iterator<Item = usize> {
    ks.into_iter()
        .flat_map(|k| [k * SYMBOLS - 1, k * SYMBOLS, k * SYMBOLS + 1])
}

#[test]
fn symbol_size_boundaries() {
    check_lengths(1000, boundaries(1..=8));
}

#[test]
fn large_symbol_size_boundaries() {
    check_lengths(1000, boundaries([9, 15, 16, 17, 64]));
    check_lengths(
        1000,
        [
            255 * SYMBOLS + 1,
            256 * SYMBOLS,
            256 * SYMBOLS + 1,
            257 * SYMBOLS,
        ],
    );
}

#[test]
#[ignore = "allocates and hashes about 1 GB in total; run with --ignored"]
fn largest_symbol_size_boundaries() {
    check_lengths(1000, boundaries([1000]));
}

/// Small committees have small matrices, so their symbol-size steps can be checked across the
/// whole `u16` range: every step up to symbol size 512, then every 97th, and the last ones.
#[test]
fn symbol_size_boundaries_whole_range_small_shard_counts() {
    for n_shards in [4, 7] {
        let symbols = walrus_config(n_shards).source_symbols_per_blob().get() as usize;
        let ks = (2..=512)
            .step_by(2)
            .chain((514..65_534).step_by(2 * 97))
            .chain([65_532, 65_534]);
        let lengths = ks.flat_map(|k| [k * symbols - 1, k * symbols, k * symbols + 1]);
        let max = 65_534 * symbols;
        check_lengths(n_shards, lengths.filter(|&len| len <= max));
    }
}

#[test]
fn random_lengths() {
    let lengths: Vec<usize> = generate(8 * 8, 0x5eed)
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()) as usize % (4 << 20))
        .collect();
    check_lengths(1000, lengths);
}

#[test]
fn other_shard_counts() {
    for n_shards in [4, 5, 6, 7, 10, 13, 31, 64, 100, 101, 333] {
        check_lengths(n_shards, [0, 1, 17, 1000, 123_457]);
    }
    for n_shards in [999, 1001, 2000] {
        let symbols = walrus_config(n_shards).source_symbols_per_blob().get() as usize;
        check_lengths(n_shards, [0, 1000, 2 * symbols + 1]);
    }
}

#[test]
fn max_blob_size_small_shard_counts() {
    for (case, n_shards) in [4, 7, 10].into_iter().enumerate() {
        let config = walrus_config(n_shards);
        let max = config.max_blob_size();
        let blob = generate(usize::try_from(max).unwrap() + 1, max);
        check(n_shards, case, &blob[..blob.len() - 1]);

        assert!(config.compute_metadata(&blob).is_err());
        for_each_backend(n_shards, case, |label, encoder| {
            assert_eq!(encoder.max_blob_size(), max, "{label}");
            assert_eq!(
                encoder.metadata(&blob).unwrap_err(),
                Error::DataTooLarge { len: max + 1, max },
                "{label}"
            );
        });
    }
}

#[test]
fn max_blob_size_and_symbol_size() {
    for n_shards in [4, 5, 7, 10, 101, 999, 1000, 1001, 2000, 49_155] {
        let config = walrus_config(n_shards);
        let encoder = BlobEncoder::new(NonZeroU16::new(n_shards).unwrap()).unwrap();
        let max = config.max_blob_size();
        assert_eq!(encoder.max_blob_size(), max, "n_shards={n_shards}");
        let symbols = u64::from(config.source_symbols_per_blob().get());
        let lengths = [
            0,
            1,
            2,
            symbols,
            symbols + 1,
            2 * symbols + 1,
            1 << 30,
            max - 1,
            max,
        ];
        for len in lengths.into_iter().filter(|&len| len <= max) {
            assert_eq!(
                encoder.symbol_size(len).unwrap(),
                config.symbol_size_for_blob(len).unwrap(),
                "n_shards={n_shards} len={len}"
            );
        }
        assert!(config.symbol_size_for_blob(max + 1).is_err());
        assert_eq!(
            encoder.symbol_size(max + 1),
            Err(Error::DataTooLarge { len: max + 1, max })
        );
        assert!(encoder.symbol_size(u64::MAX).is_err());
    }
}

#[test]
fn deterministic_across_thread_counts() {
    for (n_shards, len) in [
        (1000, 0),
        (1000, 1),
        (1000, SYMBOLS + 1),
        (1000, 3_000_000),
        (1000, 10 * (1 << 20) + 3),
        (7, 12_345),
        (101, 123_457),
    ] {
        let blob = generate(len, 99);
        let expected = walrus_config(n_shards).compute_metadata(&blob).unwrap();
        for_each_backend_and_pool(n_shards, |label, encoder| {
            let context = format!("n_shards={n_shards} len={len} {label}");
            assert_same(&encoder.metadata(&blob).unwrap(), &expected, &context);
        });
    }
}

#[test]
fn blob_id_text_matches_walrus() {
    let bytes = generate(32 * 2000, 3);
    for chunk in bytes.chunks_exact(32) {
        let array: [u8; 32] = chunk.try_into().unwrap();
        let ours = BlobId::from(array);
        let theirs = walrus_core::BlobId(array);
        let text = theirs.to_string();
        assert_eq!(ours.to_string(), text);
        assert_eq!(format!("{ours:?}"), format!("{theirs:?}"));
        assert_eq!(text.parse::<BlobId>(), Ok(ours));
    }

    // Single-character substitutions, truncations and extensions of valid IDs.
    let alphabet: Vec<char> = (0u8..128).map(char::from).chain(['é', '€']).collect();
    for (i, chunk) in bytes.chunks_exact(32).take(50).enumerate() {
        let text = walrus_core::BlobId(chunk.try_into().unwrap()).to_string();
        let mut candidates = vec![
            text[..42].to_string(),
            format!("{text}A"),
            format!("{text}="),
            format!("{text}=="),
            String::new(),
        ];
        for (j, &c) in alphabet.iter().enumerate() {
            let position = (i * 7 + j) % text.len();
            let mut candidate: Vec<char> = text.chars().collect();
            candidate[position] = c;
            candidates.push(candidate.into_iter().collect());
            candidates.push(format!("{}{c}", &text[..42]));
        }
        for candidate in candidates {
            let theirs = candidate.parse::<walrus_core::BlobId>().ok().map(|id| id.0);
            let ours = candidate.parse::<BlobId>().ok().map(<[u8; 32]>::from);
            assert_eq!(ours, theirs, "{candidate:?}");
        }
    }
}
