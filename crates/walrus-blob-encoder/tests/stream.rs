// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The streaming encoder against walrus-core and against the in-memory encoder, for budgets that
//! put batch edges everywhere the plan changes.

#[allow(dead_code, reason = "only the input generator is used")]
mod common;

use std::num::NonZeroU16;

use blake2::{Blake2b, Digest as _, digest::consts::U32};
use common::generate;
use walrus_blob_encoder::{BlobEncoder, BlobMetadata, SeekSource, StreamStats};
use walrus_core::{
    DEFAULT_ENCODING,
    encoding::{EncodingConfig, EncodingFactory as _},
    metadata::{BlobMetadataApi as _, VerifiedBlobMetadataWithId},
};

fn hash(parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Blake2b::<U32>::new();
    parts.iter().for_each(|part| hasher.update(part));
    hasher.finalize().into()
}

/// The root of the sliver-pair tree, built directly from the pair hashes.
fn metadata_root(metadata: &BlobMetadata) -> [u8; 32] {
    let mut level: Vec<[u8; 32]> = metadata
        .sliver_pairs()
        .iter()
        .map(|pair| hash(&[&[0], &pair.primary_hash, &pair.secondary_hash]))
        .collect();
    while level.len() > 1 {
        if level.len() % 2 == 1 {
            level.push([0; 32]);
        }
        level = level
            .chunks_exact(2)
            .map(|pair| hash(&[&[1], &pair[0], &pair[1]]))
            .collect();
    }
    level[0]
}

fn assert_same(ours: &BlobMetadata, theirs: &VerifiedBlobMetadataWithId, context: &str) {
    assert_eq!(
        ours.blob_id().as_bytes(),
        &theirs.blob_id().0,
        "blob ID, {context}"
    );
    let metadata = theirs.metadata();
    assert_eq!(
        metadata_root(ours),
        metadata.compute_root_hash().bytes(),
        "root, {context}"
    );
    let hashes = metadata.hashes();
    assert_eq!(ours.sliver_pairs().len(), hashes.len(), "{context}");
    for (i, (ours, theirs)) in ours.sliver_pairs().iter().zip(hashes).enumerate() {
        assert_eq!(
            ours.primary_hash,
            theirs.primary_hash.bytes(),
            "primary {i}, {context}"
        );
        assert_eq!(
            ours.secondary_hash,
            theirs.secondary_hash.bytes(),
            "secondary {i}, {context}"
        );
    }
}

/// Streams `blob` with budgets of `column_multiples` columns (`n_rows * s` bytes each) and of
/// `absolute` bytes, checking each against walrus-core and the in-memory encoder.
fn check(n_shards: u16, blob: &[u8], column_multiples: &[usize], absolute: &[usize]) -> usize {
    let n_shards = NonZeroU16::new(n_shards).unwrap();
    let config = EncodingConfig::new(n_shards).get_for_type(DEFAULT_ENCODING);
    let expected = config.compute_metadata(blob).unwrap();
    let encoder = BlobEncoder::new(n_shards).unwrap();
    let in_memory = encoder.metadata(blob).unwrap();
    assert_same(&in_memory, &expected, "in memory");
    assert_eq!(encoder.metadata_sequential(blob).unwrap(), in_memory, "sequential");
    assert_eq!(encoder.metadata_parallel(blob).unwrap(), in_memory, "parallel");
    let n = usize::from(n_shards.get());
    let f = (n - 1) / 3;
    let column = (n - 2 * f) * usize::from(encoder.symbol_size(blob.len() as u64).unwrap().get());
    let budgets: Vec<usize> = column_multiples
        .iter()
        .map(|k| k * column)
        .chain(absolute.iter().copied())
        .collect();
    for &budget in &budgets {
        let mut stream = encoder.stream(blob.len() as u64, budget).unwrap();
        while let Some(read) = stream.next_read() {
            let offset = usize::try_from(read.offset).unwrap();
            stream.supply(&blob[offset..offset + read.len]);
        }
        let StreamStats {
            batch_columns,
            full_passes,
            bytes_read,
            ..
        } = stream.stats();
        let context = format!(
            "n={n} L={} W={budget} k={batch_columns} passes={full_passes} read={bytes_read}",
            blob.len()
        );
        let metadata = stream.finish();
        assert_same(&metadata, &expected, &context);
        assert_eq!(metadata, in_memory, "{context}");
        println!("ok {context}");
    }
    budgets.len()
}

#[test]
fn small_sizes_1000_shards() {
    let multiples = [1, 2, 7, 332, 333, 334, 666, 667, 668, 999, 1000];
    let cases: usize = [1, 1000, 222_777, 222_778, 222_779]
        .into_iter()
        .map(|len| check(1000, &generate(len, len as u64), &multiples, &[0, 64 << 20]))
        .sum();
    assert_eq!(cases, 5 * 13);
}

#[test]
fn five_mb_1000_shards() {
    let blob = generate(5_000_000, 10);
    check(
        1000,
        &blob,
        &[1, 50, 333, 500, 667, 1000],
        &[1 << 20, 32 << 20],
    );
    let blob = generate(5 << 20, 11);
    check(1000, &blob, &[3, 666, 668], &[8 << 20, 64 << 20]);
}

#[test]
fn hundred_mb_plus_one_1000_shards() {
    let blob = generate(100_000_001, 12);
    check(
        1000,
        &blob,
        &[40, 333],
        &[8 << 20, 32 << 20, 64 << 20, 256 << 20],
    );
}

#[test]
fn small_shard_counts() {
    for n_shards in [4u16, 7, 10, 13, 100] {
        let n = usize::from(n_shards);
        let f = (n - 1) / 3;
        let source = (n - 2 * f) * (n - f);
        let lens = [1, source - 1, source, source + 1, 3 * source + 5, 100_000];
        for (i, len) in lens.into_iter().enumerate() {
            let blob = generate(len, 100 + i as u64);
            check(n_shards, &blob, &[1, 2, 3, f, n - f, n / 2, n], &[]);
        }
    }
}

#[test]
fn empty_blob() {
    let encoder = BlobEncoder::mainnet();
    let stream = encoder.stream(0, 1 << 20).unwrap();
    assert!(stream.is_done(), "an empty blob needs no reads");
    let stats = stream.stats();
    assert_eq!((stats.read_calls, stats.bytes_read), (0, 0));
    let metadata = stream.finish();
    assert_eq!(
        metadata.blob_id().to_string(),
        "3GPQL3HZNnFhN_F7l5Aa-X1VOtAKwIsg9zuWk8R81v4"
    );
    assert_eq!(metadata, encoder.metadata(&[]).unwrap());
}

#[test]
fn reads_and_sources() {
    let encoder = BlobEncoder::mainnet();
    let blob = generate(3_000_000, 5);
    let expected = encoder.metadata(&blob).unwrap();
    let column = 334 * 14;
    let mut slice = &blob[..];
    assert_eq!(
        encoder.metadata_from_source(&mut slice, 100 * column),
        Ok(expected.clone())
    );
    let mut file = SeekSource::new(std::io::Cursor::new(&blob), blob.len() as u64);
    let from_reader = encoder
        .metadata_from_source(&mut file, 10 * column)
        .unwrap();
    assert_eq!(from_reader, expected);

    // 70 columns per batch: 10 systematic batches of 67 columns, then 5 passes of 67 columns.
    let mut stream = encoder.stream(blob.len() as u64, 70 * column).unwrap();
    let mut reads = Vec::new();
    while let Some(read) = stream.next_read() {
        assert_eq!(stream.dest().len(), read.len);
        reads.push(read);
        let offset = usize::try_from(read.offset).unwrap();
        stream.supply(&blob[offset..offset + read.len]);
    }
    let stats = stream.stats();
    assert_eq!(stats.symbol_size, 14);
    assert_eq!(
        (
            stats.batch_columns,
            stats.systematic_batches,
            stats.full_passes
        ),
        (67, 10, 5)
    );
    assert_eq!(stats.read_calls, reads.len() as u64);
    assert_eq!(
        stats.bytes_read,
        reads.iter().map(|read| read.len as u64).sum::<u64>()
    );
    assert_eq!(stats.bytes_read, 6 * blob.len() as u64);
    assert_eq!(stats.row_encodes, 5 * 334);
    assert_eq!(stats.column_encodes, 1000);
    assert!(
        reads
            .iter()
            .all(|read| read.offset + read.len as u64 <= blob.len() as u64)
    );
    assert_eq!(stream.finish(), expected);
}

#[test]
fn rejects_too_large_blobs() {
    let encoder = BlobEncoder::new(NonZeroU16::new(4).unwrap()).unwrap();
    let max = encoder.max_blob_size();
    assert!(encoder.stream(max, 0).is_ok());
    assert!(encoder.stream(max + 1, 0).is_err());
}
