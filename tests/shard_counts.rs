// Copyright (c) Unconfirmed Labs, LLC
// SPDX-License-Identifier: Apache-2.0

//! The supported shard counts are exactly those walrus-core can encode.

use std::{num::NonZeroU16, panic};

use walrus_blob_encoder::{BlobEncoder, Error};
use walrus_core::{
    DEFAULT_ENCODING,
    encoding::{
        EncodingConfig, EncodingFactory as _, ReedSolomonEncoder, source_symbols_for_n_shards,
    },
};

fn nonzero(n: u16) -> NonZeroU16 {
    NonZeroU16::new(n).unwrap()
}

/// Whether walrus-core can create the Reed-Solomon encoders `compute_metadata` uses for `n`.
fn walrus_has_encoders(n: u16) -> bool {
    let n_shards = nonzero(n);
    let (primary, secondary) = source_symbols_for_n_shards(n_shards);
    let symbol_size = nonzero(2);
    ReedSolomonEncoder::new(symbol_size, primary, n_shards).is_ok()
        && ReedSolomonEncoder::new(symbol_size, secondary, n_shards).is_ok()
}

/// The metadata walrus-core computes for `blob` with `n` shards, or `None` if it panics.
fn walrus_metadata(n: u16, blob: &[u8]) -> Option<[u8; 32]> {
    panic::catch_unwind(|| {
        let config = EncodingConfig::new(nonzero(n)).get_for_type(DEFAULT_ENCODING);
        config.compute_blob_id(blob).unwrap().0
    })
    .ok()
}

#[test]
fn shard_counts_match_walrus() {
    let encoder_counts: Vec<u16> = (1..=2100)
        .chain(49_000..=49_300)
        .chain((2100..=u16::MAX).step_by(97))
        .chain([u16::MAX - 1, u16::MAX])
        .collect();
    let blob = [1, 2, 3];
    let metadata_counts = 1..=40;

    // walrus-core panics for unsupported counts; silence the hook while collecting results.
    let hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let walrus_encoders: Vec<Option<bool>> = encoder_counts
        .iter()
        .map(|&n| panic::catch_unwind(|| walrus_has_encoders(n)).ok())
        .collect();
    let walrus_ids: Vec<Option<[u8; 32]>> = metadata_counts
        .clone()
        .map(|n| walrus_metadata(n, &blob))
        .collect();
    panic::set_hook(hook);

    for (&n, walrus) in encoder_counts.iter().zip(walrus_encoders) {
        let ours = BlobEncoder::new(nonzero(n));
        assert_eq!(ours.is_ok(), walrus == Some(true), "n_shards={n}");
        if let Err(error) = ours {
            assert_eq!(error, Error::InvalidShardCount(n));
        }
    }
    for (n, walrus) in metadata_counts.zip(walrus_ids) {
        let ours = BlobEncoder::new(nonzero(n)).map(|encoder| encoder.blob_id(&blob).unwrap());
        assert_eq!(ours.ok().map(<[u8; 32]>::from), walrus, "n_shards={n}");
    }

    let supported: Vec<u16> = (1..=u16::MAX)
        .filter(|&n| BlobEncoder::new(nonzero(n)).is_ok())
        .collect();
    assert_eq!(supported, (4..=49_155).collect::<Vec<_>>());
}
