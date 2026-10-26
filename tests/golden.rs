// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Blob IDs computed by the `walrus` CLI 1.58.1 for 1000 shards.

mod common;

use std::path::PathBuf;

use common::{for_each_backend, for_each_backend_and_pool, generate};
use walrus_blob_encoder::BlobId;

/// Real files, checked when `$WALRUS_BLOB_ENCODER_FILES` names a directory containing them.
const FILES: [(&str, u64, &str); 4] = [
    (
        "s16_44k_010s.flac",
        710_603,
        "Ok4DpBJWtNRodlPBcA-NvyH8_OjgOd-TpCIXMljIzHw",
    ),
    (
        "s16_44k_240s.flac",
        17_066_416,
        "fiCjCZSUTIdcosdPghw2kT6OpzL-BHKcHQ26RiaCP-8",
    ),
    (
        "s24_96k_240s.flac",
        69_741_071,
        "X6x0CM1KzaZKdwaHcs-wtLufKBusm86YMPX_JeHhh0E",
    ),
    (
        "s24_192k_480s.flac",
        259_209_074,
        "Q10ZSoSuV3wP4WUt4PIDKm_u96AQHGnpURDHKEIh9tM",
    ),
];

#[test]
fn generated_inputs() {
    let vectors: Vec<(usize, u64, BlobId)> = include_str!("vectors/cli-1000.txt")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| {
            let fields: Vec<&str> = line.split(' ').collect();
            let [len, seed, id] = fields[..] else {
                panic!("malformed line {line:?}");
            };
            (
                len.parse().unwrap(),
                seed.parse().unwrap(),
                id.parse().unwrap(),
            )
        })
        .collect();
    assert_eq!(vectors.len(), 32);
    for (case, (len, seed, expected)) in vectors.into_iter().enumerate() {
        let blob = generate(len, seed);
        for_each_backend(1000, case, |label, encoder| {
            assert_eq!(
                encoder.blob_id(&blob).unwrap(),
                expected,
                "len={len} seed={seed} {label}"
            );
        });
    }
}

#[test]
fn real_files() {
    let Some(dir) = std::env::var_os("WALRUS_BLOB_ENCODER_FILES") else {
        eprintln!("skipped: set WALRUS_BLOB_ENCODER_FILES to a directory with the test files");
        return;
    };
    let dir = PathBuf::from(dir);
    for (name, len, expected) in FILES {
        let blob = std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(blob.len() as u64, len, "{name}");
        let expected: BlobId = expected.parse().unwrap();
        for_each_backend_and_pool(1000, |label, encoder| {
            assert_eq!(encoder.blob_id(&blob).unwrap(), expected, "{name} {label}");
        });
    }
}
