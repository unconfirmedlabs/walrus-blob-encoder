// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Prints the blob ID walrus-core computes for each file, the reference for cross-checks.
//!
//! ```text
//! cargo run --release --example upstream_id -- [--shards N] <files...>
//! ```

use std::{num::NonZeroU16, process::exit};

use walrus_core::{
    DEFAULT_ENCODING,
    encoding::{EncodingConfig, EncodingFactory as _},
};

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut n_shards = 1000;
    if let Some(i) = args.iter().position(|arg| arg == "--shards") {
        n_shards = args
            .get(i + 1)
            .and_then(|n| n.parse().ok())
            .unwrap_or_else(|| {
                eprintln!("--shards needs a shard count");
                exit(2);
            });
        args.drain(i..i + 2);
    }
    let config = EncodingConfig::new(NonZeroU16::new(n_shards).expect("shard count"))
        .get_for_type(DEFAULT_ENCODING);
    for path in &args {
        let blob = std::fs::read(path).unwrap_or_else(|error| {
            eprintln!("{path}: {error}");
            exit(1);
        });
        let id = config
            .compute_blob_id(&blob)
            .expect("walrus-core computes the blob ID");
        println!("{} {path}", walrus_blob_encoder::BlobId::from(id.0));
    }
}
