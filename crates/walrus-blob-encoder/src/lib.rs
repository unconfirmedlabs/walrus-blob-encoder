// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Byte-exact, parallel computation of Walrus blob IDs and blob metadata.
//!
//! [`BlobEncoder`] computes the same blob ID and sliver-pair hashes as walrus-core and the
//! `walrus` CLI for the RS2 (RedStuff / Reed-Solomon) encoding, without storing the encoded
//! slivers. The work runs on rayon threads, hashes many BLAKE2b messages at once with the best
//! [`Backend`] of the CPU, and encodes with `reed-solomon-simd`, using an AVX-512 (and GFNI) engine
//! of this crate under [`Backend::Avx512`].
//!
//! ```
//! use walrus_blob_encoder::BlobEncoder;
//!
//! let encoder = BlobEncoder::mainnet();
//! let metadata = encoder.metadata(b"hello walrus")?;
//! assert_eq!(metadata.sliver_pairs().len(), 1000);
//! assert_eq!(encoder.blob_id(b"hello walrus")?, metadata.blob_id());
//! # Ok::<(), walrus_blob_encoder::Error>(())
//! ```

mod blake2b;
mod blob_id;
mod encoder;
mod error;
mod layout;
mod merkle;
mod messages;
mod metadata;
mod reed_solomon;

pub use blake2b::{Backend, ParseBackendError};
pub use blob_id::{BlobId, ParseBlobIdError};
pub use encoder::{BlobEncoder, MAINNET_N_SHARDS};
pub use error::Error;
pub use metadata::{BlobMetadata, EncodingType, SliverPairMetadata};
