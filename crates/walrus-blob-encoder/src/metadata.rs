// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Blob metadata: the sliver-pair hashes a blob ID commits to.

use std::{fmt, num::NonZeroU16};

use crate::BlobId;

/// Erasure-coding scheme of a blob.
#[non_exhaustive]
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EncodingType {
    /// RedStuff with Reed-Solomon codes, the default Walrus encoding.
    RS2 = 1,
}

impl From<EncodingType> for u8 {
    /// The encoding type's byte in the blob ID preimage and in BCS.
    fn from(encoding_type: EncodingType) -> Self {
        encoding_type as u8
    }
}

/// Merkle roots of one sliver pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SliverPairMetadata {
    /// Root of the primary sliver: row `i` of the expanded matrix.
    pub primary_hash: [u8; 32],
    /// Root of the secondary sliver: column `n_shards - 1 - i` of the expanded matrix.
    pub secondary_hash: [u8; 32],
}

/// Metadata of a blob, equal to walrus-core's `VerifiedBlobMetadataWithId`.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct BlobMetadata {
    pub(crate) blob_id: BlobId,
    pub(crate) encoding_type: EncodingType,
    pub(crate) unencoded_length: u64,
    pub(crate) sliver_pairs: Vec<SliverPairMetadata>,
}

impl BlobMetadata {
    /// The blob ID.
    pub fn blob_id(&self) -> BlobId {
        self.blob_id
    }

    /// The encoding type.
    pub fn encoding_type(&self) -> EncodingType {
        self.encoding_type
    }

    /// The length of the blob in bytes.
    pub fn unencoded_length(&self) -> u64 {
        self.unencoded_length
    }

    /// The sliver-pair hashes; index `i` is sliver pair `i`, one per shard.
    pub fn sliver_pairs(&self) -> &[SliverPairMetadata] {
        &self.sliver_pairs
    }

    /// The number of shards the blob was encoded for.
    pub fn n_shards(&self) -> NonZeroU16 {
        u16::try_from(self.sliver_pairs.len())
            .ok()
            .and_then(NonZeroU16::new)
            .expect("one sliver pair per shard")
    }
}

impl fmt::Debug for BlobMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BlobMetadata")
            .field("blob_id", &self.blob_id)
            .field("encoding_type", &self.encoding_type)
            .field("unencoded_length", &self.unencoded_length)
            .field("n_shards", &self.sliver_pairs.len())
            .finish_non_exhaustive()
    }
}
