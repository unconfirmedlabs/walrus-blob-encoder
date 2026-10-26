// Copyright (c) Unconfirmed Labs, LLC
// SPDX-License-Identifier: Apache-2.0

use std::{error, fmt};

use crate::Backend;

/// Errors returned by [`BlobEncoder`](crate::BlobEncoder).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The blob is longer than the encoding supports for this number of shards.
    DataTooLarge {
        /// Length of the blob in bytes.
        len: u64,
        /// Maximum blob length in bytes.
        max: u64,
    },
    /// Walrus cannot encode blobs for this number of shards.
    InvalidShardCount(u16),
    /// The BLAKE2b backend cannot run on this CPU.
    UnsupportedBackend(Backend),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::DataTooLarge { len, max } => {
                write!(f, "blob of {len} bytes exceeds the maximum of {max} bytes")
            }
            Error::InvalidShardCount(n) => write!(f, "blobs cannot be encoded for {n} shards"),
            Error::UnsupportedBackend(backend) => {
                write!(f, "BLAKE2b backend {backend} is not available on this CPU")
            }
        }
    }
}

impl error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        assert_eq!(
            Error::DataTooLarge { len: 7, max: 6 }.to_string(),
            "blob of 7 bytes exceeds the maximum of 6 bytes"
        );
        assert_eq!(
            Error::InvalidShardCount(3).to_string(),
            "blobs cannot be encoded for 3 shards"
        );
        assert_eq!(
            Error::UnsupportedBackend(Backend::Neon).to_string(),
            "BLAKE2b backend neon is not available on this CPU"
        );
    }
}
