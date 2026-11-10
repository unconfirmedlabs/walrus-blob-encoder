// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The blob encoder: shard parameters, backend and thread pool, and the in-memory entry points.
//!
//! An in-memory blob is a seekable source whose reads are free, so its metadata comes from the
//! same column-by-column [`Expansion`] as the streaming encoder's, in a single pass: the
//! systematic columns are read from the blob in place, the rows are encoded once into their
//! recovery symbols, and the parity columns are read from those. With the `rayon` feature and
//! more than one thread, rows and columns are encoded in parallel (see [`expansion`]). The
//! result does not depend on the number of threads or the backend.

use std::num::NonZeroU16;
#[cfg(feature = "rayon")]
use std::sync::Arc;

#[cfg(feature = "rayon")]
use rayon::ThreadPool;

use crate::{
    Backend, BlobId, BlobMetadata, Error,
    blake2b::Hasher,
    expansion::{self, Expansion},
    layout::{MessageMatrix, ShardLayout},
    reed_solomon::Engine,
};

/// Number of shards of Walrus mainnet.
pub const MAINNET_N_SHARDS: NonZeroU16 = NonZeroU16::new(1000).unwrap();

/// Computes Walrus blob IDs and blob metadata for a fixed number of shards.
///
/// The results are byte-identical to walrus-core's `compute_blob_id` and `compute_metadata`
/// with the default encoding, RS2.
///
/// ```
/// use walrus_blob_encoder::BlobEncoder;
///
/// let encoder = BlobEncoder::mainnet();
/// let id = encoder.blob_id(b"")?;
/// assert_eq!(id.to_string(), "3GPQL3HZNnFhN_F7l5Aa-X1VOtAKwIsg9zuWk8R81v4");
/// # Ok::<(), walrus_blob_encoder::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct BlobEncoder {
    n_shards: NonZeroU16,
    layout: ShardLayout,
    hasher: Hasher,
    #[cfg(feature = "rayon")]
    pool: Option<Arc<ThreadPool>>,
}

impl BlobEncoder {
    /// Creates an encoder for `n_shards` shards, hashing with [`Backend::detect`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidShardCount`] if walrus-core cannot encode for `n_shards`, which is
    /// the case for 1 to 3 and for more than 49,155 shards.
    pub fn new(n_shards: NonZeroU16) -> Result<Self, Error> {
        let layout = ShardLayout::new(n_shards).ok_or(Error::InvalidShardCount(n_shards.get()))?;
        Ok(BlobEncoder {
            n_shards,
            layout,
            hasher: Hasher::new(Backend::detect()).expect("the detected backend is available"),
            #[cfg(feature = "rayon")]
            pool: None,
        })
    }

    /// Creates an encoder for [`MAINNET_N_SHARDS`].
    pub fn mainnet() -> Self {
        Self::new(MAINNET_N_SHARDS).expect("mainnet shard count is valid")
    }

    /// Selects the BLAKE2b backend.
    ///
    /// [`Backend::Scalar`] also selects the portable Reed-Solomon engine, giving a path free of
    /// explicit SIMD code that is several times slower for large blobs; [`Backend::Avx512`]
    /// selects the AVX-512 engine of `reed-solomon-simd`, with GFNI where available; the other
    /// backends use its `Wide` engine if the target enables SSSE3, `simd128` or NEON at compile
    /// time, and the portable engine otherwise. Only the AVX-512 engine is detected at runtime,
    /// with the `runtime-detection` feature.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnsupportedBackend`] if the backend cannot run on this CPU.
    pub fn with_backend(mut self, backend: Backend) -> Result<Self, Error> {
        self.hasher = Hasher::new(backend).ok_or(Error::UnsupportedBackend(backend))?;
        Ok(self)
    }

    #[cfg(feature = "rayon")]
    /// Runs the computations on `pool` instead of the current rayon pool.
    pub fn with_thread_pool(mut self, pool: Arc<ThreadPool>) -> Self {
        self.pool = Some(pool);
        self
    }

    /// The number of shards.
    pub fn n_shards(&self) -> NonZeroU16 {
        self.n_shards
    }

    /// The BLAKE2b backend.
    pub fn backend(&self) -> Backend {
        self.hasher.backend()
    }

    /// The maximum blob length in bytes: `n_rows * n_cols * 65534`, where the message matrix has
    /// `n_rows = n - 2f` rows and `n_cols = n - f` columns for `n` shards and
    /// `f = (n - 1) / 3`.
    pub fn max_blob_size(&self) -> u64 {
        self.layout.max_blob_size()
    }

    /// The symbol size in bytes for a blob of `len` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DataTooLarge`] if `len` exceeds [`Self::max_blob_size`].
    pub fn symbol_size(&self, len: u64) -> Result<NonZeroU16, Error> {
        self.layout.symbol_size(len).ok_or(Error::DataTooLarge {
            len,
            max: self.max_blob_size(),
        })
    }

    /// Computes the blob ID of `blob`.
    ///
    /// This computes the full [`metadata`](Self::metadata), with the same cost.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DataTooLarge`] if the blob exceeds [`Self::max_blob_size`].
    pub fn blob_id(&self, blob: &[u8]) -> Result<BlobId, Error> {
        self.metadata(blob).map(|metadata| metadata.blob_id())
    }

    /// Computes the metadata of `blob`: its blob ID and the Merkle roots of all sliver pairs.
    ///
    /// Besides the caller's blob, the computation allocates, for `n` shards,
    /// `f = (n - 1) / 3`, `n_rows = n - 2f`, `n_cols = n - f` and symbol size `s`:
    ///
    /// - the recovery symbols of the row encoding, `n_rows * f * s` bytes (about half the blob);
    /// - the rows not entirely inside the blob, zero padded, at most
    ///   `2 * n_rows * n_cols + n_cols * s` bytes;
    /// - the row trees, about `64 * n * (log2(n) + 2)` bytes (0.8 MB for 1000 shards);
    /// - per thread, one Reed-Solomon encoder with about `2 * next_power_of_two(f) * s` bytes of
    ///   work space (symbols rounded up to 64 bytes) and multiplication tables of at most
    ///   384 KiB;
    /// - with more than one thread, the leaf hashes of 256 columns at a time, `8192 * n` bytes
    ///   (8 MB for 1000 shards).
    ///
    /// It runs on the encoder's thread pool (see [`Self::with_thread_pool`]) or the current
    /// rayon pool, and on the calling thread alone without the `rayon` feature.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DataTooLarge`] if the blob exceeds [`Self::max_blob_size`].
    pub fn metadata(&self, blob: &[u8]) -> Result<BlobMetadata, Error> {
        let len = blob.len() as u64;
        let s = usize::from(self.symbol_size(len)?.get());
        let compute = || {
            #[cfg(feature = "rayon")]
            let parallel = rayon::current_num_threads() > 1;
            #[cfg(not(feature = "rayon"))]
            let parallel = false;
            let matrix = MessageMatrix::new(blob, &self.layout, s);
            let expansion = Expansion::new(self.hasher, self.engine(), self.layout, s);
            let row = |r| matrix.row(r);
            expansion::in_memory(expansion, matrix.rows(), row, len, parallel)
        };
        #[cfg(feature = "rayon")]
        if let Some(pool) = &self.pool {
            return Ok(pool.install(compute));
        }
        Ok(compute())
    }

    pub(crate) fn engine(&self) -> Engine {
        match self.hasher.backend() {
            Backend::Scalar => Engine::Portable,
            #[cfg(target_arch = "x86_64")]
            Backend::Avx512 => Engine::Avx512,
            _ => Engine::Simd,
        }
    }

    pub(crate) fn hasher(&self) -> Hasher {
        self.hasher
    }

    pub(crate) fn layout(&self) -> ShardLayout {
        self.layout
    }
}

impl Default for BlobEncoder {
    /// Same as [`BlobEncoder::mainnet`].
    fn default() -> Self {
        Self::mainnet()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blake2b::test_backends;

    #[test]
    fn empty_blob_id() {
        for backend in test_backends() {
            let encoder = BlobEncoder::mainnet().with_backend(backend).unwrap();
            assert_eq!(
                encoder.blob_id(&[]).unwrap().to_string(),
                "3GPQL3HZNnFhN_F7l5Aa-X1VOtAKwIsg9zuWk8R81v4"
            );
        }
    }

    #[test]
    fn rejects_unavailable_backends() {
        for &backend in Backend::ALL {
            let result = BlobEncoder::mainnet().with_backend(backend);
            match result {
                Ok(encoder) => assert_eq!(encoder.backend(), backend),
                Err(error) => {
                    assert!(!backend.is_available());
                    assert_eq!(error, Error::UnsupportedBackend(backend));
                }
            }
        }
    }

    #[test]
    fn rejects_invalid_shard_counts() {
        for n in [1, 2, 3, 49_156, u16::MAX] {
            let n_shards = NonZeroU16::new(n).unwrap();
            assert_eq!(
                BlobEncoder::new(n_shards).unwrap_err(),
                Error::InvalidShardCount(n)
            );
        }
    }

    #[test]
    fn rejects_too_large_blobs() {
        let encoder = BlobEncoder::new(NonZeroU16::new(4).unwrap()).unwrap();
        let max = encoder.max_blob_size();
        assert_eq!(max, 6 * 65_534);
        assert_eq!(encoder.symbol_size(max).unwrap().get(), 65_534);
        let error = Error::DataTooLarge { len: max + 1, max };
        assert_eq!(encoder.symbol_size(max + 1), Err(error.clone()));
        let blob = vec![0; usize::try_from(max).unwrap() + 1];
        assert_eq!(encoder.blob_id(&blob), Err(error));
    }
}
