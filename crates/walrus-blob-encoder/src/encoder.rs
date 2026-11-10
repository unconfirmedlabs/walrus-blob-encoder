// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The blob encoder and its parallel schedule.
//!
//! 1. Rows of the message matrix are Reed-Solomon encoded into the recovery buffer, while the
//!    systematic columns (which only need the blob) are encoded and their leaves hashed.
//! 2. The non-systematic columns are encoded from the recovery buffer and their leaves hashed.
//! 3. The Merkle roots of all rows and columns of the expanded matrix are computed, several trees
//!    per task so that every batch of inner-node hashes is wide.
//! 4. The sliver-pair tree and the blob ID are computed.
//!
//! Every output slot is written by exactly one task, so the result does not depend on the
//! number of threads or the backend.

use std::{num::NonZeroU16, sync::Arc};

use rayon::{ThreadPool, prelude::*};

use crate::{
    Backend, BlobId, BlobMetadata, EncodingType, Error, SliverPairMetadata,
    blake2b::{DIGEST_LEN, Digest, Hasher},
    layout::{MessageMatrix, Rows, ShardLayout},
    merkle::{self, LEAF_PREFIX, TreeScratch},
    messages::MessageBuf,
    reed_solomon::{Encoder, Engine},
};

/// Number of shards of Walrus mainnet.
pub const MAINNET_N_SHARDS: NonZeroU16 = NonZeroU16::new(1000).unwrap();

/// Merkle trees reduced together per task, so that the narrow top levels still fill the SIMD
/// lanes of the hasher.
const TREES_PER_TASK: usize = 16;

const ZERO_NODE: Digest = [0; DIGEST_LEN];

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
    /// selects this crate's AVX-512 engine, with GFNI where available; the other backends use the
    /// best engine of `reed-solomon-simd` up to AVX2. Without the `runtime-detection` feature,
    /// engines are limited to the target's static features.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnsupportedBackend`] if the backend cannot run on this CPU.
    pub fn with_backend(mut self, backend: Backend) -> Result<Self, Error> {
        self.hasher = Hasher::new(backend).ok_or(Error::UnsupportedBackend(backend))?;
        Ok(self)
    }

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
    /// - the leaf hashes of the expanded matrix, `32 * n^2` bytes (32 MB for 1000 shards);
    /// - the rows not entirely inside the blob, zero padded, at most
    ///   `2 * n_rows * n_cols + n_cols * s` bytes;
    /// - per thread, up to two Reed-Solomon encoders of about `n * s` bytes of work space each
    ///   (symbols rounded up to 64 bytes), and Merkle scratch of about `3 * 32 * 16 * n` bytes.
    ///
    /// The recovery symbols are freed before the Merkle phase.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DataTooLarge`] if the blob exceeds [`Self::max_blob_size`].
    pub fn metadata(&self, blob: &[u8]) -> Result<BlobMetadata, Error> {
        let len = blob.len() as u64;
        let symbol_size = usize::from(self.symbol_size(len)?.get());
        let compute = || self.compute(blob, symbol_size);
        Ok(match &self.pool {
            Some(pool) => pool.install(compute),
            None => compute(),
        })
    }

    fn engine(&self) -> Engine {
        match self.hasher.backend() {
            Backend::Scalar => Engine::Portable,
            #[cfg(target_arch = "x86_64")]
            Backend::Avx512 => Engine::Avx512,
            _ => Engine::Simd,
        }
    }

    fn compute(&self, blob: &[u8], symbol_size: usize) -> BlobMetadata {
        let layout = self.layout;
        let n = layout.n_shards;
        let schedule = Schedule {
            hasher: self.hasher,
            engine: self.engine(),
            layout,
            symbol_size,
        };

        let matrix = MessageMatrix::new(blob, &layout, symbol_size);
        let mut recovery = vec![0; layout.n_rows * layout.n_faulty * symbol_size];
        let mut leaves = vec![ZERO_NODE; n * n];
        let (systematic, non_systematic) = leaves.split_at_mut(layout.n_cols * n);
        rayon::join(
            || schedule.encode_rows(&matrix, &mut recovery),
            || schedule.encode_columns(matrix.rows(), systematic),
        );
        drop(matrix);
        let recovery_rows = Rows::new(&recovery, layout.n_faulty * symbol_size);
        schedule.encode_columns(recovery_rows, non_systematic);
        drop(recovery);

        let (primary, secondary) = sliver_roots(self.hasher, &leaves, n);
        drop(leaves);
        finish(self.hasher, &primary, &secondary, blob.len() as u64)
    }
}

impl Default for BlobEncoder {
    /// Same as [`BlobEncoder::mainnet`].
    fn default() -> Self {
        Self::mainnet()
    }
}

/// Parameters shared by the encoding phases.
#[derive(Clone, Copy)]
struct Schedule {
    hasher: Hasher,
    engine: Engine,
    layout: ShardLayout,
    symbol_size: usize,
}

impl Schedule {
    /// Writes the `f` recovery symbols of every row of the message matrix to `recovery`, row
    /// major.
    fn encode_rows(self, matrix: &MessageMatrix<'_>, recovery: &mut [u8]) {
        let Schedule {
            engine,
            layout,
            symbol_size,
            ..
        } = self;
        recovery
            .par_chunks_mut(layout.n_faulty * symbol_size)
            .enumerate()
            .for_each_init(
                || Encoder::new(engine, layout.n_cols, layout.n_faulty, symbol_size),
                |encoder, (row, out)| {
                    let result = encoder.encode(matrix.row(row).chunks_exact(symbol_size));
                    out.chunks_exact_mut(symbol_size)
                        .zip(result.recovery_iter())
                        .for_each(|(out, symbol)| out.copy_from_slice(symbol));
                },
            );
    }

    /// Encodes column `c` of `source` for every `n`-leaf chunk `c` of `leaves` and writes the
    /// leaf hashes of all `n` symbols of the expanded column.
    fn encode_columns(self, source: Rows<'_>, leaves: &mut [Digest]) {
        let Schedule {
            hasher,
            engine,
            layout,
            symbol_size,
        } = self;
        let n = layout.n_shards;
        leaves.par_chunks_mut(n).enumerate().for_each_init(
            || {
                let encoder = Encoder::new(engine, layout.n_rows, n - layout.n_rows, symbol_size);
                (encoder, MessageBuf::default())
            },
            |(encoder, messages), (col, leaves)| {
                let mut symbols = messages.take();
                symbols.extend(source.column(col, symbol_size));
                let result = encoder.encode(symbols.iter().copied());
                symbols.extend(result.recovery_iter());
                hasher.hash_batch(LEAF_PREFIX, &symbols, leaves);
                messages.put(symbols);
            },
        );
    }
}

/// Merkle roots of the rows (primary slivers) and columns of the column-major leaf matrix.
fn sliver_roots(hasher: Hasher, leaves: &[Digest], n: usize) -> (Vec<Digest>, Vec<Digest>) {
    let mut row_roots = vec![ZERO_NODE; n];
    let mut column_roots = vec![ZERO_NODE; n];
    rayon::join(
        || {
            row_roots
                .par_chunks_mut(TREES_PER_TASK)
                .enumerate()
                .for_each_init(
                    || (TreeScratch::default(), Vec::new()),
                    |(scratch, rows), (task, roots)| {
                        let first = task * TREES_PER_TASK;
                        rows.clear();
                        rows.resize(roots.len() * n, ZERO_NODE);
                        for (col, column) in leaves.chunks_exact(n).enumerate() {
                            rows[col..]
                                .iter_mut()
                                .step_by(n)
                                .zip(&column[first..first + roots.len()])
                                .for_each(|(out, leaf)| *out = *leaf);
                        }
                        merkle::roots(hasher, rows, n, roots, scratch);
                    },
                );
        },
        || {
            column_roots
                .par_chunks_mut(TREES_PER_TASK)
                .zip(leaves.par_chunks(TREES_PER_TASK * n))
                .for_each_init(TreeScratch::default, |scratch, (roots, columns)| {
                    merkle::roots(hasher, columns, n, roots, scratch);
                });
        },
    );
    (row_roots, column_roots)
}

/// Builds the metadata from the row roots and column roots; sliver pair `i` holds row `i` and
/// column `n - 1 - i`.
fn finish(
    hasher: Hasher,
    row_roots: &[Digest],
    column_roots: &[Digest],
    unencoded_length: u64,
) -> BlobMetadata {
    let pairs: Vec<[Digest; 2]> = row_roots
        .iter()
        .zip(column_roots.iter().rev())
        .map(|(&primary, &secondary)| [primary, secondary])
        .collect();
    let messages: Vec<&[u8]> = pairs.iter().map(|pair| pair.as_flattened()).collect();
    let mut pair_leaves = vec![ZERO_NODE; pairs.len()];
    hasher.hash_batch(LEAF_PREFIX, &messages, &mut pair_leaves);

    let mut root = [ZERO_NODE];
    let mut scratch = TreeScratch::default();
    merkle::roots(hasher, &pair_leaves, pairs.len(), &mut root, &mut scratch);

    let encoding_type = EncodingType::RS2;
    let mut preimage = [0; 8 + DIGEST_LEN];
    preimage[..8].copy_from_slice(&unencoded_length.to_le_bytes());
    preimage[8..].copy_from_slice(&root[0]);
    let mut blob_id = [ZERO_NODE];
    hasher.hash_batch(encoding_type.into(), &[&preimage], &mut blob_id);

    BlobMetadata {
        blob_id: BlobId::from(blob_id[0]),
        encoding_type,
        unencoded_length,
        sliver_pairs: pairs
            .into_iter()
            .map(|[primary_hash, secondary_hash]| SliverPairMetadata {
                primary_hash,
                secondary_hash,
            })
            .collect(),
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
