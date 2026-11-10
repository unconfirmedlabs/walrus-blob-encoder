// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The expanded matrix of a blob, hashed column by column.
//!
//! For `n` shards the expanded matrix is `n x n` symbols: rows `0..n_rows` are the rows of the
//! message matrix expanded from `n_cols` to `n` symbols (secondary encoding), and every column is
//! expanded from its first `n_rows` symbols to `n` (primary encoding). Columns are appended in
//! ascending order from their `n_rows` source symbols; each is encoded, its `n` leaves are hashed
//! in one batch, its root is built at once, and its leaves are appended to the row trees, which
//! all grow together in a [`Forest`]. No leaf matrix is kept.

use std::{ops::Range, slice};

use crate::{
    BlobId, BlobMetadata, EncodingType, SliverPairMetadata,
    blake2b::{DIGEST_LEN, Digest, Hasher},
    layout::ShardLayout,
    merkle::{self, Forest, LEAF_PREFIX, TreeScratch},
    messages::MessageBuf,
    reed_solomon::{Encoder, Engine},
};

const ZERO_NODE: Digest = [0; DIGEST_LEN];

/// Milliseconds spent per phase, measured with the optional clock.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Timings {
    pub(crate) row_encoding: f64,
    pub(crate) column_encoding: f64,
    pub(crate) hashing: f64,
}

/// Which encoding the Reed-Solomon encoder is set up for.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    Row,
    Column,
}

/// The state of a blob's expanded matrix while its columns are appended.
pub(crate) struct Expansion {
    hasher: Hasher,
    layout: ShardLayout,
    symbol_size: usize,
    encoder: Encoder,
    shape: Shape,
    forest: Forest,
    column_roots: Vec<Digest>,
    leaves: Vec<Digest>,
    messages: MessageBuf,
    scratch: TreeScratch,
    row_encodes: u64,
    clock: Option<fn() -> f64>,
    timings: Timings,
}

impl Expansion {
    /// One Reed-Solomon encoder serves both encodings: the work space has the same size for
    /// both, and rows and columns are never encoded at the same time.
    pub(crate) fn new(
        hasher: Hasher,
        engine: Engine,
        layout: ShardLayout,
        symbol_size: usize,
        clock: Option<fn() -> f64>,
    ) -> Self {
        let n = layout.n_shards;
        Expansion {
            hasher,
            layout,
            symbol_size,
            encoder: Encoder::new(
                engine,
                layout.n_rows,
                n - layout.n_rows,
                symbol_size,
                layout.skew_limit(),
            ),
            shape: Shape::Column,
            forest: Forest::new(n, n),
            column_roots: Vec::with_capacity(n),
            leaves: vec![ZERO_NODE; n],
            messages: MessageBuf::default(),
            scratch: TreeScratch::default(),
            row_encodes: 0,
            clock,
            timings: Timings::default(),
        }
    }

    pub(crate) fn set_clock(&mut self, clock: fn() -> f64) {
        self.clock = Some(clock);
    }

    fn now(&self) -> f64 {
        self.clock.map_or(0.0, |clock| clock())
    }

    fn set_shape(&mut self, shape: Shape) {
        if self.shape != shape {
            let ShardLayout {
                n_shards,
                n_faulty,
                n_rows,
                n_cols,
            } = self.layout;
            match shape {
                Shape::Row => self.encoder.reset(n_cols, n_faulty, self.symbol_size),
                Shape::Column => self
                    .encoder
                    .reset(n_rows, n_shards - n_rows, self.symbol_size),
            }
            self.shape = shape;
        }
    }

    /// Encodes a row of the message matrix, `n_cols` symbols, and writes its recovery symbols
    /// `recovery` (indices into its `n_faulty` recovery symbols) to `out`, back to back.
    pub(crate) fn encode_row(&mut self, row: &[u8], recovery: Range<usize>, out: &mut [u8]) {
        let start = self.now();
        self.set_shape(Shape::Row);
        let s = self.symbol_size;
        let result = self.encoder.encode(row.chunks_exact(s));
        for (out, symbol) in out
            .chunks_exact_mut(s)
            .zip(result.recovery_iter().skip(recovery.start))
            .take(recovery.len())
        {
            out.copy_from_slice(symbol);
        }
        drop(result);
        self.row_encodes += 1;
        self.timings.row_encoding += self.now() - start;
    }

    /// Appends the next column of the expanded matrix, given its `n_rows` source symbols.
    pub(crate) fn push_column<'a>(&mut self, source: impl IntoIterator<Item = &'a [u8]>) {
        let start = self.now();
        self.set_shape(Shape::Column);
        let mut symbols = self.messages.take();
        for symbol in source {
            symbols.push(symbol);
        }
        let result = self.encoder.encode(symbols.iter().copied());
        let encoded = self.clock.map_or(0.0, |clock| clock());
        symbols.extend(result.recovery_iter());
        self.hasher
            .hash_batch(LEAF_PREFIX, &symbols, &mut self.leaves);
        self.messages.put(symbols);
        drop(result);

        let mut root = ZERO_NODE;
        let n = self.layout.n_shards;
        merkle::roots(
            self.hasher,
            &self.leaves,
            n,
            slice::from_mut(&mut root),
            &mut self.scratch,
        );
        self.column_roots.push(root);
        self.forest.push(self.hasher, &self.leaves);
        self.timings.column_encoding += encoded - start;
        self.timings.hashing += self.now() - encoded;
    }

    /// The number of columns appended.
    pub(crate) fn columns(&self) -> usize {
        self.column_roots.len()
    }

    pub(crate) fn row_encodes(&self) -> u64 {
        self.row_encodes
    }

    pub(crate) fn timings(&self) -> Timings {
        self.timings
    }

    /// The metadata of a blob of `unencoded_length` bytes, once all `n` columns are appended.
    ///
    /// # Panics
    ///
    /// If columns are missing.
    pub(crate) fn finish(self, unencoded_length: u64) -> BlobMetadata {
        let mut row_roots = vec![ZERO_NODE; self.layout.n_shards];
        self.forest.finish(self.hasher, &mut row_roots);
        metadata(
            self.hasher,
            &row_roots,
            &self.column_roots,
            unencoded_length,
        )
    }
}

/// Builds the metadata from the row roots and column roots; sliver pair `i` holds row `i` and
/// column `n - 1 - i`.
pub(crate) fn metadata(
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
