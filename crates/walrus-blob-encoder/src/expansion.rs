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
//!
//! With the `rayon` feature, [`Expansion::encode_rows_parallel`] and
//! [`Expansion::push_columns_parallel`] spread the same work over the current rayon pool: rows
//! and columns are independent, and the row trees are split into groups of rows that receive
//! each group of columns' leaves in order. The result does not depend on the number of threads.

use std::{ops::Range, slice};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::{
    BlobId, BlobMetadata, EncodingType, SliverPairMetadata,
    blake2b::{DIGEST_LEN, Digest, Hasher},
    layout::{Rows, ShardLayout},
    merkle::{self, Forest, LEAF_PREFIX, TreeScratch},
    messages::MessageBuf,
    reed_solomon::{Encoder, Engine},
};

const ZERO_NODE: Digest = [0; DIGEST_LEN];

/// Columns encoded in parallel before their leaves are appended to the row trees: the leaves of
/// a group take `32 * n * COLUMN_GROUP` bytes (8 MB for 1000 shards).
#[cfg(feature = "rayon")]
const COLUMN_GROUP: usize = 256;

/// Rows per tree group in parallel mode: few enough for many tasks, enough for full SIMD lanes.
#[cfg(feature = "rayon")]
const ROWS_PER_FOREST: usize = 64;

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

/// What one thread needs to encode rows and columns: one Reed-Solomon encoder serves both
/// encodings, since the work space has the same size for both.
struct Worker {
    hasher: Hasher,
    layout: ShardLayout,
    symbol_size: usize,
    encoder: Encoder,
    shape: Shape,
    messages: MessageBuf,
    scratch: TreeScratch,
    clock: Option<fn() -> f64>,
    timings: Timings,
}

impl Worker {
    fn new(hasher: Hasher, engine: Engine, layout: ShardLayout, symbol_size: usize) -> Self {
        let rows = layout.n_rows;
        let limit = layout.skew_limit();
        Worker {
            hasher,
            layout,
            symbol_size,
            encoder: Encoder::new(engine, rows, layout.n_shards - rows, symbol_size, limit),
            shape: Shape::Column,
            messages: MessageBuf::default(),
            scratch: TreeScratch::default(),
            clock: None,
            timings: Timings::default(),
        }
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
            let s = self.symbol_size;
            match shape {
                Shape::Row => self.encoder.reset(n_cols, n_faulty, s),
                Shape::Column => self.encoder.reset(n_rows, n_shards - n_rows, s),
            }
            self.shape = shape;
        }
    }

    /// See [`Expansion::encode_row`].
    fn encode_row(&mut self, row: &[u8], recovery: Range<usize>, out: &mut [u8]) {
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
        self.timings.row_encoding += self.now() - start;
    }

    /// Encodes a column from its `n_rows` source symbols, writes the leaf hashes of its `n`
    /// symbols to `leaves`, and returns its root.
    fn encode_column<'a>(
        &mut self,
        source: impl IntoIterator<Item = &'a [u8]>,
        leaves: &mut [Digest],
    ) -> Digest {
        let start = self.now();
        self.set_shape(Shape::Column);
        let mut symbols = self.messages.take();
        for symbol in source {
            symbols.push(symbol);
        }
        let result = self.encoder.encode(symbols.iter().copied());
        let encoded = self.clock.map_or(0.0, |clock| clock());
        symbols.extend(result.recovery_iter());
        self.hasher.hash_batch(LEAF_PREFIX, &symbols, leaves);
        self.messages.put(symbols);
        drop(result);

        let mut root = ZERO_NODE;
        let n = self.layout.n_shards;
        let roots = slice::from_mut(&mut root);
        merkle::roots(self.hasher, leaves, n, roots, &mut self.scratch);
        self.timings.column_encoding += encoded - start;
        self.timings.hashing += self.now() - encoded;
        root
    }
}

/// The state of a blob's expanded matrix while its columns are appended.
pub(crate) struct Expansion {
    hasher: Hasher,
    #[cfg(feature = "rayon")]
    engine: Engine,
    layout: ShardLayout,
    symbol_size: usize,
    worker: Worker,
    /// The row trees, in groups of consecutive rows.
    forests: Vec<(Range<usize>, Forest)>,
    column_roots: Vec<Digest>,
    leaves: Vec<Digest>,
    row_encodes: u64,
}

impl Expansion {
    pub(crate) fn new(
        hasher: Hasher,
        engine: Engine,
        layout: ShardLayout,
        symbol_size: usize,
    ) -> Self {
        let n = layout.n_shards;
        Expansion {
            hasher,
            #[cfg(feature = "rayon")]
            engine,
            layout,
            symbol_size,
            worker: Worker::new(hasher, engine, layout, symbol_size),
            forests: vec![(0..n, Forest::new(n, n))],
            column_roots: Vec::with_capacity(n),
            leaves: vec![ZERO_NODE; n],
            row_encodes: 0,
        }
    }

    pub(crate) fn set_clock(&mut self, clock: fn() -> f64) {
        self.worker.clock = Some(clock);
    }

    /// Encodes a row of the message matrix, `n_cols` symbols, and writes its recovery symbols
    /// `recovery` (indices into its `n_faulty` recovery symbols) to `out`, back to back.
    pub(crate) fn encode_row(&mut self, row: &[u8], recovery: Range<usize>, out: &mut [u8]) {
        self.worker.encode_row(row, recovery, out);
        self.row_encodes += 1;
    }

    /// Appends the next column of the expanded matrix, given its `n_rows` source symbols.
    pub(crate) fn push_column<'a>(&mut self, source: impl IntoIterator<Item = &'a [u8]>) {
        let root = self.worker.encode_column(source, &mut self.leaves);
        self.column_roots.push(root);
        let start = self.worker.now();
        let n = self.layout.n_shards;
        for (rows, forest) in &mut self.forests {
            forest.push(self.hasher, &self.leaves[rows.start..rows.end.min(n)]);
        }
        self.worker.timings.hashing += self.worker.now() - start;
    }

    /// The number of columns appended.
    pub(crate) fn columns(&self) -> usize {
        self.column_roots.len()
    }

    pub(crate) fn row_encodes(&self) -> u64 {
        self.row_encodes
    }

    pub(crate) fn timings(&self) -> Timings {
        self.worker.timings
    }

    /// The metadata of a blob of `unencoded_length` bytes, once all `n` columns are appended.
    ///
    /// # Panics
    ///
    /// If columns are missing.
    pub(crate) fn finish(self, unencoded_length: u64) -> BlobMetadata {
        let mut row_roots = vec![ZERO_NODE; self.layout.n_shards];
        for (rows, forest) in self.forests {
            forest.finish(self.hasher, &mut row_roots[rows]);
        }
        metadata(
            self.hasher,
            &row_roots,
            &self.column_roots,
            unencoded_length,
        )
    }
}

#[cfg(feature = "rayon")]
impl Expansion {
    fn new_worker(&self) -> Worker {
        Worker::new(self.hasher, self.engine, self.layout, self.symbol_size)
    }

    /// [`Self::encode_row`] for every row of `rows`, writing all `n_faulty` recovery symbols of
    /// row `r` to row `r` of `out`, on the current rayon pool.
    pub(crate) fn encode_rows_parallel<'a>(
        &mut self,
        rows: impl Fn(usize) -> &'a [u8] + Sync,
        out: &mut [u8],
    ) {
        let f = self.layout.n_faulty;
        let row_len = f * self.symbol_size;
        out.par_chunks_mut(row_len).enumerate().for_each_init(
            || self.new_worker(),
            |worker, (row, out)| worker.encode_row(rows(row), 0..f, out),
        );
        self.row_encodes += (out.len() / row_len) as u64;
    }

    /// Appends the next `count` columns, column `c` of them given by `source(c)`, on the
    /// current rayon pool.
    pub(crate) fn push_columns_parallel<'a, I>(
        &mut self,
        count: usize,
        source: impl Fn(usize) -> I + Sync,
    ) where
        I: IntoIterator<Item = &'a [u8]>,
    {
        let n = self.layout.n_shards;
        if self.forests.len() == 1 && n > ROWS_PER_FOREST {
            assert_eq!(
                self.columns(),
                0,
                "the row trees are split before any column"
            );
            self.forests = (0..n)
                .step_by(ROWS_PER_FOREST)
                .map(|start| {
                    let rows = start..(start + ROWS_PER_FOREST).min(n);
                    (rows.clone(), Forest::new(rows.len(), n))
                })
                .collect();
        }
        let mut leaves = vec![ZERO_NODE; n * COLUMN_GROUP.min(count)];
        let mut roots = vec![ZERO_NODE; COLUMN_GROUP.min(count)];
        for first in (0..count).step_by(COLUMN_GROUP) {
            let group = COLUMN_GROUP.min(count - first);
            let (leaves, roots) = (&mut leaves[..group * n], &mut roots[..group]);
            leaves
                .par_chunks_mut(n)
                .zip(roots.par_iter_mut())
                .enumerate()
                .for_each_init(
                    || self.new_worker(),
                    |worker, (column, (leaves, root))| {
                        *root = worker.encode_column(source(first + column), leaves);
                    },
                );
            self.column_roots.extend_from_slice(roots);
            let hasher = self.hasher;
            let leaves = &*leaves;
            self.forests.par_iter_mut().for_each(|(rows, forest)| {
                for column in leaves.chunks_exact(n) {
                    forest.push(hasher, &column[rows.clone()]);
                }
            });
        }
    }
}

/// The metadata of a blob held in memory as the message matrix `rows`, whose row `r` is
/// `row(r)`: the systematic columns are read in place, and the only buffer is the
/// `n_rows * n_faulty * s` bytes of the rows' recovery symbols. With `parallel` (and the `rayon`
/// feature), the work runs on the current rayon pool.
pub(crate) fn in_memory<'a>(
    mut expansion: Expansion,
    rows: Rows<'a>,
    row: impl Fn(usize) -> &'a [u8] + Sync,
    unencoded_length: u64,
    parallel: bool,
) -> BlobMetadata {
    let ShardLayout {
        n_rows,
        n_cols,
        n_faulty,
        ..
    } = expansion.layout;
    let s = expansion.symbol_size;
    let mut recovery = vec![0; n_rows * n_faulty * s];
    #[cfg(feature = "rayon")]
    if parallel {
        expansion.push_columns_parallel(n_cols, |col| rows.column(col, s));
        expansion.encode_rows_parallel(&row, &mut recovery);
        let recovery_rows = Rows::new(&recovery, n_faulty * s);
        expansion.push_columns_parallel(n_faulty, |col| recovery_rows.column(col, s));
        return expansion.finish(unencoded_length);
    }
    let _ = parallel;
    for col in 0..n_cols {
        expansion.push_column(rows.column(col, s));
    }
    for (r, out) in recovery.chunks_exact_mut(n_faulty * s).enumerate() {
        expansion.encode_row(row(r), 0..n_faulty, out);
    }
    let recovery_rows = Rows::new(&recovery, n_faulty * s);
    for col in 0..n_faulty {
        expansion.push_column(recovery_rows.column(col, s));
    }
    expansion.finish(unencoded_length)
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
