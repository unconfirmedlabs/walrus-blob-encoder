// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Shard parameters, symbol size and the row-major view of the message matrix.

use std::num::NonZeroU16;

use crate::reed_solomon;

/// Largest RS2 symbol size: the largest even `u16`.
const MAX_SYMBOL_SIZE: u16 = u16::MAX - 1;
const SYMBOL_ALIGNMENT: u64 = 2;

/// Dimensions of the RS2 encoding for a number of shards `n` with `f = (n - 1) / 3`.
///
/// The message matrix has `n_rows = n - 2f` rows and `n_cols = n - f` columns. Rows are encoded
/// with `n_cols` source and `f` recovery symbols (secondary encoding), columns with `n_rows`
/// source and `n - n_rows` recovery symbols (primary encoding).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ShardLayout {
    pub(crate) n_shards: usize,
    pub(crate) n_faulty: usize,
    pub(crate) n_rows: usize,
    pub(crate) n_cols: usize,
}

impl ShardLayout {
    /// Returns the layout, or `None` if Reed-Solomon cannot encode one of the two axes.
    pub(crate) fn new(n_shards: NonZeroU16) -> Option<Self> {
        let n = usize::from(n_shards.get());
        let f = (n - 1) / 3;
        let layout = ShardLayout {
            n_shards: n,
            n_faulty: f,
            n_rows: n - 2 * f,
            n_cols: n - f,
        };
        (reed_solomon::supports(layout.n_cols, layout.n_faulty)
            && reed_solomon::supports(layout.n_rows, n - layout.n_rows))
        .then_some(layout)
    }

    pub(crate) fn n_source_symbols(&self) -> u64 {
        (self.n_rows * self.n_cols) as u64
    }

    pub(crate) fn max_blob_size(&self) -> u64 {
        self.n_source_symbols() * u64::from(MAX_SYMBOL_SIZE)
    }

    /// Symbol size for a blob of `len` bytes, or `None` if it exceeds [`Self::max_blob_size`].
    pub(crate) fn symbol_size(&self, len: u64) -> Option<NonZeroU16> {
        let size = len
            .max(1)
            .div_ceil(self.n_source_symbols())
            .next_multiple_of(SYMBOL_ALIGNMENT);
        NonZeroU16::new(u16::try_from(size).ok()?)
    }
}

/// A row-major matrix of equally sized rows stored in up to two consecutive buffers.
#[derive(Clone, Copy)]
pub(crate) struct Rows<'a> {
    parts: [&'a [u8]; 2],
    row_len: usize,
}

impl<'a> Rows<'a> {
    pub(crate) fn new(rows: &'a [u8], row_len: usize) -> Self {
        Rows {
            parts: [rows, &[]],
            row_len,
        }
    }

    pub(crate) fn iter(self) -> impl Iterator<Item = &'a [u8]> {
        let row_len = self.row_len;
        self.parts
            .into_iter()
            .flat_map(move |part| part.chunks_exact(row_len))
    }

    /// The `symbol_size`-byte symbols of column `col`, top to bottom.
    pub(crate) fn column(self, col: usize, symbol_size: usize) -> impl Iterator<Item = &'a [u8]> {
        let range = col * symbol_size..(col + 1) * symbol_size;
        self.iter().map(move |row| &row[range.clone()])
    }
}

/// The message matrix of a blob: `n_rows` rows of `n_cols` symbols, zero padded past the end.
///
/// The rows lying entirely inside the blob are borrowed; only the remaining rows are copied into
/// a padded tail. RS2 pads by less than `2 * n_rows * n_cols` bytes, so the tail stays small
/// (at most `2 * n_rows * n_cols + n_cols * symbol_size` bytes).
pub(crate) struct MessageMatrix<'a> {
    body: &'a [u8],
    tail: Vec<u8>,
    row_len: usize,
}

impl<'a> MessageMatrix<'a> {
    pub(crate) fn new(blob: &'a [u8], layout: &ShardLayout, symbol_size: usize) -> Self {
        let row_len = layout.n_cols * symbol_size;
        let n_body_rows = (blob.len() / row_len).min(layout.n_rows);
        let (body, rest) = blob.split_at(n_body_rows * row_len);
        let mut tail = vec![0; (layout.n_rows - n_body_rows) * row_len];
        tail[..rest.len()].copy_from_slice(rest);
        MessageMatrix {
            body,
            tail,
            row_len,
        }
    }

    pub(crate) fn rows(&self) -> Rows<'_> {
        Rows {
            parts: [self.body, &self.tail],
            row_len: self.row_len,
        }
    }

    pub(crate) fn row(&self, row: usize) -> &[u8] {
        let n_body_rows = self.body.len() / self.row_len;
        let (part, index) = if row < n_body_rows {
            (self.body, row)
        } else {
            (self.tail.as_slice(), row - n_body_rows)
        };
        &part[index * self.row_len..][..self.row_len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(n: u16) -> Option<ShardLayout> {
        ShardLayout::new(NonZeroU16::new(n).unwrap())
    }

    #[test]
    fn mainnet_parameters() {
        let layout = layout(1000).unwrap();
        assert_eq!(
            (layout.n_faulty, layout.n_rows, layout.n_cols),
            (333, 334, 667)
        );
        assert_eq!(layout.n_source_symbols(), 222_778);
        assert_eq!(layout.max_blob_size(), 222_778 * 65_534);
    }

    #[test]
    fn small_parameters() {
        for (n, f, rows, cols) in [(4, 1, 2, 3), (5, 1, 3, 4), (6, 1, 4, 5), (7, 2, 3, 5)] {
            let layout = layout(n).unwrap();
            assert_eq!(
                (layout.n_faulty, layout.n_rows, layout.n_cols),
                (f, rows, cols)
            );
        }
    }

    #[test]
    fn supported_shard_counts() {
        for n in 1..=3 {
            assert!(layout(n).is_none(), "n = {n}");
        }
        for n in [4, 5, 1000, 49_155] {
            assert!(layout(n).is_some(), "n = {n}");
        }
        for n in [49_156, 50_000, u16::MAX] {
            assert!(layout(n).is_none(), "n = {n}");
        }
    }

    #[test]
    fn symbol_size() {
        let layout = layout(1000).unwrap();
        let n = 222_778;
        for (len, size) in [
            (0, 2),
            (1, 2),
            (n, 2),
            (2 * n, 2),
            (2 * n + 1, 4),
            (3 * n, 4),
            (4 * n, 4),
            (4 * n + 1, 6),
            (layout.max_blob_size(), 65_534),
        ] {
            assert_eq!(layout.symbol_size(len).unwrap().get(), size, "len = {len}");
        }
        assert_eq!(layout.symbol_size(layout.max_blob_size() + 1), None);
        assert_eq!(layout.symbol_size(u64::MAX), None);
    }

    #[test]
    fn message_matrix_pads_with_zeros() {
        let layout = layout(4).unwrap();
        let blob: Vec<u8> = (1..=9).collect();
        let matrix = MessageMatrix::new(&blob, &layout, 2);
        let rows: Vec<&[u8]> = matrix.rows().iter().collect();
        assert_eq!(rows, [&[1, 2, 3, 4, 5, 6][..], &[7, 8, 9, 0, 0, 0]]);
        assert_eq!(matrix.row(1), rows[1]);
        let column: Vec<&[u8]> = matrix.rows().column(1, 2).collect();
        assert_eq!(column, [&[3, 4][..], &[9, 0]]);
        assert_eq!(matrix.body.len(), 6);
    }

    #[test]
    fn message_matrix_of_empty_blob() {
        let layout = layout(1000).unwrap();
        let matrix = MessageMatrix::new(&[], &layout, 2);
        assert!(matrix.rows().iter().all(|row| row == [0; 1334]));
        assert_eq!(matrix.rows().iter().count(), 334);
    }
}
