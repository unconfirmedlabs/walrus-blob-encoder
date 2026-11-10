// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Blob metadata from a seekable source of known length, in bounded memory.
//!
//! The expanded matrix is walked in batches of columns, in ascending column order (see
//! [`Expansion`]). A batch is held in a buffer of `n_rows * k * s` bytes, at most the memory
//! budget for `k` columns of symbol size `s` (at least one column):
//!
//! - a batch of systematic columns (below `n_cols`) is read with `n_rows` strided reads of
//!   `k * s` bytes, one per row;
//! - a batch of parity columns needs the recovery symbols of every row, so it is a sequential pass
//!   over the whole blob that encodes each row and keeps the batch's slice of its recovery
//!   symbols.
//!
//! Systematic batches stop at column `n_cols`, so parity batches start exactly there, and the
//! `n - n_cols` parity columns are split into the fewest passes the budget allows, balanced:
//! with `P = ceil((n - n_cols) / k)` passes, each holds `ceil((n - n_cols) / P)` columns. A budget
//! for all `n` columns gives a single pass that reads every row once.
//!
//! [`StreamEncoder`] is a pull state machine, so the reads can be served asynchronously: the
//! caller asks for the [next read](StreamEncoder::next_read), fills its
//! [destination](StreamEncoder::dest), and [completes](StreamEncoder::complete_read) it, which
//! runs all computation up to the next read.

use std::{
    error, fmt,
    io::{Read, Seek, SeekFrom},
};

use crate::{BlobEncoder, BlobMetadata, Error, expansion::Expansion, layout::ShardLayout};

/// A read the [`StreamEncoder`] needs: `len` bytes at `offset`, always within the blob.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadRequest {
    /// Offset of the first byte in the blob.
    pub offset: u64,
    /// The number of bytes, the length of [`StreamEncoder::dest`].
    pub len: usize,
}

/// Counters of a [`StreamEncoder`], and phase times if it has a clock.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StreamStats {
    /// The symbol size in bytes.
    pub symbol_size: usize,
    /// Columns of the widest batch.
    pub batch_columns: usize,
    /// Size of the batch buffer in bytes.
    pub batch_bytes: usize,
    /// Size of the row buffer of parity passes in bytes (zero for a single pass).
    pub row_buffer_bytes: usize,
    /// Batches of systematic columns, read with strided reads.
    pub systematic_batches: usize,
    /// Sequential passes over the whole blob, one per batch with parity columns.
    pub full_passes: usize,
    /// Bytes requested so far.
    pub bytes_read: u64,
    /// Reads completed so far, empty reads past the end of the blob excluded.
    pub read_calls: u64,
    /// Rows encoded so far.
    pub row_encodes: u64,
    /// Columns encoded so far.
    pub column_encodes: u64,
    /// Milliseconds encoding rows.
    pub row_encoding_ms: f64,
    /// Milliseconds encoding columns.
    pub column_encoding_ms: f64,
    /// Milliseconds hashing leaves and inner nodes.
    pub hashing_ms: f64,
}

/// A batch of columns `start..end`, either read with strided reads (systematic) or computed in a
/// full pass over the rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Batch {
    start: usize,
    end: usize,
    pass: bool,
}

impl Batch {
    fn width(self) -> usize {
        self.end - self.start
    }
}

/// The batches for `layout`, symbol size `s` and a budget of `budget` bytes.
fn plan(layout: &ShardLayout, s: usize, budget: usize) -> Vec<Batch> {
    let n = layout.n_shards;
    let k = (budget / (layout.n_rows * s)).clamp(1, n);
    if k == n {
        return vec![Batch {
            start: 0,
            end: n,
            pass: true,
        }];
    }
    let mut batches = Vec::new();
    for (columns, pass) in [(0..layout.n_cols, false), (layout.n_cols..n, true)] {
        let count = columns.len().div_ceil(k);
        let width = columns.len().div_ceil(count);
        batches.extend(columns.clone().step_by(width).map(|start| Batch {
            start,
            end: (start + width).min(columns.end),
            pass,
        }));
    }
    batches
}

/// Computes blob metadata from reads the caller performs; see the [module](self) documentation.
///
/// ```
/// use walrus_blob_encoder::BlobEncoder;
///
/// let blob = vec![7u8; 100_000];
/// let encoder = BlobEncoder::mainnet();
/// let mut stream = encoder.stream(blob.len() as u64, 1 << 20)?;
/// while let Some(read) = stream.next_read() {
///     let offset = read.offset as usize;
///     stream.dest().copy_from_slice(&blob[offset..offset + read.len]);
///     stream.complete_read();
/// }
/// assert_eq!(stream.finish(), encoder.metadata(&blob)?);
/// # Ok::<(), walrus_blob_encoder::Error>(())
/// ```
pub struct StreamEncoder {
    len: u64,
    layout: ShardLayout,
    symbol_size: usize,
    expansion: Expansion,
    batches: Vec<Batch>,
    batch: usize,
    row: usize,
    buffer: Vec<u8>,
    row_buffer: Vec<u8>,
    stats: StreamStats,
}

impl fmt::Debug for StreamEncoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamEncoder")
            .field("len", &self.len)
            .field("next_read", &self.next_read())
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

impl StreamEncoder {
    pub(crate) fn new(encoder: &BlobEncoder, len: u64, budget: usize) -> Result<Self, Error> {
        let symbol_size = usize::from(encoder.symbol_size(len)?.get());
        let layout = encoder.layout();
        let batches = plan(&layout, symbol_size, budget);
        let widest = batches.iter().map(|batch| batch.width()).max();
        let batch_columns = widest.expect("at least one batch");
        let separate_rows = batches.iter().any(|batch| batch.pass && batch.start > 0);
        let row_len = layout.n_cols * symbol_size;
        let mut stream = StreamEncoder {
            len,
            layout,
            symbol_size,
            expansion: Expansion::new(encoder.hasher(), encoder.engine(), layout, symbol_size),
            stats: StreamStats {
                symbol_size,
                batch_columns,
                batch_bytes: layout.n_rows * batch_columns * symbol_size,
                row_buffer_bytes: if separate_rows { row_len } else { 0 },
                ..StreamStats::default()
            },
            batches,
            batch: 0,
            row: 0,
            buffer: vec![0; layout.n_rows * batch_columns * symbol_size],
            row_buffer: vec![0; if separate_rows { row_len } else { 0 }],
        };
        stream.skip_empty_reads();
        Ok(stream)
    }

    /// Measures the time of each phase with `clock`, in milliseconds, for [`Self::stats`].
    pub fn with_clock(mut self, clock: fn() -> f64) -> Self {
        self.expansion.set_clock(clock);
        self
    }

    /// The counters so far.
    pub fn stats(&self) -> StreamStats {
        let timings = self.expansion.timings();
        StreamStats {
            row_encodes: self.expansion.row_encodes(),
            column_encodes: self.expansion.columns() as u64,
            row_encoding_ms: timings.row_encoding,
            column_encoding_ms: timings.column_encoding,
            hashing_ms: timings.hashing,
            ..self.stats
        }
    }

    /// The length of the blob.
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Whether the blob is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether every read is complete, so that [`Self::finish`] can be called.
    pub fn is_done(&self) -> bool {
        self.batch == self.batches.len()
    }

    /// Offset and padded length of the current read, and whether it is in the row buffer.
    fn current(&self) -> (u64, usize) {
        let batch = self.batches[self.batch];
        let s = self.symbol_size as u64;
        let row_start = self.row as u64 * self.layout.n_cols as u64 * s;
        if batch.pass {
            (row_start, self.layout.n_cols * self.symbol_size)
        } else {
            (
                row_start + batch.start as u64 * s,
                batch.width() * self.symbol_size,
            )
        }
    }

    fn valid_len(&self) -> usize {
        let (offset, full) = self.current();
        // The minimum is at most `full`, a `usize`.
        self.len.saturating_sub(offset).min(full as u64) as usize
    }

    /// The next read to perform, or `None` once all reads are complete.
    pub fn next_read(&self) -> Option<ReadRequest> {
        (!self.is_done()).then(|| ReadRequest {
            offset: self.current().0,
            len: self.valid_len(),
        })
    }

    /// The whole destination of the current read, including the zero padding past the blob.
    fn dest_padded(&mut self) -> &mut [u8] {
        let (_, full) = self.current();
        let batch = self.batches[self.batch];
        if batch.pass && batch.start > 0 {
            &mut self.row_buffer[..full]
        } else {
            let start = self.row * batch.width() * self.symbol_size;
            &mut self.buffer[start..start + full]
        }
    }

    /// The destination of the pending read, exactly [`ReadRequest::len`] bytes, to be filled with
    /// the blob's bytes at [`ReadRequest::offset`].
    ///
    /// # Panics
    ///
    /// If all reads are complete.
    pub fn dest(&mut self) -> &mut [u8] {
        assert!(!self.is_done(), "no pending read");
        let valid = self.valid_len();
        &mut self.dest_padded()[..valid]
    }

    /// Marks the pending read as filled and runs all computation up to the next read.
    ///
    /// # Panics
    ///
    /// If all reads are complete.
    pub fn complete_read(&mut self) {
        assert!(!self.is_done(), "no pending read");
        self.stats.bytes_read += self.valid_len() as u64;
        self.stats.read_calls += 1;
        self.process_read();
        self.skip_empty_reads();
    }

    /// Copies `data` to the destination and completes the read.
    ///
    /// # Panics
    ///
    /// If all reads are complete or `data` is not exactly [`ReadRequest::len`] bytes.
    pub fn supply(&mut self, data: &[u8]) {
        self.dest().copy_from_slice(data);
        self.complete_read();
    }

    /// Completes reads that lie entirely past the end of the blob, which are all zeros.
    fn skip_empty_reads(&mut self) {
        while !self.is_done() && self.valid_len() == 0 {
            self.process_read();
        }
    }

    fn process_read(&mut self) {
        let valid = self.valid_len();
        self.dest_padded()[valid..].fill(0);
        let batch = self.batches[self.batch];
        if batch.pass {
            self.encode_row(batch);
        }
        self.row += 1;
        if self.row == self.layout.n_rows {
            if batch.pass {
                self.stats.full_passes += 1;
            } else {
                self.stats.systematic_batches += 1;
            }
            self.push_columns(batch);
            self.batch += 1;
            self.row = 0;
        }
    }

    /// The current row is complete: keep the batch's systematic and recovery symbols of it.
    fn encode_row(&mut self, batch: Batch) {
        let (s, cols) = (self.symbol_size, self.layout.n_cols);
        let row_start = self.row * batch.width() * s;
        let first_recovery = batch.start.max(cols);
        let recovery = first_recovery - cols..batch.end - cols;
        let out_start = row_start + (first_recovery - batch.start) * s;
        let out_len = recovery.len() * s;
        if batch.start == 0 {
            let (row, out) = self.buffer.split_at_mut(out_start);
            let row = &row[row_start..];
            self.expansion
                .encode_row(row, recovery, &mut out[..out_len]);
        } else {
            let out = &mut self.buffer[out_start..out_start + out_len];
            self.expansion.encode_row(&self.row_buffer, recovery, out);
        }
    }

    /// The batch buffer holds every row of the batch's columns: append them.
    fn push_columns(&mut self, batch: Batch) {
        let (s, width) = (self.symbol_size, batch.width());
        let buffer = &self.buffer;
        for column in 0..width {
            let symbols = (0..self.layout.n_rows).map(|row| {
                let start = (row * width + column) * s;
                &buffer[start..start + s]
            });
            self.expansion.push_column(symbols);
        }
    }

    /// The metadata of the blob.
    ///
    /// # Panics
    ///
    /// If reads are pending (see [`Self::is_done`]).
    pub fn finish(self) -> BlobMetadata {
        assert!(self.is_done(), "reads are pending");
        self.expansion.finish(self.len)
    }
}

/// A seekable byte source of known length.
pub trait Source {
    /// The error of a failed read.
    type Error;

    /// The length of the source in bytes.
    fn len(&self) -> u64;

    /// Whether the source is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Fills `buf` with the bytes at `offset..offset + buf.len()`, which lie within the source.
    ///
    /// # Errors
    ///
    /// Any error of the underlying source.
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<(), Self::Error>;
}

impl Source for &[u8] {
    type Error = std::convert::Infallible;

    fn len(&self) -> u64 {
        <[u8]>::len(self) as u64
    }

    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<(), Self::Error> {
        let start = usize::try_from(offset).expect("offset within the slice");
        buf.copy_from_slice(&self[start..start + buf.len()]);
        Ok(())
    }
}

/// A [`Source`] over a reader that can seek, such as a file, of a given length.
#[derive(Debug)]
pub struct SeekSource<R> {
    reader: R,
    len: u64,
}

impl<R: Read + Seek> SeekSource<R> {
    /// A source of the first `len` bytes of `reader`.
    pub fn new(reader: R, len: u64) -> Self {
        SeekSource { reader, len }
    }

    /// The reader.
    pub fn into_inner(self) -> R {
        self.reader
    }
}

impl<R: Read + Seek> Source for SeekSource<R> {
    type Error = std::io::Error;

    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<(), Self::Error> {
        self.reader.seek(SeekFrom::Start(offset))?;
        self.reader.read_exact(buf)
    }
}

/// The error of [`BlobEncoder::metadata_from_source`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceError<E> {
    /// The blob cannot be encoded.
    Encoding(Error),
    /// A read of the source failed.
    Read(E),
}

impl<E: fmt::Display> fmt::Display for SourceError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SourceError::Encoding(error) => error.fmt(f),
            SourceError::Read(error) => write!(f, "read failed: {error}"),
        }
    }
}

impl<E: error::Error + 'static> error::Error for SourceError<E> {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            SourceError::Encoding(error) => Some(error),
            SourceError::Read(error) => Some(error),
        }
    }
}

impl BlobEncoder {
    /// A [`StreamEncoder`] for a blob of `len` bytes, with a buffer of at most `budget` bytes
    /// for its batches of columns (at least one column, `n_rows * s` bytes for symbol size `s`).
    ///
    /// Besides the batch buffer, it allocates for symbol size `s` and `n` shards about
    /// `next_power_of_two(n - n_rows) * s` bytes of Reed-Solomon work space (rounded to 64-byte
    /// chunks per symbol), one row of `n_cols * s` bytes if parity columns need more than one
    /// pass, and `64 * n * (log2(n) + 2)` bytes of Merkle state; see
    /// [`StreamStats`] for the actual buffer sizes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DataTooLarge`] if `len` exceeds [`Self::max_blob_size`].
    pub fn stream(&self, len: u64, budget: usize) -> Result<StreamEncoder, Error> {
        StreamEncoder::new(self, len, budget)
    }

    /// Computes the metadata of the blob in `source`, reading it through a [`StreamEncoder`]
    /// with a batch budget of `budget` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`SourceError::Encoding`] if the source is too long, and [`SourceError::Read`] if
    /// a read fails.
    pub fn metadata_from_source<S: Source>(
        &self,
        source: &mut S,
        budget: usize,
    ) -> Result<BlobMetadata, SourceError<S::Error>> {
        let mut stream = self
            .stream(source.len(), budget)
            .map_err(SourceError::Encoding)?;
        while let Some(read) = stream.next_read() {
            source
                .read_at(read.offset, stream.dest())
                .map_err(SourceError::Read)?;
            stream.complete_read();
        }
        Ok(stream.finish())
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU16;

    use super::*;

    fn mainnet() -> ShardLayout {
        ShardLayout::new(NonZeroU16::new(1000).unwrap()).unwrap()
    }

    fn passes(batches: &[Batch]) -> usize {
        batches.iter().filter(|batch| batch.pass).count()
    }

    #[test]
    fn batches_cover_all_columns_in_order() {
        for n in [4, 7, 10, 100, 1000] {
            let layout = ShardLayout::new(NonZeroU16::new(n).unwrap()).unwrap();
            for budget in [0, 1, 2 * layout.n_rows, 1000, 1 << 20, usize::MAX] {
                let batches = plan(&layout, 2, budget);
                assert_eq!(batches[0].start, 0);
                assert_eq!(batches.last().unwrap().end, layout.n_shards);
                for pair in batches.windows(2) {
                    assert_eq!(pair[0].end, pair[1].start);
                }
                for batch in &batches {
                    assert!(batch.start < batch.end);
                    assert!(batch.pass || batch.end <= layout.n_cols);
                    assert!(!batch.pass || batch.start >= layout.n_cols || batches.len() == 1);
                }
            }
        }
    }

    #[test]
    fn parity_passes_are_balanced() {
        let layout = mainnet();
        let columns = |k: usize| k * layout.n_rows * 2;
        // 100 MB at 32 MiB: symbol size 450, 223 columns per batch.
        let batches = plan(&layout, 450, 32 << 20);
        assert_eq!(batches.len(), 3 + 2);
        assert_eq!(passes(&batches), 2);
        assert_eq!(
            batches[3],
            Batch {
                start: 667,
                end: 834,
                pass: true
            }
        );
        // 1 GB at 256 MiB: symbol size 4490, 178 columns per batch.
        assert_eq!(passes(&plan(&layout, 4490, 256 << 20)), 2);
        assert_eq!(passes(&plan(&layout, 4490, 32 << 20)), 16);
        assert_eq!(passes(&plan(&layout, 2, columns(332))), 2);
        assert_eq!(passes(&plan(&layout, 2, columns(333))), 1);
        assert_eq!(plan(&layout, 2, columns(999)).len(), 2);
        assert_eq!(plan(&layout, 2, columns(1000)).len(), 1);
    }
}
