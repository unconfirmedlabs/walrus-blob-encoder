// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! A raw `extern "C"` WebAssembly ABI over [`walrus_blob_encoder::StreamEncoder`], without
//! wasm-bindgen, for JavaScript hosts such as Cloudflare Workers, browsers, Node and Bun.
//!
//! The host drives the pull protocol; lengths and offsets are `f64` so that JavaScript numbers
//! pass unchanged (exact up to 2^53):
//!
//! ```text
//! h = enc_new(len, n_shards, budget, flags)          // 0 on failure
//! while (enc_next(h)) {
//!     read enc_req_len(h) bytes at enc_req_offset(h) into memory at enc_dest(h)
//!     enc_complete(h)
//! }
//! out = enc_finish(h)                                // blob ID, sliver hashes, stats
//! enc_free(h)
//! ```
//!
//! The reads may be served asynchronously between calls. `flags`: bit 0 measures phase times
//! with the imported `env.now_ms` (milliseconds, as from `performance.now()`); bit 1 forces the
//! scalar BLAKE2b backend and portable Reed-Solomon engine. `enc_finish` returns a pointer to
//! [`output_len`] bytes, valid until `enc_free`: the 32-byte blob ID, then for each sliver pair
//! its 32-byte primary and secondary hashes, then the [`STATS`] counters as little-endian `f64`.
//!
//! `blob_alloc` and `blob_metadata` compute the metadata of a blob held in wasm memory with the
//! in-memory encoder, returning the same layout without stats (all zeros).

use std::num::NonZeroU16;

use walrus_blob_encoder::{Backend, BlobEncoder, BlobMetadata, StreamEncoder, StreamStats};

/// The names of the `f64` counters after the hashes in the output of `enc_finish`.
pub const STATS: [&str; 13] = [
    "symbol_size",
    "batch_columns",
    "batch_bytes",
    "row_buffer_bytes",
    "systematic_batches",
    "full_passes",
    "bytes_read",
    "read_calls",
    "row_encodes",
    "column_encodes",
    "row_encoding_ms",
    "column_encoding_ms",
    "hashing_ms",
];

/// The length of the output of `enc_finish` for `n_shards` shards.
pub const fn output_len(n_shards: usize) -> usize {
    32 + 64 * n_shards + 8 * STATS.len()
}

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "env")]
unsafe extern "C" {
    fn now_ms() -> f64;
}

#[cfg(target_arch = "wasm32")]
fn clock() -> f64 {
    // SAFETY: `now_ms` takes no arguments and returns a number; the host provides it whenever
    // a module with timing is instantiated.
    unsafe { now_ms() }
}

#[cfg(not(target_arch = "wasm32"))]
fn clock() -> f64 {
    use std::{sync::OnceLock, time::Instant};
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64() * 1e3
}

/// The state behind a handle.
pub struct Handle {
    stream: Option<StreamEncoder>,
    out: Vec<u8>,
}

fn encoder(n_shards: u32, scalar: bool) -> Option<BlobEncoder> {
    let n_shards = NonZeroU16::new(u16::try_from(n_shards).ok()?)?;
    let encoder = BlobEncoder::new(n_shards).ok()?;
    if scalar {
        encoder.with_backend(Backend::Scalar).ok()
    } else {
        Some(encoder)
    }
}

fn output(metadata: &BlobMetadata, stats: StreamStats) -> Vec<u8> {
    let mut out = Vec::with_capacity(output_len(metadata.sliver_pairs().len()));
    out.extend_from_slice(metadata.blob_id().as_bytes());
    for pair in metadata.sliver_pairs() {
        out.extend_from_slice(&pair.primary_hash);
        out.extend_from_slice(&pair.secondary_hash);
    }
    let counters = [
        stats.symbol_size as f64,
        stats.batch_columns as f64,
        stats.batch_bytes as f64,
        stats.row_buffer_bytes as f64,
        stats.systematic_batches as f64,
        stats.full_passes as f64,
        stats.bytes_read as f64,
        stats.read_calls as f64,
        stats.row_encodes as f64,
        stats.column_encodes as f64,
        stats.row_encoding_ms,
        stats.column_encoding_ms,
        stats.hashing_ms,
    ];
    for value in counters {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

/// # Safety
///
/// `h` must come from [`enc_new`] and not be freed.
unsafe fn handle<'a>(h: *mut Handle) -> &'a mut Handle {
    // SAFETY: by the caller's contract `h` is a live pointer from `Box::into_raw`, and the host
    // is single-threaded, so no other reference exists during the call.
    unsafe { &mut *h }
}

/// Starts a stream for a blob of `len` bytes with a batch budget of `budget` bytes; returns 0 if
/// the shard count or length is invalid.
#[unsafe(no_mangle)]
pub extern "C" fn enc_new(len: f64, n_shards: u32, budget: f64, flags: u32) -> *mut Handle {
    let Some(encoder) = encoder(n_shards, flags & 2 != 0) else {
        return std::ptr::null_mut();
    };
    // `as` saturates: a negative or NaN budget is 0, one column.
    let Ok(stream) = encoder.stream(len as u64, budget as usize) else {
        return std::ptr::null_mut();
    };
    let stream = if flags & 1 != 0 {
        stream.with_clock(clock)
    } else {
        stream
    };
    Box::into_raw(Box::new(Handle {
        stream: Some(stream),
        out: Vec::new(),
    }))
}

/// 1 if a read is pending, 0 once all reads are complete.
///
/// # Safety
///
/// `h` must come from [`enc_new`] and not be freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn enc_next(h: *mut Handle) -> i32 {
    // SAFETY: forwarded from the caller.
    let handle = unsafe { handle(h) };
    i32::from(handle.stream.as_ref().is_some_and(|s| !s.is_done()))
}

/// The offset of the pending read, or -1.
///
/// # Safety
///
/// `h` must come from [`enc_new`] and not be freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn enc_req_offset(h: *mut Handle) -> f64 {
    // SAFETY: forwarded from the caller.
    let handle = unsafe { handle(h) };
    let read = handle.stream.as_ref().and_then(StreamEncoder::next_read);
    read.map_or(-1.0, |read| read.offset as f64)
}

/// The length of the pending read, or 0.
///
/// # Safety
///
/// `h` must come from [`enc_new`] and not be freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn enc_req_len(h: *mut Handle) -> u32 {
    // SAFETY: forwarded from the caller.
    let handle = unsafe { handle(h) };
    let read = handle.stream.as_ref().and_then(StreamEncoder::next_read);
    read.map_or(0, |read| {
        u32::try_from(read.len).expect("wasm32 lengths fit in u32")
    })
}

/// Where the host writes the pending read, or null if none is pending. Valid until the next
/// call into the module, which may grow its memory.
///
/// # Safety
///
/// `h` must come from [`enc_new`] and not be freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn enc_dest(h: *mut Handle) -> *mut u8 {
    // SAFETY: forwarded from the caller.
    let handle = unsafe { handle(h) };
    match &mut handle.stream {
        Some(stream) if !stream.is_done() => stream.dest().as_mut_ptr(),
        _ => std::ptr::null_mut(),
    }
}

/// Completes the pending read; 0 if none was pending.
///
/// # Safety
///
/// `h` must come from [`enc_new`] and not be freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn enc_complete(h: *mut Handle) -> i32 {
    // SAFETY: forwarded from the caller.
    let handle = unsafe { handle(h) };
    match &mut handle.stream {
        Some(stream) if !stream.is_done() => {
            stream.complete_read();
            1
        }
        _ => 0,
    }
}

/// Finishes the stream; returns the output, or null if reads are pending.
///
/// # Safety
///
/// `h` must come from [`enc_new`] and not be freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn enc_finish(h: *mut Handle) -> *const u8 {
    // SAFETY: forwarded from the caller.
    let handle = unsafe { handle(h) };
    match handle.stream.take() {
        Some(stream) if stream.is_done() => {
            let stats = stream.stats();
            handle.out = output(&stream.finish(), stats);
            handle.out.as_ptr()
        }
        stream => {
            handle.stream = stream;
            std::ptr::null()
        }
    }
}

/// Frees a handle and its output.
///
/// # Safety
///
/// `h` must come from [`enc_new`] and not be freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn enc_free(h: *mut Handle) {
    // SAFETY: by the caller's contract `h` came from `Box::into_raw` and is freed only here.
    drop(unsafe { Box::from_raw(h) });
}

/// Allocates `len` bytes for a blob, to be freed by [`blob_metadata`].
#[unsafe(no_mangle)]
pub extern "C" fn blob_alloc(len: u32) -> *mut u8 {
    let blob = vec![0u8; len as usize].into_boxed_slice();
    Box::into_raw(blob).cast()
}

/// Frees the blob of `len` bytes at `ptr` from [`blob_alloc`] after computing its metadata with
/// the in-memory encoder; returns the output (owned by the module until the next call) or null.
///
/// # Safety
///
/// `ptr` and `len` must come from one call of [`blob_alloc`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blob_metadata(ptr: *mut u8, len: u32, n_shards: u32) -> *const u8 {
    // SAFETY: by the caller's contract this is the boxed slice `blob_alloc` leaked.
    let blob = unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len as usize)) };
    let Some(encoder) = encoder(n_shards, false) else {
        return std::ptr::null();
    };
    let Ok(metadata) = encoder.metadata(&blob) else {
        return std::ptr::null();
    };
    drop(blob);
    thread_local!(static OUT: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) });
    OUT.with_borrow_mut(|out| {
        *out = output(&metadata, StreamStats::default());
        out.as_ptr()
    })
}

/// The number of BLAKE2b mismatches between the fastest available backend, the scalar backend
/// and RustCrypto's `blake2`, over block-boundary lengths and batch sizes.
#[cfg(feature = "selftest")]
#[unsafe(no_mangle)]
pub extern "C" fn selftest_blake2b() -> u32 {
    use blake2::{Blake2b, Digest as _, digest::consts::U32};
    use walrus_blob_encoder::__private::hash_batch;

    let mut lengths: Vec<usize> = vec![0, 1, 2, 31, 32, 33, 63, 64, 65, 126];
    for k in 1..=12 {
        lengths.extend([128 * k - 2, 128 * k - 1, 128 * k, 128 * k + 1]);
    }
    lengths.extend([1164, 1206, 4490, 4491]);
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let buffer: Vec<u8> = (0..8192)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect();
    let backends = [Backend::detect(), Backend::Scalar];
    let mut failures = 0;
    for &len in &lengths {
        for count in [1, 2, 3, 5, 8, 33] {
            let messages: Vec<&[u8]> = (0..count)
                .map(|i| &buffer[(i * 97) % (8192 - len)..][..len])
                .collect();
            for prefix in [0u8, 1, 0xff] {
                let expected: Vec<[u8; 32]> = messages
                    .iter()
                    .map(|m| {
                        let digest = Blake2b::<U32>::new()
                            .chain_update([prefix])
                            .chain_update(m)
                            .finalize();
                        digest.into()
                    })
                    .collect();
                for backend in backends {
                    let mut out = vec![[0xaa; 32]; count];
                    assert!(hash_batch(backend, prefix, &messages, &mut out));
                    failures += out.iter().zip(&expected).filter(|(a, b)| a != b).count();
                }
            }
        }
    }
    u32::try_from(failures).unwrap_or(u32::MAX)
}

/// The BLAKE2b backend the module uses: the index in [`Backend::ALL`].
#[unsafe(no_mangle)]
pub extern "C" fn backend() -> u32 {
    let backend = Backend::detect();
    let index = Backend::ALL.iter().position(|&b| b == backend);
    index.map_or(u32::MAX, |i| i as u32)
}
