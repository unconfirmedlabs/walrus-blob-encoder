// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Multi-buffer BLAKE2b-256 (RFC 7693) with SIMD backends.
//!
//! Every backend hashes `prefix || message` for a batch of equal-length messages, several
//! messages at a time, one message per SIMD lane.
//!
//! A backend whose target features are enabled at compile time is called directly. With the
//! `runtime-detection` feature, faster backends are compiled in as well and used when the
//! running CPU supports them; without it, no CPU detection is compiled at all.

#[cfg(all(
    target_arch = "x86_64",
    any(feature = "runtime-detection", target_feature = "avx2")
))]
mod avx2;
#[cfg(all(
    target_arch = "x86_64",
    any(
        feature = "runtime-detection",
        all(
            target_feature = "avx2",
            target_feature = "avx512f",
            target_feature = "avx512bw"
        )
    )
))]
mod avx512;
#[cfg(all(
    target_arch = "aarch64",
    target_endian = "little",
    any(feature = "runtime-detection", target_feature = "neon")
))]
mod neon;
mod scalar;
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "runtime-detection", target_feature = "sse4.1")
))]
mod sse41;
#[cfg(test)]
mod tests;

use core::fmt;
use core::str::FromStr;
#[cfg(feature = "runtime-detection")]
use std::sync::OnceLock;

pub(crate) const DIGEST_LEN: usize = 32;
pub(crate) type Digest = [u8; DIGEST_LEN];

const BLOCK_LEN: usize = 128;
type Block = [u8; BLOCK_LEN];

const IV: [u64; 8] = [
    0x6a09_e667_f3bc_c908,
    0xbb67_ae85_84ca_a73b,
    0x3c6e_f372_fe94_f82b,
    0xa54f_f53a_5f1d_36f1,
    0x510e_527f_ade6_82d1,
    0x9b05_688c_2b3e_6c1f,
    0x1f83_d9ab_fb41_bd6b,
    0x5be0_cd19_137e_2179,
];

const SIGMA: [[usize; 16]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

/// Initial state: the IV xored with parameter block word 0 (digest length 32, no key, fanout 1,
/// depth 1). The other parameter words are zero.
const H0: [u64; 8] = {
    let mut h = IV;
    h[0] ^= 0x0101_0000 ^ DIGEST_LEN as u64;
    h
};

/// A BLAKE2b implementation strategy.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    /// Portable code, one message at a time.
    Scalar,
    /// x86-64 AVX2, four messages at a time.
    Avx2,
    /// x86-64 AVX-512 (F and BW, with AVX2 for remainders), eight messages at a time.
    Avx512,
    /// AArch64 NEON, two messages at a time.
    Neon,
    /// x86-64 SSE4.1, two messages at a time.
    Sse41,
}

impl Backend {
    /// Every backend, fastest first.
    pub const ALL: &'static [Backend] = &[
        Backend::Avx512,
        Backend::Avx2,
        Backend::Neon,
        Backend::Sse41,
        Backend::Scalar,
    ];

    /// The fastest available backend (see [`Backend::is_available`]). With the
    /// `runtime-detection` feature it is detected once per process, unless the fastest backend
    /// for the target is enabled at compile time; without it, it is a compile-time constant.
    pub fn detect() -> Backend {
        #[cfg(feature = "runtime-detection")]
        {
            static DETECTED: OnceLock<Backend> = OnceLock::new();
            *DETECTED.get_or_init(Self::fastest_available)
        }
        #[cfg(not(feature = "runtime-detection"))]
        Self::fastest_available()
    }

    fn fastest_available() -> Backend {
        Self::ALL
            .iter()
            .copied()
            .find(|backend| backend.is_available())
            .unwrap_or(Backend::Scalar)
    }

    /// Whether this backend can run: its target features are enabled at compile time or, with
    /// the `runtime-detection` feature, the running CPU supports them. Always `false` for
    /// backends of other architectures.
    pub fn is_available(self) -> bool {
        #[cfg(feature = "runtime-detection")]
        {
            self.is_enabled_statically() || self.is_detected()
        }
        #[cfg(not(feature = "runtime-detection"))]
        self.is_enabled_statically()
    }

    /// Whether the target features of this backend are enabled at compile time.
    const fn is_enabled_statically(self) -> bool {
        match self {
            Backend::Scalar => true,
            Backend::Avx2 => cfg!(all(target_arch = "x86_64", target_feature = "avx2")),
            Backend::Avx512 => cfg!(all(
                target_arch = "x86_64",
                target_feature = "avx2",
                target_feature = "avx512f",
                target_feature = "avx512bw"
            )),
            Backend::Neon => cfg!(all(
                target_arch = "aarch64",
                target_endian = "little",
                target_feature = "neon"
            )),
            Backend::Sse41 => cfg!(all(target_arch = "x86_64", target_feature = "sse4.1")),
        }
    }

    /// Whether the running CPU supports this backend.
    #[cfg(feature = "runtime-detection")]
    fn is_detected(self) -> bool {
        match self {
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2 => std::is_x86_feature_detected!("avx2"),
            #[cfg(target_arch = "x86_64")]
            Backend::Avx512 => {
                std::is_x86_feature_detected!("avx2")
                    && std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512bw")
            }
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Backend::Neon => std::arch::is_aarch64_feature_detected!("neon"),
            #[cfg(target_arch = "x86_64")]
            Backend::Sse41 => std::is_x86_feature_detected!("sse4.1"),
            _ => false,
        }
    }

    /// The lowercase name of the backend, as accepted by [`FromStr`].
    pub fn name(self) -> &'static str {
        match self {
            Backend::Scalar => "scalar",
            Backend::Avx2 => "avx2",
            Backend::Avx512 => "avx512",
            Backend::Neon => "neon",
            Backend::Sse41 => "sse41",
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Backend {
    type Err = ParseBackendError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|backend| backend.name() == s)
            .ok_or(ParseBackendError(()))
    }
}

/// The error returned when parsing a string that is not the name of a [`Backend`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseBackendError(());

impl fmt::Display for ParseBackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("unknown backend, expected one of: ")?;
        for (i, backend) in Backend::ALL.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            f.write_str(backend.name())?;
        }
        Ok(())
    }
}

impl std::error::Error for ParseBackendError {}

/// Proof that `backend` is available (see [`Backend::is_available`]). Only constructible through [`Hasher::new`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct Hasher {
    backend: Backend,
}

impl Hasher {
    /// A hasher for `backend`, or `None` if the running CPU does not support it.
    pub(crate) fn new(backend: Backend) -> Option<Hasher> {
        backend.is_available().then_some(Hasher { backend })
    }

    pub(crate) fn backend(self) -> Backend {
        self.backend
    }

    /// Sets `out[i] = BLAKE2b-256(prefix || messages[i])` for every `i`.
    ///
    /// # Panics
    ///
    /// If `messages.len() != out.len()` or the messages are not all the same length.
    pub(crate) fn hash_batch(self, prefix: u8, messages: &[&[u8]], out: &mut [Digest]) {
        assert_eq!(messages.len(), out.len(), "one output per message");
        let Some(first) = messages.first() else {
            return;
        };
        assert!(
            messages.iter().all(|m| m.len() == first.len()),
            "messages of a batch must have equal lengths"
        );
        match self.backend {
            Backend::Scalar => scalar::hash_batch(prefix, messages, out),
            #[cfg(all(
                target_arch = "x86_64",
                any(feature = "runtime-detection", target_feature = "avx2")
            ))]
            Backend::Avx2 => avx2::hash_batch(self, prefix, messages, out),
            #[cfg(all(
                target_arch = "x86_64",
                any(
                    feature = "runtime-detection",
                    all(
                        target_feature = "avx2",
                        target_feature = "avx512f",
                        target_feature = "avx512bw"
                    )
                )
            ))]
            Backend::Avx512 => avx512::hash_batch(self, prefix, messages, out),
            #[cfg(all(
                target_arch = "aarch64",
                target_endian = "little",
                any(feature = "runtime-detection", target_feature = "neon")
            ))]
            Backend::Neon => neon::hash_batch(self, prefix, messages, out),
            #[cfg(all(
                target_arch = "x86_64",
                any(feature = "runtime-detection", target_feature = "sse4.1")
            ))]
            Backend::Sse41 => sse41::hash_batch(self, prefix, messages, out),
            _ => unreachable!("a Hasher only holds available backends"),
        }
    }
}

/// The block structure of `prefix || message`, the same for every message of a batch.
#[derive(Clone, Copy)]
struct Geometry {
    len: usize,
    blocks: usize,
}

impl Geometry {
    fn new(len: usize) -> Self {
        Geometry {
            len,
            blocks: (len + 1).div_ceil(BLOCK_LEN),
        }
    }

    #[inline]
    fn is_last(self, k: usize) -> bool {
        k + 1 == self.blocks
    }

    /// The byte counter after block `k`. A message of at most `isize::MAX` bytes keeps the
    /// counter below 2^64, so the high counter word is always zero.
    #[inline]
    fn counter(self, k: usize) -> u64 {
        if self.is_last(k) {
            self.len as u64 + 1
        } else {
            (BLOCK_LEN * (k + 1)) as u64
        }
    }
}

/// Splits `prefix || message` into blocks for `N` equal-length messages at a time.
///
/// Block `k` of a lane is `(prefix || message)[128 k .. 128 k + 128]`, zero padded. Full blocks
/// after the first are borrowed from the message; the first and the last are staged in buffers
/// whose fixed bytes (the prefix and the zero padding) are written once per batch.
struct Lanes<const N: usize> {
    geometry: Geometry,
    first: [Block; N],
    last: [Block; N],
}

impl<const N: usize> Lanes<N> {
    fn new(prefix: u8, len: usize) -> Self {
        let geometry = Geometry::new(len);
        let mut first = [[0; BLOCK_LEN]; N];
        let mut last = first;
        for block in &mut first {
            block[0] = prefix;
        }
        if geometry.blocks == 1 {
            last = first;
        }
        Lanes {
            geometry,
            first,
            last,
        }
    }

    /// Block `k` of every lane. All `messages` must have the length given to [`Lanes::new`].
    #[inline]
    fn block<'s, 'm: 's>(&'s mut self, messages: &[&'m [u8]; N], k: usize) -> [&'s Block; N] {
        let start = BLOCK_LEN * k;
        if self.geometry.is_last(k) {
            let head = usize::from(k == 0);
            let from = start + head - 1;
            for (block, message) in self.last.iter_mut().zip(messages) {
                let tail = &message[from..];
                block[head..head + tail.len()].copy_from_slice(tail);
            }
            self.last.each_ref()
        } else if k == 0 {
            for (block, message) in self.first.iter_mut().zip(messages) {
                block[1..].copy_from_slice(&message[..BLOCK_LEN - 1]);
            }
            self.first.each_ref()
        } else {
            messages.map(|message| {
                message[start - 1..start - 1 + BLOCK_LEN]
                    .try_into()
                    .expect("slice of block length")
            })
        }
    }
}

/// Hashes `messages` in groups of `N` with `hash_group`. A ragged remainder is hashed as one
/// more group padded with copies of its first message, and only its own digests are kept.
#[inline]
fn for_each_group<const N: usize>(
    messages: &[&[u8]],
    out: &mut [Digest],
    mut hash_group: impl FnMut(&[&[u8]; N]) -> [Digest; N],
) {
    let mut message_groups = messages.chunks_exact(N);
    let mut out_groups = out.chunks_exact_mut(N);
    for (group, out) in (&mut message_groups).zip(&mut out_groups) {
        let group = group.try_into().expect("chunk of N messages");
        out.copy_from_slice(&hash_group(group));
    }
    let rest = message_groups.remainder();
    if let Some(&first) = rest.first() {
        let mut group = [first; N];
        group[..rest.len()].copy_from_slice(rest);
        let digests = hash_group(&group);
        out_groups
            .into_remainder()
            .copy_from_slice(&digests[..rest.len()]);
    }
}

/// The digest of a final state given as its first four words.
#[inline]
fn digest(words: [u64; 4]) -> Digest {
    let mut out = [0; DIGEST_LEN];
    for (chunk, word) in out.chunks_exact_mut(8).zip(words) {
        chunk.copy_from_slice(&word.to_le_bytes());
    }
    out
}

/// The backends tests run against: all available ones, or exactly the one named by
/// `WALRUS_BLOB_ENCODER_BACKEND`, which must be available.
#[cfg(test)]
pub(crate) fn test_backends() -> Vec<Backend> {
    match std::env::var("WALRUS_BLOB_ENCODER_BACKEND") {
        Ok(name) => {
            let backend: Backend = name
                .parse()
                .unwrap_or_else(|e| panic!("WALRUS_BLOB_ENCODER_BACKEND={name}: {e}"));
            assert!(
                backend.is_available(),
                "WALRUS_BLOB_ENCODER_BACKEND={name}: not available on this CPU"
            );
            vec![backend]
        }
        Err(_) => Backend::ALL
            .iter()
            .copied()
            .filter(|backend| backend.is_available())
            .collect(),
    }
}
