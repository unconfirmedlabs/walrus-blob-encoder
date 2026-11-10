// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The blob ID and its URL-safe base64 text form.

use std::{error, fmt, str::FromStr};

const LEN: usize = 32;
/// Characters of a 32-byte value in unpadded base64.
const TEXT_LEN: usize = 43;
const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/// The ID of a Walrus blob.
///
/// The bytes are the BLAKE2b-256 digest committing to the blob's encoding type, length and
/// sliver hashes, in the order walrus-core and the CLI use. Interpreted as an integer they are
/// little endian: the on-chain `u256` blob ID is `u256::from_le_bytes(id.as_bytes())`.
///
/// The text form ([`Display`](fmt::Display) and [`FromStr`]) is URL-safe base64 without
/// padding, the same as the `walrus` CLI.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlobId([u8; LEN]);

impl BlobId {
    /// Returns the 32 bytes of the ID.
    pub const fn as_bytes(&self) -> &[u8; LEN] {
        &self.0
    }
}

impl From<[u8; LEN]> for BlobId {
    fn from(bytes: [u8; LEN]) -> Self {
        BlobId(bytes)
    }
}

impl From<BlobId> for [u8; LEN] {
    fn from(id: BlobId) -> Self {
        id.0
    }
}

impl fmt::Display for BlobId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = encode(&self.0);
        f.pad(std::str::from_utf8(&text).expect("base64 is ASCII"))
    }
}

impl fmt::Debug for BlobId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BlobId({self})")
    }
}

/// Error returned when a string is not a URL-safe, unpadded base64 encoding of 32 bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseBlobIdError;

impl fmt::Display for ParseBlobIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("invalid blob ID: expected 43 characters of URL-safe base64 without padding")
    }
}

impl error::Error for ParseBlobIdError {}

impl FromStr for BlobId {
    type Err = ParseBlobIdError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        decode(s.as_bytes()).map(BlobId).ok_or(ParseBlobIdError)
    }
}

fn encode(bytes: &[u8; LEN]) -> [u8; TEXT_LEN] {
    let mut text = [0; TEXT_LEN];
    let (groups, rest) = bytes.as_chunks::<3>();
    let char_of = |bits: u32| ALPHABET[(bits & 0x3f) as usize];
    for (out, group) in text.chunks_exact_mut(4).zip(groups) {
        let bits = u32::from_be_bytes([0, group[0], group[1], group[2]]);
        out.copy_from_slice(&[18, 12, 6, 0].map(|shift| char_of(bits >> shift)));
    }
    let bits = u32::from_be_bytes([0, rest[0], rest[1], 0]);
    text[TEXT_LEN - 3..].copy_from_slice(&[18, 12, 6].map(|shift| char_of(bits >> shift)));
    text
}

fn decode(text: &[u8]) -> Option<[u8; LEN]> {
    let text: &[u8; TEXT_LEN] = text.try_into().ok()?;
    let value_of = |c: u8| -> Option<u32> {
        let value = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'-' => 62,
            b'_' => 63,
            _ => return None,
        };
        Some(value.into())
    };
    let (quads, rest) = text.as_chunks::<4>();
    let mut bytes = [0; LEN];
    for (out, quad) in bytes.chunks_exact_mut(3).zip(quads) {
        let bits = quad
            .iter()
            .try_fold(0, |bits, &c| Some(bits << 6 | value_of(c)?))?;
        out.copy_from_slice(&bits.to_be_bytes()[1..]);
    }
    let bits = rest
        .iter()
        .try_fold(0, |bits, &c| Some(bits << 6 | value_of(c)?))?;
    // 18 bits carry 16 bytes' worth of data; canonical encodings leave the last two bits zero.
    if bits & 0b11 != 0 {
        return None;
    }
    bytes[LEN - 2..].copy_from_slice(&((bits >> 2) as u16).to_be_bytes());
    Some(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EMPTY_BLOB_ID: &str = "3GPQL3HZNnFhN_F7l5Aa-X1VOtAKwIsg9zuWk8R81v4";

    #[test]
    fn round_trips() {
        for seed in 0..=255u8 {
            let bytes: [u8; LEN] =
                std::array::from_fn(|i| (i as u8).wrapping_mul(97).wrapping_add(seed));
            let id = BlobId::from(bytes);
            let text = id.to_string();
            assert_eq!(text.len(), TEXT_LEN);
            assert_eq!(text.parse::<BlobId>(), Ok(id));
        }
        assert_eq!(
            BlobId::from([0; LEN]).to_string(),
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        );
        assert_eq!(
            BlobId::from([0xff; LEN]).to_string(),
            "__________________________________________8"
        );
        let id: BlobId = EMPTY_BLOB_ID.parse().unwrap();
        assert_eq!(id.to_string(), EMPTY_BLOB_ID);
        assert_eq!(format!("{id:?}"), format!("BlobId({EMPTY_BLOB_ID})"));
    }

    #[test]
    fn rejects_invalid_text() {
        for text in [
            "",
            &EMPTY_BLOB_ID[..42],
            &format!("{EMPTY_BLOB_ID}A"),
            &format!("{EMPTY_BLOB_ID}="),
            "3GPQL3HZNnFhN_F7l5Aa-X1VOtAKwIsg9zuWk8R81v5",
            "3GPQL3HZNnFhN/F7l5Aa-X1VOtAKwIsg9zuWk8R81v4",
            "3GPQL3HZNnFhN+F7l5Aa-X1VOtAKwIsg9zuWk8R81v4",
            "3GPQL3HZNnFhN_F7l5Aa-X1VOtAKwIsg9zuWk8R81v=",
            "3GPQL3HZNnFhN_F7l5Aa-X1VOtAKwIsg9zuWk8R81 4",
            "3GPQL3HZNnFhN_F7l5Aa-X1VOtAKwIsg9zuWk8R81vé",
        ] {
            assert_eq!(text.parse::<BlobId>(), Err(ParseBlobIdError), "{text:?}");
        }
    }

    #[test]
    fn displays_padded() {
        let id: BlobId = EMPTY_BLOB_ID.parse().unwrap();
        assert_eq!(format!("{id:>45}"), format!("  {EMPTY_BLOB_ID}"));
    }
}
