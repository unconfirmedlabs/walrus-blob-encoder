// Copyright (c) Unconfirmed Labs, LLC
// SPDX-License-Identifier: Apache-2.0

//! Merkle roots of many equally sized trees, hashed level by level in one batch per level.
//!
//! Leaves are `BLAKE2b-256(0x00 || data)`, inner nodes `BLAKE2b-256(0x01 || left || right)`,
//! and a level of odd length is completed with an all-zero node.

use std::mem;

use crate::{
    blake2b::{DIGEST_LEN, Digest, Hasher},
    messages::MessageBuf,
};

pub(crate) const LEAF_PREFIX: u8 = 0x00;
pub(crate) const INNER_PREFIX: u8 = 0x01;

const ZERO_NODE: Digest = [0; DIGEST_LEN];

/// Scratch space for [`roots`], reused across calls.
#[derive(Default)]
pub(crate) struct TreeScratch {
    level: Vec<Digest>,
    next: Vec<Digest>,
    odd_pairs: Vec<[u8; 2 * DIGEST_LEN]>,
    messages: MessageBuf,
}

/// Computes the roots of `roots.len()` trees with `n_leaves` leaves each, stored back to back in
/// `leaves`.
pub(crate) fn roots(
    hasher: Hasher,
    leaves: &[Digest],
    n_leaves: usize,
    roots: &mut [Digest],
    scratch: &mut TreeScratch,
) {
    assert_eq!(leaves.len(), roots.len() * n_leaves);
    if n_leaves == 1 {
        roots.copy_from_slice(leaves);
        return;
    }
    let TreeScratch {
        level,
        next,
        odd_pairs,
        messages,
    } = scratch;
    let mut width = reduce(hasher, leaves, n_leaves, level, odd_pairs, messages);
    while width > 1 {
        width = reduce(hasher, level, width, next, odd_pairs, messages);
        mem::swap(level, next);
    }
    roots.copy_from_slice(level);
}

/// Hashes one level of every tree (`width` nodes per tree) into `out`; returns the new width.
fn reduce(
    hasher: Hasher,
    level: &[Digest],
    width: usize,
    out: &mut Vec<Digest>,
    odd_pairs: &mut Vec<[u8; 2 * DIGEST_LEN]>,
    messages: &mut MessageBuf,
) -> usize {
    let n_pairs = width / 2;
    let next_width = width.div_ceil(2);
    let trees = level.chunks_exact(width);
    out.clear();
    out.resize(trees.len() * next_width, ZERO_NODE);

    odd_pairs.clear();
    if width % 2 == 1 {
        odd_pairs.extend(trees.clone().map(|tree| {
            let mut pair = [0; 2 * DIGEST_LEN];
            pair[..DIGEST_LEN].copy_from_slice(&tree[width - 1]);
            pair
        }));
    }

    let mut list = messages.take();
    let mut odd_pairs = odd_pairs.iter();
    for tree in trees {
        list.extend(
            tree[..2 * n_pairs]
                .as_flattened()
                .chunks_exact(2 * DIGEST_LEN),
        );
        list.extend(odd_pairs.next().map(<[u8; 2 * DIGEST_LEN]>::as_slice));
    }
    hasher.hash_batch(INNER_PREFIX, &list, out);
    messages.put(list);
    next_width
}

#[cfg(test)]
mod tests {
    use blake2::{Blake2b, Digest as _, digest::consts::U32};

    use super::*;
    use crate::blake2b::test_backends;

    fn hash(parts: &[&[u8]]) -> Digest {
        let mut hasher = Blake2b::<U32>::new();
        parts.iter().for_each(|part| hasher.update(part));
        hasher.finalize().into()
    }

    /// Straightforward bottom-up construction, one tree at a time.
    fn reference_root(leaves: &[Digest]) -> Digest {
        let mut level = leaves.to_vec();
        while level.len() > 1 {
            if level.len() % 2 == 1 {
                level.push(ZERO_NODE);
            }
            level = level
                .chunks_exact(2)
                .map(|pair| hash(&[&[INNER_PREFIX], &pair[0], &pair[1]]))
                .collect();
        }
        level[0]
    }

    #[test]
    fn matches_reference() {
        for backend in test_backends() {
            let hasher = Hasher::new(backend).unwrap();
            let mut scratch = TreeScratch::default();
            for n_leaves in (1..=70).chain([999, 1000, 1001]) {
                for n_trees in [1, 3, 16] {
                    let leaves: Vec<Digest> = (0..n_trees * n_leaves)
                        .map(|i| hash(&[&[LEAF_PREFIX], &(i as u64).to_le_bytes()]))
                        .collect();
                    let mut got = vec![ZERO_NODE; n_trees];
                    roots(hasher, &leaves, n_leaves, &mut got, &mut scratch);
                    let expected: Vec<Digest> =
                        leaves.chunks(n_leaves).map(reference_root).collect();
                    assert_eq!(
                        got, expected,
                        "{backend}: {n_trees} trees of {n_leaves} leaves"
                    );
                }
            }
        }
    }
}
