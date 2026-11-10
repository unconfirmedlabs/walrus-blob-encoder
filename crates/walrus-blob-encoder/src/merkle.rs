// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Merkle roots of many equally sized trees, hashed level by level in one batch per level:
//! either all leaves at once ([`roots`]) or leaf by leaf, every tree receiving its next leaf at
//! the same time ([`Forest`]).
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

/// `n_trees` Merkle trees of `n_leaves` leaves each, built leaf by leaf.
///
/// Every call to [`Forest::push`] appends one leaf to every tree, so all trees have the same shape
/// at all times and each completed pair of nodes, on any level, is hashed for all trees in one
/// batch. Only one pending `left || right` pair per tree and level is kept: about
/// `64 * n_trees * log2(n_leaves)` bytes.
pub(crate) struct Forest {
    n_trees: usize,
    n_leaves: usize,
    /// The number of nodes pushed to each level so far, the same for every tree.
    pushed: Vec<usize>,
    /// Per level, the pending pair of every tree; its left half is set at even positions.
    pairs: Vec<Vec<[u8; 2 * DIGEST_LEN]>>,
    hashed: Vec<Digest>,
    messages: MessageBuf,
}

impl Forest {
    pub(crate) fn new(n_trees: usize, n_leaves: usize) -> Self {
        assert!(n_leaves > 0, "a tree has at least one leaf");
        let levels = (usize::BITS - (n_leaves - 1).leading_zeros()) as usize + 1;
        Forest {
            n_trees,
            n_leaves,
            pushed: vec![0; levels],
            pairs: vec![vec![[0; 2 * DIGEST_LEN]; n_trees]; levels],
            hashed: vec![ZERO_NODE; n_trees],
            messages: MessageBuf::default(),
        }
    }

    /// The number of leaves pushed to every tree.
    pub(crate) fn len(&self) -> usize {
        self.pushed[0]
    }

    /// Appends `leaves[t]` to tree `t`, for every tree.
    pub(crate) fn push(&mut self, hasher: Hasher, leaves: &[Digest]) {
        assert!(self.len() < self.n_leaves, "too many leaves");
        self.push_at(hasher, 0, Some(leaves));
    }

    /// Appends one node to every tree at `level`, hashing completed pairs upwards: `nodes`, or the
    /// result of the previous level's hashes when `None`.
    fn push_at(&mut self, hasher: Hasher, mut level: usize, mut nodes: Option<&[Digest]>) {
        loop {
            let index = self.pushed[level];
            self.pushed[level] += 1;
            let half = (index % 2) * DIGEST_LEN;
            let source = nodes.take().unwrap_or(&self.hashed);
            assert_eq!(source.len(), self.n_trees);
            for (pair, node) in self.pairs[level].iter_mut().zip(source) {
                pair[half..half + DIGEST_LEN].copy_from_slice(node);
            }
            if index.is_multiple_of(2) {
                return;
            }
            let mut list = self.messages.take();
            list.extend(
                self.pairs[level]
                    .iter()
                    .map(<[u8; 2 * DIGEST_LEN]>::as_slice),
            );
            hasher.hash_batch(INNER_PREFIX, &list, &mut self.hashed);
            self.messages.put(list);
            level += 1;
        }
    }

    /// Completes every tree, padding each level of odd length with an all-zero node, and writes
    /// the root of tree `t` to `roots[t]`.
    ///
    /// # Panics
    ///
    /// If fewer than `n_leaves` leaves were pushed.
    pub(crate) fn finish(mut self, hasher: Hasher, roots: &mut [Digest]) {
        assert_eq!(self.len(), self.n_leaves, "missing leaves");
        assert_eq!(roots.len(), self.n_trees);
        let (mut level, mut width) = (0, self.n_leaves);
        while width > 1 {
            if width % 2 == 1 {
                self.hashed.fill(ZERO_NODE);
                self.push_at(hasher, level, None);
            }
            width = width.div_ceil(2);
            level += 1;
        }
        for (root, pair) in roots.iter_mut().zip(&self.pairs[level]) {
            root.copy_from_slice(&pair[..DIGEST_LEN]);
        }
    }
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

                    let mut forest = Forest::new(n_trees, n_leaves);
                    for leaf in 0..n_leaves {
                        let column: Vec<Digest> =
                            (0..n_trees).map(|t| leaves[t * n_leaves + leaf]).collect();
                        forest.push(hasher, &column);
                    }
                    let mut got = vec![ZERO_NODE; n_trees];
                    forest.finish(hasher, &mut got);
                    assert_eq!(got, expected, "{backend}: forest of {n_trees} x {n_leaves}");
                }
            }
        }
    }
}
