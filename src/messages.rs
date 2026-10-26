// Copyright (c) Unconfirmed Labs, LLC
// SPDX-License-Identifier: Apache-2.0

//! Reusable allocation for lists of borrowed hash messages.

use std::mem;

/// Keeps the allocation of a `Vec<&[u8]>` alive between uses with unrelated lifetimes.
#[derive(Default)]
pub(crate) struct MessageBuf(Vec<&'static [u8]>);

impl MessageBuf {
    /// Returns the empty list, keeping its capacity.
    pub(crate) fn take<'a>(&mut self) -> Vec<&'a [u8]> {
        recycle(mem::take(&mut self.0))
    }

    /// Stores the list's allocation for the next [`Self::take`].
    pub(crate) fn put(&mut self, messages: Vec<&[u8]>) {
        self.0 = recycle(messages);
    }
}

/// Empties `messages` and changes the lifetime of its element type; collecting into a vector of
/// the same layout reuses the allocation in place.
fn recycle<'b>(mut messages: Vec<&[u8]>) -> Vec<&'b [u8]> {
    messages.clear();
    messages.into_iter().map(|_| -> &'b [u8] { &[] }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_capacity() {
        let mut buf = MessageBuf::default();
        let data = [1u8; 10];
        let mut messages = buf.take();
        messages.extend(data.chunks(1));
        let capacity = messages.capacity();
        buf.put(messages);
        let messages = buf.take();
        assert!(messages.is_empty());
        assert_eq!(messages.capacity(), capacity);
    }
}
