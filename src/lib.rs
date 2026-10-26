// Copyright (c) Unconfirmed Labs, LLC
// SPDX-License-Identifier: Apache-2.0

//! Byte-exact, parallel computation of Walrus blob IDs and blob metadata.

mod blake2b;

pub use blake2b::{Backend, ParseBackendError};
