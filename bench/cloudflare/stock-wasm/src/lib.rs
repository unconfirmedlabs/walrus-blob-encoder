// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroU16;

use walrus_core::{
    DEFAULT_ENCODING,
    encoding::{EncodingConfig, EncodingFactory as _},
};

static mut OUT: [u8; 32] = [0; 32];

#[unsafe(no_mangle)]
pub extern "C" fn blob_alloc(len: u32) -> *mut u8 {
    Box::into_raw(vec![0u8; len as usize].into_boxed_slice()).cast()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn blob_id(ptr: *mut u8, len: u32, n_shards: u32) -> *const u8 {
    let blob = unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len as usize)) };
    let Some(n) = NonZeroU16::new(n_shards as u16) else { return std::ptr::null() };
    let config = EncodingConfig::new(n).get_for_type(DEFAULT_ENCODING);
    match config.compute_blob_id(&blob) {
        Ok(id) => unsafe {
            let out = &raw mut OUT;
            (*out).copy_from_slice(&id.0);
            out.cast()
        },
        Err(_) => std::ptr::null(),
    }
}

fn no_entropy(_: &mut [u8]) -> Result<(), getrandom::Error> {
    Err(getrandom::Error::UNSUPPORTED)
}
getrandom::register_custom_getrandom!(no_entropy);
