// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

#![feature(slice_pattern)]
#![feature(let_chains)]

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

mod arena;
mod key;
mod list;

const MAX_HEIGHT: usize = 20;

pub use key::{ByteWiseComparator, FixedLengthSuffixComparator, KeyComparator};
pub use list::{IterRef, Skiplist, MAX_NODE_SIZE};
