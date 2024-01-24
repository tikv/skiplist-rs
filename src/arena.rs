// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{alloc::Layout, sync::Arc};

use crate::list::{MemoryLimiter, ReclaimableNode, U_SIZE};

#[derive(Clone)]
pub struct Arena<M: MemoryLimiter> {
    pub limiter: Arc<M>,
}

impl<M: MemoryLimiter> Arena<M> {
    pub fn new(limiter: Arc<M>) -> Self {
        Arena { limiter }
    }

    /// Alloc 8-byte aligned memory.
    pub fn alloc(&self, size: usize) -> usize {
        // todo: alloc returns Result
        assert!(self.limiter.acquire(size));

        let layout = Layout::from_size_align(size, U_SIZE).unwrap();
        let addr = unsafe { std::alloc::alloc(layout) as usize };
        self.limiter.alloc(addr, size);
        addr
    }

    pub fn free<N: ReclaimableNode>(&self, node_addr: *mut N) {
        let size = {
            let node = unsafe { &(*node_addr) };
            node.size()
        };

        self.limiter.free(node_addr as usize, size);
        let layout = Layout::from_size_align(size, U_SIZE).unwrap();
        unsafe {
            (*node_addr).drop_key_value();
            std::alloc::dealloc(node_addr as *mut u8, layout);
        }
        self.limiter.reclaim(size);
    }
}
