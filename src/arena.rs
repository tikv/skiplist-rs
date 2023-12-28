// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{alloc::Layout, ptr};

use crate::list::{MemoryLimiter, ReclaimableNode, U_SIZE};

#[derive(Clone)]
pub struct Arena<M: MemoryLimiter> {
    pub limiter: M,
}

static NO_TAG: usize = !((1 << 3) /* alignment */ - 1);

pub fn tag(offset: usize) -> usize {
    offset & 1
}

pub fn without_tag(offset: usize) -> usize {
    offset & NO_TAG
}

impl<M: MemoryLimiter> Arena<M> {
    pub fn new(limiter: M) -> Self {
        Arena { limiter }
    }

    /// Alloc 8-byte aligned memory.
    pub fn alloc(&self, size: usize) -> usize {
        // todo: alloc returns Result
        assert!(self.limiter.acquire(size));

        let layout = Layout::from_size_align(size, U_SIZE).unwrap();
        unsafe { std::alloc::alloc(layout) as usize }
    }

    pub fn free<N: ReclaimableNode>(&self, node_addr: *mut N) {
        let size = {
            let node = unsafe { &(*node_addr) };
            node.size()
        };

        let layout = Layout::from_size_align(size, U_SIZE).unwrap();
        unsafe {
            (*node_addr).drop_key_value();
            std::alloc::dealloc(node_addr as *mut u8, layout);
        }
        self.limiter.reclaim(size);
    }

    pub unsafe fn get_mut<N>(&self, offset: usize) -> *mut N {
        let offset = without_tag(offset);
        if offset == 0 {
            return ptr::null_mut();
        }

        offset as _
    }

    pub fn offset<N>(&self, ptr: *const N) -> usize {
        ptr as usize
    }
}
