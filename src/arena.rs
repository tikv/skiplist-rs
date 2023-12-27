// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    alloc::Layout,
    cell::Cell,
    mem, ptr,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::list::{NodeSize, U_SIZE};

// Thread-local Arena
pub struct Arena {
    len: AtomicUsize,
    cap: Cell<usize>,
    ptr: Cell<*mut u8>,
}

impl Drop for Arena {
    fn drop(&mut self) {
        let ptr = self.ptr.get() as *mut u64;
        let cap = self.cap.get() / 8;
        unsafe {
            Vec::from_raw_parts(ptr, 0, cap);
        }
    }
}

static NO_TAG: usize = !((1 << 3) /* alignment */ - 1);

pub fn tag(offset: usize) -> usize {
    offset & 1
}

pub fn without_tag(offset: usize) -> usize {
    offset & NO_TAG
}

impl Arena {
    pub fn with_capacity(_cap: usize) -> Arena {
        let mut buf: Vec<u64> = Vec::with_capacity(0);
        let ptr = buf.as_mut_ptr() as *mut u8;
        let cap = buf.capacity() * 8;
        mem::forget(buf);
        Arena {
            // Offset 0 is invalid value for func `offset` and `get_mut`, initialize the
            // len 8 to guarantee the allocated memory addr is always align with 8 bytes.
            len: AtomicUsize::new(8),
            cap: Cell::new(cap),
            ptr: Cell::new(ptr),
        }
    }

    pub fn len(&self) -> usize {
        self.len.load(Ordering::SeqCst)
    }

    pub fn cap(&self) -> usize {
        self.cap.get()
    }

    /// Alloc 8-byte aligned memory.
    pub fn alloc(&self, size: usize) -> usize {
        let layout = Layout::from_size_align(size, U_SIZE).unwrap();
        unsafe { std::alloc::alloc(layout) as usize }
    }

    pub fn free<N: NodeSize>(&self, node_addr: *mut N) {
        let size = {
            let node = unsafe { &(*node_addr) };
            node.size()
        };
        let layout = Layout::from_size_align(size, U_SIZE).unwrap();
        unsafe {
            std::alloc::dealloc(node_addr as *mut u8, layout);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena() {
        // There is enough space
        let arena = Arena::with_capacity(128);
        let offset = arena.alloc(8);
        assert_eq!(offset, 8);
        assert_eq!(arena.len(), 16);
        unsafe {
            let ptr = arena.get_mut::<u64>(offset);
            let offset = arena.offset::<u64>(ptr);
            assert_eq!(offset, 8);
        }

        // There is not enough space, grow buf and then return the offset
        let offset = arena.alloc(256);
        assert_eq!(offset, 16);
    }
}
