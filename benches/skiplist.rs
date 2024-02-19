#![feature(test)]

extern crate skiplist_rs;
extern crate test;

use std::sync::Arc;

use test::Bencher;

use skiplist_rs::{AllocationRecorder, ByteWiseComparator, MemoryLimiter, Skiplist};

#[derive(Clone, Default)]
struct DummyLimiter {}

impl AllocationRecorder for DummyLimiter {
    fn allocated(&self, _: usize, _: usize) {}

    fn freed(&self, _: usize, _: usize) {}
}

impl MemoryLimiter for DummyLimiter {
    fn acquire(&self, _: usize) -> bool {
        true
    }

    fn mem_usage(&self) -> usize {
        0
    }

    fn reclaim(&self, _: usize) {}
}

fn construct_key(i: u64) -> Vec<u8> {
    format!("key-{:08}", i).into_bytes()
}

#[bench]
fn insert(b: &mut Bencher) {
    b.iter(|| {
        let map = Skiplist::new(ByteWiseComparator {}, Arc::new(DummyLimiter::default()));

        let mut num = 0 as u64;
        for _ in 0..1_000 {
            num = num.wrapping_mul(17).wrapping_add(255);
            let k = construct_key(num);
            map.put(k.clone(), construct_key(!num));
            map.get(&k);
        }
    });
}
