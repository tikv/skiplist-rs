use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use loom::thread;
use skiplist_rs::{AllocationRecorder, ByteWiseComparator, MemoryLimiter, Skiplist};

#[derive(Clone, Default)]
struct RecorderLimiter {
    recorder: Arc<Mutex<HashMap<usize, usize>>>,
}

impl Drop for RecorderLimiter {
    fn drop(&mut self) {
        let recorder = self.recorder.lock().unwrap();
        assert!(recorder.is_empty());
    }
}

impl AllocationRecorder for RecorderLimiter {
    fn allocated(&self, addr: usize, size: usize) {
        let mut recorder = self.recorder.lock().unwrap();
        assert!(!recorder.contains_key(&addr));
        recorder.insert(addr, size);
    }

    fn freed(&self, addr: usize, size: usize) {
        let mut recorder = self.recorder.lock().unwrap();
        assert_eq!(recorder.remove(&addr).unwrap(), size);
    }
}

impl MemoryLimiter for RecorderLimiter {
    fn acquire(&self, _: usize) -> bool {
        true
    }

    fn mem_usage(&self) -> usize {
        0
    }

    fn reclaim(&self, _: usize) {}
}

#[test]
fn concurrent_put_and_remove() {
    loom::model(|| {
        let sl = Skiplist::<ByteWiseComparator, RecorderLimiter>::new(
            ByteWiseComparator {},
            Arc::default(),
            crossbeam_epoch::default_collector().clone(),
        );
        let n = 50;
        for i in (0..n).step_by(3) {
            let guard = &crossbeam_epoch::pin();
            let k = format!("k{:04}", i).into_bytes();
            let v = format!("v{:04}", i).into_bytes();
            sl.put(k, v, guard);
        }
        let sl1 = sl.clone();
        let h1 = thread::spawn(move || {
            for i in (1..n).step_by(3) {
                let guard = &crossbeam_epoch::pin();
                let k = format!("k{:04}", i).into_bytes();
                let v = format!("v{:04}", i).into_bytes();
                sl1.put(k, v, guard);
            }
        });
        let sl1 = sl.clone();
        let h2 = thread::spawn(move || {
            for i in (0..n).step_by(3) {
                let guard = &crossbeam_epoch::pin();
                let k = format!("k{:04}", i);
                sl1.remove(k.as_bytes(), guard);
            }
        });

        let sl1 = sl.clone();
        let h3 = thread::spawn(move || {
            for i in (0..n).step_by(3) {
                let guard = &crossbeam_epoch::pin();
                let k = format!("k{:04}", i);
                sl1.remove(k.as_bytes(), guard);
            }
        });

        let sl1 = sl.clone();
        let h4 = thread::spawn(move || {
            for i in (2..n).step_by(3) {
                let guard = &crossbeam_epoch::pin();
                let k = format!("k{:04}", i);
                sl1.remove(k.as_bytes(), guard);
            }
        });

        h1.join().unwrap();
        h2.join().unwrap();
        h3.join().unwrap();
        h4.join().unwrap();

        for i in (1..n).step_by(3) {
            let guard = &crossbeam_epoch::pin();
            let k = format!("k{:04}", i);
            let v = format!("v{:04}", i);
            assert_eq!(sl.get(k.as_bytes(), guard).unwrap().value(), v.as_bytes());
        }
    });
}
