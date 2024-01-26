use std::sync::{mpsc::sync_channel, Arc, Mutex};

use skiplist_rs::{ByteWiseComparator, RecorderLimiter, Skiplist};

#[test]
fn test_concurrent_delete_with_get_none() {
    let sl =
        Skiplist::<ByteWiseComparator, RecorderLimiter>::new(ByteWiseComparator {}, Arc::default());

    sl.put(b"aaa".to_vec(), b"val-a".to_vec());

    let (tx, rx) = sync_channel(0);
    let (tx2, rx2) = sync_channel(0);
    let rx2 = Arc::new(Mutex::new(rx2));

    let sl_clone = sl.clone();
    let h = std::thread::spawn(move || {
        fail::cfg_callback("on_try_acquire", move || {
            tx.send(1).unwrap();
            let _ = rx2.lock().unwrap().recv().unwrap();
        })
        .unwrap();
        assert!(sl_clone.get(b"aaa").is_none());
    });

    let _ = rx.recv().unwrap();

    let sl_clone = sl.clone();
    let h2 = std::thread::spawn(move || {
        sl_clone.remove(b"aaa");
        tx2.send(1).unwrap();
    });

    h.join().unwrap();
    h2.join().unwrap();
}

#[test]
fn test_concurrent_delete_with_next() {
    let sl =
        Skiplist::<ByteWiseComparator, RecorderLimiter>::new(ByteWiseComparator {}, Arc::default());
    sl.put(b"aaa".to_vec(), b"val-a".to_vec());
    sl.put(b"ccc".to_vec(), b"val-c".to_vec());
    sl.put(b"bbb".to_vec(), b"val-b".to_vec());
    sl.put(b"eee".to_vec(), b"val-e".to_vec());
    sl.put(b"ddd".to_vec(), b"val-d".to_vec());

    let (tx, rx) = sync_channel(0);
    let rx = Arc::new(Mutex::new(rx));
    let (tx2, rx2) = sync_channel(0);
    let rx2 = Arc::new(Mutex::new(rx2));
    let mut iter = sl.iter();
    let h = std::thread::spawn(move || {
        fail::cfg_callback("on_try_acquire", move || {
            let _ = tx.send(1);
            let _ = rx2.lock().unwrap().recv();
        })
        .unwrap();
        iter.seek_to_first();
        assert_eq!(iter.value(), &b"val-c".to_vec());

        iter.next();
        assert_eq!(iter.value(), &b"val-e".to_vec());

        iter.next();
        assert!(!iter.valid());
    });

    let sl_clone = sl.clone();
    let h2 = std::thread::spawn(move || {
        let _ = rx.lock().unwrap().recv().unwrap();
        sl_clone.remove(b"aaa");
        tx2.send(1).unwrap();

        let _ = rx.lock().unwrap().recv().unwrap();
        sl_clone.remove(b"bbb");
        tx2.send(1).unwrap();

        // do nothing, so the iter can read "ccc"
        let _ = rx.lock().unwrap().recv().unwrap();
        tx2.send(1).unwrap();

        let _ = rx.lock().unwrap().recv().unwrap();
        sl_clone.remove(b"ddd");
        tx2.send(1).unwrap();

        // do nothing, so the iter can read "eee"
        let _ = rx.lock().unwrap().recv().unwrap();
        tx2.send(1).unwrap();
    });

    h.join().unwrap();
    h2.join().unwrap();
}

#[test]
fn test_concurrent_delete_with_seek() {
    let sl =
        Skiplist::<ByteWiseComparator, RecorderLimiter>::new(ByteWiseComparator {}, Arc::default());
    sl.put(b"aaa".to_vec(), b"val-a".to_vec());
    sl.put(b"ccc".to_vec(), b"val-c".to_vec());
    sl.put(b"bbb".to_vec(), b"val-b".to_vec());
    sl.put(b"eee".to_vec(), b"val-e".to_vec());
    sl.put(b"ddd".to_vec(), b"val-d".to_vec());

    let (tx, rx) = sync_channel(0);
    let rx = Arc::new(Mutex::new(rx));
    let (tx2, rx2) = sync_channel(0);
    let rx2 = Arc::new(Mutex::new(rx2));
    let mut iter = sl.iter();
    let h = std::thread::spawn(move || {
        fail::cfg_callback("on_try_acquire", move || {
            let _ = tx.send(1);
            let _ = rx2.lock().unwrap().recv();
        })
        .unwrap();
        iter.seek(b"bbb");
        assert_eq!(iter.value(), &b"val-e".to_vec());

        iter.seek(b"aaa");
        assert_eq!(iter.value(), &b"val-e".to_vec());
    });

    let sl_clone = sl.clone();
    let h2 = std::thread::spawn(move || {
        let _ = rx.lock().unwrap().recv().unwrap();
        sl_clone.remove(b"bbb");
        tx2.send(1).unwrap();

        let _ = rx.lock().unwrap().recv().unwrap();
        sl_clone.remove(b"ccc");
        tx2.send(1).unwrap();

        let _ = rx.lock().unwrap().recv().unwrap();
        sl_clone.remove(b"ddd");
        tx2.send(1).unwrap();

        // do nothing, so the iter can read "ccc"
        let _ = rx.lock().unwrap().recv().unwrap();
        tx2.send(1).unwrap();

        let _ = rx.lock().unwrap().recv().unwrap();
        sl_clone.remove(b"aaa");
        tx2.send(1).unwrap();

        // do nothing, so the iter can read "eee"
        let _ = rx.lock().unwrap().recv().unwrap();
        tx2.send(1).unwrap();
    });

    h.join().unwrap();
    h2.join().unwrap();
}

#[test]
fn test_concurrent_delete_with_prev() {
    let sl =
        Skiplist::<ByteWiseComparator, RecorderLimiter>::new(ByteWiseComparator {}, Arc::default());
    sl.put(b"aaa".to_vec(), b"val-a".to_vec());
    sl.put(b"ccc".to_vec(), b"val-c".to_vec());
    sl.put(b"bbb".to_vec(), b"val-b".to_vec());
    sl.put(b"eee".to_vec(), b"val-e".to_vec());
    sl.put(b"ddd".to_vec(), b"val-d".to_vec());

    let (tx, rx) = sync_channel(0);
    let rx = Arc::new(Mutex::new(rx));
    let (tx2, rx2) = sync_channel(0);
    let rx2 = Arc::new(Mutex::new(rx2));
    let mut iter = sl.iter();
    let h = std::thread::spawn(move || {
        fail::cfg_callback("on_try_acquire", move || {
            let _ = tx.send(1);
            let _ = rx2.lock().unwrap().recv();
        })
        .unwrap();
        iter.seek_for_prev(b"z");
        assert_eq!(iter.value(), &b"val-c".to_vec());

        iter.prev();
        assert_eq!(iter.value(), &b"val-a".to_vec());

        iter.prev();
        assert!(!iter.valid());
    });

    let sl_clone = sl.clone();
    let h2 = std::thread::spawn(move || {
        let _ = rx.lock().unwrap().recv().unwrap();
        sl_clone.remove(b"eee");
        tx2.send(1).unwrap();

        let _ = rx.lock().unwrap().recv().unwrap();
        sl_clone.remove(b"ddd");
        tx2.send(1).unwrap();

        // do nothing, so the iter can read "ccc"
        let _ = rx.lock().unwrap().recv().unwrap();
        tx2.send(1).unwrap();

        let _ = rx.lock().unwrap().recv().unwrap();
        sl_clone.remove(b"bbb");
        tx2.send(1).unwrap();

        // do nothing, so the iter can read "ccc"
        let _ = rx.lock().unwrap().recv().unwrap();
        tx2.send(1).unwrap();
    });

    h.join().unwrap();
    h2.join().unwrap();
}
