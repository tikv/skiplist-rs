// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    cmp, mem,
    ops::{Deref, Index},
    ptr,
    sync::{
        atomic::{fence, AtomicUsize, Ordering},
        Arc,
    },
};

use bytes::Bytes;
use crossbeam_epoch::{self, Collector, Guard, Shared};
use crossbeam_utils::cache_padded::CachePadded;

use crossbeam_epoch::Atomic;

use crate::Bound;

use super::{arena::Arena, KeyComparator};

/// Number of bits needed to store height.
const HEIGHT_BITS: usize = 5;
const MAX_HEIGHT: usize = 1 << HEIGHT_BITS;
const HEIGHT_MASK: usize = (1 << HEIGHT_BITS) - 1;

const U64_MOD_BITS: usize = !(mem::size_of::<u64>() - 1);

/// The bits of `refs_and_height` that keep the height.

/// The tower of atomic pointers.
///
/// The actual size of the tower will vary depending on the height that a node
/// was allocated with.
#[repr(C)]
#[derive(Debug)]
struct Tower {
    pointers: [Atomic<Node>; 0],
}

impl Index<usize> for Tower {
    type Output = Atomic<Node>;
    fn index(&self, index: usize) -> &Atomic<Node> {
        // This implementation is actually unsafe since we don't check if the
        // index is in-bounds. But this is fine since this is only used internally.
        unsafe { self.pointers.get_unchecked(index) }
    }
}

/// A search result.
///
/// The result indicates whether the key was found, as well as what were the adjacent nodes to the
/// key on each level of the skip list.
struct Position<'a> {
    /// Reference to a node with the given key, if found.
    ///
    /// If this is `Some` then it will point to the same node as `right[0]`.
    found: Option<&'a Node>,

    /// Adjacent nodes with smaller keys (predecessors).
    left: [&'a Tower; MAX_HEIGHT],

    /// Adjacent nodes with equal or greater keys (successors).
    right: [Shared<'a, Node>; MAX_HEIGHT],
}

/// Tower at the head of a skip list.
///
/// This is located in the `SkipList` struct itself and holds a full height
/// tower.
#[repr(C)]
struct Head {
    pointers: [Atomic<Node>; MAX_HEIGHT],
}

impl Head {
    /// Initializes a `Head`.
    #[inline]
    fn new() -> Head {
        // Initializing arrays in rust is a pain...
        Head {
            pointers: Default::default(),
        }
    }
}

impl Deref for Head {
    type Target = Tower;
    fn deref(&self) -> &Tower {
        unsafe { &*(self as *const _ as *const Tower) }
    }
}

// Uses C layout to make sure tower is at the bottom
#[derive(Debug)]
#[repr(C)]
pub struct Node {
    pub(crate) key: Bytes,
    pub(crate) value: Bytes,
    /// Keeps the reference count and the height of its tower.
    ///
    /// The reference count is equal to the number of `Entry`s pointing to this node, plus the
    /// number of levels in which this node is installed.
    refs_and_height: AtomicUsize,

    /// The tower of atomic pointers.
    tower: Tower,
}

impl Node {
    pub fn key(&self) -> &[u8] {
        &self.key
    }
}

pub const U_SIZE: usize = mem::size_of::<AtomicUsize>();

pub trait ReclaimableNode {
    fn size(&self) -> usize;

    fn drop_key_value(&mut self);
}

impl ReclaimableNode for Node {
    fn size(&self) -> usize {
        Node::node_size(self.height())
    }

    fn drop_key_value(&mut self) {
        println!("{:?}", self.key);
        unsafe {
            ptr::drop_in_place(&mut self.key);
            ptr::drop_in_place(&mut self.value);
        }
    }
}

pub trait MemoryLimiter: AllocationRecorder {
    fn acquire(&self, n: usize) -> bool;
    fn reclaim(&self, n: usize);
    fn mem_usage(&self) -> usize;
}

// todo(SpadeA): This is used for the purpose of recording the memory footprint.
// It should be removed in the future.
pub trait AllocationRecorder: Clone {
    fn allocated(&self, addr: usize, size: usize);
    fn freed(&self, addr: usize, size: usize);
}

impl Node {
    fn alloc<M: MemoryLimiter>(
        arena: &Arena<M>,
        key: Bytes,
        value: Bytes,
        height: usize,
        ref_count: usize,
    ) -> *mut Self {
        // Not all values in Node::tower will be utilized.
        let node_size = Node::node_size(height);
        let node_ptr = arena.alloc(node_size) as *mut Node;
        unsafe {
            let node = &mut *node_ptr;
            ptr::write(&mut node.key, key);
            ptr::write(&mut node.value, value);
            ptr::write(
                &mut node.refs_and_height,
                AtomicUsize::new((height - 1) | ref_count << HEIGHT_BITS),
            );
            ptr::write_bytes(node.tower.pointers.as_mut_ptr(), 0, height);
        }
        node_ptr
    }

    /// Returns the size of a node with tower of given `height` measured in `u64`s.
    fn node_size(height: usize) -> usize {
        assert!(1 <= height && height <= MAX_HEIGHT);
        assert!(mem::align_of::<Self>() <= mem::align_of::<u64>());

        let size_base = mem::size_of::<Self>();
        let size_ptr = mem::size_of::<Atomic<Self>>();

        let size_u64 = mem::size_of::<u64>();
        let size_self = size_base + size_ptr * height;

        (size_self + size_u64 - 1) & U64_MOD_BITS
    }

    /// Returns the height of this node's tower.
    #[inline]
    fn height(&self) -> usize {
        (self.refs_and_height.load(Ordering::Relaxed) & HEIGHT_MASK) + 1
    }

    fn mark_tower(&self) -> bool {
        let height = self.height();

        for level in (0..height).rev() {
            let tag = unsafe {
                // We're loading the pointer only for the tag, so it's okay to use
                // `epoch::unprotected()` in this situation.
                self.tower[level]
                    .fetch_or(1, Ordering::SeqCst, crossbeam_epoch::unprotected())
                    .tag()
            };

            // If the level 0 pointer was already marked, somebody else removed the node.
            if level == 0 && tag == 1 {
                return false;
            }
        }

        // We marked the level 0 pointer, therefore we removed the node.
        true
    }

    #[allow(dead_code)]
    #[inline]
    unsafe fn try_increment(&self) -> bool {
        let mut refs_and_height = self.refs_and_height.load(Ordering::Relaxed);

        loop {
            // If the reference count is zero, then the node has already been
            // queued for deletion. Incrementing it again could lead to a
            // double-free.
            if refs_and_height & !HEIGHT_MASK == 0 {
                return false;
            }

            // If all bits in the reference count are ones, we're about to overflow it.
            let new_refs_and_height = refs_and_height
                .checked_add(1 << HEIGHT_BITS)
                .expect("SkipList reference count overflow");

            // Try incrementing the count.
            match self.refs_and_height.compare_exchange_weak(
                refs_and_height,
                new_refs_and_height,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(current) => refs_and_height = current,
            }
        }
    }

    /// Decrements the reference count of a node, destroying it if the count becomes zero.
    #[inline]
    unsafe fn decrement<M: MemoryLimiter>(&self, guard: &Guard, arena: Arena<M>) {
        let current_ref = self
            .refs_and_height
            .fetch_sub(1 << HEIGHT_BITS, Ordering::Release)
            >> HEIGHT_BITS;
        if current_ref == 1 {
            fence(Ordering::Acquire);
            guard.defer(move || Self::finalize(self, &arena));
        }
    }

    /// Drops the key and value of a node, then deallocates it.
    #[cold]
    unsafe fn finalize<M: MemoryLimiter>(ptr: *const Self, arena: &Arena<M>) {
        let ptr = ptr as *mut Self;
        arena.free(ptr);
    }
}

/// Frequently modified data associated with a skip list.
struct HotData {
    /// The seed for random height generation.
    seed: AtomicUsize,

    /// Highest tower currently in use. This value is used as a hint for where
    /// to start lookups and never decreases.
    max_height: AtomicUsize,
}

struct SkiplistInner<M: MemoryLimiter> {
    /// The head of the skip list (just a dummy node, not a real entry).
    head: Head,
    /// Hot data associated with the skip list, stored in a dedicated cache line.
    hot_data: CachePadded<HotData>,
    /// The `Collector` associated with this skip list.
    collector: Collector,
    /// <emory management unit
    arena: Arena<M>,
}

// impl<M: MemoryLimiter> SkiplistInner<M> {
//     pub fn print(&self) {
//         println!("print the skiplist");
//         unsafe {
//             for i in (0..=self.height.load(Ordering::Relaxed)).rev() {
//                 let mut node_off = self.head.as_ref().tower[i].load(Ordering::Relaxed);

//                 print!("level {}", i);
//                 while node_off != 0 {
//                     let node: *mut Node = self.arena.get_mut(node_off);
//                     print!("  {:?} -->", (*node).key);
//                     node_off = (*node).tower[i].load(Ordering::Relaxed);
//                 }
//                 println!()
//             }
//         }
//     }
// }

unsafe impl<M: MemoryLimiter> Send for SkiplistInner<M> {}
unsafe impl<M: MemoryLimiter> Sync for SkiplistInner<M> {}

#[derive(Clone)]
pub struct Skiplist<C: KeyComparator, M: MemoryLimiter> {
    inner: Arc<SkiplistInner<M>>,
    c: C,
}

impl<C: KeyComparator, M: MemoryLimiter> Skiplist<C, M> {
    pub fn new(c: C, mem_limiter: Arc<M>, collector: Collector) -> Skiplist<C, M> {
        let arena = Arena::new(mem_limiter);
        Skiplist {
            inner: Arc::new(SkiplistInner {
                hot_data: CachePadded::new(HotData {
                    seed: AtomicUsize::new(1),
                    max_height: AtomicUsize::new(1),
                }),
                collector,
                head: Head::new(),
                arena,
            }),
            c,
        }
    }

    /// Generates a random height and returns it.
    fn random_height(&self) -> usize {
        // Pseudorandom number generation from "Xorshift RNGs" by George Marsaglia.
        //
        // This particular set of operations generates 32-bit integers. See:
        // https://en.wikipedia.org/wiki/Xorshift#Example_implementation
        let mut num = self.inner.hot_data.seed.load(Ordering::Relaxed);
        num ^= num << 13;
        num ^= num >> 17;
        num ^= num << 5;
        self.inner.hot_data.seed.store(num, Ordering::Relaxed);

        let mut height = cmp::min(MAX_HEIGHT, num.trailing_zeros() as usize + 1);
        unsafe {
            // Keep decreasing the height while it's much larger than all towers currently
            // in the skip list.
            //
            // Note that we're loading the pointer only to check whether it is null, so it's
            // okay to use `epoch::unprotected()` in this situation.
            while height >= 4
                && self.inner.head[height - 2]
                    .load(Ordering::Relaxed, crossbeam_epoch::unprotected())
                    .is_null()
            {
                height -= 1;
            }
        }

        // Track the max height to speed up lookups
        let mut max_height = self.inner.hot_data.max_height.load(Ordering::Relaxed);
        while height > max_height {
            match self.inner.hot_data.max_height.compare_exchange_weak(
                max_height,
                height,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(h) => max_height = h,
            }
        }
        height
    }

    /// Ensures that all `Guard`s used with the skip list come from the same
    /// `Collector`.
    fn check_guard(&self, guard: &Guard) {
        if let Some(c) = guard.collector() {
            assert!(c == &self.inner.collector);
        }
    }

    #[inline]
    fn height(&self) -> usize {
        self.inner.hot_data.max_height.load(Ordering::Relaxed)
    }
}

impl<C: KeyComparator, M: MemoryLimiter> Skiplist<C, M> {
    /// If we encounter a deleted node while searching, help with the deletion
    /// by attempting to unlink the node from the list.
    ///
    /// If the unlinking is successful then this function returns the next node
    /// with which the search should continue on the current level.
    unsafe fn help_unlink<'a>(
        &'a self,
        pred: &'a Atomic<Node>,
        curr: &'a Node,
        succ: Shared<'a, Node>,
        guard: &'a Guard,
    ) -> Option<Shared<'a, Node>> {
        // If `succ` is marked, that means `curr` is removed. Let's try
        // unlinking it from the skip list at this level.
        match pred.compare_and_set(
            Shared::from(curr as *const _),
            succ.with_tag(0),
            Ordering::Release,
            guard,
        ) {
            Ok(_) => {
                curr.decrement(guard, self.inner.arena.clone());
                Some(succ.with_tag(0))
            }
            Err(_) => None,
        }
    }

    /// Finds the node near to key.
    ///
    /// If upper_bound=true, it finds rightmost node such that node.key < key
    /// (if allow_equal=false) or node.key <= key (if allow_equal=true).
    /// If upper_bound=false, it finds leftmost node such that node.key > key
    /// (if allow_equal=false) or node.key >= key (if allow_equal=true).
    unsafe fn search_bound<'a>(
        &'a self,
        bound: Bound<&[u8]>,
        upper_bound: bool,
        guard: &'a Guard,
    ) -> Option<&'a Node> {
        'search: loop {
            let mut level = self.height();
            // Fast loop to skip empty tower levels.
            while level >= 1
                && self.inner.head[level - 1]
                    .load(Ordering::Relaxed, guard)
                    .is_null()
            {
                level -= 1;
            }

            let mut found = None;
            let mut pred = &*self.inner.head;

            while level >= 1 {
                level -= 1;

                // Two adjacent nodes at the current level.
                let mut curr = pred[level].load_consume(guard);

                // If `curr` is marked, that means `pred` is removed and we have to restart the
                // search.
                if curr.tag() == 1 {
                    continue 'search;
                }

                // Iterate through the current level until we reach a node with a key greater
                // than or equal to `key`.
                while let Some(c) = curr.as_ref() {
                    let succ = c.tower[level].load_consume(guard);

                    if succ.tag() == 1 {
                        if let Some(c) = self.help_unlink(&pred[level], c, succ, guard) {
                            curr = c;
                            continue;
                        } else {
                            // On failure, we cannot do anything reasonable to continue
                            // searching from the current position. Restart the search.
                            continue 'search;
                        }
                    }

                    // If `curr` contains a key that is greater than (or equal) to `key`, we're
                    // done with this level.
                    //
                    // The condition determines whether we should stop the search. For the upper
                    // bound, we return the last node before the condition became true. For the
                    // lower bound, we return the first node after the condition became true.
                    if upper_bound {
                        if !below_upper_bound(&self.c, &bound, &c.key) {
                            break;
                        }
                        found = Some(c);
                    } else if above_lower_bound(&self.c, &bound, &c.key) {
                        found = Some(c);
                        break;
                    }

                    // Move one step forward.
                    pred = &c.tower;
                    curr = succ;
                }
            }
            return found;
        }
    }

    unsafe fn search_position<'a>(&'a self, key: &[u8], guard: &'a Guard) -> Position<'a> {
        unsafe {
            'search: loop {
                // The result of this search.
                let mut result = Position {
                    found: None,
                    left: [&*self.inner.head; MAX_HEIGHT],
                    right: [Shared::null(); MAX_HEIGHT],
                };

                let mut level = self.height();

                // Fast loop to skip empty tower levels.
                while level >= 1
                    && self.inner.head[level - 1]
                        .load(Ordering::Relaxed, guard)
                        .is_null()
                {
                    level -= 1;
                }

                let mut pred = &*self.inner.head;

                while level >= 1 {
                    level -= 1;

                    // Two adjacent nodes at the current level.
                    let mut curr = pred[level].load_consume(guard);

                    // If `curr` is marked, that means `pred` is removed and we have to restart the
                    // search.
                    if curr.tag() == 1 {
                        continue 'search;
                    }

                    // Iterate through the current level until we reach a node with a key greater
                    // than or equal to `key`.
                    while let Some(c) = curr.as_ref() {
                        let succ = c.tower[level].load_consume(guard);

                        if curr.tag() == 1 {
                            if let Some(c) = self.help_unlink(&pred[level], c, succ, guard) {
                                // On success, continue searching through the current level.
                                curr = c;
                                continue;
                            } else {
                                // On failure, we cannot do anything reasonable to continue
                                // searching from the current position. Restart the search.
                                continue 'search;
                            }
                        }

                        // If `curr` contains a key that is greater than or equal to `key`, we're
                        // done with this level.
                        match self.c.compare_key(&c.key, key) {
                            cmp::Ordering::Greater => break,
                            cmp::Ordering::Equal => {
                                result.found = Some(c);
                                break;
                            }
                            cmp::Ordering::Less => {}
                        }

                        // Move one step forward.
                        pred = &c.tower;
                        curr = succ;
                    }

                    // Store the position at the current level into the result.
                    result.left[level] = pred;
                    result.right[level] = curr;
                }

                return result;
            }
        }
    }

    pub fn remove(&self, key: &[u8], guard: &Guard) -> bool {
        self.check_guard(guard);

        unsafe {
            // Rebind the guard to the lifetime of self. This is a bit of a
            // hack but it allows us to return references that are not bound to
            // the lifetime of the guard.
            let guard = &*(guard as *const _);

            loop {
                // Try searching for the key.
                let search = self.search_position(key, guard);

                let Some(n) = search.found else {
                    return false;
                };

                // Try removing the node by marking its tower.
                if n.mark_tower() {
                    for level in (0..n.height()).rev() {
                        let succ = n.tower[level].load(Ordering::SeqCst, guard).with_tag(0);

                        if search.left[level][level]
                            .compare_and_set(
                                Shared::from(n as *const _),
                                succ,
                                Ordering::SeqCst,
                                guard,
                            )
                            .is_ok()
                        {
                            // Success! Decrement the reference count.
                            n.decrement(guard, self.inner.arena.clone());
                        } else {
                            self.search_bound(Bound::Included(key), false, guard);
                            break;
                        }
                    }

                    return true;
                }
            }
        }
    }

    pub fn put(&self, key: impl Into<Bytes>, value: impl Into<Bytes>, guard: &Guard) -> bool {
        let (key, value) = (key.into(), value.into());
        self.check_guard(guard);

        unsafe {
            // Rebind the guard to the lifetime of self. This is a bit of a
            // hack but it allows us to return references that are not bound to
            // the lifetime of the guard.
            let guard = &*(guard as *const _);

            let mut search;
            // First try searching for the key.
            // Note that the `Ord` implementation for `K` may panic during the search.
            search = self.search_position(&key, guard);
            if let Some(_) = search.found {
                // panic!("Overwrite is not supported, {:?}", (*r).key);
                return false;
            }

            let height = self.random_height();
            let (node, n) = {
                let n = Node::alloc(&self.inner.arena, key, value, height, 1);
                (Shared::<Node>::from(n as *const _), &*n)
            };
            loop {
                // Set the lowest successor of `n` to `search.right[0]`.
                n.tower[0].store(search.right[0], Ordering::Relaxed);

                // Try installing the new node into the skip list (at level 0).
                if search.left[0][0]
                    .compare_and_set(search.right[0], node, Ordering::SeqCst, guard)
                    .is_ok()
                {
                    break;
                }

                // We failed. Let's search for the key and try again.
                {
                    // Create a guard that destroys the new node in case search panics.
                    let sg = scopeguard::guard((), |_| {
                        Node::finalize(node.as_raw(), &self.inner.arena.clone())
                    });
                    search = self.search_position(&n.key, guard);
                    mem::forget(sg);
                }

                if let Some(_) = search.found {
                    // panic!("Overwrite is not supported, {:?}", (*r).key);
                    return false;
                }
            }

            // Build the rest of the tower above level 0.
            'build: for level in 1..height {
                loop {
                    // Obtain the predecessor and successor at the current level.
                    let pred = search.left[level];
                    let succ = search.right[level];

                    // Load the current value of the pointer in the tower at this level.
                    let next = n.tower[level].load(Ordering::SeqCst, guard);

                    // If the current pointer is marked, that means another thread is already
                    // removing the node we've just inserted. In that case, let's just stop
                    // building the tower.
                    if next.tag() == 1 {
                        break 'build;
                    }

                    // Change the pointer at the current level from `next` to `succ`. If this CAS
                    // operation fails, that means another thread has marked the pointer and we
                    // should stop building the tower.
                    if n.tower[level]
                        .compare_and_set(next, succ, Ordering::SeqCst, guard)
                        .is_err()
                    {
                        break 'build;
                    }

                    // Increment the reference count. Heigher means more ref.
                    n.refs_and_height
                        .fetch_add(1 << HEIGHT_BITS, Ordering::Relaxed);

                    // Try installing the new node at the current level.
                    if pred[level]
                        .compare_and_set(succ, node, Ordering::SeqCst, guard)
                        .is_ok()
                    {
                        // Success! Continue on the next level.
                        break;
                    }

                    // Installation failed. Decrement the reference count.
                    (*n).refs_and_height
                        .fetch_sub(1 << HEIGHT_BITS, Ordering::Relaxed);

                    // We don't have the most up-to-date search results. Repeat the search.
                    //
                    // If this search panics, we simply stop building the tower without breaking
                    // any invariants. Note that building higher levels is completely optional.
                    // Only the lowest level really matters, and all the higher levels are there
                    // just to make searching faster.
                    search = self.search_position(&n.key, guard);
                }
            }

            // If any pointer in the tower is marked, that means our node is in the process of
            // removal or already removed. It is possible that another thread (either partially or
            // completely) removed the new node while we were building the tower, and just after
            // that we installed the new node at one of the higher levels. In order to undo that
            // installation, we must repeat the search, which will unlink the new node at that
            // level.
            // TODO(Amanieu): can we use relaxed ordering here?
            if n.tower[height - 1].load(Ordering::SeqCst, guard).tag() == 1 {
                self.search_bound(Bound::Included(&n.key), false, guard);
            }
        }

        true
    }

    pub fn is_empty(&self) -> bool {
        true
    }

    pub fn get<'a>(&self, key: &[u8], guard: &'a Guard) -> Option<Entry<'a>> {
        self.check_guard(guard);
        if let Some(n) = unsafe { self.search_bound(Bound::Included(key), false, guard) } {
            if self.c.same_key(&n.key, key) {
                return Some(unsafe { Entry::new(n, guard) });
            }
        }

        None
    }

    fn next_node<'a>(
        &self,
        pred: &Tower,
        lower_bound: Bound<&[u8]>,
        guard: &'a Guard,
    ) -> Option<&'a Node> {
        unsafe {
            // Load the level 0 successor of the current node.
            let mut curr = pred[0].load_consume(guard);

            // If `curr` is marked, that means `pred` is removed and we have to use
            // a key search.
            if curr.tag() == 1 {
                return self
                    .search_bound(lower_bound, false, guard)
                    .map(|n| Entry::lifetime_with_guard(n, guard));
            }

            while let Some(c) = curr.as_ref() {
                let succ = c.tower[0].load_consume(guard);

                if succ.tag() == 1 {
                    if let Some(c) = self.help_unlink(&pred[0], c, succ, guard) {
                        // On success, continue searching through the current level.
                        curr = c;
                        continue;
                    } else {
                        // On failure, we cannot do anything reasonable to continue
                        // searching from the current position. Restart the search.
                        return self
                            .search_bound(lower_bound, false, guard)
                            .map(|n| Entry::lifetime_with_guard(n, guard));
                    }
                }

                return Some(Entry::lifetime_with_guard(c, guard));
            }

            None
        }
    }

    // pub fn iter_ref(&self) -> IterRef<&Skiplist<C, M>, C, M> {
    //     IterRef {
    //         list: self,
    //         cursor: NodeWrap(ptr::null()),
    //         _key_cmp: std::marker::PhantomData,
    //         _limiter: std::marker::PhantomData,
    //     }
    // }

    pub fn iter<'a>(&self, guard: &'a Guard) -> IterRef<'a, Skiplist<C, M>, C, M> {
        IterRef {
            list: self.clone(),
            cursor: None,
            guard,
            _key_cmp: std::marker::PhantomData,
            _limiter: std::marker::PhantomData,
        }
    }

    pub fn mem_size(&self) -> usize {
        self.inner.arena.limiter.mem_usage()
    }
}

impl<C: KeyComparator, M: MemoryLimiter> AsRef<Skiplist<C, M>> for Skiplist<C, M> {
    fn as_ref(&self) -> &Skiplist<C, M> {
        self
    }
}

impl<M: MemoryLimiter> Drop for SkiplistInner<M> {
    fn drop(&mut self) {
        unsafe {
            let mut node = self.head[0]
                .load(Ordering::Relaxed, crossbeam_epoch::unprotected())
                .as_ref();

            // Iterate through the whole skip list and destroy every node.
            while let Some(n) = node {
                // Unprotected loads are okay because this function is the only one currently using
                // the skip list.
                let next = n.tower[0]
                    .load(Ordering::Relaxed, crossbeam_epoch::unprotected())
                    .as_ref();

                // Deallocate every node.
                Node::finalize(n, &self.arena);

                node = next;
            }
        }
    }
}

unsafe impl<C: Send + KeyComparator, M: MemoryLimiter> Send for Skiplist<C, M> {}
unsafe impl<C: Sync + KeyComparator, M: MemoryLimiter> Sync for Skiplist<C, M> {}

/// An entry in a skip list, protected by a `Guard`.
///
/// The lifetimes of the key and value are the same as that of the `Guard`
/// used when creating the `Entry` (`'g`).
pub struct Entry<'g> {
    node: &'g Node,
}

impl<'g> Entry<'g> {
    unsafe fn new(node: &Node, _: &'g Guard) -> Self {
        Self {
            node: &*(node as *const _),
        }
    }

    unsafe fn lifetime_with_guard(node: &Node, _: &'g Guard) -> &'g Node {
        &*(node as *const _)
    }

    pub fn value(&self) -> &Bytes {
        &self.node.value
    }
}

pub struct IterRef<'g, T, C: KeyComparator, M: MemoryLimiter>
where
    T: AsRef<Skiplist<C, M>>,
{
    list: T,
    cursor: Option<&'g Node>,
    guard: &'g Guard,
    _key_cmp: std::marker::PhantomData<C>,
    _limiter: std::marker::PhantomData<M>,
}

impl<'g, T: AsRef<Skiplist<C, M>>, C: KeyComparator, M: MemoryLimiter> IterRef<'g, T, C, M> {
    pub fn valid(&self) -> bool {
        self.cursor.is_some()
    }

    pub fn key(&self) -> &Bytes {
        assert!(self.valid());
        &(&**self.cursor.as_ref().unwrap()).key
    }

    pub fn value(&self) -> &Bytes {
        assert!(self.valid());
        // &self.cursor.as_ref().unwrap().value
        unimplemented!()
    }

    pub fn next(&mut self) {
        assert!(self.valid());
        self.cursor = match self.cursor {
            Some(n) => self
                .list
                .as_ref()
                .next_node(&n.tower, Bound::Excluded(&n.key), self.guard),
            None => unreachable!(),
        }
    }

    pub fn prev(&mut self) {
        assert!(self.valid());
        unsafe {
            self.cursor = match self.cursor {
                Some(n) => self
                    .list
                    .as_ref()
                    .search_bound(Bound::Excluded(&n.key), true, self.guard)
                    .map(|n| Entry::lifetime_with_guard(n, self.guard)),
                None => None,
            };
        }
    }

    pub fn seek(&mut self, target: &[u8]) {
        unsafe {
            self.cursor = self
                .list
                .as_ref()
                .search_bound(Bound::Included(target), false, self.guard)
                .map(|n| Entry::lifetime_with_guard(n, self.guard));
        }
    }

    pub fn seek_for_prev(&mut self, target: &[u8]) {
        unsafe {
            self.cursor = self
                .list
                .as_ref()
                .search_bound(Bound::Included(target), true, self.guard)
                .map(|n| Entry::lifetime_with_guard(n, self.guard));
        }
    }

    pub fn seek_to_first(&mut self, guard: &'g Guard) {
        self.list.as_ref().check_guard(guard);
        let pred = &self.list.as_ref().inner.head;
        self.cursor = self.list.as_ref().next_node(pred, Bound::Unbounded, guard);
    }
}

/// Helper function to check if a value is above a lower bound
fn above_lower_bound<C: KeyComparator>(c: &C, bound: &Bound<&[u8]>, key: &[u8]) -> bool {
    match *bound {
        Bound::Unbounded => true,
        Bound::Included(bound) => matches!(
            c.compare_key(key, bound),
            cmp::Ordering::Greater | cmp::Ordering::Equal
        ),
        Bound::Excluded(bound) => matches!(c.compare_key(key, bound), cmp::Ordering::Greater),
    }
}

/// Helper function to check if a value is below an upper bound
fn below_upper_bound<C: KeyComparator>(c: &C, bound: &Bound<&[u8]>, key: &[u8]) -> bool {
    match *bound {
        Bound::Unbounded => true,
        Bound::Included(bound) => matches!(
            c.compare_key(bound, key),
            cmp::Ordering::Greater | cmp::Ordering::Equal
        ),
        Bound::Excluded(bound) => matches!(c.compare_key(bound, key), cmp::Ordering::Greater),
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Mutex, thread};

    use super::*;
    use crate::key::ByteWiseComparator;

    #[cfg(not(target_env = "msvc"))]
    use tikv_jemallocator::Jemalloc;

    #[cfg(not(target_env = "msvc"))]
    #[global_allocator]
    static GLOBAL: Jemalloc = Jemalloc;

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

    #[derive(Debug, Clone)]
    struct DummyLimiter;

    impl Drop for DummyLimiter {
        fn drop(&mut self) {}
    }

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

    fn construct_key(i: i32) -> Vec<u8> {
        format!("key-{:08}", i).into_bytes()
    }

    fn construct_val(i: i32) -> Vec<u8> {
        format!("val-{}", i).into_bytes()
    }

    fn default_list() -> Skiplist<ByteWiseComparator, RecorderLimiter> {
        Skiplist::new(
            ByteWiseComparator {},
            Arc::default(),
            crossbeam_epoch::default_collector().clone(),
        )
    }

    fn sl_insert(
        sl: &Skiplist<ByteWiseComparator, RecorderLimiter>,
        k: i32,
        v: i32,
        guard: &Guard,
    ) -> bool {
        let k = construct_key(k);
        let v = construct_val(v);
        sl.put(k, v, guard)
    }

    fn sl_remove(
        sl: &Skiplist<ByteWiseComparator, RecorderLimiter>,
        k: i32,
        guard: &Guard,
    ) -> bool {
        let k = construct_key(k);
        sl.remove(&k, guard)
    }

    fn sl_get_assert<'a>(
        sl: &'a Skiplist<ByteWiseComparator, RecorderLimiter>,
        k: i32,
        v: Option<i32>,
        guard: &Guard,
    ) {
        let k = construct_key(k);
        let res = sl.get(&k, guard);
        if let Some(v) = v {
            let v = construct_val(v);
            assert_eq!(res.unwrap().value(), &v);
        } else {
            assert!(res.is_none());
        }
    }

    #[test]
    fn insert() {
        let insert = [0, 4, 2, 12, 8, 7, 11, 5];
        let not_present = [1, 3, 6, 9, 10];
        let s = default_list();

        let guard = &crossbeam_epoch::pin();
        for i in insert {
            sl_insert(&s, i, i * 10, guard);
            sl_get_assert(&s, i, Some(i * 10), guard);
        }

        for i in not_present {
            sl_get_assert(&s, i, None, guard);
        }
    }

    #[test]
    fn remove() {
        let guard = &crossbeam_epoch::pin();
        let insert = [0, 4, 2, 12, 8, 7, 11, 5];
        let not_present = [1, 3, 6, 9, 10];
        let remove = [2, 12, 8];
        let remaining = [0, 4, 5, 7, 11];

        let s = default_list();

        for &x in &insert {
            sl_insert(&s, x, x * 10, guard);
        }
        for &x in &not_present {
            sl_remove(&s, x, guard);
        }
        for &x in &remove {
            sl_remove(&s, x, guard);
        }

        let mut v = vec![];
        let mut iter = s.iter(guard);
        iter.seek_to_first(guard);
        iter.valid();
        while iter.valid() {
            v.push(iter.key().to_vec());
            iter.next();
        }

        for (&remain, k) in remaining.iter().zip(v.into_iter()) {
            let remain = construct_key(remain);
            assert_eq!(remain, k);
        }

        for &x in &insert {
            sl_remove(&s, x, guard);
        }
        assert!(s.is_empty());
    }

    fn assert_keys(
        s: &Skiplist<ByteWiseComparator, RecorderLimiter>,
        expected: Vec<i32>,
        guard: &Guard,
    ) {
        let mut iter = s.iter(guard);
        iter.seek_to_first(guard);
        let mut expect_iter = expected.iter();
        while iter.valid() {
            let expect_k = construct_key(*expect_iter.next().unwrap());
            assert_eq!(iter.key(), &expect_k);
            iter.next();
        }
        assert!(expect_iter.next().is_none());
    }

    #[test]
    fn insert_and_remove() {
        let collector = crossbeam_epoch::default_collector();
        let handle = collector.register();

        {
            let guard = &handle.pin();
            let s = default_list();
            sl_insert(&s, 3, 0, guard);
            sl_insert(&s, 5, 0, guard);
            sl_insert(&s, 1, 0, guard);
            sl_insert(&s, 4, 0, guard);
            sl_insert(&s, 2, 0, guard);
            assert_keys(&s, vec![1, 2, 3, 4, 5], guard);

            assert!(sl_remove(&s, 4, guard));
            assert_keys(&s, vec![1, 2, 3, 5], guard);
            assert!(sl_remove(&s, 3, guard));
            assert_keys(&s, vec![1, 2, 5], guard);
            assert!(sl_remove(&s, 1, guard));
            assert_keys(&s, vec![2, 5], guard);

            assert!(!sl_remove(&s, 1, guard));
            assert_keys(&s, vec![2, 5], guard);
            assert!(!sl_remove(&s, 3, guard));
            assert_keys(&s, vec![2, 5], guard);

            assert!(sl_remove(&s, 2, guard));
            assert_keys(&s, vec![5], guard);
            assert!(sl_remove(&s, 5, guard));
            assert_keys(&s, vec![], guard);

            sl_insert(&s, 3, 0, guard);
            assert_keys(&s, vec![3], guard);
            sl_insert(&s, 1, 0, guard);
            assert_keys(&s, vec![1, 3], guard);
            // overwrite
            assert!(!sl_insert(&s, 3, 0, guard));
            assert_keys(&s, vec![1, 3], guard);
            sl_insert(&s, 5, 0, guard);
            assert_keys(&s, vec![1, 3, 5], guard);

            assert!(sl_remove(&s, 3, guard));
            assert_keys(&s, vec![1, 5], guard);
            assert!(sl_remove(&s, 1, guard));
            assert_keys(&s, vec![5], guard);
            assert!(!sl_remove(&s, 3, guard));
            assert_keys(&s, vec![5], guard);
            assert!(sl_remove(&s, 5, guard));
            assert_keys(&s, vec![], guard);
        }
    }

    #[test]
    fn get() {
        let guard = &crossbeam_epoch::pin();
        let s = default_list();
        sl_insert(&s, 30, 3, guard);
        sl_insert(&s, 50, 5, guard);
        sl_insert(&s, 10, 1, guard);
        sl_insert(&s, 40, 4, guard);
        sl_insert(&s, 20, 2, guard);

        sl_get_assert(&s, 10, Some(1), guard);
        sl_get_assert(&s, 20, Some(2), guard);
        sl_get_assert(&s, 30, Some(3), guard);
        sl_get_assert(&s, 40, Some(4), guard);
        sl_get_assert(&s, 50, Some(5), guard);

        sl_get_assert(&s, 7, None, guard);
        sl_get_assert(&s, 27, None, guard);
        sl_get_assert(&s, 31, None, guard);
        sl_get_assert(&s, 97, None, guard);
    }

    #[test]
    fn iter() {
        let collector = crossbeam_epoch::default_collector();
        let handle = collector.register();

        {
            let guard = &handle.pin();
            let s = default_list();
            for &x in &[4, 2] {
                sl_insert(&s, x, x * 10, guard);
            }

            assert_keys(&s, vec![2, 4], guard);

            let mut it = s.iter(guard);
            it.seek_to_first(guard);
            // `it` is already pointing on 2
            sl_remove(&s, 2, guard);
            let k = construct_key(2);
            assert_eq!(it.key(), &k);
            // deleting the next key that the iterator would point to makes the iterator skip the key
            sl_remove(&s, 4, guard);
            it.next();
        }

        handle.pin().flush();
        handle.pin().flush();
    }

    #[test]
    fn test_seek() {
        let insert = [0, 4, 2, 12, 8, 7, 11, 5];
        let s = default_list();

        let guard = &crossbeam_epoch::pin();
        for i in insert {
            sl_insert(&s, i, i * 10, guard);
        }

        let mut iter = s.iter(guard);
        let k = construct_key(3);
        iter.seek(&k);
        let expected_k = construct_key(4);
        assert_eq!(iter.key(), &expected_k);

        let k = construct_key(12);
        iter.seek(&k);
        assert_eq!(iter.key(), &k);

        let k = construct_key(13);
        iter.seek(&k);
        assert!(!iter.valid());
    }

    #[test]
    fn test_prev() {
        let insert = [0, 4, 2, 12, 8, 7, 11, 5];
        let s = default_list();

        let guard = &crossbeam_epoch::pin();
        for i in insert {
            sl_insert(&s, i, i * 10, guard);
        }

        let mut iter = s.iter(guard);
        let mut iter2 = s.iter(guard);
        let k = construct_key(20);
        iter.seek_for_prev(&k);
        for &i in [0, 2, 4, 5, 7, 8, 11, 12].iter().rev() {
            let k = construct_key(i);
            assert_eq!(iter.key(), &k);
            iter2.seek_for_prev(&k);
            assert_eq!(iter2.key(), &k);
            iter.prev()
        }
    }

    #[test]
    fn concurrent_put_and_remove() {
        let sl = default_list();
        let n = 100000;
        for i in (0..n).step_by(2) {
            let guard = &crossbeam_epoch::pin();
            let k = format!("k{:04}", i).into_bytes();
            let v = format!("v{:04}", i).into_bytes();
            sl.put(k, v, guard);
        }
        let sl1 = sl.clone();
        let h1 = thread::spawn(move || {
            for i in (1..n).step_by(2) {
                let guard = &crossbeam_epoch::pin();
                let k = format!("k{:04}", i).into_bytes();
                let v = format!("v{:04}", i).into_bytes();
                sl1.put(k, v, guard);
            }
        });
        let sl1 = sl.clone();
        let h2 = thread::spawn(move || {
            for i in (0..n).step_by(2) {
                let guard = &crossbeam_epoch::pin();
                let k = format!("k{:04}", i);
                sl1.remove(k.as_bytes(), guard);
            }
        });
        h1.join().unwrap();
        h2.join().unwrap();

        for i in (1..n).step_by(2) {
            let guard = &crossbeam_epoch::pin();
            let k = format!("k{:04}", i);
            let v = format!("v{:04}", i);
            assert_eq!(sl.get(k.as_bytes(), guard).unwrap().value(), v.as_bytes());
        }
    }
}
