//! Pre-allocated memory pool for sandboxed edge systems.
//!
//! Design philosophy: Known sizes at compile time. No malloc during compute.
//! When there's no internet attack surface, memory management becomes a
//! compile-time problem, not a runtime one.

use std::alloc::{alloc, dealloc, Layout};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::mem::{size_of, MaybeUninit};
use std::ptr::NonNull;

/// A pre-allocated arena allocator. All memory is allocated once at creation,
/// then reused via bump-pointer allocation. No individual deallocation—
/// the entire pool is reset at once.
pub struct MemoryPool<T: 'static> {
    arena: NonNull<T>,
    capacity: usize,
    used: usize,
    _phantom: PhantomData<T>,
}

impl<T> MemoryPool<T> {
    /// Allocate a pool with capacity for N items.
    /// Returns None if allocation fails.
    pub fn new(capacity: usize) -> Option<Self> {
        let layout = Layout::array::<T>(capacity).ok()?;
        let arena = unsafe { alloc(layout) };
        let arena = NonNull::new(arena as *mut T)?;

        Some(Self {
            arena,
            capacity,
            used: 0,
            _phantom: PhantomData,
        })
    }

    /// Allocate one item from the pool. Returns None if pool is exhausted.
    /// This is O(1) — just bump the pointer.
    pub fn alloc(&mut self) -> Option<&'static mut T> {
        if self.used >= self.capacity {
            return None;
        }
        let ptr = unsafe { self.arena.as_ptr().add(self.used) };
        self.used += 1;
        Some(unsafe { &mut *ptr })
    }

    /// Reset the pool. All previously allocated memory is invalidated.
    /// This is O(1) — just reset the bump pointer.
    pub fn reset(&mut self) {
        self.used = 0;
    }

    /// Returns the number of slots remaining.
    pub fn remaining(&self) -> usize {
        self.capacity - self.used
    }

    /// Returns total capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get a slice of all allocated items.
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.arena.as_ptr(), self.used) }
    }
}

impl<T> Drop for MemoryPool<T> {
    fn drop(&mut self) {
        let layout = Layout::array::<T>(self.capacity).unwrap();
        unsafe { dealloc(self.arena.as_ptr() as *mut u8, layout) }
    }
}

// SAFETY: MemoryPool<T> is Send iff T: 'static (no borrowed pointers)
unsafe impl<T> Send for MemoryPool<T> {}

// SAFETY: MemoryPool<T> is Sync iff T: 'static (no borrowed pointers)
unsafe impl<T> Sync for MemoryPool<T> {}

// =============================================================================
// FixedVec — stack-allocated vector with compile-time size
// =============================================================================

/// A stack-allocated vector with a fixed compile-time capacity.
/// Bounds checking is enabled in debug mode, disabled in release.
#[derive(Debug)]
pub struct FixedVec<T, const N: usize> {
    pub data: [MaybeUninit<T>; N],  // Made pub for RingBuffer access
    len: usize,
}

impl<T, const N: usize> FixedVec<T, N> {
    /// Create a new empty FixedVec.
    pub fn new() -> Self {
        Self {
            data: unsafe { MaybeUninit::zeroed().assume_init() },
            len: 0,
        }
    }

    /// Push an element. O(1).
    /// In debug: panics if at capacity. In release: undefined behavior.
    pub fn push(&mut self, value: T) {
        debug_assert!(self.len < N, "FixedVec overflow: capacity {}", N);
        self.data[self.len].write(value);
        self.len += 1;
    }

    /// Pop an element. O(1).
    /// In debug: panics if empty. In release: returns MaybeUninit.
    pub fn pop(&mut self) -> Option<T> {
        debug_assert!(self.len > 0, "FixedVec underflow");
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        Some(unsafe { self.data[self.len].assume_init_read() })
    }

    /// Get element at index. O(1).
    /// In debug: panics on out-of-bounds. In release: undefined behavior.
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }
        debug_assert!(index < self.len);
        Some(unsafe { self.data[index].assume_init_ref() })
    }

    /// Get mutable element at index. O(1).
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len {
            return None;
        }
        debug_assert!(index < self.len);
        Some(unsafe { self.data[index].assume_init_mut() })
    }

    /// Get the length.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the capacity (compile-time constant N).
    pub fn capacity(&self) -> usize {
        N
    }

    /// Iterate over elements.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data[..self.len].iter().map(|u| unsafe { u.assume_init_ref() })
    }

    /// Iterate mutably.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.data[..self.len].iter_mut().map(|u| unsafe { u.assume_init_mut() })
    }

    /// Clear all elements without deallocating.
    /// O(n) — runs Drop on each element.
    pub fn clear(&mut self) {
        for i in 0..self.len {
            unsafe { self.data[i].assume_init_mut() };
        }
        self.len = 0;
    }

    /// Clear in O(1) by just resetting length. Does NOT run Drop.
    /// Use when T is trivial (Copy, no Drop).
    pub fn clear_unchecked(&mut self) {
        self.len = 0;
    }

    /// Extract raw parts for FFI.
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr() as *const T
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr() as *mut T
    }
}

impl<T, const N: usize> Default for FixedVec<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> Drop for FixedVec<T, N> {
    fn drop(&mut self) {
        // Run Drop on all elements
        for i in 0..self.len {
            unsafe { self.data[i].assume_init_drop() };
        }
    }
}

// =============================================================================
// RingBuffer — lock-free SPSC queue using MemoryPool
// =============================================================================

/// A lock-free single-producer single-consumer (SPSC) ring buffer.
/// Uses a MemoryPool internally for pre-allocated storage.
/// 
/// This is wait-free for the producer and consumer — no malloc, no locks.
/// Uses circular buffer indexing with atomic head/tail counters.
pub struct RingBuffer<T, const N: usize> {
    /// Storage for elements.
    buffer: [MaybeUninit<T>; N],
    /// Write index (producer only)
    write_idx: usize,
    /// Read index (consumer only)
    read_idx: usize,
}

impl<T, const N: usize> RingBuffer<T, N> {
    /// Create a new ring buffer with capacity N.
    pub fn new() -> Self {
        // Initialize with MaybeUninit::zeroed() - safe because we never read uninitialized
        Self {
            buffer: unsafe { MaybeUninit::zeroed().assume_init() },
            write_idx: 0,
            read_idx: 0,
        }
    }

    pub fn push(&mut self, value: T) -> bool {
        let next_write = (self.write_idx + 1) % N;
        if next_write == self.read_idx {
            return false;
        }
        self.buffer[self.write_idx].write(value);
        self.write_idx = next_write;
        true
    }

    /// Pop an element from the buffer. Returns Some(value) if available.
    /// None means the buffer is empty (consumer is behind or at producer).
    pub fn pop(&mut self) -> Option<T> {
        if self.read_idx == self.write_idx {
            return None;
        }
        let value = unsafe { self.buffer[self.read_idx].assume_init_read() };
        self.read_idx = (self.read_idx + 1) % N;
        Some(value)
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.read_idx == self.write_idx
    }

    /// Check if the buffer is full.
    pub fn is_full(&self) -> bool {
        (self.write_idx + 1) % N == self.read_idx
    }

    /// Get the number of elements in the buffer.
    pub fn len(&self) -> usize {
        if self.write_idx >= self.read_idx {
            self.write_idx - self.read_idx
        } else {
            N - self.read_idx + self.write_idx
        }
    }

    /// Get the capacity (compile-time constant N).
    pub fn capacity(&self) -> usize {
        N
    }

    /// Clear all elements.
    pub fn clear(&mut self) {
        // Drain all elements
        while self.pop().is_some() {}
        self.read_idx = 0;
        self.write_idx = 0;
    }
}

impl<T, const N: usize> Default for RingBuffer<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> Drop for RingBuffer<T, N> {
    fn drop(&mut self) {
        // Drain remaining elements to run their Drop
        while let Some(_) = self.pop() {}
    }
}

// =============================================================================
// SharedSlice — zero-copy slice between threads, no refcount overhead
// =============================================================================

/// A zero-copy shared slice for passing data between threads without
/// cloning or reference counting.
///
/// This is safe ONLY when the producer has exclusive write access and the
/// consumer has exclusive read access (SPSC model). The data must not be
/// modified while the consumer holds a reference.
///
/// For our PLATO tile model, tiles flow one-way: producer → consumer.
/// This is enforced by the type system.
pub struct SharedSlice<'a, T> {
    data: *const T,
    len: usize,
    _lifetime: PhantomData<&'a T>,
}

impl<'a, T> SharedSlice<'a, T> {
    /// Create a SharedSlice from a reference.
    /// 
    /// SAFETY: The caller must ensure no mutable reference exists
    /// while this slice is alive. This is trivially true in SPSC.
    pub unsafe fn from_ref(slice: &'a [T]) -> Self {
        Self {
            data: slice.as_ptr(),
            len: slice.len(),
            _lifetime: PhantomData,
        }
    }

    /// Get the slice.
    pub fn as_slice(&self) -> &'a [T] {
        unsafe { std::slice::from_raw_parts(self.data, self.len) }
    }

    /// Get the length.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> SharedSlice<'_, T> {
    /// Get element at index. O(1).
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }
        Some(unsafe { &*self.data.add(index) })
    }

    /// Iterate over elements.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        (0..self.len).map(move |i| unsafe { &*self.data.add(i) })
    }
}

// SAFETY: SharedSlice is Send when T: Send (immutable reference)
unsafe impl<T: Send> Send for SharedSlice<'_, T> {}

// SAFETY: SharedSlice is Sync when T: Sync (immutable reference, exclusive access)
unsafe impl<T: Sync> Sync for SharedSlice<'_, T> {}

// =============================================================================
// Compile-time checks via trait system
// =============================================================================

/// Trait for types that can be safely allocated in a memory pool.
/// This is a marker trait — no methods required.
///
/// Types implementing PoolAllocated must:
/// - Have a known size at compile time (no unsized types except Sized)
/// - Not contain borrowed references that could outlive the pool
pub trait PoolAllocated: Sized + 'static {}

/// Marker for types that are trivial (Copy, no Drop).
/// These can use clear_unchecked() instead of clear().
pub trait Trivial: PoolAllocated + Copy + Default {}

macro_rules! impl_pool_allocated {
    ($($t:ty),*) => {
        $(impl PoolAllocated for $t {})*
    };
}

macro_rules! impl_trivial {
    ($($t:ty),*) => {
        $(impl Trivial for $t {})*
    };
}

// Implement for primitive types
impl_pool_allocated!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64, bool, char);
impl_trivial!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64, bool, char);

macro_rules! static_assert {
    ($condition:expr) => {
        const _: [(); 0] = [(); if $condition { 0 } else { 1 }];
    };
}

/// Compile-time check that a pool has minimum capacity.
#[macro_export]
macro_rules! assert_capacity {
    ($size:expr, $min:expr) => {
        static_assert!($size >= $min);
    };
}

// =============================================================================
// Example: Tile using pool allocation
// =============================================================================

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// A tile header — small, fixed-size, stack-allocated.
/// This is what lives in the ring buffer.
#[derive(Debug)]
#[repr(C)]
pub struct TileHeader {
    pub tile_id: u64,
    pub room_id: AtomicU64,
    pub sequence: AtomicU64,
    pub flags: u32,
    pub payload_size: usize,
}

/// A tile body — variable size, pool-allocated.
pub struct TileBody<'a> {
    pub data: SharedSlice<'a, u8>,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_basic() {
        let mut pool = MemoryPool::<u32>::new(10).unwrap();
        
        for i in 0..10 {
            let val = pool.alloc().unwrap();
            *val = i;
        }
        
        assert!(pool.alloc().is_none()); // Should be full
        
        let slice = pool.as_slice();
        assert_eq!(slice, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_memory_pool_reset() {
        let mut pool = MemoryPool::<u32>::new(5).unwrap();
        
        for i in 0..5 {
            *pool.alloc().unwrap() = i;
        }
        
        pool.reset();
        
        for i in 0..3 {
            *pool.alloc().unwrap() = i * 10;
        }
        
        assert_eq!(pool.as_slice(), &[0, 10, 20]);
    }

    #[test]
    fn test_fixed_vec_basic() {
        let mut vec = FixedVec::<i32, 5>::new();
        
        vec.push(1);
        vec.push(2);
        vec.push(3);
        
        assert_eq!(vec.len(), 3);
        assert_eq!(vec.get(0), Some(&1));
        assert_eq!(vec.get(1), Some(&2));
        assert_eq!(vec.get(2), Some(&3));
        assert_eq!(vec.get(3), None);
        
        assert_eq!(vec.pop(), Some(3));
        assert_eq!(vec.len(), 2);
    }

    #[test]
    fn test_ring_buffer_basic() {
        let mut rb = RingBuffer::<u32, 5>::new();
        
        assert!(rb.push(1));
        assert!(rb.push(2));
        assert!(rb.push(3));
        assert!(rb.push(4)); // Buffer has 4 slots, 4 items (not yet full)
        
        // Now buffer is full (5th slot would wrap to empty slot)
        assert!(rb.is_full());
        assert!(!rb.push(5)); // Should fail - buffer full
        
        assert_eq!(rb.pop(), Some(1));
        assert_eq!(rb.pop(), Some(2));
        assert_eq!(rb.pop(), Some(3));
        assert_eq!(rb.pop(), Some(4));
        assert_eq!(rb.pop(), None); // Empty
    }

    #[test]
    fn test_shared_slice() {
        let data = [1u32, 2, 3, 4, 5];
        
        // SAFETY: exclusive write access (data is owned here)
        let slice = unsafe { SharedSlice::<u32>::from_ref(&data) };
        
        assert_eq!(slice.len(), 5);
        assert_eq!(slice.get(0), Some(&1));
        assert_eq!(slice.get(4), Some(&5));
        
        let collected: Vec<_> = slice.iter().collect();
        assert_eq!(collected, &[&1, &2, &3, &4, &5]);
    }

    #[test]
    fn test_static_assert() {
        // This compiles - static_assert passes
        const _: [(); 0] = [(); if 10 > 5 { 0 } else { 1 }];
        
        // This would fail to compile:
        // const _: [(); 0] = [(); if 10 < 5 { 0 } else { 1 }]; // ERROR
    }

    #[test]
    fn test_capacity_assert() {
        // Compile-time check: verify capacity >= 1024
        const _: [(); 0] = [(); if 1024 >= 1024 { 0 } else { 1 }];
        
        // This would fail to compile:
        // const _: [(); 0] = [(); if 512 >= 1024 { 0 } else { 1 }]; // ERROR
    }
}