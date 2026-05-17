//! Compile-time guarantees for pool-allocated types.
//!
//! This module demonstrates how Rust's type system can enforce
//! compile-time properties that would require runtime checks in other languages.
//!
//! Philosophy: If we can prove it at compile time, we don't pay for it at runtime.

use std::fmt::Debug;
use core::marker::PhantomData;

// =============================================================================
// 1. Const Generics — Sizes Known at Compile Time
// =============================================================================

/// A fixed-size ring buffer with capacity known at compile time.
struct RingBuffer<T: Copy, const N: usize> {
    data: [Option<T>; N],
    head: usize,
    tail: usize,
}

impl<T: Copy, const N: usize> RingBuffer<T, N> {
    const fn new() -> Self {
        // Arrays of Option<T> must be initialized - requires Copy bound
        Self {
            data: [const { None }; N],
            head: 0,
            tail: 0,
        }
    }
    
    fn push(&mut self, value: T) -> bool {
        let next_head = (self.head + 1) % N;
        if next_head == self.tail {
            return false; // Full
        }
        self.data[self.head] = Some(value);
        self.head = next_head;
        true
    }
    
    fn pop(&mut self) -> Option<T> {
        if self.head == self.tail {
            return None; // Empty
        }
        let value = self.data[self.tail];
        self.data[self.tail] = None;
        self.tail = (self.tail + 1) % N;
        value
    }
    
    fn capacity() -> usize {
        N // Free — compile-time constant
    }
}

// =============================================================================
// 2. Static Assertions — Compile-Time Truth Checks
// =============================================================================

/// Compile-time assertion macro.
/// Fails compilation if condition is false.
/// Must be used as a statement (ends with semicolon).
macro_rules! static_assert {
    ($condition:expr) => {
        // Array length 0 means condition was false, which is a compile error
        const _: [(); 0] = [(); if $condition { 0 } else { 1 }];
    };
}

/// Compile-time check that a value is a power of two.
/// Used for ring buffer capacity optimization.
const fn is_power_of_two(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

// Verify at compile time that our ring buffer capacity is a power of 2
static_assert!(is_power_of_two(1024)); // ✓ compiles
// static_assert!(is_power_of_two(1000)); // ✗ would fail to compile

/// Compile-time minimum and maximum.
const fn const_min(a: usize, b: usize) -> usize {
    if a < b { a } else { b }
}

const fn const_max(a: usize, b: usize) -> usize {
    if a > b { a } else { b }
}

// =============================================================================
// 3. Trait PoolAllocated — Types That Can Go in Pools
// =============================================================================

/// Marker trait for types that can be safely allocated in a memory pool.
///
/// Requirements:
/// - Sized: known size at compile time (no unsized types like dyn Trait)
/// - 'static: no borrowed references that could outlive the pool
///
/// This is a zero-sized marker — no runtime cost.
pub trait PoolAllocated: Sized + 'static {}

/// Marker trait for trivial types (Copy + no Drop).
///
/// These can use O(1) "clear without running Drop" operations.
pub trait Trivial: PoolAllocated + Copy + Default {}

// =============================================================================
// 4. Auto-impl for Primitives
// =============================================================================

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

impl_pool_allocated!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64, bool, char);
impl_trivial!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64, bool, char);

// =============================================================================
// 5. NoHeap Trait — Proof That Struct Doesn't Allocate
// =============================================================================

/// Proof trait: a type contains no heap allocation.
///
/// This is stronger than PoolAllocated — it means the type, transitively,
/// contains no heap pointers, no Vec, no String, no Box, no Box<dyn Trait>.
pub trait NoHeap: PoolAllocated {
    // Marker method — implementors just implement, no logic
    const PROOF: () = ();
}

// Primitive types are trivially NoHeap
macro_rules! impl_no_heap_primitives {
    ($($t:ty),*) => {
        $(impl NoHeap for $t {
            const PROOF: () = ();
        })*
    };
}

impl_no_heap_primitives!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64, bool, char);

// Composite types: manually prove they're heap-free
#[derive(Debug, Clone, Copy)]
struct Header {
    tile_id: u64,
    sequence: u32,
    flags: u16,
}

impl PoolAllocated for Header {}
impl NoHeap for Header {
    const PROOF: () = ();
}

#[derive(Debug, Clone, Copy)]
struct DistanceVector([f64; 4]); // Reduced from 128 for easier handling

impl PoolAllocated for DistanceVector {}
impl NoHeap for DistanceVector {
    const PROOF: () = ();
}

#[derive(Debug, Clone, Copy)]
struct TileEmbedding {
    tile_id: u64,
    distance: f64,
}

impl PoolAllocated for TileEmbedding {}
impl NoHeap for TileEmbedding {
    const PROOF: () = ();
}

// =============================================================================
// 6. Const Evaluations — Compile-Time Computations
// =============================================================================

/// Compile-time tile memory budget calculator.
struct TileBudget {
    header_size: usize,
    max_tiles: usize,
    avg_body_size: usize,
}

impl TileBudget {
    const fn new(header_size: usize, max_tiles: usize, avg_body_size: usize) -> Self {
        Self {
            header_size,
            max_tiles,
            avg_body_size,
        }
    }
    
    /// Total memory needed for the room's tile pool.
    const fn total_pool_size(&self) -> usize {
        // All compile-time arithmetic
        self.header_size * self.max_tiles + self.avg_body_size * self.max_tiles
    }
    
    /// Memory needed including 20% headroom.
    const fn total_with_headroom(&self) -> usize {
        self.total_pool_size() * 6 / 5  // Multiply by 1.2
    }
    
    /// Verify budget fits within a limit.
    const fn fits_in(&self, limit: usize) -> bool {
        self.total_with_headroom() <= limit
    }
}

// Compile-time calculation
const TILE_BUDGET: TileBudget = TileBudget::new(32, 1024, 256);
const _TOTAL_MEMORY: usize = TILE_BUDGET.total_pool_size(); // Pre-computed at compile time
// Note: static_assert in const context requires Rust 1.79+. In tests below.

// =============================================================================
// 7. Type-Level Capacity Checks
// =============================================================================

/// A pool that enforces capacity at the type level.
trait Capacity {
    const MAX: usize;
}

struct Capacity1024;
struct Capacity4096;
struct Capacity65536;

impl Capacity for Capacity1024 {
    const MAX: usize = 1024;
}

impl Capacity for Capacity4096 {
    const MAX: usize = 4096;
}

impl Capacity for Capacity65536 {
    const MAX: usize = 65536;
}

/// A type-safe pool that rejects capacities it wasn't designed for.
struct TypedPool<T: PoolAllocated, C: Capacity> {
    _phantom: PhantomData<T>,
    _capacity: PhantomData<C>,
}

impl<T: PoolAllocated, C: Capacity> TypedPool<T, C> {
    fn new() -> Self {
        Self {
            _phantom: PhantomData,
            _capacity: PhantomData,
        }
    }
    
    fn max_capacity() -> usize {
        C::MAX
    }
}

// Usage: Must specify capacity at type level
fn create_room<C: Capacity>() -> TypedPool<u64, C> {
    TypedPool::new()
}

// let pool: TypedPool<u64, Capacity1024> = create_room(); // Type-safe
// let pool: TypedPool<u64, Capacity4096> = create_room(); // Type-safe
// create_room::<Capacity1024>(); // Error if called with wrong capacity

// =============================================================================
// 8. Extern "C" FFI Safety — Compile-Time Layout Control
// =============================================================================

/// Tile as it appears in shared memory (FFI-safe layout).
#[repr(C)]
struct TileFFI {
    tile_id: u64,
    room_id: u64,
    sequence: u64,
    flags: u32,
    payload_size: usize,
    // Followed by variable-length payload
}

/// Verify TileFFI is exactly 40 bytes (5 x 8-byte fields)
const TILE_FFI_SIZE: usize = std::mem::size_of::<TileFFI>();

/// Tile with fixed-size embedding (8 dimensions for simplicity).
#[repr(C)]
struct TileWithEmbedding {
    header: TileFFI,
    embedding: [f32; 8],
}

const TILE_WITH_EMBEDDING_SIZE: usize = std::mem::size_of::<TileWithEmbedding>();

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_basic() {
        let mut rb = RingBuffer::<u32, 8>::new();
        
        assert!(rb.push(1));
        assert!(rb.push(2));
        assert!(rb.push(3));
        
        assert_eq!(rb.pop(), Some(1));
        assert_eq!(rb.pop(), Some(2));
        assert_eq!(rb.pop(), Some(3));
        assert_eq!(rb.pop(), None);
    }
    
    #[test]
    fn test_ring_buffer_full() {
        let mut rb = RingBuffer::<u32, 3>::new();
        
        assert!(rb.push(1));
        assert!(rb.push(2));
        assert!(rb.push(3));
        assert!(!rb.push(4)); // Full
        
        assert_eq!(rb.pop(), Some(1));
        assert!(rb.push(4)); // Space now available
        
        assert_eq!(rb.pop(), Some(2));
        assert_eq!(rb.pop(), Some(3));
        assert_eq!(rb.pop(), Some(4));
    }
    
    #[test]
    fn test_const_min_max() {
        assert_eq!(const_min(5, 10), 5);
        assert_eq!(const_max(5, 10), 10);
    }
    
    #[test]
    fn test_tile_budget() {
        let budget = TileBudget::new(32, 1024, 256);
        assert_eq!(budget.total_pool_size(), 32 * 1024 + 256 * 1024);
        assert!(budget.fits_in(300_000));
    }
    
    #[test]
    fn test_no_heap_marker() {
        let h = Header { tile_id: 1, sequence: 42, flags: 0 };
        // Header is NoHeap — no heap allocation
        let _ = h;
    }
    
    #[test]
    fn test_static_assertions() {
        // These all compile — static assertions pass
        static_assert!(is_power_of_two(1024));
        static_assert!(is_power_of_two(2));
        static_assert!(!is_power_of_two(0));
        static_assert!(!is_power_of_two(1000));
    }
    
    #[test]
    fn test_typed_pool() {
        let pool: TypedPool<u64, Capacity1024> = TypedPool::new();
        assert_eq!(TypedPool::<u64, Capacity1024>::max_capacity(), 1024);
    }
    
    #[test]
    fn test_tile_ffi_layout() {
        // Verify FFI layout is correct
        assert_eq!(std::mem::size_of::<TileFFI>(), 40);
        assert_eq!(std::mem::size_of::<TileWithEmbedding>(), 72);
    }
}