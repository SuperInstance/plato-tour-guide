# Tile Memory Model — How Tiles Flow Through the System

The fundamental insight: tiles are **ephemeral** (in flight) or **persistent** (at rest), and each state has different memory requirements.

## Tile Anatomy

Every tile has two parts:

```
┌─────────────────────────────────────┐
│           TILE HEADER               │  Stack-allocated (FixedVec)
│  tile_id, room_id, seq, flags, size │
└─────────────────────────────────────┘
              ↓ (pointer)
┌─────────────────────────────────────┐
│           TILE BODY                 │  Pool-allocated (MemoryPool)
│  raw bytes, distance vectors, etc   │
└─────────────────────────────────────┘
```

**Header:** Small, fixed-size, always stack-allocated. Lives in ring buffer entries.

**Body:** Variable size, pool-allocated. Lives in arena per PLATO room.

## Tile Creation Pipeline

```
1. User code creates tile metadata
   └── FixedVec<u8, HEADER_SIZE> header (stack)

2. Header pushed to ring buffer
   └── O(1) pointer copy, no data movement

3. Worker thread pops header
   └── Reads tile_id, calculates body size

4. Worker allocates body from room pool
   └── MemoryPool::alloc() — O(1) bump
   └── Copies payload or computes distance vector

5. Worker stores result
   └── Body pointer stored back in header or separate result queue
```

**Key insight:** Steps 1-2 and 3-5 happen on different threads. The ring buffer is the boundary.

## Tile Queue: Ring Buffer with Fixed Capacity

```rust
pub struct RingBuffer<TileHeader, const N: usize> {
    buffer: FixedVec<MaybeUninit<TileHeader>, N>,
    write_idx: usize,  // Producer only
    read_idx: usize,    // Consumer only
}
```

**Capacity:** Compile-time constant `N` (e.g., 1024)

**Behavior:**
- Full buffer → producer blocked or drops (caller choice)
- Empty buffer → consumer blocked or returns None

**No malloc:** Buffer slots pre-allocated at creation. Push just writes to slot and increments pointer.

## Tile Deletion: Bump Pointer Reset

When a room is cleared or a tile's lifetime ends:

```
Before: Room has 5 tiles allocated in pool
        [Tile1][Tile2][Tile3][Tile4][Tile5][    ][    ][    ]

After reset: bump pointer moves back
        [    ][    ][    ][    ][    ][    ][    ][    ]
                    ↑
              used = 0
```

**O(1)** — just reset the counter. No individual deallocation, no GC pause.

**Tradeoff:** Can only reset entire room, not individual tiles. This is intentional:
- Tiles in PLATO rooms have deterministic lifetimes (one message exchange)
- If a tile lives longer, it stays; if shorter, its memory is recovered at room reset
- No use-after-free possible (enclave isolation)

## PLATO Room Memory: One Pool Per Room

```rust
struct PlatoRoom {
    id: u64,
    pool: MemoryPool<TileBody>,      // Pre-allocated arena
    tile_queue: RingBuffer<TileHeader, 1024>,
    history: MemoryPool<TileHistory>, // Separate pool for archived tiles
}
```

**Isolation:** Each room has its own pool. A room's memory pressure doesn't affect other rooms.

**Capacity planning:** Each room knows its max tile count at compile time (derived from room config). Pool size = `max_tiles * avg_tile_size * 1.2` (20% headroom).

## Memory Layout Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                        STACK (1MB)                               │
│  Tile headers (stack-allocated FixedVec)                        │
│  Small buffers, return addresses                                 │
│  Access: ~1ns                                                    │
├─────────────────────────────────────────────────────────────────┤
│                        L1/L2 CACHE                               │
│  Hot tile pointers, hot distance values                          │
│  Current room ID, active tile sequence numbers                   │
│  Access: L1 ~1ns, L2 ~4ns                                        │
├─────────────────────────────────────────────────────────────────┤
│                        L3 CACHE                                  │
│  Partial distance matrices (working set)                        │
│  Room metadata, recent tiles                                     │
│  Access: ~15ns                                                   │
├─────────────────────────────────────────────────────────────────┤
│                      MAIN MEMORY                                 │
│  Embedding vectors (full model weights in DRAM)                 │
│  PLATO room pools (all tile bodies)                             │
│  Access: ~100ns                                                  │
├─────────────────────────────────────────────────────────────────┤
│                        GPU VRAM                                  │
│  Full distance matrices for consensus                            │
│  Consensus kernels (batch processing)                          │
│  Access: ~400ns (PCIe latency)                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Why This Is Safe: Enclave Isolation

The standard objection to bump-allocated pools: **use-after-free** if a dangling pointer escapes.

**Our answer:** Enclave isolation eliminates the attack surface.

```
Without isolation:
  Tile freed → pointer to Tile escapes → use-after-free vulnerability

With enclave isolation:
  Tile freed → pointer to Tile cannot escape enclave
  Even if pointer exists, other processes can't access it
```

**In our edge system:**
- No internet-facing attack surface (no external processes)
- All tile access happens within the same enclave
- Tile pointers stay within the PLATO room
- "Free" means bump-pointer reset, which is deterministic and safe

**The safety guarantee:**
> A tile's memory is only reused after the tile's room is cleared, and no references to that tile's memory exist outside the room's control flow.

## Tile Lifecycle State Machine

```
                    ┌──────────────┐
         create      │   ALLOCATED   │
        ──────────→  │  (in pool)    │
                    └──────────────┘
                         │
                         │ process
                         ↓
                    ┌──────────────┐
                    │   ACTIVE      │
                    │ (in flight)   │
                    └──────────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
              ↓                     ↓
        ┌───────────┐        ┌────────────┐
        │  ARCHIVED │        │  DISCARDED │
        │(room pool)│        │ (no action) │
        └───────────┘        └────────────┘
              │                     │
              │ room reset          │ tile dropped
              ↓                     ↓
        ┌───────────┐        ┌────────────┐
        │   FREED   │        │   FREED    │
        │(bump reset)│        │ (implicit) │
        └───────────┘        └────────────┘
```

**Archived tiles:** Explicitly saved to history pool for replay/debugging.

**Discarded tiles:** Expired naturally (room timeout), memory recovered at room reset.

**Key property:** No individual deallocation. All memory recovered in bulk at room boundaries.

## Compile-Time Contracts

```rust
// Size known at compile time
const MAX_TILES_PER_ROOM: usize = 1024;
const TILE_HEADER_SIZE: usize = 32;  // bytes

// Pool capacity derived from config
struct RoomConfig {
    max_tiles: usize,
    avg_tile_size: usize,
}

impl RoomConfig {
    fn pool_size(&self) -> usize {
        self.max_tiles * self.avg_tile_size * 2  // 2x headroom
    }
}

// Static assertion: ring buffer must be power of 2 for fast modulo
static_assert!(MAX_TILES_PER_ROOM.is_power_of_two());
```

This is the Fortran philosophy applied to Rust: **if it's known at compile time, the compiler enforces it**.

## Comparison: Our Model vs GC vs malloc

| Aspect | Our Model | GC (Go/Rust) | malloc |
|--------|-----------|--------------|--------|
| Allocation | O(1) bump | O(1) bump | O(log n) |
| Deallocation | O(1) bulk | O(n) incremental | O(1) individual |
| Fragmentation | Zero | Low | High |
| Pause times | Zero (deterministic) | Variable | N/A |
| Predictability | Deterministic | Non-deterministic | Best-effort |
| Memory overhead | Minimal | Meta-data per object | Minimal |
| Thread safety | Lock-free | GC marks | Lock-free (tcmalloc) |

**Bottom line:** For our workload (many short-lived tiles, known sizes, room boundaries), bump allocation + bulk reset is optimal.

The GC analogy is apt but incomplete: our "GC" runs at compile time, not runtime. We're paying the complexity cost upfront in exchange for predictable, zero-overhead execution.