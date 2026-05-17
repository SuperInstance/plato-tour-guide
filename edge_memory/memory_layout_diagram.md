# Memory Layout Diagram — Edge System Architecture

Visual representation of how tiles flow through the memory hierarchy in our sandboxed edge system.

## Overall Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GPU VRAM (Dedicated)                              │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  Full Distance Matrices (n×n float32)                              │  │
│  │  Consensus Kernels (batch of 1024 tiles)                            │  │
│  │  Embedding Lookup Tables                                            │  │
│  │                                                                       │  │
│  │  Access Latency: ~400ns (PCIe 4.0 x16)                             │  │
│  │  Bandwidth: ~64 GB/s                                                 │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────── │
│                                                                             │
│                           MAIN MEMORY (DDR5)                                │
│  ┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐│
│  │   Embedding     │   Room Pools    │   History       │   Code/Stack    ││
│  │   Vectors       │   (per-room     │   Archives      │                 ││
│  │   (model        │   MemoryPool)   │   (mmap)        │   Working       ││
│  │   weights)      │                 │                 │   Memory        ││
│  │                 │                 │                 │                 ││
│  │  Access Latency: ~100ns                                             ││
│  │  Bandwidth: ~100 GB/s                                                ││
│  └─────────────────┴─────────────────┴─────────────────┴─────────────────┘│
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────── │
│                                                                             │
│                           L3 CACHE (LLC)                                    │
│  ┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐│
│  │  Room Metadata  │  Partial        │  Hot Tile       │  LRU Cache      ││
│  │  (current       │  Distance       │  Headers        │  of Recent      ││
│  │  room ID,       │  Matrices       │  (active        │  Embeddings     ││
│  │  sequence)      │  (working       │  tiles)         │                 ││
│  │                 │  set, top-k)     │                 │                 ││
│  │                                                                       │  │
│  │  Access Latency: ~15ns                                              │  │
│  │  Size: ~20 MB (typical)                                             │  │
│  └─────────────────┴─────────────────┴─────────────────┴─────────────────┘│
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────── │
│                                                                             │
│                           L2 CACHE                                          │
│  ┌─────────────────┬─────────────────┬────────────────────────────────────┤
│  │  Ring Buffer    │  Tile Pointers   │  Current Room Working Set         │
│  │  Slots (8-16    │  (hot paths)     │  (small)                          │
│  │  entries)       │                  │                                   │
│  │                 │                 │                                   │
│  │  Access Latency: ~4ns                                               │  │
│  └─────────────────┴─────────────────┴────────────────────────────────────┘│
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────── │
│                                                                             │
│                           L1 CACHE                                          │
│  ┌─────────────────┬──────────────────────────────────────────────────────┤
│  │  Ring Buffer    │  Active Tile Header                                   │
│  │  Head/Tail      │  (tile being processed)                               │
│  │  Pointers      │  Sequence counters                                    │
│  │                 │                                                       │
│  │  Access Latency: ~1ns                                                  │
│  │  Size: ~32 KB per core                                                 │
│  └─────────────────┴──────────────────────────────────────────────────────┘│
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────── │
│                                                                             │
│                           STACK                                              │
│  ┌─────────────────┬──────────────────────────────────────────────────────┤
│  │  Tile Headers    │  Return Addresses                                    │
│  │  (FixedVec,     │  Local Variables                                     │
│  │  stack-alloc)    │  Function Arguments                                  │
│  │                 │                                                       │
│  │  Access Latency: ~1ns                                                  │
│  │  Size: ~1 MB typical                                                   │
│  │                                                                       │  │
│  │  "Fortran philosophy: small, fixed-size, stack-allocated"            │  │
│  └─────────────────┴──────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

## Tile Flow Through Memory

```
                         USER SPACE
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     TILE CREATION (Producer Thread)                   │
│                                                                      │
│   1. Create tile header (stack-allocated FixedVec)                  │
│      ┌─────────────────┐                                             │
│      │ tile_id: u64    │  ← Stack (1ns access)                       │
│      │ room_id: u64    │                                             │
│      │ sequence: u64   │                                             │
│      └─────────────────┘                                             │
│                              │                                        │
│                              ▼                                        │
│   2. Push header to ring buffer (lock-free, O(1))                    │
│      ┌───────────────────────────────────────────────────────────┐  │
│      │  RingBuffer[head] ──────────────────────────────────────→  │  │
│      │     ↓                                                       │  │
│      │  L1/L2 cache write (4ns)                                   │  │
│      └───────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              │ Ring buffer (cache-to-cache transfer)
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     TILE PROCESSING (Consumer Thread)                │
│                                                                      │
│   3. Pop header from ring buffer (L1 cache read, 1ns)                │
│                                                                      │
│   4. Allocate tile body from room pool (bump pointer, ~5ns)          │
│      ┌───────────────────────────────────────────────────────────┐  │
│      │  MemoryPool<Room123>        TILE BODIES                   │  │
│      │  ┌────────────────┐    ┌────────────────┐                 │  │
│      │  │ TileBody #1   │    │ TileBody #2   │                 │  │
│      │  │ (256 bytes)   │    │ (256 bytes)   │                 │  │
│      │  └────────────────┘    └────────────────┘                 │  │
│      │       ↑                       ↑                            │  │
│      │  bump pointer          bump pointer                        │  │
│      └───────────────────────────────────────────────────────────┘  │
│                              │                                        │
│                              ▼                                        │
│   5. Compute distance vectors (use L3/L2 for partial results)        │
│      ┌───────────────────────────────────────────────────────────┐  │
│      │  L3 Cache: Partial distance matrix (top-k candidates)     │  │
│      │     Access: 15ns                                           │  │
│      └───────────────────────────────────────────────────────────┘  │
│                              │                                        │
│                              ▼                                        │
│   6. Send to GPU for full matrix computation                         │
│      ┌───────────────────────────────────────────────────────────┐  │
│      │  GPU VRAM: Full distance matrix, consensus kernels        │  │
│      │     Access: 400ns                                          │  │
│      └───────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              │ Result returned
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     TILE ARCHIVAL (Room History)                     │
│                                                                      │
│   7. Archive result to room's history pool                           │
│      ┌───────────────────────────────────────────────────────────┐  │
│      │  MemoryPool<History>                                      │  │
│      │  ┌────────────────┐ ┌────────────────┐                   │  │
│      │  │ TileHistory #1 │ │ TileHistory #2 │ → (mmap to disk)  │  │
│      │  └────────────────┘ └────────────────┘                   │  │
│      └───────────────────────────────────────────────────────────┘  │
│                                                                      │
│   8. Room reset: bump pointer reset, all history freed (O(1))        │
└──────────────────────────────────────────────────────────────────────┘
```

## Latency Breakdown by Operation

```
┌────────────────────────────────────────────────────────────────────────┐
│                     LATENCY SUMMARY                                    │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  STACK OPERATIONS                                                      │
│  ├─ Allocate tile header (FixedVec push):       ~1ns                    │
│  ├─ Stack-to-stack copy:                       ~1ns                    │
│  └─ Return address push/pop:                  ~0.5ns                   │
│                                                                        │
│  L1 CACHE (32 KB per core)                                             │
│  ├─ Ring buffer head/tail read:                 ~1ns                    │
│  ├─ Hot tile pointer access:                   ~1ns                    │
│  └─ L1 hit rate target: >95%                  (for hot path)          │
│                                                                        │
│  L2 CACHE (256 KB per core)                                            │
│  ├─ Ring buffer slot read:                     ~4ns                    │
│  ├─ Tile header read (non-hot):                ~4ns                    │
│  └─ L2 hit rate target: >90%                  (for working set)      │
│                                                                        │
│  L3 CACHE (shared, ~20 MB)                                              │
│  ├─ Room metadata read:                        ~15ns                   │
│  ├─ Partial distance matrix access:            ~15ns                   │
│  └─ L3 hit rate target: >80%                   (for recent tiles)   │
│                                                                        │
│  DRAM (main memory, ~64 GB)                                             │
│  ├─ Embedding vector read (256 floats):         ~100ns                 │
│  ├─ Room pool tile body access:                  ~100ns                 │
│  └─ MemoryPool bump pointer update:              ~5ns (adjacent alloc) │
│                                                                        │
│  GPU VRAM (PCIe 4.0 x16)                                               │
│  ├─ Full distance matrix transfer:               ~400ns (PCIe)         │
│  ├─ Consensus kernel execution:                   ~1000ns (GPU)         │
│  └─ Result transfer back:                        ~400ns (PCIe)         │
│                                                                        │
│  NETWORK (N/A for isolated edge system)                                │
│  └─ N/A — no network latency in compute path                           │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

## Memory Layout Per Room

```
┌──────────────────────────────────────────────────────────────────────┐
│                          PLATO ROOM #123                              │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                      HEADER (Stack)                             │ │
│  │  room_id: u64 = 123                                            │ │
│  │  max_tiles: usize = 1024                                       │ │
│  │  tile_count: AtomicUsize                                       │ │
│  │  bump_ptr: *mut TileBody                                       │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                        │
│                              ▼                                        │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                  TILE POOL (MemoryPool)                        │ │
│  │                                                                │ │
│  │  Capacity: 1024 tiles × 256 bytes = 256 KB                    │ │
│  │                                                                │ │
│  │  ┌─────────┬─────────┬─────────┬─────────┬─────────┐         │ │
│  │  │ Tile #0 │ Tile #1 │ Tile #2 │ Tile #3 │ Tile #4 │ ...      │ │
│  │  │ 256 B   │ 256 B   │ 256 B   │ 256 B   │ 256 B   │         │ │
│  │  └─────────┴─────────┴─────────┴─────────┴─────────┘         │ │
│  │       ↑                                                           │ │
│  │   bump pointer (alloc here)                                      │ │
│  │                                                                 │ │
│  │  Layout: [Tile][Tile][Tile][Tile][Tile][    ][    ][    ]       │ │
│  │                         allocated     │          │    │         │ │
│  │                                    bump_ptr ──────┘    │         │ │
│  └──────────────────────────────────────────────────────────┘         │
│                              │                                        │
│                              ▼                                        │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                  RING BUFFER (TileQueue)                       │ │
│  │                                                                │ │
│  │  Capacity: 1024 entries (compile-time)                        │ │
│  │  Type: SPSC lock-free                                          │ │
│  │                                                                │ │
│  │  ┌───┬───┬───┬───┬───┬───┬───┬───┐                             │ │
│  │  │ H │ H │ H │   │   │   │   │   │ → head (producer)         │ │
│  │  └───┴───┴───┴───┴───┴───┴───┴───┘                             │ │
│  │    ↑                                                         │ │
│  │  tail (consumer)                                             │ │
│  │                                                                 │ │
│  │  Entry size: 32 bytes (TileHeader, no heap)                   │ │
│  │  Full latency: ~4ns (L2 cache)                                │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                        │
│                              ▼                                        │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                  HISTORY POOL (MemoryPool)                     │ │
│  │                                                                │ │
│  │  For archived tiles that need replay/debugging                 │ │
│  │  Capacity: 512 tiles × 512 bytes = 256 KB                      │ │
│  │  Reset: triggered by room archival policy                      │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

## Memory Budget Calculation

```
┌────────────────────────────────────────────────────────────────────────┐
│                     COMPILE-TIME MEMORY BUDGET                         │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  CONST TILE_BUDGET = TileBudget {                                     │
│      header_size: 32,          // bytes (5 × 8-byte fields)          │
│      max_tiles: 1024,          // per room                             │
│      avg_body_size: 256,       // bytes per tile body                 │
│  }                                                                      │
│                                                                        │
│  Per-Room Memory:                                                      │
│  ├─ Tile Pool:      32 × 1024 + 256 × 1024 = 288 KB                    │
│  ├─ Ring Buffer:    32 × 1024 = 32 KB                                  │
│  ├─ History Pool:  512 × 512 = 256 KB (separate)                      │
│  └─ Overhead:      ~4 KB                                              │
│                                                                        │
│  TOTAL PER ROOM:     ~580 KB (with headroom)                           │
│                                                                        │
│  For 16 concurrent rooms:                                             │
│  └─ Total: ~9.3 MB (fits in L3 cache)                                 │
│                                                                        │
│  For 256 max rooms (system limit):                                     │
│  └─ Total: ~150 MB (fits in main memory, L3 working set)              │
│                                                                        │
│  NOTE: All of this is pre-allocated at startup.                        │
│        No malloc during compute.                                       │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

## Key Insight: The Closer to the Core, the Faster

```
         Producer                     Consumer
         Thread                       Thread
            │                           ▲
            ▼                           │
┌─────────────────────────────────────────────────────────────┐
│                      L1 Cache                              │
│   Hot: ring buffer head/tail, active tile header           │
│   Latency: 1ns                                             │
└─────────────────────────────────────────────────────────────┘
            │                           ▲
            ▼                           │
┌─────────────────────────────────────────────────────────────┐
│                      L2 Cache                              │
│   Warm: ring buffer slots, recent tile pointers            │
│   Latency: 4ns                                             │
└─────────────────────────────────────────────────────────────┘
            │                           ▲
            ▼                           │
┌─────────────────────────────────────────────────────────────┐
│                      L3 Cache                              │
│   Active: room metadata, partial distance matrices          │
│   Latency: 15ns                                            │
└─────────────────────────────────────────────────────────────┘
            │                           ▲
            ▼                           │
┌─────────────────────────────────────────────────────────────┐
│                      DRAM                                   │
│   Idle: embedding vectors, room pools (cold)               │
│   Latency: 100ns                                           │
└─────────────────────────────────────────────────────────────┘
            │                           ▲
            ▼                           │
┌─────────────────────────────────────────────────────────────┐
│                      GPU VRAM                               │
│   Batched: full distance matrix, consensus results         │
│   Latency: 400ns + GPU compute                              │
└─────────────────────────────────────────────────────────────┘
```

**This is why bump-allocated pools work:** We pay ~5ns to allocate from an adjacent slot (CPU knows the memory is there), versus ~100ns to allocate from DRAM. The GPU offload path (~400ns) is the only truly expensive operation, and we batch to minimize overhead.