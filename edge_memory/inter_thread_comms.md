# Inter-Thread Communication for Sandbox Edge Systems

Survey of IPC mechanisms for our tile-based PLATO architecture.

## Lock-Free Queues

### SPSC (Single Producer Single Consumer)

Best for our tile flow model where one thread produces tiles and another consumes them.

**Implementation: RingBuffer in memory_pool.rs**

```
Producer → [RingBuffer] → Consumer
          (lock-free, wait-free)
```

**Characteristics:**
- Wait-free for producer (never blocks)
- Wait-free for consumer (never blocks)
- O(1) push/pop operations
- No malloc during operation (pre-allocated)
- Cache-friendly circular buffer

**Rust crates:**
- `crossbeam-queue` — SpscQueue, SpscQueue::new() for unlimited
- `ringbuf` — pure Rust, no unsafe
- Our implementation in memory_pool.rs — minimal, no dependencies

### MPSC (Multi Producer Single Consumer)

When multiple edge agents produce tiles for one aggregator.

**Implementation:**
- Use `crossbeam-channel` — mpsc channel backed by lock-free ring buffer
- Or roll your own with atomic compare-and-swap

**Tradeoff vs SPSC:**
- More complex atomic ops (CAS vs simple head/tail)
- Still lock-free but not wait-free
- ~2x slower than SPSC in benchmarks

## Shared Memory (Linux shmget)

For sharing large data structures between processes on the same machine.

```c
// Create shared memory segment
int shmid = shmget(key, size, IPC_CREAT | 0666);
void *shm = shmat(shmid, NULL, 0);

// Attach to process
struct Tile *tiles = (struct Tile *)shm;
```

**Use case:** Sharing full distance matrices between edge nodes.

**Problem for our tile model:** Tiles are small, frequent, and ephemeral. Shared memory requires:
1. Synchronization (semaphores or mutexes)
2. Kernel syscalls for shmat/shmdt
3. Manual garbage collection

**Verdict:** Not ideal for tile flow. Good for bulk data at rest.

## Memory-Mapped Files for PLATO Tile Exchange

For persisting tiles to disk while allowing zero-copy reads.

```rust
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::FileExt;

// Create or open tile storage
let file = OpenOptions::new()
    .read(true)
    .write(true)
    .create(true)
    .open("plato_tiles.dat")?;

let mmap = unsafe { MmapOptions::new().len(size).map(&file)? };
```

**Use case:** PLATO room history — tiles that survive restarts.

**Not for:** Hot tile path (too slow for compute loop).

## Message Passing vs Shared Memory

The fundamental question: should tiles be:

1. **Data in flight** (message passing) — serialized, deserialized, moved
2. **Data at rest** (shared memory) — accessed in place, more complex sync

### Why Message Passing Wins for Our Tile Model

**1. Isolation = Safety**

```
Tile is produced → serialized to queue → deserialized by consumer
                                ↓
                    Producer and consumer CANNOT
                    access same memory simultaneously
```

No use-after-free, no data races by construction. The queue is the boundary.

**2. Locality = Performance**

```
Producer (core 0)          Queue            Consumer (core 1)
    └── writes to ──→  [L1 cache line]  ←── reads from
                        (cache-to-cache transfer)
```

Cache-coherent inter-core communication via ring buffer is ~40ns latency.

**3. Determinism = Testability**

Message passing is deterministic. Same sequence of sends → same sequence of receives. Shared memory has race conditions by nature.

**4. Flow Control = Backpressure**

Ring buffer with fixed capacity provides natural backpressure:
- Full buffer → producer must wait or drop
- Empty buffer → consumer must wait or idle

### When Shared Memory Makes Sense

- **Bulk data at rest:** Distance matrices that consumers read but don't modify
- **Cross-process communication:** Different executables, not just threads
- **Zero-copy persistence:** mmapped files for durable state

## eBPF Rings vs Unix Domain Sockets vs Shared Memory

### eBPF Rings (perf_event)

Kernel-to-userpace communication for tracing/metrics.

**Pros:**
- Zero-copy from kernel to user
- Lock-free ring buffer
- Efficient for events, not bulk data

**Cons:**
- Complex setup
- Requires kernel module or capabilities
- Not for general tile exchange

**Use case:** Performance monitoring, not tile flow.

### Unix Domain Sockets

IPC between processes on the same machine.

```bash
# Create socket
nc -U /tmp/plato.sock

# Or in Rust
UnixStream::connect("/tmp/plato.sock")?;
```

**Pros:**
- Stream or datagram modes
- Works across containers (with volume mount)
- Simple programming model

**Cons:**
- Serialization overhead (JSON/binary)
- Kernel involvement in data path
- Not lock-free (kernel mediates)

**Use case:** Agent-to-agent communication in distributed systems. Not for hot path.

### Shared Memory Summary

| Mechanism | Latency | Throughput | Complexity | Best For |
|-----------|---------|------------|------------|----------|
| SPSC Ring Buffer | ~40ns | Very high | Low | Hot tile path |
| MPSC Channel | ~60ns | High | Medium | Multi-producer |
| shmget | ~200ns | High | High | Bulk matrices |
| mmap | ~100ns | High | Medium | Durable state |
| Unix Socket | ~500ns | Medium | Low | Cross-process |
| eBPF Ring | ~100ns | High | Very high | Tracing |

## Our Architecture Decision

**Primary:** SPSC ring buffer for tile flow (memory_pool.rs RingBuffer)

**Secondary:** Shared memory for room state pools (MemoryPool per PLATO room)

**Tertiary:** mmap for persistent history (PLATO room archives)

**Rejected:** shmget for tiles (overkill), Unix sockets (too slow), eBPF (wrong use case)

This matches the Fortran philosophy: known sizes at compile time, no runtime surprises.