#!/usr/bin/env python3
"""
python_binding.py — Python bindings for Plato CUDA Consensus.

Provides two paths:
1. **Rust native** (via PyO3 / maturin): `pip install plato-cuda-consensus`
2. **Pure Python fallback** (NumPy/Numba): works without Rust

## Usage

```python
from plato_cuda_consensus import compute_consensus_spread, is_cuda_available

# Or use the pure-Python fallback
from python_binding import compute_spread_numpy, compute_spread_numba

embeddings = np.random.randn(100, 384).astype(np.float32)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

spread = compute_spread_numpy(embeddings)  # NumPy path
spread = compute_spread_numba(embeddings)  # Numba GPU path (if CUDA available)
```

## Switching Logic

1. Try `plato_cuda_consensus` (Rust native) — fastest
2. Fall back to `numba.cuda` — medium (if CUDA + Numba available)
3. Fall back to `numpy` — universal fallback
"""

import warnings
import numpy as np
from typing import Optional, Tuple, List

# Optional imports — gracefully degrade
try:
    from numpy.linalg import norm as _np_norm
except ImportError:
    _np_norm = None

try:
    import numba
    from numba import cuda as _numba_cuda
    _HAS_NUMBA_CUDA = _numba_cuda.is_available()
except (ImportError, Exception):
    _HAS_NUMBA_CUDA = False

try:
    from plato_cuda_consensus import compute_consensus_spread as _rust_compute
    from plato_cuda_consensus import is_cuda_available as _rust_cuda_check
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

# ---------------------------------------------------------------------------
# CUDA Kernels (Numba)
# ---------------------------------------------------------------------------

if _HAS_NUMBA_CUDA:
    @_numba_cuda.jit
    def _cosine_distance_kernel(embeddings, output):
        """CUDA kernel: compute pairwise cosine distances."""
        i = _numba_cuda.grid(1)
        n = embeddings.shape[0]
        dim = embeddings.shape[1]

        if i >= n:
            return

        for j in range(i + 1, n):
            dot = 0.0
            norm_i = 0.0
            norm_j = 0.0

            for k in range(dim):
                vi = embeddings[i, k]
                vj = embeddings[j, k]
                dot += vi * vj
                norm_i += vi * vi
                norm_j += vj * vj

            denom = max(np.sqrt(norm_i) * np.sqrt(norm_j), 1e-8)
            dist = 1.0 - dot / denom
            output[i, j] = dist
            output[j, i] = dist

    @_numba_cuda.jit
    def _reduce_max_kernel(input_flat, output):
        """CUDA kernel: tree reduction to find max."""
        tid = _numba_cuda.threadIdx.x
        idx = _numba_cuda.blockIdx.x * _numba_cuda.blockDim.x + tid
        shared = _numba_cuda.shared.array(256, dtype=np.float32)

        shared[tid] = input_flat[idx] if idx < input_flat.shape[0] else -1.0
        _numba_cuda.syncthreads()

        s = 128
        while s > 0:
            if tid < s:
                shared[tid] = max(shared[tid], shared[tid + s])
            _numba_cuda.syncthreads()
            s //= 2

        if tid == 0:
            output[_numba_cuda.blockIdx.x] = shared[0]

# ---------------------------------------------------------------------------
# High-Level API
# ---------------------------------------------------------------------------

def is_cuda_available() -> bool:
    """
    Check if GPU acceleration is available via any path.

    Returns:
        True if Rust CUDA or Numba CUDA is available.
    """
    if _HAS_RUST:
        try:
            return _rust_cuda_check()
        except Exception:
            pass
    if _HAS_NUMBA_CUDA:
        try:
            return _numba_cuda.is_available()
        except Exception:
            pass
    return False


def compute_consensus_spread(
    embeddings: np.ndarray,
    threshold: float = 0.5,
    force_cpu: bool = False,
) -> float:
    """
    Compute the consensus spread (max pairwise cosine distance).

    Automatically selects the best available backend:
        Rust CUDA → Numba CUDA → NumPy

    Args:
        embeddings: Float32 array of shape (n_agents, dim)
        threshold: Consensus threshold
        force_cpu: Force CPU even if GPU is available

    Returns:
        Maximum pairwise cosine distance (spread)
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    n_agents, dim = embeddings.shape

    if n_agents < 2:
        raise ValueError(f"Need at least 2 agents, got {n_agents}")

    # Check for NaN/Inf
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinity")

    # Route to best backend
    if not force_cpu and _HAS_RUST and n_agents > 20:
        try:
            flat = embeddings.ravel()
            spread = _rust_compute(flat, n_agents, dim)
            return spread
        except Exception as e:
            warnings.warn(f"Rust backend failed, falling back: {e}")

    if not force_cpu and _HAS_NUMBA_CUDA and n_agents > 50:
        try:
            spread = _compute_spread_numba_cuda(embeddings)
            return spread
        except Exception as e:
            warnings.warn(f"Numba CUDA failed, falling back: {e}")

    return compute_spread_numpy(embeddings)


def compute_spread_numpy(embeddings: np.ndarray) -> float:
    """
    Compute consensus spread using NumPy (batched matrix multiply).

    This is the universal fallback — works everywhere.

    Complexity: O(n²·d) but uses BLAS for the matrix multiply.

    Args:
        embeddings: Float32 array of shape (n_agents, dim)

    Returns:
        Maximum pairwise cosine distance
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = embeddings / norms

    # Cosine similarity matrix via matrix multiply
    sim = normalized @ normalized.T  # (n, n)

    # Clip numerical noise
    sim = np.clip(sim, -1.0, 1.0)

    # Distance matrix: d = 1 - sim
    dist = 1.0 - sim

    # Spread = max off-diagonal
    np.fill_diagonal(dist, 0.0)
    spread = float(dist.max())

    return spread


def compute_spread_numba_cuda(
    embeddings: np.ndarray,
    block_size: int = 256,
) -> float:
    """
    Compute consensus spread using Numba CUDA.

    Much faster than NumPy for large swarms (n > 50).

    Args:
        embeddings: Float32 array of shape (n_agents, dim)
        block_size: CUDA block size

    Returns:
        Maximum pairwise cosine distance
    """
    if not _HAS_NUMBA_CUDA:
        raise RuntimeError("Numba CUDA not available")

    n, dim = embeddings.shape

    # Upload to GPU
    d_emb = _numba_cuda.to_device(embeddings)
    d_dist = _numba_cuda.device_array((n, n), dtype=np.float32)

    # Launch distance kernel — 1D grid of n agents
    blocks = (n + block_size - 1) // block_size
    _cosine_distance_kernel[blocks, block_size](d_emb, d_dist)

    # Sync after kernel
    _numba_cuda.synchronize()

    # Download and verify
    h_dist = d_dist.copy_to_host()

    # Compute spread
    np.fill_diagonal(h_dist, 0.0)
    spread = float(h_dist.max())

    return spread


def compute_spread_numba_cpu(embeddings: np.ndarray) -> float:
    """
    Compute consensus spread using Numba JIT (CPU, no CUDA).

    Faster than NumPy for moderate sizes due to loop optimizations.

    Args:
        embeddings: Float32 array of shape (n_agents, dim)

    Returns:
        Maximum pairwise cosine distance
    """
    @numba.jit(nopython=True, parallel=True)
    def _spread_jit(data: np.ndarray) -> float:
        n = data.shape[0]
        dim = data.shape[1]
        max_dist = 0.0

        for i in numba.prange(n):
            row_i = data[i]
            sq_i = 0.0
            for k in range(dim):
                sq_i += row_i[k] * row_i[k]
            norm_i = max(np.sqrt(sq_i), 1e-8)

            for j in range(i + 1, n):
                row_j = data[j]
                sq_j = 0.0
                dot = 0.0
                for k in range(dim):
                    vj = row_j[k]
                    sq_j += vj * vj
                    dot += row_i[k] * vj
                norm_j = max(np.sqrt(sq_j), 1e-8)
                dist = 1.0 - dot / (norm_i * norm_j)
                if dist > max_dist:
                    max_dist = dist

        return max_dist

    return float(_spread_jit(embeddings))


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def compute_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute the full pairwise cosine distance matrix.

    Args:
        embeddings: Float32 array of shape (n_agents, dim)

    Returns:
        Float32 array of shape (n_agents, n_agents)
    """
    return 1.0 - cosine_similarity_matrix(embeddings)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute the full pairwise cosine similarity matrix.

    Args:
        embeddings: Float32 array of shape (n_agents, dim)

    Returns:
        Float32 array of shape (n_agents, n_agents), values in [-1, 1]
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = embeddings / norms
    sim = normalized @ normalized.T
    return np.clip(sim, -1.0, 1.0)


def consensus_is_reached(
    embeddings: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[bool, float]:
    """
    Check if consensus is reached (spread <= threshold).

    Args:
        embeddings: Float32 array of shape (n_agents, dim)
        threshold: Maximum allowed spread

    Returns:
        (consensus_reached, current_spread)
    """
    spread = compute_consensus_spread(embeddings, threshold)
    return spread <= threshold, spread


# ---------------------------------------------------------------------------
# Benchmark Helper
# ---------------------------------------------------------------------------

def benchmark_backends(
    sizes: Optional[List[int]] = None,
    dim: int = 384,
    samples: int = 3,
) -> dict:
    """
    Benchmark all available backends across swarm sizes.

    Args:
        sizes: List of swarm sizes to test
        dim: Embedding dimension
        samples: Number of timing samples per size

    Returns:
        Dict mapping backend names to {size: time_ms}
    """
    if sizes is None:
        sizes = [10, 50, 100, 200, 500, 1000]

    import time

    backends = {}
    backends["numpy"] = compute_spread_numpy

    if _HAS_NUMBA_CUDA:
        backends["numba_cuda"] = compute_spread_numba_cuda

    results = {name: {} for name in backends}

    for name, func in backends.items():
        print(f"\nBenchmarking: {name}")

        for n in sizes:
            emb = np.random.randn(n, dim).astype(np.float32)
            emb /= np.linalg.norm(emb, axis=1, keepdims=True)

            times = []
            for _ in range(samples):
                start = time.perf_counter()
                func(emb)
                elapsed = (time.perf_counter() - start) * 1000  # ms
                times.append(elapsed)

            mean_time = np.mean(times)
            results[name][n] = mean_time
            print(f"  n={n:>5}: {mean_time:.3f} ms")

    return results


# ---------------------------------------------------------------------------
# Command-line Interface
# ---------------------------------------------------------------------------

def main():
    """Quick demo from the command line."""
    import sys

    print("Plato CUDA Consensus — Python Bindings")
    print("=" * 45)
    print(f"  Rust backend:      {'✓' if _HAS_RUST else '✗'}")
    print(f"  Numba CUDA:        {'✓' if _HAS_NUMBA_CUDA else '✗'}")
    print(f"  NumPy:             {'✓' if _np_norm else '✗'}")
    print()

    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        dim = 128
        sizes = [10, 50, 100, 200, 500]
        benchmark_backends(sizes, dim)
        return

    # Quick demo
    for n in [10, 100, 500]:
        emb = np.random.randn(n, 384).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)

        spread = compute_consensus_spread(emb)
        _, reached = consensus_is_reached(emb, 0.5)

        print(
            f"  n={n:>4}: spread={spread:.4f}  "
            f"consensus={'✓' if reached else '✗'}"
        )


if __name__ == "__main__":
    main()
