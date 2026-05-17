#!/usr/bin/env python3
"""
benchmark_cuda.py

Python harness using ctypes to call CUDA kernels and compare against NumPy/Numba.
Falls back gracefully if no GPU is available.

Fortran philosophy: no classes in kernels — this harness can be class-free too.
But for benchmarking organization, a simple struct dict is used (not a class).
"""

import ctypes
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# CUDA availability check
# ---------------------------------------------------------------------------
CUDA_AVAILABLE = False
cuda_clib = None

try:
    cuda_clib = ctypes.CDLL("libcudart.so")
    CUDA_AVAILABLE = True
except (OSError, AttributeError):
    CUDA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Kernel handle (ctypes void pointer)
# ---------------------------------------------------------------------------
class KernelHandle:
    """Minimal handle — Fortran COMMON block equivalent."""
    def __init__(self):
        self.module: Optional[ctypes.c_void_p] = None
        self.cosine_distance_fn: Optional[ctypes.c_void_p] = None
        self.spread_reduce_fn: Optional[ctypes.c_void_p] = None
        self.consensus_snap_fn: Optional[ctypes.c_void_p] = None

CUDA_KERNELS = KernelHandle()

# ---------------------------------------------------------------------------
# Device-side kernels — compiled ahead-of-time
# For RTX 4050 (sm_75), kernels must be pre-compiled.
# This script uses runtime loading of pre-built .so files.
# ---------------------------------------------------------------------------

def cuda_init() -> bool:
    """Initialize CUDA. Returns True if GPU is available."""
    if not CUDA_AVAILABLE:
        print("[CPU fallback] CUDA runtime not found.")
        return False

    result = cuda_clib.cudaInit(0)
    if result != 0:
        print(f"[CPU fallback] cudaInit failed with code {result}")
        return False

    print("[GPU] CUDA initialized successfully.")
    return True

def cuda_free(handle: KernelHandle):
    """Free CUDA resources."""
    if handle.module:
        cuda_clib.cudaModuleUnload(handle.module)
        handle.module = None

# ---------------------------------------------------------------------------
# NumPy CPU fallback — Fortran-style: pure arithmetic, no GPU
# ---------------------------------------------------------------------------

def cosine_distance_cpu_np(embeddings: np.ndarray) -> np.ndarray:
    """
    NumPy implementation — same semantics as CUDA kernel.
    Each row of embeddings is an agent's vector.
    Returns cosine distance matrix.
    """
    n_agents, dim = embeddings.shape
    # Normalize rows
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = embeddings / norms

    # Cosine similarity = dot product of normalized vectors
    cos_sim = normed @ normed.T

    # Clamp for numerical safety
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    # Cosine distance = 1 - cosine similarity
    dist_matrix = 1.0 - cos_sim
    np.fill_diagonal(dist_matrix, 0.0)

    return dist_matrix

def spread_compute(dist_matrix: np.ndarray, threshold: float):
    """
    Compute spread statistics — Fortran-style: explicit arithmetic.
    Returns (spread, should_snap, is_full_snap).
    """
    n_agents = dist_matrix.shape[0]

    # Mean pairwise distance
    total = np.sum(dist_matrix)
    spread = total / (n_agents * n_agents - n_agents)  # exclude diagonal

    max_dist = np.max(dist_matrix)
    should_snap = 1 if max_dist < threshold else 0
    is_full_snap = 1 if np.all(dist_matrix < threshold) else 0

    return spread, should_snap, is_full_snap

def find_medoid(dist_matrix: np.ndarray) -> tuple[int, float]:
    """
    Find medoid — Fortran-style: explicit loop, no templates.
    Returns (medoid_index, medoid_score).
    """
    n_agents = dist_matrix.shape[0]
    scores = np.sum(dist_matrix, axis=1)  # sum of distances from each agent
    medoid = int(np.argmin(scores))
    return medoid, float(scores[medoid])

def numba_cosine_distance(embeddings: np.ndarray) -> np.ndarray:
    """
    Numba JIT implementation — GPU-like parallel semantics, CPU execution.
    """
    try:
        from numba import njit, prange

        @njit(parallel=True)
        def compute_dist(emb: np.ndarray) -> np.ndarray:
            n, dim = emb.shape
            result = np.zeros((n, n), dtype=np.float32)

            for i in prange(n):
                # ||vi||
                norm_i = 0.0
                for k in range(dim):
                    x = emb[i, k]
                    norm_i += x * x
                norm_i = np.sqrt(norm_i)
                if norm_i == 0.0:
                    norm_i = 1.0

                for j in range(n):
                    if j == i:
                        result[i, j] = 0.0
                        continue

                    # dot product
                    dot = 0.0
                    for k in range(dim):
                        dot += emb[i, k] * emb[j, k]

                    # ||vj||
                    norm_j = 0.0
                    for k in range(dim):
                        x = emb[j, k]
                        norm_j += x * x
                    norm_j = np.sqrt(norm_j)
                    if norm_j == 0.0:
                        norm_j = 1.0

                    cos_sim = dot / (norm_i * norm_j)
                    if cos_sim > 1.0: cos_sim = 1.0
                    if cos_sim < -1.0: cos_sim = -1.0

                    result[i, j] = 1.0 - cos_sim

            return result

        return compute_dist(embeddings.astype(np.float32))
    except ImportError:
        return cosine_distance_cpu_np(embeddings)

# ---------------------------------------------------------------------------
# CPU baseline using ctypes (loads the .so if it exists)
# ---------------------------------------------------------------------------

def load_cuda_kernels(cu_so_path: str) -> KernelHandle:
    """
    Load pre-compiled CUDA kernels from .so file.
    Fortran style: explicit void* handles, no class wrapping of device code.
    """
    if not os.path.exists(cu_so_path):
        print(f"[WARNING] {cu_so_path} not found. CPU fallback only.")
        return CUDA_KERNELS

    try:
        cuda_lib = ctypes.CDLL(cu_so_path)
        CUDA_KERNELS.module = cuda_lib
        # Mark functions as available (simplified — real impl would resolve symbols)
        print(f"[GPU] Loaded kernels from {cu_so_path}")
        return CUDA_KERNELS
    except OSError as e:
        print(f"[CPU fallback] Failed to load {cu_so_path}: {e}")
        return CUDA_KERNELS

# ---------------------------------------------------------------------------
# Benchmark runner — Fortran philosophy: struct-of-arrays, no classes
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Simple result struct — Fortran derived type equivalent."""
    name: str
    time_sec: float
    spread: float
    should_snap: int
    is_full_snap: int
    medoid: int
    medoid_score: float
    max_diff_from_ref: float  # max absolute diff from NumPy baseline

def run_benchmark(
    name: str,
    embeddings: np.ndarray,
    threshold: float,
    compute_fn
) -> BenchmarkResult:
    """Run one benchmark variant. Fortran style: function pointer dispatch."""
    t0 = time.perf_counter()
    dist_matrix = compute_fn(embeddings)
    elapsed = time.perf_counter() - t0

    spread, should_snap, is_full_snap = spread_compute(dist_matrix, threshold)
    medoid, medoid_score = find_medoid(dist_matrix)

    # Compare against NumPy reference
    ref_dist = cosine_distance_cpu_np(embeddings)
    max_diff = float(np.max(np.abs(dist_matrix - ref_dist)))

    print(f"  {name}: {elapsed:.4f}s | spread={spread:.4f} | "
          f"snap={should_snap}/{is_full_snap} | "
          f"medoid={medoid}({medoid_score:.2f}) | "
          f"Δref={max_diff:.2e}")

    return BenchmarkResult(
        name=name, time_sec=elapsed,
        spread=spread, should_snap=should_snap, is_full_snap=is_full_snap,
        medoid=medoid, medoid_score=medoid_score, max_diff_from_ref=max_diff
    )

# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("CUDA Consensus Kernel Benchmark")
    print("=" * 60)

    cuda_init()

    # Test configurations — Fortran DATA statements equivalent
    configs = [
        # (n_agents, dim, threshold)
        (16,    64,    0.3),
        (32,    128,   0.3),
        (64,    256,   0.25),
        (128,   512,   0.25),
        (256,   768,   0.2),
        (512,   1024,  0.2),
    ]

    for n_agents, dim, threshold in configs:
        print(f"\n--- Config: {n_agents} agents x {dim} dim (threshold={threshold}) ---")

        rng = np.random.default_rng(seed=42)
        embeddings = rng.normal(0.0, 1.0, (n_agents, dim)).astype(np.float32)

        # ---- NumPy reference ----
        ref_result = run_benchmark("NumPy (ref)", embeddings, threshold, cosine_distance_cpu_np)

        # ---- Numba ----
        nb_result = run_benchmark("Numba JIT", embeddings, threshold, numba_cosine_distance)

        # ---- CUDA .so (if available) ----
        cu_so = os.path.join(os.path.dirname(__file__), "cuda_consensus_kernel.so")
        if os.path.exists(cu_so):
            load_cuda_kernels(cu_so)
            # Note: actual CUDA kernel calls would need full ctypes signature setup.
            # For now, just acknowledge the .so exists.
            print(f"  [GPU] cuda_consensus_kernel.so found (kernel calls omitted for brevity)")
        else:
            print(f"  [GPU] .so not built yet — run ./build.sh first")

        print(f"  Reference spread={ref_result.spread:.4f} | Numba Δ={nb_result.max_diff_from_ref:.2e}")

    print("\n" + "=" * 60)
    print("Benchmark complete.")
    print("To build CUDA kernels: ./build.sh")
    print("=" * 60)

if __name__ == "__main__":
    main()