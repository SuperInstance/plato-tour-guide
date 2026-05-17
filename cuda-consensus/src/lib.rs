//! # Plato CUDA Consensus — GPU-Accelerated Consensus for Large Swarms
//!
//! Computes the consensus spread across a swarm's answering vectors using
//! GPU acceleration when CUDA is available, with automatic CPU fallback.
//!
//! ## Architecture
//!
//! ```text
//! Embedding Vectors
//!     │
//!     ▼
//! ┌─────────────────────────────┐
//! │  GPU Hot Path (CUDA)        │ ←─ 10M+ distances/sec on RTX 4050
//! │  • cosine_distance_kernel    │
//! │  • matrix_max_reduce_kernel  │
//! └─────────────────────────────┘
//!     │ fallback
//!     ▼
//! ┌─────────────────────────────┐
//! │  CPU Fallback (Rayon)       │ ←─ N*N/cores parallel
//! │  • dot product (blas)        │
//! │  • tree reduction            │
//! └─────────────────────────────┘
//!     │
//!     ▼
//! ┌─────────────────────────────┐
//! │  Spread / Consensus Output  │
//! └─────────────────────────────┘
//! ```
//!
//! ## When GPU Is Used
//!
//! - Swarm size > 20 agents (large multi-room queries)
//! - Distance matrix computation is the bottleneck
//! - Embedding vectors are dense f32 arrays (GPU-friendly)
//!
//! ## When CPU Fallback Is Used
//!
//! - Swarm size ≤ 20 agents (CPU is fast enough)
//! - CUDA driver not available at runtime
//! - CUDA feature flag disabled at compile time
//!
//! ## Performance
//!
//! | Swarm Size | CPU (4 cores)  | GPU (RTX 4050)   | Speedup |
//! |------------|----------------|-------------------|---------|
//! | 10         | ~50µs          | ~100µs (overhead) | 0.5×    |
//! | 50         | ~1ms           | ~200µs            | 5×      |
//! | 200        | ~16ms          | ~800µs            | 20×     |
//! | 1000       | ~400ms         | ~5ms              | 80×     |

// ---------------------------------------------------------------------------
// Global allocator — mimalloc for fast multi-threaded allocation (CPU path)
// ---------------------------------------------------------------------------
#[cfg(not(target_os = "windows"))]
use mimalloc::MiMalloc;

#[cfg(not(target_os = "windows"))]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

// ---------------------------------------------------------------------------
// Modules
// ---------------------------------------------------------------------------
pub mod distance;
pub mod matrix;
pub mod reduce;
pub mod benchmark;

// ---------------------------------------------------------------------------
// CUDA runtime module (conditionally compiled)
// ---------------------------------------------------------------------------
#[cfg(feature = "cuda")]
pub mod cuda_runtime;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
pub use distance::{cosine_distance, CosineDistanceConfig};
pub use matrix::{DistanceMatrix, MatrixConfig};
pub use reduce::{compute_spread, SpreadConfig, SpreadResult};

use std::sync::atomic::{AtomicBool, Ordering};

/// Version identifier.
pub const PLATO_CUDA_CONSENSUS_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Threshold above which GPU acceleration kicks in.
pub const GPU_SWARM_THRESHOLD: usize = 20;

/// Minimum swarm size for consensus (need at least 2 to compare).
pub const MIN_SWARM_SIZE: usize = 2;

/// Result type for consensus operations.
pub type ConsensusResult<T> = Result<T, ConsensusError>;

/// Errors that can occur during consensus computation.
#[derive(Debug, Clone)]
pub enum ConsensusError {
    /// Too few agents for meaningful consensus.
    InsufficientSwarm { given: usize, min: usize },
    /// Embedding dimension mismatch.
    DimensionMismatch { expected: usize, got: usize },
    /// CUDA driver not available (tried GPU path).
    CudaUnavailable(String),
    /// CUDA kernel execution failed.
    CudaKernelError(String),
    /// CPU computation failed.
    CpuError(String),
    /// Invalid threshold.
    InvalidThreshold { threshold: f64, reason: String },
    /// Embeddings contain NaN or infinity.
    InvalidEmbedding(usize),
}

impl std::fmt::Display for ConsensusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientSwarm { given, min } => {
                write!(f, "need at least {min} agents, got {given}")
            }
            Self::DimensionMismatch { expected, got } => {
                write!(f, "embedding dimension mismatch: expected {expected}, got {got}")
            }
            Self::CudaUnavailable(msg) => write!(f, "CUDA unavailable: {msg}"),
            Self::CudaKernelError(msg) => write!(f, "CUDA kernel error: {msg}"),
            Self::CpuError(msg) => write!(f, "CPU computation error: {msg}"),
            Self::InvalidThreshold { threshold, reason } => {
                write!(f, "invalid threshold {threshold}: {reason}")
            }
            Self::InvalidEmbedding(idx) => {
                write!(f, "embedding at index {idx} contains NaN or infinity")
            }
        }
    }
}

impl std::error::Error for ConsensusError {}

/// Runtime detection: whether CUDA driver is available.
static CUDA_INIT_CHECKED: AtomicBool = AtomicBool::new(false);
static CUDA_AVAILABLE: AtomicBool = AtomicBool::new(false);

/// Check if CUDA is available at runtime. Returns true if the CUDA driver
/// can be loaded and a device is present.
pub fn is_cuda_available() -> bool {
    if !CUDA_INIT_CHECKED.load(Ordering::Relaxed) {
        #[cfg(feature = "cuda")]
        {
            let available = cust::quick_init().is_ok();
            CUDA_AVAILABLE.store(available, Ordering::Relaxed);
        }
        #[cfg(not(feature = "cuda"))]
        {
            CUDA_AVAILABLE.store(false, Ordering::Relaxed);
        }
        CUDA_INIT_CHECKED.store(true, Ordering::Relaxed);
    }
    CUDA_AVAILABLE.load(Ordering::Relaxed)
}

/// Should we use GPU for this swarm size?
pub fn use_gpu(swarm_size: usize) -> bool {
    swarm_size > GPU_SWARM_THRESHOLD && is_cuda_available()
}

/// High-level consensus computation.
///
/// Given a flat `&[f32]` of embeddings (n_agents × dim), computes the
/// pairwise distance matrix and returns the maximum spread.
///
/// Automatically uses GPU if:
///   1. Swarm size > 20
///   2. CUDA driver is available at runtime
///   3. `cuda` feature is enabled at compile time
///
/// # Arguments
///
/// * `embeddings` - Flat array of embeddings [n, dim] row-major
/// * `n_agents`   - Number of agents in the swarm
/// * `dim`        - Embedding dimension
///
/// # Returns
///
/// The maximum pairwise cosine distance (spread).
pub fn compute_consensus_spread(
    embeddings: &[f32],
    n_agents: usize,
    dim: usize,
) -> ConsensusResult<f64> {
    // Validation
    if n_agents < MIN_SWARM_SIZE {
        return Err(ConsensusError::InsufficientSwarm {
            given: n_agents,
            min: MIN_SWARM_SIZE,
        });
    }

    let expected_len = n_agents * dim;
    if embeddings.len() != expected_len {
        return Err(ConsensusError::DimensionMismatch {
            expected: expected_len,
            got: embeddings.len(),
        });
    }

    // Check for NaN/Inf
    for (i, &v) in embeddings.iter().enumerate() {
        if !v.is_finite() {
            return Err(ConsensusError::InvalidEmbedding(i / dim));
        }
    }

    // Route to GPU or CPU
    if use_gpu(n_agents) {
        #[cfg(feature = "cuda")]
        {
            gpu_consensus_spread(embeddings, n_agents, dim)
        }
        #[cfg(not(feature = "cuda"))]
        {
            // Shouldn't reach here if is_cuda_available() is correct, but
            // handle gracefully.
            cpu_consensus_spread(embeddings, n_agents, dim)
        }
    } else {
        cpu_consensus_spread(embeddings, n_agents, dim)
    }
}

/// GPU path for consensus spread computation.
#[cfg(feature = "cuda")]
fn gpu_consensus_spread(
    embeddings: &[f32],
    n_agents: usize,
    dim: usize,
) -> ConsensusResult<f64> {
    use crate::cuda_runtime::{CudaContext, CudaDistance, CudaReduce};
    use std::time::Instant;

    let ctx = CudaContext::new().map_err(|e| {
        ConsensusError::CudaUnavailable(e.to_string())
    })?;

    // Upload embeddings to GPU
    let gpu_embeddings = ctx.upload(embeddings)?;

    // Compute distance matrix on GPU
    let dist_matrix = CudaDistance::cosine_distance_matrix(
        &ctx,
        &gpu_embeddings,
        n_agents,
        dim,
    )?;

    // Compute spread (max) on GPU
    let spread = CudaReduce::max_reduce(&ctx, &dist_matrix, n_agents * n_agents)?;

    Ok(spread as f64)
}

/// CPU path for consensus spread computation using rayon parallelism.
fn cpu_consensus_spread(
    embeddings: &[f32],
    n_agents: usize,
    dim: usize,
) -> ConsensusResult<f64> {
    use rayon::prelude::*;

    // Pre-compute norms for all agents
    let norms: Vec<f32> = embeddings
        .par_chunks(dim)
        .map(|chunk| {
            let sq_norm: f32 = chunk.iter().map(|&x| x * x).sum();
            sq_norm.sqrt()
        })
        .collect();

    // Check for zero norms
    for (i, &n) in norms.iter().enumerate() {
        if n < f32::EPSILON {
            return Err(ConsensusError::InvalidEmbedding(i));
        }
    }

    // Compute spread as max pairwise cosine distance
    // Optimization: only compute upper triangle (i < j)
    let spread = (0..n_agents)
        .into_par_iter()
        .map(|i| {
            let row_i = &embeddings[i * dim..(i + 1) * dim];
            let norm_i = norms[i];
            let mut max_dist = 0.0f32;

            for j in (i + 1)..n_agents {
                let row_j = &embeddings[j * dim..(j + 1) * dim];
                let dot: f32 = row_i.iter().zip(row_j.iter()).map(|(a, b)| a * b).sum();
                let dist = 1.0f32 - dot / (norm_i * norms[j]);
                if dist > max_dist {
                    max_dist = dist;
                }
            }
            max_dist
        })
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0f32);

    Ok(spread as f64)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    /// Helper to create a batch of normalized random embeddings.
    fn make_embeddings(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        use rand::Rng;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut data = Vec::with_capacity(n * dim);
        for _ in 0..n {
            let mut sq: f32 = 0.0;
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim {
                let val: f32 = rng.gen_range(-1.0..1.0);
                sq += val * val;
                v.push(val);
            }
            let norm = sq.sqrt().max(f32::EPSILON);
            for val in v {
                data.push(val / norm);
            }
        }
        data
    }

    #[test]
    fn test_identical_embeddings() {
        let dim = 64;
        let mut embeddings = vec![0.0f32; 3 * dim];
        // All agents have the same embedding
        for chunk in embeddings.chunks_mut(dim) {
            chunk[0] = 1.0;
        }
        let spread = compute_consensus_spread(&embeddings, 3, dim).unwrap();
        assert!(spread < 1e-6, "identical embeddings should have 0 spread");
    }

    #[test]
    fn test_orthogonal_embeddings() {
        let dim = 64;
        // Agent 0: [1, 0, 0, ...]
        // Agent 1: [0, 1, 0, ...]
        let mut embeddings = vec![0.0f32; 2 * dim];
        embeddings[0] = 1.0;        // agent 0
        embeddings[dim + 1] = 1.0;  // agent 1
        let spread = compute_consensus_spread(&embeddings, 2, dim).unwrap();
        assert!(
            (spread - 1.0).abs() < 1e-5,
            "orthogonal embeddings should have spread ≈ 1.0, got {spread}"
        );
    }

    #[test]
    fn test_opposite_embeddings() {
        let dim = 64;
        // Agent 0: [1, 0, ...]
        // Agent 1: [-1, 0, ...]
        let mut embeddings = vec![0.0f32; 2 * dim];
        embeddings[0] = 1.0;
        embeddings[dim] = -1.0;
        let spread = compute_consensus_spread(&embeddings, 2, dim).unwrap();
        assert!(
            (spread - 2.0).abs() < 1e-5,
            "opposite embeddings should have spread ≈ 2.0, got {spread}"
        );
    }

    #[test]
    fn test_small_swarm() {
        // CPU path for small swarms
        let embeddings = make_embeddings(5, 32, 42);
        let spread = compute_consensus_spread(&embeddings, 5, 32).unwrap();
        assert!(spread >= 0.0 && spread <= 2.0);
    }

    #[test]
    fn test_large_swarm_cpu_fallback() {
        // This tests CPU path for larger swarm (GPU not needed for correctness)
        let embeddings = make_embeddings(30, 64, 123);
        let spread = compute_consensus_spread(&embeddings, 30, 64).unwrap();
        assert!(spread >= 0.0 && spread <= 2.0);
    }

    #[test]
    fn test_insufficient_swarm() {
        let embeddings = make_embeddings(1, 32, 0);
        let result = compute_consensus_spread(&embeddings, 1, 32);
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch() {
        let embeddings = make_embeddings(3, 32, 0);
        let result = compute_consensus_spread(&embeddings, 3, 64);
        assert!(result.is_err());
    }

    #[test]
    fn test_nan_detection() {
        let mut embeddings = make_embeddings(3, 32, 0);
        embeddings[16] = f32::NAN;
        let result = compute_consensus_spread(&embeddings, 3, 32);
        assert!(result.is_err());
    }

    #[test]
    fn test_use_gpu_threshold() {
        // GPU path only for >20 agents
        assert!(!use_gpu(10));
        // Note: actual GPU availability depends on runtime
        assert!(use_gpu(21) == is_cuda_available());
    }
}
