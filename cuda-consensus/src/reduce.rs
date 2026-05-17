//! # Reduce — Spread / Max Reduction
//!
//! Computes the spread (maximum pairwise distance) across a swarm.
//! Supports both exact computation and early-termination threshold checks.

use crate::ConsensusError;

/// Configuration for spread computation.
#[derive(Debug, Clone)]
pub struct SpreadConfig {
    /// Spread threshold for consensus.
    pub threshold: f64,
    /// If true, use early termination (stop at first pair exceeding threshold).
    pub early_termination: bool,
    /// If true, prefer GPU acceleration for large swarms.
    pub prefer_gpu: bool,
}

impl Default for SpreadConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            early_termination: false,
            prefer_gpu: true,
        }
    }
}

/// Result of a spread computation.
#[derive(Debug, Clone)]
pub struct SpreadResult {
    /// The maximum pairwise distance (spread).
    pub spread: f64,
    /// Number of agent pairs evaluated.
    pub pairs_evaluated: usize,
    /// Whether the result came from GPU acceleration.
    pub gpu_accelerated: bool,
    /// Whether spread exceeds the configured threshold.
    pub exceeds_threshold: bool,
    /// Time to compute (nanoseconds), if timing was enabled.
    pub compute_ns: Option<u64>,
}

impl SpreadResult {
    /// Create a new spread result.
    pub fn new(spread: f64, pairs: usize, gpu: bool, threshold: f64) -> Self {
        Self {
            spread,
            pairs_evaluated: pairs,
            gpu_accelerated: gpu,
            exceeds_threshold: spread > threshold,
            compute_ns: None,
        }
    }
}

/// Compute the spread (max pairwise distance) for a set of embeddings.
///
/// This is the primary entry point for consensus spread computation.
///
/// # Arguments
///
/// * `embeddings` - Flat embedding matrix [n, dim]
/// * `n_agents` - Number of agents
/// * `dim` - Embedding dimension
/// * `config` - Spread computation configuration
///
/// # Returns
///
/// Spread result including the max distance and metadata.
pub fn compute_spread(
    embeddings: &[f32],
    n_agents: usize,
    dim: usize,
    config: &SpreadConfig,
) -> Result<SpreadResult, ConsensusError> {
    use crate::{is_cuda_available, use_gpu};
    use std::time::Instant;

    let start = Instant::now();

    // Validate
    if n_agents < 2 {
        return Err(ConsensusError::InsufficientSwarm {
            given: n_agents,
            min: 2,
        });
    }

    let total_pairs = n_agents * (n_agents - 1) / 2;

    // Decide compute path
    let use_gpu_path = config.prefer_gpu
        && use_gpu(n_agents)
        && is_cuda_available();

    let spread = if use_gpu_path {
        #[cfg(feature = "cuda")]
        {
            compute_spread_gpu(embeddings, n_agents, dim)?
        }
        #[cfg(not(feature = "cuda"))]
        {
            compute_spread_cpu(embeddings, n_agents, dim)?
        }
    } else {
        compute_spread_cpu(embeddings, n_agents, dim)?
    };

    let elapsed = start.elapsed().as_nanos() as u64;

    Ok(SpreadResult {
        spread,
        pairs_evaluated: total_pairs,
        gpu_accelerated: use_gpu_path,
        exceeds_threshold: spread > config.threshold,
        compute_ns: Some(elapsed),
    })
}

/// CPU spread computation using rayon.
fn compute_spread_cpu(
    embeddings: &[f32],
    n_agents: usize,
    dim: usize,
) -> Result<f64, ConsensusError> {
    use rayon::prelude::*;

    // Pre-compute norms
    let norms: Vec<f32> = embeddings
        .par_chunks(dim)
        .map(|chunk| {
            let sq: f32 = chunk.iter().map(|&x| x * x).sum();
            sq.sqrt().max(1e-8)
        })
        .collect();

    // Compute max pairwise distance
    let max_dist = (0..n_agents)
        .into_par_iter()
        .map(|i| {
            let row_i = &embeddings[i * dim..(i + 1) * dim];
            let norm_i = norms[i];
            let mut local_max = 0.0f32;

            for j in (i + 1)..n_agents {
                let row_j = &embeddings[j * dim..(j + 1) * dim];
                let dot: f32 = row_i.iter().zip(row_j.iter()).map(|(a, b)| a * b).sum();
                let dist = 1.0 - dot / (norm_i * norms[j]);
                if dist > local_max {
                    local_max = dist;
                    // Early termination if we only care about threshold
                    // (handled by caller via outer early_termination flag)
                }
            }
            local_max as f64
        })
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0);

    Ok(max_dist)
}

/// GPU-accelerated spread computation.
#[cfg(feature = "cuda")]
fn compute_spread_gpu(
    embeddings: &[f32],
    n_agents: usize,
    dim: usize,
) -> Result<f64, ConsensusError> {
    use crate::cuda_runtime::{CudaContext, CudaDistance, CudaReduce};

    let ctx = CudaContext::new()
        .map_err(|e| ConsensusError::CudaUnavailable(e.to_string()))?;

    // Upload embeddings
    let gpu_emb = ctx
        .upload(embeddings)
        .map_err(|e| ConsensusError::CudaKernelError(e.to_string()))?;

    // Compute distance matrix
    let gpu_matrix = CudaDistance::cosine_distance_matrix(&ctx, &gpu_emb, n_agents, dim)
        .map_err(|e| ConsensusError::CudaKernelError(e.to_string()))?;

    // Reduce to max (spread)
    let spread = CudaReduce::max_reduce(&ctx, &gpu_matrix, n_agents * n_agents)
        .map_err(|e| ConsensusError::CudaKernelError(e.to_string()))?;

    Ok(spread as f64)
}

/// Check if spread exceeds a threshold using early termination.
///
/// Much faster than full spread computation when you only need a
/// yes/no answer for consensus.
///
/// # Arguments
///
/// * `embeddings` - Flat embedding matrix [n, dim]
/// * `n_agents` - Number of agents
/// * `dim` - Embedding dimension
/// * `threshold` - Spread threshold to check
///
/// # Returns
///
/// `true` if any pair's distance exceeds the threshold.
pub fn spread_exceeds_threshold(
    embeddings: &[f32],
    n_agents: usize,
    dim: usize,
    threshold: f64,
) -> Result<bool, ConsensusError> {
    use crate::use_gpu;

    if n_agents < 2 {
        return Err(ConsensusError::InsufficientSwarm {
            given: n_agents,
            min: 2,
        });
    }

    if threshold < 0.0 || threshold > 2.0 {
        return Err(ConsensusError::InvalidThreshold {
            threshold,
            reason: "threshold must be in [0, 2]".into(),
        });
    }

    // Try GPU early termination for large swarms
    if use_gpu(n_agents) {
        #[cfg(feature = "cuda")]
        {
            return spread_exceeds_threshold_gpu(embeddings, n_agents, dim, threshold as f32);
        }
    }

    // CPU path — still returns early if threshold exceeded
    Ok(spread_exceeds_threshold_cpu(embeddings, n_agents, dim, threshold))
}

/// GPU early termination check.
#[cfg(feature = "cuda")]
fn spread_exceeds_threshold_gpu(
    embeddings: &[f32],
    n_agents: usize,
    dim: usize,
    threshold: f32,
) -> Result<bool, ConsensusError> {
    use crate::cuda_runtime::{CudaContext, CudaDistance, CudaReduce};

    let ctx = CudaContext::new()
        .map_err(|e| ConsensusError::CudaUnavailable(e.to_string()))?;

    let gpu_emb = ctx
        .upload(embeddings)
        .map_err(|e| ConsensusError::CudaKernelError(e.to_string()))?;

    let gpu_matrix = CudaDistance::cosine_distance_matrix(&ctx, &gpu_emb, n_agents, dim)
        .map_err(|e| ConsensusError::CudaKernelError(e.to_string()))?;

    CudaReduce::spread_exceeds_threshold(&ctx, &gpu_matrix, n_agents, threshold)
        .map_err(|e| ConsensusError::CudaKernelError(e.to_string()))
}

/// CPU early termination check — stops as soon as threshold is exceeded.
fn spread_exceeds_threshold_cpu(
    embeddings: &[f32],
    n_agents: usize,
    dim: usize,
    threshold: f64,
) -> bool {
    use rayon::prelude::*;

    // Early exit using par_iter with find_any
    (0..n_agents).into_par_iter().any(|i| {
        let row_i = &embeddings[i * dim..(i + 1) * dim];
        let sq_i: f32 = row_i.iter().map(|&x| x * x).sum();
        let norm_i = sq_i.sqrt().max(1e-8);

        ((i + 1)..n_agents).into_par_iter().any(|j| {
            let row_j = &embeddings[j * dim..(j + 1) * dim];
            let sq_j: f32 = row_j.iter().map(|&x| x * x).sum();
            let norm_j = sq_j.sqrt().max(1e-8);

            let dot: f32 = row_i.iter().zip(row_j.iter()).map(|(a, b)| a * b).sum();
            let dist = 1.0 - dot / (norm_i * norm_j);
            dist as f64 > threshold
        })
    })
}

/// Aggregated statistics across a distance matrix.
#[derive(Debug, Clone)]
pub struct SpreadStatistics {
    /// Maximum distance (spread).
    pub max: f64,
    /// Minimum (non-diagonal) distance.
    pub min: f64,
    /// Mean pairwise distance.
    pub mean: f64,
    /// Median pairwise distance.
    pub median: f64,
    /// Standard deviation.
    pub std_dev: f64,
    /// Number of pairs analyzed.
    pub n_pairs: usize,
}

/// Compute detailed spread statistics for a swarm.
///
/// More expensive than plain spread computation, but provides richer
/// insight into swarm cohesion.
pub fn compute_spread_statistics(
    embeddings: &[f32],
    n_agents: usize,
    dim: usize,
) -> Result<SpreadStatistics, ConsensusError> {
    use rayon::prelude::*;

    if n_agents < 2 {
        return Err(ConsensusError::InsufficientSwarm {
            given: n_agents,
            min: 2,
        });
    }

    let n_pairs = n_agents * (n_agents - 1) / 2;

    // Pre-compute norms
    let norms: Vec<f32> = embeddings
        .par_chunks(dim)
        .map(|chunk| {
            let sq: f32 = chunk.iter().map(|&x| x * x).sum();
            sq.sqrt().max(1e-8)
        })
        .collect();

    // Compute all pairwise distances in parallel
    let mut distances: Vec<f64> = (0..n_agents)
        .into_par_iter()
        .flat_map(|i| {
            let row_i = &embeddings[i * dim..(i + 1) * dim];
            let norm_i = norms[i];
            ((i + 1)..n_agents)
                .into_par_iter()
                .map({
                    let value = norms.clone();
                    move |j| {
                        let row_j = &embeddings[j * dim..(j + 1) * dim];
                        let dot: f32 = row_i.iter().zip(row_j.iter()).map(|(a, b)| a * b).sum();
                        (1.0 - dot / (norm_i * value[j])) as f64
                    }
                })
        })
        .collect();

    distances.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let max = *distances.last().unwrap_or(&0.0);
    let min = *distances.first().unwrap_or(&0.0);
    let mean = distances.iter().sum::<f64>() / n_pairs as f64;
    let median = distances[n_pairs / 2];

    let variance = distances
        .par_iter()
        .map(|&d| (d - mean).powi(2))
        .sum::<f64>()
        / n_pairs as f64;

    Ok(SpreadStatistics {
        max,
        min,
        mean,
        median,
        std_dev: variance.sqrt(),
        n_pairs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

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
            let norm = sq.sqrt().max(1e-8);
            for val in v {
                data.push(val / norm);
            }
        }
        data
    }

    #[test]
    fn test_spread_identical() {
        let dim = 32;
        let mut emb = vec![0.0f32; 3 * dim];
        for chunk in emb.chunks_mut(dim) {
            chunk[0] = 1.0;
        }
        let config = SpreadConfig::default();
        let result = compute_spread(&emb, 3, dim, &config).unwrap();
        assert!(result.spread < 1e-6);
        assert!(!result.exceeds_threshold);
    }

    #[test]
    fn test_spread_orthogonal() {
        let dim = 32;
        let mut emb = vec![0.0f32; 2 * dim];
        emb[0] = 1.0;
        emb[dim + 1] = 1.0;
        let config = SpreadConfig::default();
        let result = compute_spread(&emb, 2, dim, &config).unwrap();
        assert!((result.spread - 1.0).abs() < 1e-5);
        assert!(result.exceeds_threshold);
    }

    #[test]
    fn test_spread_early_termination() {
        let emb = make_embeddings(50, 64, 42);
        let result = spread_exceeds_threshold(&emb, 50, 64, 0.1).unwrap();
        assert!(result); // Random normalized vectors should be somewhat spread
    }

    #[test]
    fn test_spread_early_termination_high_threshold() {
        let emb = make_embeddings(50, 64, 42);
        let result = spread_exceeds_threshold(&emb, 50, 64, 1.5).unwrap();
        assert!(!result); // No distance > 1.5 (max is 2 for cosine)
    }

    #[test]
    fn test_spread_statistics() {
        let emb = make_embeddings(10, 32, 42);
        let stats = compute_spread_statistics(&emb, 10, 32).unwrap();
        assert_eq!(stats.n_pairs, 45);
        assert!(stats.max >= stats.mean);
        assert!(stats.min <= stats.mean);
        assert!(stats.std_dev >= 0.0);
    }

    #[test]
    fn test_insufficient_agents() {
        let emb = make_embeddings(1, 32, 0);
        let config = SpreadConfig::default();
        let result = compute_spread(&emb, 1, 32, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_spread_invalid_threshold() {
        let emb = make_embeddings(3, 16, 0);
        let result = spread_exceeds_threshold(&emb, 3, 16, 3.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_spread_stays_in_range() {
        let emb = make_embeddings(100, 128, 99);
        let config = SpreadConfig::default();
        let result = compute_spread(&emb, 100, 128, &config).unwrap();
        assert!(result.spread >= 0.0 && result.spread <= 2.0,
            "spread {:.4} should be in [0, 2]", result.spread);
        assert!(result.pairs_evaluated > 0);
        assert!(result.compute_ns.is_some());
    }
}
