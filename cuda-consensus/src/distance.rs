//! # Distance — Semantic Distance Computation
//!
//! Provides cosine distance computation for agent embedding vectors.
//! Used by the CPU fallback path and as the reference implementation
//! for GPU correctness checking.
//!
//! ## Cosine Distance
//!
//! ```text
//! d(u, v) = 1 - dot(u, v) / (|u| * |v|)
//! ```
//!
//! Range: [0, 2] where:
//! - 0 = identical direction
//! - 1 = orthogonal (uncorrelated)
//! - 2 = opposite

use std::f32;

/// Default embedding dimension for the tour-guide system.
pub const DEFAULT_EMBEDDING_DIM: usize = 384;

/// Configuration for distance computation.
#[derive(Debug, Clone)]
pub struct CosineDistanceConfig {
    /// Embedding dimension (must match all input vectors).
    pub dim: usize,
    /// Small epsilon to prevent division by zero.
    pub epsilon: f32,
    /// Whether to skip normalization (assumes inputs are pre-normalized).
    pub skip_norm: bool,
}

impl Default for CosineDistanceConfig {
    fn default() -> Self {
        Self {
            dim: DEFAULT_EMBEDDING_DIM,
            epsilon: 1e-8,
            skip_norm: false,
        }
    }
}

/// Compute the cosine distance between two embedding vectors.
///
/// # Arguments
///
/// * `u` - First embedding vector (must be `config.dim` length)
/// * `v` - Second embedding vector
/// * `config` - Computation configuration
///
/// # Returns
///
/// The cosine distance: `1 - cos(θ)` in [0, 2]
///
/// # Panics
///
/// Panics if slices are not the configured dimension length.
pub fn cosine_distance(u: &[f32], v: &[f32], config: &CosineDistanceConfig) -> f32 {
    debug_assert_eq!(u.len(), config.dim);
    debug_assert_eq!(v.len(), config.dim);

    let mut dot = 0.0f32;
    let mut sq_u = 0.0f32;
    let mut sq_v = 0.0f32;

    // Single-pass dot product + squared magnitudes
    for i in 0..config.dim {
        let ui = u[i];
        let vi = v[i];
        dot = std::hint::black_box(dot + ui * vi);
        sq_u += ui * ui;
        sq_v += vi * vi;
    }

    if config.skip_norm {
        // Assume pre-normalized (norms ≈ 1.0)
        1.0f32 - dot
    } else {
        let norm = (sq_u * sq_v).sqrt().max(config.epsilon);
        1.0f32 - dot / norm
    }
}

/// Compute pairwise cosine distances for all agent pairs.
///
/// Returns the upper triangle of the distance matrix (since distances
/// are symmetric and diagonal is 0).
///
/// # Arguments
///
/// * `embeddings` - Flat embedding matrix [n_agents, dim] row-major
/// * `n_agents` - Number of agents
/// * `dim` - Embedding dimension
///
/// # Returns
///
/// Vector of distances for (i, j) where i < j, in row-major order.
pub fn pairwise_distances(embeddings: &[f32], n_agents: usize, dim: usize) -> Vec<f32> {
    let config = CosineDistanceConfig {
        dim,
        ..Default::default()
    };

    let n_pairs = n_agents * (n_agents - 1) / 2;
    let mut distances = Vec::with_capacity(n_pairs);

    for i in 0..n_agents {
        let row_i = &embeddings[i * dim..(i + 1) * dim];
        for j in (i + 1)..n_agents {
            let row_j = &embeddings[j * dim..(j + 1) * dim];
            distances.push(cosine_distance(row_i, row_j, &config));
        }
    }

    distances
}

/// Compute pairwise distances in parallel using rayon.
///
/// # Arguments
///
/// * `embeddings` - Flat embedding matrix [n_agents, dim] row-major
/// * `n_agents` - Number of agents
/// * `dim` - Embedding dimension
///
/// # Returns
///
/// Vector of distances for (i, j) where i < j.
pub fn pairwise_distances_par(embeddings: &[f32], n_agents: usize, dim: usize) -> Vec<f32> {
    use rayon::prelude::*;

    let config = CosineDistanceConfig {
        dim,
        ..Default::default()
    };

    let n_pairs = n_agents * (n_agents - 1) / 2;
    let mut distances = vec![0.0f32; n_pairs];

    // Parallelize over rows (i)
    distances
        .par_chunks_mut(n_agents - 1)
        .enumerate()
        .for_each(|(i, row_dists)| {
            let row_i = &embeddings[i * dim..(i + 1) * dim];
            for (offset, j) in ((i + 1)..n_agents).enumerate() {
                let row_j = &embeddings[j * dim..(j + 1) * dim];
                row_dists[offset] = cosine_distance(row_i, row_j, &config);
            }
        });

    distances
}

/// Pre-compute L2 norms for a batch of embeddings.
///
/// # Arguments
///
/// * `embeddings` - Flat embedding matrix [n, dim] row-major
/// * `n` - Number of embeddings
/// * `dim` - Embedding dimension
///
/// # Returns
///
/// Vector of L2 norms.
pub fn compute_norms(embeddings: &[f32], n: usize, dim: usize) -> Vec<f32> {
    let mut norms = Vec::with_capacity(n);
    for i in 0..n {
        let start = i * dim;
        let sq_sum: f32 = embeddings[start..start + dim]
            .iter()
            .map(|&x| x * x)
            .sum();
        norms.push(sq_sum.sqrt().max(1e-8));
    }
    norms
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_vectors() {
        let u = vec![1.0, 0.0, 0.0];
        let v = vec![1.0, 0.0, 0.0];
        let config = CosineDistanceConfig {
            dim: 3,
            ..Default::default()
        };
        let d = cosine_distance(&u, &v, &config);
        assert!((d - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_orthogonal_vectors() {
        let u = vec![1.0, 0.0];
        let v = vec![0.0, 1.0];
        let config = CosineDistanceConfig {
            dim: 2,
            ..Default::default()
        };
        let d = cosine_distance(&u, &v, &config);
        assert!((d - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_opposite_vectors() {
        let u = vec![1.0, 0.0];
        let v = vec![-1.0, 0.0];
        let config = CosineDistanceConfig {
            dim: 2,
            ..Default::default()
        };
        let d = cosine_distance(&u, &v, &config);
        assert!((d - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_same_direction_half_magnitude() {
        let u = vec![2.0, 0.0];
        let v = vec![1.0, 0.0];
        let config = CosineDistanceConfig {
            dim: 2,
            ..Default::default()
        };
        let d = cosine_distance(&u, &v, &config);
        assert!(d < 1e-6, "same direction = 0 distance");
    }

    #[test]
    fn test_pre_normalized() {
        let u = vec![0.6, 0.8]; // L2 = 1.0
        let v = vec![-0.8, 0.6]; // L2 = 1.0
        let config = CosineDistanceConfig {
            dim: 2,
            skip_norm: true,
            epsilon: 1e-8,
        };
        // dot = 0.6 * -0.8 + 0.8 * 0.6 = -0.48 + 0.48 = 0.0
        // distance = 1 - 0 = 1 (orthogonal)
        let d = cosine_distance(&u, &v, &config);
        assert!((d - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_pairwise_simple() {
        let emb = vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0];
        let dists = pairwise_distances(&emb, 3, 2);
        // 3 agents = 3 pairs
        assert_eq!(dists.len(), 3);
        // (0,1): orthogonal ≈ 1.0
        assert!((dists[0] - 1.0).abs() < 1e-5);
        // (0,2): opposite ≈ 2.0
        assert!((dists[1] - 2.0).abs() < 1e-5);
        // (1,2): orthogonal ≈ 1.0
        assert!((dists[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_zero_vector_edge_case() {
        let u = vec![1.0, 0.0];
        let v = vec![0.0, 0.0]; // zero vector — undefined in standard cosine
        let config = CosineDistanceConfig {
            dim: 2,
            ..Default::default()
        };
        let d = cosine_distance(&u, &v, &config);
        // With epsilon, this should be 1.0 (no similarity)
        assert!((d - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_norms() {
        let embeddings = vec![3.0, 4.0, 0.0, 1.0];
        let norms = compute_norms(&embeddings, 2, 2);
        assert!((norms[0] - 5.0).abs() < 1e-5);
        assert!((norms[1] - 1.0).abs() < 1e-5);
    }
}
