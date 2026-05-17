//! # Matrix — Distance Matrix Batch Computation
//!
//! High-level batch operations for constructing and querying the pairwise
//! distance matrix. Routes to GPU or CPU based on swarm size.

use crate::distance::{self, CosineDistanceConfig};
use crate::ConsensusError;

use rayon::prelude::*;

/// Configuration for matrix computation.
#[derive(Debug, Clone)]
pub struct MatrixConfig {
    /// Use GPU if available and swarm is large enough.
    pub prefer_gpu: bool,
    /// Swarm size threshold for GPU acceleration.
    pub gpu_threshold: usize,
    /// Embedding dimension.
    pub dim: usize,
}

impl Default for MatrixConfig {
    fn default() -> Self {
        Self {
            prefer_gpu: true,
            gpu_threshold: crate::GPU_SWARM_THRESHOLD,
            dim: distance::DEFAULT_EMBEDDING_DIM,
        }
    }
}

/// A pairwise distance matrix for a swarm of agents.
///
/// Stores the full N×N matrix (row-major) for fast lookup.
pub struct DistanceMatrix {
    /// Number of agents.
    pub n_agents: usize,
    /// Embedding dimension.
    pub dim: usize,
    /// Full N×N distance matrix (row-major, f32).
    pub data: Vec<f32>,
    /// Whether this was computed on GPU.
    pub computed_on_gpu: bool,
}

impl DistanceMatrix {
    /// Compute the full distance matrix for a batch of embeddings.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - Flat embedding matrix [n_agents, dim]
    /// * `n_agents` - Number of agents
    /// * `dim` - Embedding dimension
    /// * `config` - Matrix configuration
    pub fn compute(
        embeddings: &[f32],
        n_agents: usize,
        dim: usize,
        config: &MatrixConfig,
    ) -> Result<Self, ConsensusError> {
        // Validate
        if n_agents < 2 {
            return Err(ConsensusError::InsufficientSwarm {
                given: n_agents,
                min: 2,
            });
        }

        if embeddings.len() != n_agents * dim {
            return Err(ConsensusError::DimensionMismatch {
                expected: n_agents * dim,
                got: embeddings.len(),
            });
        }

        // Decide compute path
        let use_gpu = config.prefer_gpu
            && n_agents > config.gpu_threshold
            && crate::is_cuda_available();

        let data = if use_gpu {
            #[cfg(feature = "cuda")]
            {
                Self::compute_gpu(embeddings, n_agents, dim)?
            }
            #[cfg(not(feature = "cuda"))]
            {
                Self::compute_cpu_par(embeddings, n_agents, dim)?
            }
        } else {
            Self::compute_cpu_par(embeddings, n_agents, dim)?
        };

        Ok(Self {
            n_agents,
            dim,
            data,
            computed_on_gpu: use_gpu,
        })
    }

    /// GPU-accelerated matrix computation.
    #[cfg(feature = "cuda")]
    fn compute_gpu(
        embeddings: &[f32],
        n_agents: usize,
        dim: usize,
    ) -> Result<Vec<f32>, ConsensusError> {
        use crate::cuda_runtime::{CudaContext, CudaDistance};

        let ctx = CudaContext::new()
            .map_err(|e| ConsensusError::CudaUnavailable(e.to_string()))?;

        let gpu_emb = ctx
            .upload(embeddings)
            .map_err(|e| ConsensusError::CudaKernelError(e.to_string()))?;

        let gpu_matrix = CudaDistance::cosine_distance_matrix(&ctx, &gpu_emb, n_agents, dim)
            .map_err(|e| ConsensusError::CudaKernelError(e.to_string()))?;

        ctx.download(&gpu_matrix, n_agents * n_agents)
            .map_err(|e| ConsensusError::CudaKernelError(e.to_string()))
    }

    /// CPU parallel matrix computation using rayon.
    fn compute_cpu_par(
        embeddings: &[f32],
        n_agents: usize,
        dim: usize,
    ) -> Result<Vec<f32>, ConsensusError> {
        let config = CosineDistanceConfig {
            dim,
            ..Default::default()
        };

        // Pre-compute norms
        let norms: Vec<f32> = embeddings
            .par_chunks(dim)
            .map(|chunk| {
                let sq: f32 = chunk.iter().map(|&x| x * x).sum();
                sq.sqrt().max(config.epsilon)
            })
            .collect();

        let n = n_agents;
        let mut matrix = vec![0.0f32; n * n];

        // Compute upper triangle in parallel (symmetric)
        matrix
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, row)| {
                let row_i = &embeddings[i * dim..(i + 1) * dim];
                let norm_i = norms[i];

                for j in 0..n {
                    if i == j {
                        row[j] = 0.0;
                    } else if j > i {
                        let row_j = &embeddings[j * dim..(j + 1) * dim];
                        let dot: f32 = row_i.iter().zip(row_j.iter()).map(|(a, b)| a * b).sum();
                        let dist = 1.0 - dot / (norm_i * norms[j]);
                        row[j] = dist;
                    } else {
                        // Skip lower triangle for now; filled after parallel pass
                    }
                }
            });

        // Fill lower triangle (symmetric) — sequential, O(n²/2)
        for i in 0..n {
            for j in 0..i {
                matrix[i * n + j] = matrix[j * n + i];
            }
        }

        Ok(matrix)
    }

    /// Get the distance between two agents.
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds.
    pub fn get(&self, i: usize, j: usize) -> f32 {
        debug_assert!(i < self.n_agents && j < self.n_agents);
        self.data[i * self.n_agents + j]
    }

    /// Get the spread (maximum pairwise distance).
    pub fn spread(&self) -> f32 {
        self.data
            .par_iter()
            .cloned()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0)
    }

    /// Get the minimum pairwise distance.
    pub fn min_distance(&self) -> f32 {
        self.data
            .iter()
            .cloned()
            .filter(|&d| d > 1e-8) // skip diagonal
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0)
    }

    /// Check if spread exceeds threshold.
    pub fn spread_exceeds(&self, threshold: f32) -> bool {
        self.data.iter().any(|&d| d > threshold)
    }

    /// Get agents whose distance to all others is below threshold
    /// (the consensus clique).
    pub fn consensus_clique(&self, threshold: f32) -> Vec<usize> {
        let n = self.n_agents;
        let mut clique: Vec<usize> = (0..n).collect();

        // Iteratively remove agents whose max distance to clique > threshold
        loop {
            let before = clique.len();

            let mut new_clique: Vec<usize> = Vec::with_capacity(clique.len());
            for &i in &clique {
                let mut ok = true;
                for &j in &clique {
                    if i == j { continue; }
                    if self.get(i, j) > threshold {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    new_clique.push(i);
                }
            }

            clique = new_clique;

            if clique.len() == before || clique.len() <= 1 {
                break;
            }
        }

        clique
    }

    /// Get the number of agents within threshold of a given agent.
    pub fn neighbors_within(&self, agent: usize, threshold: f32) -> Vec<usize> {
        let n = self.n_agents;
        let row_start = agent * n;
        (0..n)
            .filter(|&j| j != agent && self.data[row_start + j] <= threshold)
            .collect()
    }

    /// Get a row of the distance matrix.
    pub fn row(&self, i: usize) -> &[f32] {
        let start = i * self.n_agents;
        &self.data[start..start + self.n_agents]
    }

    /// Convert to a full square matrix (already stored this way).
    pub fn to_square(&self) -> Vec<Vec<f32>> {
        let n = self.n_agents;
        (0..n)
            .map(|i| self.data[i * n..(i + 1) * n].to_vec())
            .collect()
    }
}

impl std::fmt::Display for DistanceMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DistanceMatrix({}×{}, spread={:.4}, gpu={})",
            self.n_agents,
            self.n_agents,
            self.spread(),
            self.computed_on_gpu
        )
    }
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
    fn test_matrix_symmetry() {
        let emb = make_embeddings(5, 32, 42);
        let matrix = DistanceMatrix::compute(&emb, 5, 32, &MatrixConfig::default()).unwrap();

        for i in 0..5 {
            for j in 0..5 {
                assert!(
                    (matrix.get(i, j) - matrix.get(j, i)).abs() < 1e-5,
                    "matrix should be symmetric: ({i},{j}) ≠ ({j},{i})"
                );
            }
        }
    }

    #[test]
    fn test_matrix_diagonal_zero() {
        let emb = make_embeddings(5, 32, 42);
        let matrix = DistanceMatrix::compute(&emb, 5, 32, &MatrixConfig::default()).unwrap();

        for i in 0..5 {
            assert!(
                matrix.get(i, i).abs() < 1e-6,
                "diagonal should be 0"
            );
        }
    }

    #[test]
    fn test_matrix_spread() {
        // Two opposite vectors should give spread ≈ 2.0
        let dim = 64;
        let mut emb = vec![0.0f32; 2 * dim];
        emb[0] = 1.0;
        emb[dim] = -1.0;

        let matrix = DistanceMatrix::compute(&emb, 2, dim, &MatrixConfig::default()).unwrap();
        assert!((matrix.spread() - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_consensus_clique() {
        // Use identical embeddings so all distances are 0
        let dim = 4;
        let mut emb = vec![0.0f32; 3 * dim];
        for i in 0..3 { emb[i * dim] = 1.0; }
        let matrix = DistanceMatrix::compute(&emb, 3, dim, &MatrixConfig::default()).unwrap();
        let clique = matrix.consensus_clique(0.5);
        assert!(!clique.is_empty(), "identical embeddings should form full clique");
        assert_eq!(clique.len(), 3);
    }

    #[test]
    fn test_consensus_clique_identical() {
        let dim = 4;
        let mut emb = vec![0.0f32; 4 * dim];
        // All agents = [1, 0, 0, 0]
        for i in 0..4 {
            emb[i * dim] = 1.0;
        }
        let matrix = DistanceMatrix::compute(&emb, 4, dim, &MatrixConfig::default()).unwrap();
        let clique = matrix.consensus_clique(0.1);
        assert_eq!(clique.len(), 4, "all identical → clique = all agents");
    }

    #[test]
    fn test_spread_exceeds() {
        let emb = make_embeddings(5, 16, 7);
        let matrix = DistanceMatrix::compute(&emb, 5, 16, &MatrixConfig::default()).unwrap();
        let spread = matrix.spread();
        assert!(matrix.spread_exceeds(spread - 0.01));
        assert!(!matrix.spread_exceeds(spread + 0.01));
    }

    #[test]
    fn test_matrix_display() {
        let emb = make_embeddings(3, 8, 0);
        let matrix = DistanceMatrix::compute(&emb, 3, 8, &MatrixConfig::default()).unwrap();
        let display = format!("{matrix}");
        assert!(display.contains("DistanceMatrix"));
        assert!(display.contains("3×3"));
    }
}
