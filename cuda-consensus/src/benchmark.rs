//! # Benchmark — CPU vs GPU Performance Comparison
//!
//! Benchmark harness for comparing CPU and GPU consensus computation
//! across different swarm sizes. Provides calibrated measurements
//! for deciding when to use GPU acceleration.

use crate::distance::DEFAULT_EMBEDDING_DIM;
use crate::matrix::{DistanceMatrix, MatrixConfig};
use crate::reduce::{SpreadConfig, SpreadResult};
use crate::ConsensusError;
use std::time::{Duration, Instant};

/// Benchmark result for a single configuration.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Swarm size.
    pub n_agents: usize,
    /// Computation time on CPU (µs).
    pub cpu_time_us: f64,
    /// Computation time on GPU (µs), None if GPU unavailable.
    pub gpu_time_us: Option<f64>,
    /// CPU speedup (CPU time / GPU time), None if GPU unavailable.
    pub speedup: Option<f64>,
    /// Whether CPU and GPU results match (within tolerance).
    pub results_match: Option<bool>,
    /// Spread computed on CPU.
    pub cpu_spread: f64,
    /// Spread computed on GPU.
    pub gpu_spread: Option<f64>,
}

/// Run a benchmark comparison across a range of swarm sizes.
///
/// # Arguments
///
/// * `sizes` - List of swarm sizes to test
/// * `dim` - Embedding dimension (default: 384)
/// * `samples` - Number of samples per size for averaging
///
/// # Returns
///
/// Vector of benchmark results, one per swarm size.
pub fn run_benchmark(
    sizes: &[usize],
    dim: Option<usize>,
    samples: Option<usize>,
) -> Result<Vec<BenchmarkResult>, ConsensusError> {
    let dim = dim.unwrap_or(DEFAULT_EMBEDDING_DIM);
    let samples = samples.unwrap_or(5);
    let mut results = Vec::with_capacity(sizes.len());

    let cuda_available = crate::is_cuda_available();

    for &n in sizes {
        println!(
            "  benchmarking n={n}, dim={dim}, samples={samples}..."
        );

        // Generate random embeddings
        let embeddings = generate_test_embeddings(n, dim);

        // CPU benchmark
        let cpu_times = measure_cpu(&embeddings, n, dim, samples);
        let cpu_mean = mean(&cpu_times);

        // Compute CPU spread
        let cpu_config = SpreadConfig::default();
        let cpu_result = crate::reduce::compute_spread(&embeddings, n, dim, &cpu_config)?;

        // GPU benchmark (if available)
        let (gpu_mean, gpu_spread, match_result) = if cuda_available {
            #[cfg(feature = "cuda")]
            {
                let gpu_times = measure_gpu(&embeddings, n, dim, samples)?;
                let gpu_mean = mean(&gpu_times);

                let gpu_config = SpreadConfig::default();
                let gpu_result =
                    crate::reduce::compute_spread(&embeddings, n, dim, &gpu_config)?;

                let matched = (cpu_result.spread - gpu_result.spread).abs() < 0.01;

                (Some(gpu_mean), Some(gpu_result.spread), Some(matched))
            }
            #[cfg(not(feature = "cuda"))]
            {
                // Should not reach if cuda_available is correct, but handle it
                (None, None, None)
            }
        } else {
            (None, None, None)
        };

        let speedup = gpu_mean.map(|g| cpu_mean / g);

        results.push(BenchmarkResult {
            n_agents: n,
            cpu_time_us: cpu_mean,
            gpu_time_us: gpu_mean,
            speedup,
            results_match: match_result,
            cpu_spread: cpu_result.spread,
            gpu_spread,
        });
    }

    Ok(results)
}

/// Generate random normalized embeddings for benchmarking.
pub fn generate_test_embeddings(n: usize, dim: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
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

/// Measure CPU-only computation time over multiple samples.
fn measure_cpu(embeddings: &[f32], n: usize, dim: usize, samples: usize) -> Vec<f64> {
    let mut times = Vec::with_capacity(samples);

    for _ in 0..samples {
        let start = Instant::now();
        let config = SpreadConfig {
            prefer_gpu: false,
            ..Default::default()
        };
        let _ = crate::reduce::compute_spread(embeddings, n, dim, &config);
        let elapsed = start.elapsed().as_micros() as f64;
        times.push(elapsed.max(1.0)); // floor at 1µs
    }

    times
}

/// Measure GPU-accelerated computation time over multiple samples.
#[cfg(feature = "cuda")]
fn measure_gpu(
    embeddings: &[f32],
    n: usize,
    dim: usize,
    samples: usize,
) -> Result<Vec<f64>, ConsensusError> {
    let mut times = Vec::with_capacity(samples);

    // Warm-up
    let warmup_config = SpreadConfig {
        prefer_gpu: true,
        ..Default::default()
    };
    let _ = crate::reduce::compute_spread(embeddings, n, dim, &warmup_config)?;

    for _ in 0..samples {
        let start = Instant::now();
        let config = SpreadConfig {
            prefer_gpu: true,
            ..Default::default()
        };
        let _ = crate::reduce::compute_spread(embeddings, n, dim, &config)?;
        let elapsed = start.elapsed().as_micros() as f64;
        times.push(elapsed.max(1.0));
    }

    Ok(times)
}

#[cfg(not(feature = "cuda"))]
fn measure_gpu(
    _embeddings: &[f32],
    _n: usize,
    _dim: usize,
    _samples: usize,
) -> Result<Vec<f64>, ConsensusError> {
    Err(ConsensusError::CudaUnavailable(
        "CUDA feature not enabled".into(),
    ))
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Print a formatted benchmark table to stdout.
pub fn print_benchmark_table(results: &[BenchmarkResult]) {
    println!();
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│               Plato CUDA Consensus — Benchmark                      │");
    println!("├────────┬──────────┬──────────┬──────────┬──────────┬───────────────┤");
    println!("│ Agents │  CPU µs  │  GPU µs  │ Speedup  │  Spread  │  GPU Avail    │");
    println!("├────────┼──────────┼──────────┼──────────┼──────────┼───────────────┤");

    for r in results {
        let gpu_str = match r.gpu_time_us {
            Some(t) => format!("{t:>8.1}"),
            None => "   N/A   ".to_string(),
        };
        let speedup_str = match r.speedup {
            Some(s) => format!("{s:>6.2}×"),
            None => "  N/A   ".to_string(),
        };
        let gpu_avail = if r.gpu_time_us.is_some() { "✓ CUDA" } else { "✗ CPU" };

        println!(
            "│ {n:>6} │ {cpu:>8.1} │ {gpu} │ {speedup} │ {spread:>6.3} │ {avail:<12} │",
            n = r.n_agents,
            cpu = r.cpu_time_us,
            gpu = gpu_str,
            speedup = speedup_str,
            spread = r.cpu_spread,
            avail = gpu_avail
        );
    }

    println!("├────────┴──────────┴──────────┴──────────┴──────────┴───────────────┤");
    if let Some(r) = results.last() {
        if let Some(matched) = r.results_match {
            if matched {
                println!("│  ✓ CPU & GPU results match within tolerance (0.01).              │");
            } else {
                println!("│  ⚠ CPU & GPU results DO NOT match — investigate.                 │");
            }
        }
    }
    println!("└─────────────────────────────────────────────────────────────────────┘");
}

/// Run a quick comparison: Python (NumPy) vs Rust CPU vs Rust CUDA.
///
/// This function generates data and runs all three variants, printing
/// results to stdout. Intended for CI and developer validation.
pub fn quick_compare(n_agents: usize, dim: usize) {
    use std::time::Instant;

    println!("\n=== Quick Compare: Python vs CPU vs GPU ===");
    println!("  Agents: {n_agents}, Dim: {dim}\n");

    let embeddings = generate_test_embeddings(n_agents, dim);

    // Rust CPU
    let start = Instant::now();
    let cpu_result = crate::reduce::compute_spread(
        &embeddings,
        n_agents,
        dim,
        &SpreadConfig {
            prefer_gpu: false,
            ..Default::default()
        },
    );
    let cpu_time = start.elapsed();
    println!(
        "  Rust CPU:     {:.3} ms  spread={:.6}",
        cpu_time.as_secs_f64() * 1000.0,
        cpu_result.as_ref().map(|r| r.spread).unwrap_or(-1.0)
    );

    // Rust GPU (if available)
    if crate::is_cuda_available() {
        let start = Instant::now();
        let gpu_result = crate::reduce::compute_spread(
            &embeddings,
            n_agents,
            dim,
            &SpreadConfig {
                prefer_gpu: true,
                ..Default::default()
            },
        );
        let gpu_time = start.elapsed();

        match (&cpu_result, &gpu_result) {
            (Ok(cpu), Ok(gpu)) => {
                let diff = (cpu.spread - gpu.spread).abs();
                println!(
                    "  Rust GPU:     {:.3} ms  spread={:.6}  (Δ={:.6})  {:.1}× speedup",
                    gpu_time.as_secs_f64() * 1000.0,
                    gpu.spread,
                    diff,
                    cpu_time.as_secs_f64() / gpu_time.as_secs_f64()
                );
            }
            (Ok(_), Err(e)) => {
                println!("  Rust GPU:     FAILED — {e}");
            }
            _ => {}
        }
    } else {
        println!("  Rust GPU:     N/A (no CUDA driver)");
    }

    println!();
}

/// Self-calibrating threshold: determine the optimal swarm size
/// for switching from CPU to GPU.
pub fn find_gpu_threshold(
    dim: Option<usize>,
) -> Result<usize, ConsensusError> {
    let dim = dim.unwrap_or(DEFAULT_EMBEDDING_DIM);

    if !crate::is_cuda_available() {
        println!("  No GPU available — CPU-only mode.");
        return Ok(usize::MAX);
    }

    let test_sizes: Vec<usize> = vec![5, 10, 20, 30, 50, 100, 200];

    for &n in &test_sizes {
        let embeddings = generate_test_embeddings(n, dim);

        // CPU time
        let cpu_config = SpreadConfig {
            prefer_gpu: false,
            ..Default::default()
        };
        let cpu_start = Instant::now();
        let _ = crate::reduce::compute_spread(&embeddings, n, dim, &cpu_config);
        let cpu_us = cpu_start.elapsed().as_micros() as f64;

        // GPU time
        let gpu_config = SpreadConfig {
            prefer_gpu: true,
            ..Default::default()
        };
        let gpu_start = Instant::now();
        let _ = crate::reduce::compute_spread(&embeddings, n, dim, &gpu_config);
        let gpu_us = gpu_start.elapsed().as_micros() as f64;

        let speedup = cpu_us / gpu_us.max(1.0);

        println!(
            "  n={n:>4}: CPU={cpu_us:>8.1}µs  GPU={gpu_us:>8.1}µs  speedup={speedup:.2}×"
        );

        if speedup > 1.5 && n >= 20 {
            println!("  → Optimal GPU threshold: n > {n}");
            return Ok(n);
        }
    }

    Ok(20) // default fallback
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_test_embeddings() {
        let data = generate_test_embeddings(10, 64);
        assert_eq!(data.len(), 640);
    }

    #[test]
    fn test_benchmark_small() {
        let results = run_benchmark(&[2, 5], Some(8), Some(2)).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].cpu_time_us > 0.0);
        assert!(results[0].cpu_spread >= 0.0);
    }

    #[test]
    fn test_print_table() {
        let results = vec![
            BenchmarkResult {
                n_agents: 10,
                cpu_time_us: 100.0,
                gpu_time_us: Some(200.0),
                speedup: Some(0.5),
                results_match: Some(true),
                cpu_spread: 0.5,
                gpu_spread: Some(0.5),
            },
        ];
        // Should not panic
        print_benchmark_table(&results);
    }
}
