//! # Bench Compare — Python vs NumPy vs CPU vs GPU
//!
//! Criterion benchmark comparing consensus computation across:
//! - Rust CPU (single-threaded)
//! - Rust CPU (rayon parallel)
//! - Rust GPU (CUDA, if available)
//! - Python/NumPy (via subprocess, if Python available)
//!
//! Run with:
//! ```bash
//! cargo bench --bench bench_compare
//! ```

use criterion::{criterion_group, criterion_main, Criterion};
use plato_cuda_consensus::{
    benchmark::{self, run_benchmark},
    distance::DEFAULT_EMBEDDING_DIM,
};

/// Generate embeddings once, benchmark different sizes.
fn bench_sizes() -> Vec<usize> {
    vec![2, 5, 10, 20, 50, 100, 200, 500]
}

/// Benchmark: CPU-only consensus across multiple swarm sizes.
fn bench_cpu_consensus(c: &mut Criterion) {
    let mut group = c.benchmark_group("consensus_cpu");
    group.sample_size(10);

    for &n in &bench_sizes() {
        let dim = std::cmp::min(DEFAULT_EMBEDDING_DIM, 128); // keep benchmark fast
        let embeddings = benchmark::generate_test_embeddings(n, dim);

        group.bench_with_input(
            criterion::BenchmarkId::new("n", n),
            &(embeddings, n, dim),
            |b, (emb, n, dim)| {
                b.iter(|| {
                    let config = plato_cuda_consensus::reduce::SpreadConfig {
                        prefer_gpu: false,
                        ..Default::default()
                    };
                    criterion::black_box(
                        plato_cuda_consensus::reduce::compute_spread(
                            emb, *n, *dim, &config,
                        ),
                    )
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: GPU-accelerated consensus (only if CUDA available).
fn bench_gpu_consensus(c: &mut Criterion) {
    if !plato_cuda_consensus::is_cuda_available() {
        eprintln!("[bench] CUDA not available — skipping GPU benchmarks");
        return;
    }

    let mut group = c.benchmark_group("consensus_gpu");
    group.sample_size(10);

    for &n in &bench_sizes() {
        let dim = std::cmp::min(DEFAULT_EMBEDDING_DIM, 128);
        let embeddings = benchmark::generate_test_embeddings(n, dim);

        group.bench_with_input(
            criterion::BenchmarkId::new("n", n),
            &(embeddings, n, dim),
            |b, (emb, n, dim)| {
                b.iter(|| {
                    let config = plato_cuda_consensus::reduce::SpreadConfig {
                        prefer_gpu: true,
                        ..Default::default()
                    };
                    criterion::black_box(
                        plato_cuda_consensus::reduce::compute_spread(
                            emb, *n, *dim, &config,
                        ),
                    )
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: Distance matrix computation (CPU).
fn bench_distance_matrix_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_matrix_cpu");
    group.sample_size(10);

    for &n in &[10, 20, 50, 100] {
        let dim = 128;
        let embeddings = benchmark::generate_test_embeddings(n, dim);

        group.bench_with_input(
            criterion::BenchmarkId::new("n", n),
            &(embeddings, n, dim),
            |b, (emb, n, dim)| {
                b.iter(|| {
                    let config = plato_cuda_consensus::matrix::MatrixConfig {
                        prefer_gpu: false,
                        ..Default::default()
                    };
                    criterion::black_box(
                        plato_cuda_consensus::matrix::DistanceMatrix::compute(
                            emb, *n, *dim, &config,
                        ),
                    )
                })
            },
        );
    }

    group.finish();
}

/// Quick comparison: print a table comparing implementations.
fn bench_compare_table(_c: &mut Criterion) {
    println!("\n=== Quick Comparison Table ===\n");

    let dim = 128;
    let sizes = vec![10, 50, 200];

    match run_benchmark(&sizes, Some(dim), Some(3)) {
        Ok(results) => {
            benchmark::print_benchmark_table(&results);
        }
        Err(e) => {
            eprintln!("Benchmark failed: {e}");
        }
    }
}

/// Run Python/NumPy comparison via subprocess (optional).
fn bench_python_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("consensus_python");
    group.sample_size(5);

    for &n in &[10, 50, 100] {
        let dim = 128;

        group.bench_with_input(
            criterion::BenchmarkId::new("numpy_n", n),
            &(n, dim),
            |b, &(n, dim)| {
                b.iter(|| {
                    // Call the Python binding script
                    let output = std::process::Command::new("python3")
                        .args([
                            "-c",
                            &format!(
                                r#"
import numpy as np
np.random.seed(42)
emb = np.random.randn({n}, {dim}).astype(np.float32)
emb /= np.linalg.norm(emb, axis=1, keepdims=True)
dists = 1.0 - emb @ emb.T
spread = float(dists.max())
print(spread)
"#
                            ),
                        ])
                        .output();

                    match output {
                        Ok(out) => {
                            let _spread: f64 = String::from_utf8_lossy(&out.stdout)
                                .trim()
                                .parse()
                                .unwrap_or(0.0);
                            criterion::black_box(_spread);
                        }
                        Err(e) => {
                            eprintln!("Python subprocess failed: {e}");
                        }
                    }
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(2));
    targets = bench_cpu_consensus, bench_gpu_consensus, bench_distance_matrix_cpu,
              bench_compare_table, bench_python_comparison
);

criterion_main!(benches);
