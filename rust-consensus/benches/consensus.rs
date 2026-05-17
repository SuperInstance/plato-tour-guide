//! # Criterion Benchmarks for Plato Consensus
//!
//! Benchmarks measure throughput across the core pipeline stages:
//!   1. Semantic distance (single pair)
//!   2. Batch distance matrix construction
//!   3. Spread computation (with/without threshold)
//!   4. Maximal clique finding
//!   5. Full consensus snap (all strategies)
//!
//! Run: `cargo bench` in the crate directory.
//! Results appear in `target/criterion/report/index.html`.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

use plato_consensus::{
    consensus_snap, compute_spread, find_maximal_cliques, DistanceMatrix, SnapStrategy, Token,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a set of `n` similar tokens for benchmarking.
fn similar_tokens(n: usize) -> Vec<String> {
    let base = "hello";
    let variants = [
        "hallo", "helo", "hillo", "heaven", "heavy", "help", "held", "helm",
        "hello!", "Hello", "HELLO", "hailo", "hellp", "helow", "hllo", "heloo",
    ];
    (0..n)
        .map(|i| {
            let suffix = variants[i % variants.len()];
            format!("{}{}", base, suffix)
        })
        .collect()
}

/// Build a set of `n` diverse tokens.
fn diverse_tokens(n: usize) -> Vec<String> {
    let words = [
        "cat", "astrophysics", "quantum", "zebra", "xylophone", "bicycle",
        "elephant", "galaxy", "hurricane", "igloo", "jazz", "kangaroo",
        "lighthouse", "mountain", "nebula", "octopus", "piano", "quartz",
        "rainbow", "sunset", "tornado", "umbrella", "volcano", "waterfall",
    ];
    (0..n).map(|i| words[i % words.len()].to_string()).collect()
}

fn token_objs(strings: &[String]) -> Vec<Token> {
    strings.iter().map(|s| Token::new(s)).collect()
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_semantic_distance(c: &mut Criterion) {
    let a = Token::new("hello_world_long_string");
    let b = Token::new("hallo_world_long_string");

    c.bench_function("semantic_distance/pair", |bencher| {
        bencher.iter(|| {
            black_box(a.distance_to(&b));
        });
    });

    let a_unicode = Token::new("¹²³⁴⁵⁶⁷⁸⁹⁰");
    let b_unicode = Token::new("1234567890");

    c.bench_function("semantic_distance/unicode_superscript", |bencher| {
        bencher.iter(|| {
            black_box(a_unicode.distance_to(&b_unicode));
        });
    });
}

fn bench_distance_matrix(c: &mut Criterion) {
    let sizes = [16usize, 32, 64, 128];

    let mut group = c.benchmark_group("distance_matrix");
    group.sample_size(10);

    for &n in &sizes {
        let tokens = token_objs(&similar_tokens(n));
        group.bench_with_input(BenchmarkId::new("similar", n), &tokens, |b, t| {
            b.iter(|| {
                black_box(DistanceMatrix::new(t));
            });
        });

        let tokens = token_objs(&diverse_tokens(n));
        group.bench_with_input(BenchmarkId::new("diverse", n), &tokens, |b, t| {
            b.iter(|| {
                black_box(DistanceMatrix::new(t));
            });
        });
    }
    group.finish();
}

fn bench_spread(c: &mut Criterion) {
    let sizes = [8usize, 16, 32, 64];

    let mut group = c.benchmark_group("spread");
    group.sample_size(10);

    for &n in &sizes {
        let tokens = token_objs(&similar_tokens(n));
        let matrix = DistanceMatrix::new(&tokens);

        group.bench_with_input(BenchmarkId::new("no_threshold", n), &matrix, |b, m| {
            b.iter(|| {
                black_box(compute_spread(m, None));
            });
        });

        group.bench_with_input(BenchmarkId::new("with_threshold", n), &matrix, |b, m| {
            b.iter(|| {
                black_box(compute_spread(m, Some(0.3)));
            });
        });
    }
    group.finish();
}

fn bench_clique(c: &mut Criterion) {
    let sizes = [8usize, 16, 32, 64];

    let mut group = c.benchmark_group("clique");
    group.sample_size(10);

    for &n in &sizes {
        let tokens = token_objs(&similar_tokens(n));
        let matrix = DistanceMatrix::new(&tokens);

        group.bench_with_input(BenchmarkId::new("threshold_0.2", n), &matrix, |b, m| {
            b.iter(|| {
                black_box(find_maximal_cliques(m, 0.2, 2, None));
            });
        });

        group.bench_with_input(BenchmarkId::new("threshold_0.5", n), &matrix, |b, m| {
            b.iter(|| {
                black_box(find_maximal_cliques(m, 0.5, 2, None));
            });
        });
    }
    group.finish();
}

fn bench_full_consensus(c: &mut Criterion) {
    let sizes = [8usize, 16, 32];

    let mut group = c.benchmark_group("consensus_snap");
    group.sample_size(10);

    for &n in &sizes {
        let tokens = similar_tokens(n);

        group.bench_with_input(
            BenchmarkId::new("medoid", n),
            &tokens,
            |b, t| {
                b.iter(|| {
                    black_box(
                        consensus_snap(t, 0.3, SnapStrategy::Medoid, 2, false).ok(),
                    );
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("weighted", n),
            &tokens,
            |b, t| {
                b.iter(|| {
                    black_box(
                        consensus_snap(t, 0.3, SnapStrategy::Weighted, 2, false).ok(),
                    );
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("mean", n),
            &tokens,
            |b, t| {
                b.iter(|| {
                    black_box(
                        consensus_snap(t, 0.3, SnapStrategy::Mean, 2, false).ok(),
                    );
                });
            },
        );
    }
    group.finish();
}

fn bench_pre_tokenized(c: &mut Criterion) {
    // Simulate a bulk batch from an upstream Python pipeline
    let tokens = diverse_tokens(64);

    c.bench_function("pre_tokenized/batch_64", |bencher| {
        bencher.iter(|| {
            let _toks: Vec<Token> = tokens.iter().map(|s| Token::new(s)).collect();
            black_box(_toks.len());
        });
    });

    // Pre-normalized path
    c.bench_function("pre_tokenized/from_normalized_64", |bencher| {
        bencher.iter(|| {
            let _toks: Vec<Token> = tokens
                .iter()
                .map(|s| Token::from_normalized(s, s))
                .collect();
            black_box(_toks.len());
        });
    });
}

// ---------------------------------------------------------------------------
// Group and run
// ---------------------------------------------------------------------------

criterion_group! {
    name = consensus;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(3))
        .significance_level(0.05)
        .noise_threshold(0.05);
    targets =
        bench_semantic_distance,
        bench_distance_matrix,
        bench_spread,
        bench_clique,
        bench_full_consensus,
        bench_pre_tokenized,
}

criterion_main!(consensus);
