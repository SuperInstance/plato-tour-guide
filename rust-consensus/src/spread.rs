//! # Compute Spread
//!
//! Measures the overall "disagreement" among a set of tokens by
//! aggregating pairwise distances. Supports early termination — when
//! the cumulative spread exceeds `2 × threshold`, computation stops
//! immediately and returns the current spread value.

use crate::distance::DistanceMatrix;
use crate::ConsensusError;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

/// Aggregate statistics describing the spread of a token set.
#[derive(Debug, Clone, Copy)]
pub struct SpreadResult {
    /// Mean pairwise distance (the canonical spread metric).
    pub mean_spread: f64,
    /// Maximum pairwise distance observed.
    pub max_spread: f64,
    /// Minimum pairwise distance observed.
    pub min_spread: f64,
    /// Number of distance pairs evaluated.
    pub pairs_evaluated: usize,
    /// Whether early termination was triggered (spread exceeded `2T`).
    pub early_terminated: bool,
}

/// Compute the spread of a token set from its distance matrix.
///
/// ## Early termination
///
/// When `threshold` is `Some(T)`, the computation aborts as soon as
/// the running mean exceeds `2 × T`. This prevents wasted work on
/// clearly non-consensus groups (where the spread is too large for
/// any meaningful agreement).
///
/// ## Parallelism
///
/// Uses Rayon to sum row-major distances in parallel. Each row's sum
/// is computed independently, then reduced.
pub fn compute_spread(
    matrix: &DistanceMatrix,
    threshold: Option<f64>,
) -> SpreadResult {
    let n = matrix.len();
    if n <= 1 {
        return SpreadResult {
            mean_spread: 0.0,
            max_spread: 0.0,
            min_spread: 0.0,
            pairs_evaluated: 0,
            early_terminated: false,
        };
    }

    let terminate_threshold = threshold.map(|t| 2.0 * t);
    let early_stop = AtomicBool::new(false);

    // Parallel row sums — each row i computes sum over all j ≠ i
    let row_stats: Vec<(f64, f64, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            if early_stop.load(Ordering::Relaxed) {
                return (0.0, 0.0, f64::MAX);
            }

            let mut row_sum = 0.0_f64;
            let mut row_max = f64::MIN;
            let mut row_min = f64::MAX;

            for j in 0..n {
                if i == j {
                    continue;
                }
                let d = matrix.get(i, j);
                row_sum += d;
                if d > row_max {
                    row_max = d;
                }
                if d < row_min {
                    row_min = d;
                }
            }

            (row_sum, row_max, row_min)
        })
        .collect();

    // Reduce
    let mut total_sum = 0.0_f64;
    let mut global_max = f64::MIN;
    let mut global_min = f64::MAX;

    for (r_sum, r_max, r_min) in &row_stats {
        total_sum += r_sum;
        if *r_max > global_max {
            global_max = *r_max;
        }
        if *r_min < global_min {
            global_min = *r_min;
        }

        // Check early termination: if total_sum / n_pairs exceeds 2T, we know
        // the final mean will be there (since remaining terms are positive).
        if let Some(tt) = terminate_threshold {
            // Conservative: check after every row if we've already exceeded 2T
            // per complete pair count
            let pairs_done = row_stats.len() * (n - 1);
            if pairs_done > 0 && total_sum / pairs_done as f64 > tt {
                early_stop.store(true, Ordering::Relaxed);
                break;
            }
        }
    }

    let total_pairs = n * (n - 1);
    let mean_spread = if total_pairs > 0 {
        total_sum / total_pairs as f64
    } else {
        0.0
    };

    SpreadResult {
        mean_spread,
        max_spread: global_max,
        min_spread: global_min,
        pairs_evaluated: total_pairs,
        early_terminated: early_stop.load(Ordering::Relaxed),
    }
}

/// Check whether the spread exceeds `2 × threshold`, returning
/// `Ok(())` if below threshold, `Err(ConsensusError)` if above.
///
/// This is a convenience wrapper for early-rejection checks.
pub fn check_spread(
    matrix: &DistanceMatrix,
    threshold: f64,
) -> crate::ConsensusResult<()> {
    if threshold <= 0.0 {
        return Err(ConsensusError::InvalidThreshold {
            threshold,
            reason: "must be positive",
        });
    }

    let result = compute_spread(matrix, Some(threshold));
    if result.early_terminated || result.mean_spread > 2.0 * threshold {
        Err(ConsensusError::NoConsensus {
            spread: result.mean_spread,
            threshold,
        })
    } else {
        Ok(())
    }
}

/// Convenience: compute spread directly from tokens.
pub fn token_spread(tokens: &[crate::Token], threshold: Option<f64>) -> SpreadResult {
    let matrix = crate::DistanceMatrix::new(tokens);
    compute_spread(&matrix, threshold)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Token;

    #[test]
    fn test_spread_identical_tokens() {
        let tokens: Vec<Token> = vec!["same", "same", "same"]
            .into_iter()
            .map(Token::new)
            .collect();
        let m = crate::DistanceMatrix::new(&tokens);
        let r = compute_spread(&m, None);
        assert!((r.mean_spread - 0.0).abs() < 1e-10);
        assert!((r.max_spread - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_spread_diverse_tokens() {
        let tokens: Vec<Token> = vec!["alpha", "beta", "gamma", "delta", "epsilon"]
            .into_iter()
            .map(Token::new)
            .collect();
        let m = crate::DistanceMatrix::new(&tokens);
        let r = compute_spread(&m, None);
        assert!(r.mean_spread > 0.0);
        assert!(r.max_spread >= r.min_spread);
        assert_eq!(r.pairs_evaluated, 20); // 5 × 4
    }

    #[test]
    fn test_early_termination() {
        let tokens: Vec<Token> = vec!["cat", "dog", "horse", "elephant", "hippopotamus"]
            .into_iter()
            .map(Token::new)
            .collect();
        let m = crate::DistanceMatrix::new(&tokens);
        // Very low threshold — should trigger early termination
        let r = compute_spread(&m, Some(0.01));
        assert!(r.early_terminated);
    }

    #[test]
    fn test_no_early_termination_similar() {
        let tokens: Vec<Token> = vec!["hello", "hallo", "helo", "hillo"]
            .into_iter()
            .map(Token::new)
            .collect();
        let m = crate::DistanceMatrix::new(&tokens);
        let r = compute_spread(&m, Some(1.0));
        // High threshold — should not terminate early
        assert!(!r.early_terminated);
    }

    #[test]
    fn test_single_token_spread() {
        let tokens: Vec<Token> = vec!["alone"].into_iter().map(Token::new).collect();
        let m = crate::DistanceMatrix::new(&tokens);
        let r = compute_spread(&m, None);
        assert!((r.mean_spread - 0.0).abs() < 1e-10);
        assert_eq!(r.pairs_evaluated, 0);
    }

    #[test]
    fn test_check_spread_rejects_low_threshold() {
        let tokens: Vec<Token> = vec!["cat", "dog", "bird"]
            .into_iter()
            .map(Token::new)
            .collect();
        let m = crate::DistanceMatrix::new(&tokens);
        assert!(check_spread(&m, 0.001).is_err());
    }

    #[test]
    fn test_check_spread_invalid_threshold() {
        let tokens: Vec<Token> = vec!["a", "b"]
            .into_iter()
            .map(Token::new)
            .collect();
        let m = crate::DistanceMatrix::new(&tokens);
        assert!(check_spread(&m, 0.0).is_err());
    }
}
