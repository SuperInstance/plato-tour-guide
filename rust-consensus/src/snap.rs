//! # Consensus Snap
//!
//! The final stage: from a set of maximal cliques, produce a single
//! consensus "snap" token that best represents the group.
//!
//! ## Snap Strategies
//!
//! * **Mean** — chooses the token closest to the centroid of its clique.
//! * **Medoid** — chooses the token with minimum sum-of-distances to
//!   all other clique members (the graph-medoid).
//! * **Weighted** — same as medoid but weighted by coherence (tighter
//!   cliques have more influence).

use crate::clique::{find_maximal_cliques, CliqueResult, MaximalClique};
use crate::distance::{DistanceMatrix, Token};
use crate::spread::compute_spread;
use crate::ConsensusError;

/// Strategy for selecting the consensus token from a clique.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapStrategy {
    /// Pick the token closest to the centroid of the clique.
    Mean,
    /// Pick the token that minimises total distance to other clique members.
    Medoid,
    /// Pick the medoid, but weight by clique coherence.
    Weighted,
}

/// Detailed decision info returned with the consensus result.
#[derive(Debug, Clone)]
pub struct SnapDecision {
    /// The snap strategy used.
    pub strategy: SnapStrategy,
    /// Threshold T used.
    pub threshold: f64,
    /// Number of cliques evaluated.
    pub cliques_evaluated: usize,
    /// Size of the clique that produced the consensus.
    pub clique_size: usize,
    /// Coherence of the chosen clique (max intra-clique distance).
    pub clique_coherence: f64,
    /// Spread of the full token set (before clique finding).
    pub full_spread: f64,
    /// Whether the spread check was skipped (e.g., low token count).
    pub spread_skipped: bool,
    /// Mean distance from chosen token to its clique members.
    pub mean_distance_to_consensus: f64,
    /// Duration of computation (wall-clock, seconds, if measured).
    pub computation_time_s: Option<f64>,
}

/// The result of a consensus snap.
#[derive(Debug, Clone)]
pub struct SnapResult {
    /// The consensus token.
    pub token: String,
    /// Index of the consensus token in the original input array.
    pub token_index: usize,
    /// Tokens in the consensus clique (by original index).
    pub clique_members: Vec<usize>,
    /// Decision metadata.
    pub decision: SnapDecision,
}

/// Run a full consensus snap: compute spread, find cliques, pick consensus.
///
/// ## Parameters
///
/// * `tokens` - Input tokens (raw strings).
/// * `threshold` - The distance threshold T. Pairs closer than T are "near."
/// * `strategy` - Which snap strategy to use.
/// * `min_clique_size` - Minimum members for a clique to be considered.
/// * `skip_spread_check` - If true, skip the preliminary spread check.
///
/// ## Returns
///
/// A `SnapResult` with the consensus token and decision metadata, or
/// a `ConsensusError` if no consensus could be reached.
pub fn consensus_snap<T: AsRef<str> + std::fmt::Debug>(
    tokens: &[T],
    threshold: f64,
    strategy: SnapStrategy,
    min_clique_size: usize,
    skip_spread_check: bool,
) -> crate::ConsensusResult<SnapResult> {
    let start = std::time::Instant::now();
    let n = tokens.len();

    // Validate input
    if n < 2 {
        return Err(ConsensusError::InsufficientTokens {
            given: n,
            min: 2,
        });
    }
    if threshold <= 0.0 || threshold > 1.0 {
        return Err(ConsensusError::InvalidThreshold {
            threshold,
            reason: "must be in (0.0, 1.0]",
        });
    }

    // Build token list
    let token_objs: Vec<Token> = tokens
        .iter()
        .map(|s| Token::new(s.as_ref()))
        .collect();

    // Distance matrix
    let matrix = DistanceMatrix::new(&token_objs);

    // Spread check (early rejection)
    let (full_spread, spread_skipped) = if skip_spread_check {
        (0.0, true)
    } else {
        let spread_result = compute_spread(&matrix, Some(threshold));
        (spread_result.mean_spread, false)
    };

    if !skip_spread_check && full_spread > 2.0 * threshold {
        return Err(ConsensusError::NoConsensus {
            spread: full_spread,
            threshold,
        });
    }

    // Find maximal cliques
    let clique_result = find_maximal_cliques(
        &matrix,
        threshold,
        min_clique_size,
        None, // no pruning by default; caller can pre-filter tokens
    );

    if clique_result.cliques.is_empty() {
        return Err(ConsensusError::NoConsensus {
            spread: full_spread,
            threshold,
        });
    }

    // Pick consensus token from the best clique(s)
    let (chosen_token, chosen_index, chosen_clique) = match strategy {
        SnapStrategy::Mean => select_mean(&token_objs, &matrix, &clique_result),
        SnapStrategy::Medoid => select_medoid(&token_objs, &matrix, &clique_result),
        SnapStrategy::Weighted => select_weighted(&token_objs, &matrix, &clique_result),
    };

    // Compute mean distance from chosen token to clique members
    let mean_distance = compute_mean_distance(&matrix, chosen_index, &chosen_clique.members);

    let elapsed = start.elapsed().as_secs_f64();

    Ok(SnapResult {
        token: chosen_token,
        token_index: chosen_index,
        clique_members: chosen_clique.members.clone(),
        decision: SnapDecision {
            strategy,
            threshold,
            cliques_evaluated: clique_result.count,
            clique_size: chosen_clique.size,
            clique_coherence: chosen_clique.coherence,
            full_spread,
            spread_skipped,
            mean_distance_to_consensus: mean_distance,
            computation_time_s: Some(elapsed),
        },
    })
}

/// Select consensus by picking the token closest to the centroid.
///
/// For each clique, compute the centroid vector (mean of all member
/// distance profiles). Pick the member closest to this centroid.
fn select_mean(
    tokens: &[Token],
    matrix: &DistanceMatrix,
    result: &CliqueResult,
) -> (String, usize, MaximalClique) {
    let mut best_token = String::new();
    let mut best_index = 0;
    let mut best_clique = result.cliques[0].clone();
    let mut best_score = f64::MAX;

    for clique in &result.cliques {
        if clique.members.len() < 2 {
            continue;
        }

        // Compute centroid: mean distance profile across all clique members
        let mut centroid_sum = 0.0_f64;
        let m = clique.members.len();
        for &i in &clique.members {
            for &j in &clique.members {
                if i != j {
                    centroid_sum += matrix.get(i, j);
                }
            }
        }
        let centroid = centroid_sum / (m * (m - 1)) as f64;

        // Find member closest to centroid
        let mut member_score = f64::MAX;
        let mut member_idx = clique.members[0];
        for &i in &clique.members {
            let mut dist_to_centroid = 0.0_f64;
            let mut count = 0;
            for &j in &clique.members {
                if i != j {
                    dist_to_centroid += (matrix.get(i, j) - centroid).abs();
                    count += 1;
                }
            }
            let avg_dist = dist_to_centroid / count as f64;
            if avg_dist < member_score {
                member_score = avg_dist;
                member_idx = i;
            }
        }

        if member_score < best_score {
            best_score = member_score;
            best_index = member_idx;
            best_token = tokens[member_idx].raw.clone();
            best_clique = clique.clone();
        }
    }

    (best_token, best_index, best_clique)
}

/// Select the medoid: the member with minimum sum-of-distances to all others.
fn select_medoid(
    tokens: &[Token],
    matrix: &DistanceMatrix,
    result: &CliqueResult,
) -> (String, usize, MaximalClique) {
    let mut best_token = String::new();
    let mut best_index = 0;
    let mut best_clique = result.cliques[0].clone();
    let mut best_score = f64::MAX;

    for clique in &result.cliques {
        if clique.members.len() < 2 {
            continue;
        }

        // Find medoid: minimises total distance to other clique members
        let mut member_score = f64::MAX;
        let mut member_idx = clique.members[0];
        for &i in &clique.members {
            let total_dist: f64 = clique
                .members
                .iter()
                .filter(|&&j| j != i)
                .map(|&j| matrix.get(i, j))
                .sum();
            let avg_dist = total_dist / (clique.members.len() - 1) as f64;
            if avg_dist < member_score {
                member_score = avg_dist;
                member_idx = i;
            }
        }

        if member_score < best_score {
            best_score = member_score;
            best_index = member_idx;
            best_token = tokens[member_idx].raw.clone();
            best_clique = clique.clone();
        }
    }

    (best_token, best_index, best_clique)
}

/// Weighted medoid: score cliques by coherence weight.
///
/// Tight cliques (low coherence) are weighted more heavily.
fn select_weighted(
    tokens: &[Token],
    matrix: &DistanceMatrix,
    result: &CliqueResult,
) -> (String, usize, MaximalClique) {
    let mut best_token = String::new();
    let mut best_index = 0;
    let mut best_clique = result.cliques[0].clone();
    let mut best_score = f64::MAX;

    for clique in &result.cliques {
        if clique.members.len() < 2 {
            continue;
        }

        // Coherence weight: tighter cliques (lower coherence) get higher weight
        let coherence_weight = 1.0 / (clique.coherence + 0.01);

        let mut member_score = f64::MAX;
        let mut member_idx = clique.members[0];
        for &i in &clique.members {
            let total_dist: f64 = clique
                .members
                .iter()
                .filter(|&&j| j != i)
                .map(|&j| matrix.get(i, j) * coherence_weight)
                .sum();
            let avg_dist = total_dist / (clique.members.len() - 1) as f64;
            if avg_dist < member_score {
                member_score = avg_dist;
                member_idx = i;
            }
        }

        if member_score < best_score {
            best_score = member_score;
            best_index = member_idx;
            best_token = tokens[member_idx].raw.clone();
            best_clique = clique.clone();
        }
    }

    (best_token, best_index, best_clique)
}

/// Compute mean distance from `index` to all other members.
fn compute_mean_distance(matrix: &DistanceMatrix, index: usize, members: &[usize]) -> f64 {
    let count = members.len().saturating_sub(1);
    if count == 0 {
        return 0.0;
    }
    let total: f64 = members
        .iter()
        .filter(|&&j| j != index)
        .map(|&j| matrix.get(index, j))
        .sum();
    total / count as f64
}

// ---------------------------------------------------------------------------
// Convenience: one-shot consensus with best-defaults
// ---------------------------------------------------------------------------

/// Run consensus snap with medoid strategy and default parameters.
pub fn quick_snap<T: AsRef<str> + std::fmt::Debug>(tokens: &[T]) -> crate::ConsensusResult<SnapResult> {
    consensus_snap(tokens, 0.3, SnapStrategy::Medoid, 2, false)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_consensus() {
        let tokens = vec!["hello", "hallo", "helo", "world"];
        let result = consensus_snap(&tokens, 0.4, SnapStrategy::Medoid, 2, false);
        assert!(result.is_ok(), "consensus failed: {:?}", result.err());
        let snap = result.unwrap();
        // Consensus should be one of the similar tokens
        assert!(
            snap.token == "hello" || snap.token == "hallo" || snap.token == "helo",
            "unexpected consensus token: {}",
            snap.token
        );
        assert!(snap.clique_members.len() >= 2);
    }

    #[test]
    fn test_consensus_identical_input() {
        let tokens = vec!["exact", "exact", "exact"];
        let result = consensus_snap(&tokens, 0.1, SnapStrategy::Mean, 2, false);
        assert!(result.is_ok());
        let snap = result.unwrap();
        assert_eq!(snap.token, "exact");
        assert!(snap.decision.clique_coherence < 0.001);
    }

    #[test]
    fn test_insufficient_tokens() {
        let tokens = vec!["only_one"];
        let result = consensus_snap(&tokens, 0.3, SnapStrategy::Medoid, 2, false);
        assert!(matches!(
            result,
            Err(ConsensusError::InsufficientTokens { .. })
        ));
    }

    #[test]
    fn test_invalid_threshold() {
        let tokens = vec!["a", "b"];
        let result = consensus_snap(&tokens, 0.0, SnapStrategy::Medoid, 2, false);
        assert!(matches!(
            result,
            Err(ConsensusError::InvalidThreshold { .. })
        ));
    }

    #[test]
    fn test_no_consensus_diverse_input() {
        let tokens = vec!["cat", "astrophysics", "quantum", "zebra", "xylophone"];
        let result = consensus_snap(&tokens, 0.1, SnapStrategy::Medoid, 3, false);
        // With very low threshold, diverse tokens should fail to form a 3-clique
        assert!(result.is_err());
    }

    #[test]
    fn test_all_strategies() {
        let tokens = vec!["hello", "hallo", "helo", "heaven", "heavy"];
        for strategy in &[SnapStrategy::Mean, SnapStrategy::Medoid, SnapStrategy::Weighted] {
            let result = consensus_snap(&tokens, 0.5, *strategy, 2, false);
            assert!(
                result.is_ok(),
                "strategy {:?} failed: {:?}",
                strategy,
                result.err()
            );
        }
    }

    #[test]
    fn test_quick_snap() {
        let tokens = vec!["hello", "hallo", "helo", "world"];
        let result = quick_snap(&tokens);
        assert!(result.is_ok());
    }

    #[test]
    fn test_skip_spread_check() {
        let tokens = vec!["hello", "world", "goodbye", "space"];
        let with_check = consensus_snap(&tokens, 0.3, SnapStrategy::Medoid, 2, false);
        let without_check = consensus_snap(&tokens, 0.3, SnapStrategy::Medoid, 2, true);
        // Skipping the spread check may succeed where insufficient clique exists
        // Both should run without panic
        let _ = with_check;
        let _ = without_check;
    }

    #[test]
    fn test_decision_metadata() {
        let tokens = vec!["hello", "hallo", "helo"];
        let result = consensus_snap(&tokens, 0.4, SnapStrategy::Medoid, 2, false);
        assert!(result.is_ok());
        let snap = result.unwrap();
        let d = &snap.decision;
        assert_eq!(d.strategy, SnapStrategy::Medoid);
        assert!((d.threshold - 0.4).abs() < 1e-6);
        assert!(d.cliques_evaluated > 0);
        assert!(d.clique_size >= 2);
        assert!(d.computation_time_s.unwrap() > 0.0);
    }
}
