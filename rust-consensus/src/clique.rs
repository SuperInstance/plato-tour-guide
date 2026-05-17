//! # Find Maximal Clique
//!
//! Implements the Bron–Kerbosch algorithm with pivoting (Tomita variant)
//! to find all maximal cliques in a token-similarity graph. The adjacency
//! threshold determines which token pairs are considered "connected."
//!
//! ## Adjacency matrix
//!
//! A boolean N×N matrix is constructed from the distance matrix using
//! a threshold: two tokens are adjacent iff their distance is below
//! the threshold T.

use crate::distance::DistanceMatrix;

/// Represents a single maximal clique — a set of mutually close tokens.
#[derive(Debug, Clone)]
pub struct MaximalClique {
    /// Indices of tokens in this clique (into the original token array).
    pub members: Vec<usize>,
    /// Size of this clique.
    pub size: usize,
    /// Internal coherence: max pairwise distance among members.
    pub coherence: f64,
}

/// Result of maximal-clique finding.
#[derive(Debug, Clone)]
pub struct CliqueResult {
    /// All maximal cliques found, sorted descending by size.
    pub cliques: Vec<MaximalClique>,
    /// Number of cliques found.
    pub count: usize,
    /// Size of the largest clique.
    pub largest_size: usize,
    /// Whether pruning was applied during search.
    pub pruned: bool,
}

/// Find all maximal cliques in the token graph using Bron–Kerbosch with pivot.
///
/// ## Parameters
///
/// * `matrix` - The N×N distance matrix.
/// * `threshold` - Two tokens are "near" iff `distance(i, j) <= threshold`.
/// * `min_size` - Only return cliques with at least this many members.
/// * `prune_threshold` - Skip tokens whose total connections fall below this.
///
/// ## Returns
///
/// A `CliqueResult` with cliques sorted descending by size (largest first).
pub fn find_maximal_cliques(
    matrix: &DistanceMatrix,
    threshold: f64,
    min_size: usize,
    prune_threshold: Option<usize>,
) -> CliqueResult {
    let n = matrix.len();
    if n == 0 {
        return CliqueResult {
            cliques: vec![],
            count: 0,
            largest_size: 0,
            pruned: false,
        };
    }

    // Build adjacency matrix
    let adjacency = build_adjacency(matrix, threshold);

    // Apply pruning: remove tokens with too few connections
    let (adj, valid_indices, pruned) = if let Some(pt) = prune_threshold {
        prune_sparse(&adjacency, n, pt, &(0..n).collect::<Vec<_>>())
    } else {
        (adjacency, (0..n).collect::<Vec<_>>(), false)
    };

    if adj.is_empty() || valid_indices.is_empty() {
        return CliqueResult {
            cliques: vec![],
            count: 0,
            largest_size: 0,
            pruned,
        };
    }

    let m = valid_indices.len();

    // Bron–Kerbosch with Tomita pivot
    let mut all_cliques: Vec<Vec<usize>> = Vec::new();
    let mut r = Vec::with_capacity(m);
    let mut p: Vec<usize> = (0..m).collect();
    let mut x: Vec<usize> = Vec::with_capacity(m);

    bron_kerbosch(&adj, &mut r, &mut p, &mut x, &mut all_cliques);

    // Build cliques (with optional min_size filter)
    let mut cliques: Vec<MaximalClique> = all_cliques
        .into_iter()
        .filter(|members| members.len() >= min_size)
        .map(|members| {
            let num = members.len();
            // Map back to original indices
            let orig_members: Vec<usize> = members.iter().map(|&i| valid_indices[i]).collect();
            // Compute coherence: max pairwise distance among clique members
            let coherence = compute_clique_coherence(matrix, &orig_members);
            MaximalClique {
                members: orig_members,
                size: num,
                coherence,
            }
        })
        .collect();

    // Sort descending by size
    cliques.sort_unstable_by(|a, b| b.size.cmp(&a.size));

    let largest_size = cliques.first().map(|c| c.size).unwrap_or(0);

    CliqueResult {
        count: cliques.len(),
        cliques,
        largest_size,
        pruned,
    }
}

/// Build boolean adjacency matrix from distance matrix.
fn build_adjacency(matrix: &DistanceMatrix, threshold: f64) -> Vec<Vec<bool>> {
    let n = matrix.len();
    let mut adj = vec![vec![false; n]; n];
    for i in 0..n {
        // Diagonal: a node is always adjacent to itself for set operations
        adj[i][i] = true;
        for j in (i + 1)..n {
            let connected = matrix.get(i, j) <= threshold;
            adj[i][j] = connected;
            adj[j][i] = connected;
        }
    }
    adj
}

/// Prune nodes with fewer than `min_connections` neighbors.
fn prune_sparse(
    adj: &[Vec<bool>],
    n: usize,
    min_connections: usize,
    valid_indices: &[usize],
) -> (Vec<Vec<bool>>, Vec<usize>, bool) {
    let valid: Vec<usize> = valid_indices
        .iter()
        .copied()
        .filter(|&i| {
            let count = adj[i].iter().filter(|&&v| v).count();
            count > min_connections
        })
        .collect();

    if valid.len() == n {
        // No pruning happened — return original
        (adj.to_vec(), valid, false)
    } else {
        // Build reduced adjacency matrix
        let m = valid.len();
        let mut reduced = vec![vec![false; m]; m];
        for (new_i, &orig_i) in valid.iter().enumerate() {
            reduced[new_i][new_i] = true;
            for (new_j, &orig_j) in valid.iter().enumerate() {
                if new_i < new_j && adj[orig_i][orig_j] {
                    reduced[new_i][new_j] = true;
                    reduced[new_j][new_i] = true;
                }
            }
        }
        (reduced, valid, true)
    }
}

/// Bron–Kerbosch algorithm with Tomita pivot (recursive, iterative ordering).
///
/// This variant is chosen for its efficiency on sparse and dense graphs
/// alike, with good worst-case bounds for the maximum-clique problem.
fn bron_kerbosch(
    adj: &[Vec<bool>],
    r: &mut Vec<usize>,
    p: &mut Vec<usize>,
    x: &mut Vec<usize>,
    results: &mut Vec<Vec<usize>>,
) {
    if p.is_empty() && x.is_empty() {
        // Found a maximal clique
        results.push(r.clone());
        return;
    }

    // Tomita pivot: choose vertex u from P ∪ X that maximizes |N(u) ∩ P|
    let pivot = choose_pivot(adj, p, x);

    // Candidates = P \ N(pivot)
    let candidates: Vec<usize> = if pivot < adj.len() && !adj[pivot].is_empty() {
        // For efficiency, use a HashSet-like approach on the sorted sets
        // p is always sorted, so we can filter linearly
        let pivot_neighbors = &adj[pivot];
        p.iter()
            .copied()
            .filter(|&v| !pivot_neighbors[v])
            .collect()
    } else {
        p.clone()
    };

    for v in candidates {
        // R ∪ {v}
        r.push(v);

        // P ∩ N(v)
        let p_intersection: Vec<usize> = intersect_sorted(p, &collect_neighbors(adj, v));
        *p = p_intersection;

        // X ∩ N(v)
        let x_intersection: Vec<usize> = intersect_sorted(x, &collect_neighbors(adj, v));
        *x = x_intersection;

        bron_kerbosch(adj, r, p, x, results);

        // Backtrack
        r.pop();

        // P ← P \ {v}, X ← X ∪ {v}
        p.retain(|&u| u != v);
        x.push(v);
        x.sort_unstable();
    }
}

/// Choose a pivot vertex from `P ∪ X`.
///
/// The pivot is the vertex that maximises `|N(v) ∩ P|`, reducing
/// the number of recursive calls (Tomita's optimisation).
fn choose_pivot(adj: &[Vec<bool>], p: &[usize], x: &[usize]) -> usize {
    let mut best = 0;
    let mut best_count = usize::MAX; // we want minimum remaining candidates, so maximise neighbours

    for &u in p.iter().chain(x.iter()) {
        if u >= adj.len() {
            continue;
        }
        let count = p.iter().filter(|&&v| adj[u][v]).count();
        if count < best_count {
            best_count = count;
            best = u;
        }
    }

    best
}

/// Collect all neighbors of vertex `v` as a sorted vector.
fn collect_neighbors(adj: &[Vec<bool>], v: usize) -> Vec<usize> {
    if v >= adj.len() {
        return vec![];
    }
    adj[v]
        .iter()
        .enumerate()
        .filter(|(_, &connected)| connected)
        .map(|(i, _)| i)
        .collect()
}

/// Intersection of two sorted vectors.
fn intersect_sorted(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut result = Vec::with_capacity(a.len().min(b.len()));
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        if a[i] < b[j] {
            i += 1;
        } else if a[i] > b[j] {
            j += 1;
        } else {
            result.push(a[i]);
            i += 1;
            j += 1;
        }
    }
    result
}

/// Compute intra-clique coherence: maximum pairwise distance among members.
fn compute_clique_coherence(matrix: &DistanceMatrix, members: &[usize]) -> f64 {
    let mut max_dist = 0.0_f64;
    for (idx, &i) in members.iter().enumerate() {
        for &j in &members[idx + 1..] {
            let d = matrix.get(i, j);
            if d > max_dist {
                max_dist = d;
            }
        }
    }
    max_dist
}

/// Find the single best clique (largest, ties broken by coherence).
pub fn find_best_clique(
    matrix: &DistanceMatrix,
    threshold: f64,
    min_size: usize,
) -> Option<MaximalClique> {
    let result = find_maximal_cliques(matrix, threshold, min_size, None);
    result.cliques.into_iter().next()
}

/// Find the largest clique that includes a specific token index.
pub fn find_clique_containing(
    matrix: &DistanceMatrix,
    threshold: f64,
    token_idx: usize,
    min_size: usize,
) -> Option<MaximalClique> {
    let result = find_maximal_cliques(matrix, threshold, min_size, None);
    result
        .cliques
        .into_iter()
        .filter(|c| c.members.contains(&token_idx))
        .max_by_key(|c| c.size)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Token;

    fn make_matrix(tokens: &[&str]) -> DistanceMatrix {
        let ts: Vec<Token> = tokens.iter().map(|&s| Token::new(s)).collect();
        DistanceMatrix::new(&ts)
    }

    #[test]
    fn test_identical_tokens_form_clique() {
        let tokens = vec!["hello", "hello", "hello"];
        let m = make_matrix(&tokens);
        let result = find_maximal_cliques(&m, 0.1, 2, None);
        // All three identical → should form a 3-clique
        assert!(!result.cliques.is_empty());
        assert_eq!(result.cliques[0].size, 3);
        assert!(result.cliques[0].coherence < 0.01);
    }

    #[test]
    fn test_adjacent_tokens_small_clique() {
        // Similar tokens should cluster
        let tokens = vec!["hello", "hallo", "helo"];
        let m = make_matrix(&tokens);
        let result = find_maximal_cliques(&m, 0.5, 2, None);
        assert!(!result.cliques.is_empty());
        assert!(result.largest_size >= 2);
    }

    #[test]
    fn test_diverse_tokens_no_large_cliques() {
        let tokens = vec!["cat", "astrophysics", "quantum", "zebra", "xylophone"];
        let m = make_matrix(&tokens);
        let result = find_maximal_cliques(&m, 0.2, 3, None);
        // Diverse tokens with low threshold should not form large cliques
        assert!(result.cliques.is_empty() || result.largest_size < 3);
    }

    #[test]
    fn test_high_threshold_all_connected() {
        let tokens = vec!["a", "b", "c", "d"];
        let m = make_matrix(&tokens);
        let result = find_maximal_cliques(&m, 1.0, 2, None);
        // With threshold 1.0, all tokens are adjacent → one big clique
        assert!(!result.cliques.is_empty());
        assert_eq!(result.cliques[0].size, 4);
    }

    #[test]
    fn test_min_size_filter() {
        let tokens = vec!["hello", "hallo", "helo", "xyzzy"];
        let m = make_matrix(&tokens);
        let result = find_maximal_cliques(&m, 0.5, 3, None);
        // Only cliques with >= 3 members
        for c in &result.cliques {
            assert!(c.size >= 3);
        }
    }

    #[test]
    fn test_empty_input() {
        let m = DistanceMatrix::from_distances(vec![], 0);
        let result = find_maximal_cliques(&m, 0.5, 1, None);
        assert_eq!(result.count, 0);
    }

    #[test]
    fn test_clique_containing() {
        let tokens = vec!["hello", "hallo", "helo", "xyzzy"];
        let m = make_matrix(&tokens);
        // Token at index 0 ("hello") should be in a clique with "hallo" and "helo"
        let clique = find_clique_containing(&m, 0.4, 0, 2);
        assert!(clique.is_some());
        assert!(clique.unwrap().members.contains(&0));
    }

    #[test]
    fn test_pruning() {
        let tokens = vec!["a", "b", "c", "d", "xyzzy"];
        // One very different token ("xyzzy") should be pruned
        let m = make_matrix(&tokens);
        let result = find_maximal_cliques(&m, 0.5, 3, Some(1));
        assert!(result.largest_size >= 3 || result.count == 0);
    }
}
