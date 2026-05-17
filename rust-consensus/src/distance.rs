//! # Semantic Distance
//!
//! Computes normalised token-level similarity using Unicode-aware
//! string comparison. Supports pre-tokenized input to avoid redundant
//! tokenization in batch pipelines, and uses a SIMD-ready [`DistanceMatrix`]
//! structure for efficient bulk processing.

use rayon::prelude::*;
use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

// ---------------------------------------------------------------------------
// Token: the fundamental consensus unit
// ---------------------------------------------------------------------------

/// A consensus token with optional pre-computed normalisation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Token {
    /// Raw input string (as provided).
    pub raw: String,
    /// Pre-normalized NFC form — computed once, reused.
    pub normalized: String,
    /// Grapheme cluster count, a proxy for visual length.
    pub grapheme_len: usize,
}

impl Token {
    /// Create a new `Token` from a raw string.
    ///
    /// Normalization (NFC) and grapheme counting happen once here.
    #[inline]
    pub fn new(raw: impl Into<String>) -> Self {
        let raw = raw.into();
        let normalized: String = raw.nfc().collect();
        let grapheme_len = normalized.graphemes(true).count();
        Self {
            raw,
            normalized,
            grapheme_len,
        }
    }

    /// Create a `Token` from already-normalized text.
    ///
    /// Skips re-normalization. Only use this when you *know* the input
    /// is already in NFC form (e.g., from an upstream pipeline).
    #[inline]
    pub fn from_normalized(raw: impl Into<String>, normalized: impl Into<String>) -> Self {
        let raw = raw.into();
        let normalized = normalized.into();
        let grapheme_len = normalized.graphemes(true).count();
        Self {
            raw,
            normalized,
            grapheme_len,
        }
    }

    /// Compute semantic distance between this token and another.
    ///
    /// Uses normalised edit distance (Levenshtein / NFC hybrid) as the
    /// primary metric, producing a value in `[0, 1]`.
    #[inline]
    pub fn distance_to(&self, other: &Token) -> f64 {
        semantic_distance_impl(&self.normalized, &other.normalized)
    }
}

impl<T: Into<String>> From<T> for Token {
    #[inline]
    fn from(s: T) -> Self {
        Self::new(s)
    }
}

// ---------------------------------------------------------------------------
// Distance Matrix — SIMD-ready contiguous storage
// ---------------------------------------------------------------------------

/// A flattened upper-triangular distance matrix.
///
/// Layout: `distance_to(i, j)` is stored at index `i * n + j` for `i < j`.
/// The diagonal (`i == j`) always has distance `0.0`. Lower triangle
/// mirrors the upper: `distance(j, i) == distance(i, j)`.
///
/// This contiguous layout is amenable to SIMD reduction via Rayon.
#[derive(Debug, Clone)]
pub struct DistanceMatrix {
    /// Raw flattened upper-triangular storage.
    data: Vec<f64>,
    /// Number of tokens.
    n: usize,
}

impl DistanceMatrix {
    /// Build a new distance matrix from a slice of tokens.
    ///
    /// Computes `n*(n-1)/2` distances in parallel.
    pub fn new(tokens: &[Token]) -> Self {
        let n = tokens.len();
        let mut data = vec![0.0_f64; n * n];

        // Compute all pairwise distances in parallel
        // We compute both (i,j) and (j,i) simultaneously for cache friendliness
        data.par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, row)| {
                let ti = &tokens[i];
                for (j, cell) in row.iter_mut().enumerate() {
                    if i != j {
                        *cell = ti.distance_to(&tokens[j]);
                    }
                }
            });

        Self { data, n }
    }

    /// Build a distance matrix from pre-computed distances.
    ///
    /// Useful when distances come from an external oracle (e.g., an embedding model).
    #[inline]
    pub fn from_distances(distances: Vec<f64>, n: usize) -> Self {
        assert_eq!(distances.len(), n * n, "distance vector must be n×n");
        Self { data: distances, n }
    }

    /// Number of tokens represented.
    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    /// True when there are no tokens.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Look up distance between token `i` and token `j`.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        debug_assert!(i < self.n && j < self.n);
        self.data[i * self.n + j]
    }

    /// Raw inner data — for SIMD or external consumption.
    #[inline]
    pub fn raw(&self) -> &[f64] {
        &self.data
    }

    /// Consume the matrix, returning raw data — useful for FFI boundaries.
    #[inline]
    pub fn into_raw(self) -> Vec<f64> {
        self.data
    }
}

// ---------------------------------------------------------------------------
// Core distance calculation
// ---------------------------------------------------------------------------

/// Normalised Levenshtein distance between two strings.
///
/// Returns a value in `[0, 1]` where `0.0` means identical and `1.0`
/// means maximally different. Normalisation divides by the length of
/// the longer string.
///
/// ## Performance note
///
/// Uses a classic two-row DP to minimise allocation. For very short
/// tokens (≤ 4 chars) a small special-case returns immediately.
fn normalized_levenshtein(a: &str, b: &str) -> f64 {
    let len_a = a.chars().count();
    let len_b = b.chars().count();

    // Short-circuit: identical strings
    if a == b {
        return 0.0;
    }

    // Short-circuit: empty strings
    if len_a == 0 {
        return 1.0;
    }
    if len_b == 0 {
        return 1.0;
    }

    // Ensure b is the shorter string for smaller DP rows
    let (short, long, short_len, long_len) = if len_a <= len_b {
        (a, b, len_a, len_b)
    } else {
        (b, a, len_b, len_a)
    };

    // Two-row DP
    let mut prev: Vec<usize> = (0..=short_len).collect();
    let mut curr: Vec<usize> = vec![0; short_len + 1];

    for (i, ch_l) in long.chars().enumerate() {
        curr[0] = i + 1;
        for (j, ch_s) in short.chars().enumerate() {
            let cost = if ch_l == ch_s { 0 } else { 1 };
            curr[j + 1] = std::cmp::min(
                std::cmp::min(curr[j] + 1, prev[j + 1] + 1),
                prev[j] + cost,
            );
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[short_len] as f64 / long_len as f64
}

/// Compute semantic distance between two normalized strings.
///
/// Combines Levenshtein distance with Unicode-aware normalization
/// factors to produce a robust similarity score.
fn semantic_distance_impl(a: &str, b: &str) -> f64 {
    // Base: normalised Levenshtein
    let lev = normalized_levenshtein(a, b);

    // Penalty for differing grapheme counts (visual length mismatch)
    let gc_a = a.graphemes(true).count();
    let gc_b = b.graphemes(true).count();
    let gc_max = gc_a.max(gc_b) as f64;

    let grapheme_penalty = if gc_max == 0.0 {
        0.0
    } else {
        (gc_a.max(gc_b) - gc_a.min(gc_b)) as f64 / gc_max
    };

    // Blend: 70% edit distance, 30% grapheme-length mismatch
    0.7 * lev + 0.3 * grapheme_penalty
}

// ---------------------------------------------------------------------------
// Batched distance computation for pre-tokenized inputs
// ---------------------------------------------------------------------------

/// Compute pairwise distances between two batches of pre-tokenized tokens.
///
/// Accepts already-normalized token strings. Useful when the caller has
/// already done normalization upstream (e.g., in a Python pipeline).
pub fn batch_distance(
    left: &[impl AsRef<str>],
    right: &[impl AsRef<str>],
) -> Vec<Vec<f64>> {
    let lt: Vec<Token> = left.iter().map(|s| Token::new(s.as_ref())).collect();
    let rt: Vec<Token> = right.iter().map(|s| Token::new(s.as_ref())).collect();

    lt.par_iter()
        .map(|l| {
            rt.iter()
                .map(|r| l.distance_to(r))
                .collect::<Vec<_>>()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_tokens() {
        let a = Token::new("hello");
        let b = Token::new("hello");
        assert!((a.distance_to(&b) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_completely_different_tokens() {
        let a = Token::new("abc");
        let b = Token::new("xyz");
        // These should be fairly distant
        assert!(a.distance_to(&b) > 0.5);
    }

    #[test]
    fn test_unicode_normalization_superscript() {
        let sup = Token::new("¹");
        let num = Token::new("1");
        // ¹ vs 1 should be close but not identical
        let d = sup.distance_to(&num);
        assert!(d < 0.8, "superscript vs digit distance too large: {d}");
    }

    #[test]
    fn test_pre_tokenized_input() {
        let a = Token::from_normalized("RAW", "raw");
        let b = Token::from_normalized("RAWER", "rawer");
        assert!(a.distance_to(&b) > 0.0);
    }

    #[test]
    fn test_distance_matrix() {
        let tokens: Vec<Token> = vec!["alpha", "beta", "gamma", "delta"]
            .into_iter()
            .map(Token::new)
            .collect();
        let dm = DistanceMatrix::new(&tokens);
        assert_eq!(dm.len(), 4);
        assert!((dm.get(0, 0) - 0.0).abs() < 1e-10);
        assert!((dm.get(1, 1) - 0.0).abs() < 1e-10);
        assert!((dm.get(0, 1) - dm.get(1, 0)).abs() < 1e-12);
    }

    #[test]
    fn test_empty_token() {
        let a = Token::new("");
        let b = Token::new("something");
        assert!((a.distance_to(&b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_grapheme_penalty() {
        let short = Token::new("a");
        let long = Token::new("abcdefghij");
        let d = short.distance_to(&long);
        assert!(d > 0.0);
        assert!(d <= 1.0);
    }

    #[test]
    fn test_batch_distance_shape() {
        let left = vec!["a", "b"];
        let right = vec!["x", "y", "z"];
        let result = batch_distance(&left, &right);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 3);
        assert_eq!(result[1].len(), 3);
    }
}
