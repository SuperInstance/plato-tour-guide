//! # Plato Consensus — High-Performance Token-Level Consensus Engine
//!
//! Implements the consensus snap mechanism for the Plato tour-guide system.
//! Designed for production hot-paths where Python overhead is unacceptable.
//!
//! ## Architecture
//!
//! ```text
//! Tokens → Semantic Distance → Compute Spread → Find Maximal Clique → Snap
//! ```
//!
//! Each stage is independently usable and benchmarkable.

// ---------------------------------------------------------------------------
// Global allocator — mimalloc for faster multi-threaded allocation
// ---------------------------------------------------------------------------
#[cfg(not(target_os = "windows"))]
use mimalloc::MiMalloc;

#[cfg(not(target_os = "windows"))]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

// ---------------------------------------------------------------------------
// Core library re-exports
// ---------------------------------------------------------------------------
mod distance;
mod spread;
mod clique;
mod snap;

pub use distance::*;
pub use spread::*;
pub use clique::*;
pub use snap::*;

// ---------------------------------------------------------------------------
// Re-export for ergonomics
// ---------------------------------------------------------------------------
pub use parking_lot;

/// Version identifier embedded into binary for runtime inspection.
pub const PLATO_CONSENSUS_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Result type alias for fallible consensus operations.
pub type ConsensusResult<T> = Result<T, ConsensusError>;

/// Errors that can occur during consensus computation.
#[derive(Debug, Clone)]
pub enum ConsensusError {
    /// Empty input provided — at least 2 tokens required.
    InsufficientTokens { given: usize, min: usize },
    /// Mismatched dimensions between token pairs.
    DimensionMismatch { left: usize, right: usize },
    /// Threshold out of valid range.
    InvalidThreshold { threshold: f64, reason: &'static str },
    /// No tokens passed the spread filter.
    NoConsensus { spread: f64, threshold: f64 },
}

impl std::fmt::Display for ConsensusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientTokens { given, min } => {
                write!(f, "need at least {min} tokens, got {given}")
            }
            Self::DimensionMismatch { left, right } => {
                write!(f, "embedding dimension mismatch: {left} ≠ {right}")
            }
            Self::InvalidThreshold { threshold, reason } => {
                write!(f, "invalid threshold {threshold}: {reason}")
            }
            Self::NoConsensus { spread, threshold } => {
                write!(
                    f,
                    "no consensus: spread {spread:.4} exceeds threshold {threshold:.4}"
                )
            }
        }
    }
}

impl std::error::Error for ConsensusError {}

// ---------------------------------------------------------------------------
// Python bindings (optional, behind `python` feature)
// ---------------------------------------------------------------------------
#[cfg(feature = "python")]
pub mod python;
pub mod debug;
