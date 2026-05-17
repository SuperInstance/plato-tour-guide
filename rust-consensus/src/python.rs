//! Python bindings for the Plato consensus engine via PyO3.
//!
//! Exposes `consensus_snap`, `compute_spread`, and `find_maximal_cliques`
//! as Python-callable functions operating on lists of strings.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Python module definition
// ---------------------------------------------------------------------------

/// High-performance token-level consensus for the Plato tour-guide system.
#[pymodule]
fn plato_consensus(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_consensus_snap, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_spread, m)?)?;
    m.add_function(wrap_pyfunction!(py_find_cliques, m)?)?;
    m.add_function(wrap_pyfunction!(py_semantic_distance, m)?)?;
    m.add_function(wrap_pyfunction!(py_version, m)?)?;
    m.add_class::<PySnapStrategy>()?;
    m.add_class::<PySnapResult>()?;
    m.add_class::<PySpreadResult>()?;
    m.add_class::<PyCliqueResult>()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Python-facing enums and types
// ---------------------------------------------------------------------------

#[pyclass(eq, eq_int)]
#[derive(Clone, Copy)]
enum PySnapStrategy {
    Mean = 0,
    Medoid = 1,
    Weighted = 2,
}

#[pymethods]
impl PySnapStrategy {
    fn __repr__(&self) -> String {
        match self {
            Self::Mean => "SnapStrategy.Mean",
            Self::Medoid => "SnapStrategy.Medoid",
            Self::Weighted => "SnapStrategy.Weighted",
        }
        .to_string()
    }
}

#[pyclass]
struct PySnapResult {
    #[pyo3(get)]
    token: String,
    #[pyo3(get)]
    token_index: usize,
    #[pyo3(get)]
    clique_members: Vec<usize>,
    #[pyo3(get)]
    strategy: String,
    #[pyo3(get)]
    threshold: f64,
    #[pyo3(get)]
    cliques_evaluated: usize,
    #[pyo3(get)]
    clique_size: usize,
    #[pyo3(get)]
    clique_coherence: f64,
    #[pyo3(get)]
    full_spread: f64,
    #[pyo3(get)]
    mean_distance_to_consensus: f64,
    #[pyo3(get)]
    computation_time_s: f64,
}

#[pymethods]
impl PySnapResult {
    fn __repr__(&self) -> String {
        format!(
            "SnapResult(token='{}', clique_size={}, coherence={:.4}, time={:.4}s)",
            self.token, self.clique_size, self.clique_coherence, self.computation_time_s
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Return a JSON-serializable dictionary.
    fn to_dict(&self) -> PyResult<std::collections::HashMap<String, PyObject>> {
        use pyo3::types::PyDict;
        Python::with_gil(|py| {
            let d = PyDict::new(py);
            d.set_item("token", &self.token)?;
            d.set_item("token_index", self.token_index)?;
            d.set_item("clique_members", &self.clique_members)?;
            d.set_item("strategy", &self.strategy)?;
            d.set_item("threshold", self.threshold)?;
            d.set_item("cliques_evaluated", self.cliques_evaluated)?;
            d.set_item("clique_size", self.clique_size)?;
            d.set_item("clique_coherence", self.clique_coherence)?;
            d.set_item("full_spread", self.full_spread)?;
            d.set_item("mean_distance_to_consensus", self.mean_distance_to_consensus)?;
            d.set_item("computation_time_s", self.computation_time_s)?;
            Ok(d.into())
        })
    }
}

#[pyclass]
struct PySpreadResult {
    #[pyo3(get)]
    mean_spread: f64,
    #[pyo3(get)]
    max_spread: f64,
    #[pyo3(get)]
    min_spread: f64,
    #[pyo3(get)]
    pairs_evaluated: usize,
    #[pyo3(get)]
    early_terminated: bool,
}

#[pymethods]
impl PySpreadResult {
    fn __repr__(&self) -> String {
        format!(
            "SpreadResult(mean={:.4}, max={:.4}, min={:.4}, pairs={}, early_stop={})",
            self.mean_spread, self.max_spread, self.min_spread,
            self.pairs_evaluated, self.early_terminated
        )
    }
}

#[pyclass]
struct PyCliqueResult {
    #[pyo3(get)]
    count: usize,
    #[pyo3(get)]
    largest_size: usize,
    #[pyo3(get)]
    cliques: Vec<Vec<usize>>,
    #[pyo3(get)]
    coherences: Vec<f64>,
}

#[pymethods]
impl PyCliqueResult {
    fn __repr__(&self) -> String {
        format!(
            "CliqueResult(count={}, largest={})",
            self.count, self.largest_size
        )
    }
}

// ---------------------------------------------------------------------------
// Python-callable functions
// ---------------------------------------------------------------------------

/// Run a full consensus snap.
///
/// Args:
///     tokens: List of string tokens.
///     threshold: Distance threshold for consensus.
///     strategy: SnapStrategy enum (Mean, Medoid, or Weighted).
///     min_clique_size: Minimum clique size to consider (default 2).
///     skip_spread_check: Skip preliminary spread check (default False).
///
/// Returns:
///     PySnapResult with consensus token and metadata.
#[pyfunction]
#[pyo3(signature = (tokens, threshold, strategy = PySnapStrategy::Medoid, min_clique_size = 2, skip_spread_check = false))]
fn py_consensus_snap(
    tokens: Vec<String>,
    threshold: f64,
    strategy: PySnapStrategy,
    min_clique_size: usize,
    skip_spread_check: bool,
) -> PyResult<PySnapResult> {
    let rust_strategy = match strategy {
        PySnapStrategy::Mean => crate::snap::SnapStrategy::Mean,
        PySnapStrategy::Medoid => crate::snap::SnapStrategy::Medoid,
        PySnapStrategy::Weighted => crate::snap::SnapStrategy::Weighted,
    };

    let result = crate::snap::consensus_snap(
        &tokens,
        threshold,
        rust_strategy,
        min_clique_size,
        skip_spread_check,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PySnapResult {
        token: result.token,
        token_index: result.token_index,
        clique_members: result.clique_members,
        strategy: format!("{:?}", result.decision.strategy),
        threshold: result.decision.threshold,
        cliques_evaluated: result.decision.cliques_evaluated,
        clique_size: result.decision.clique_size,
        clique_coherence: result.decision.clique_coherence,
        full_spread: result.decision.full_spread,
        mean_distance_to_consensus: result.decision.mean_distance_to_consensus,
        computation_time_s: result.decision.computation_time_s.unwrap_or(0.0),
    })
}

/// Compute the spread of a token set.
///
/// Args:
///     tokens: List of string tokens.
///     threshold: Optional threshold for early termination (None = no limit).
///
/// Returns:
///     PySpreadResult with spread statistics.
#[pyfunction]
#[pyo3(signature = (tokens, threshold = None))]
fn py_compute_spread(tokens: Vec<String>, threshold: Option<f64>) -> PyResult<PySpreadResult> {
    let token_objs: Vec<crate::Token> = tokens.iter().map(|s| crate::Token::new(s)).collect();
    let matrix = crate::DistanceMatrix::new(&token_objs);
    let result = crate::spread::compute_spread(&matrix, threshold);

    Ok(PySpreadResult {
        mean_spread: result.mean_spread,
        max_spread: result.max_spread,
        min_spread: result.min_spread,
        pairs_evaluated: result.pairs_evaluated,
        early_terminated: result.early_terminated,
    })
}

/// Find maximal cliques from token distance data.
///
/// Args:
///     tokens: List of string tokens.
///     threshold: Adjacency threshold.
///     min_size: Minimum clique size (default 2).
///
/// Returns:
///     PyCliqueResult with cliques and metadata.
#[pyfunction]
#[pyo3(signature = (tokens, threshold, min_size = 2))]
fn py_find_cliques(
    tokens: Vec<String>,
    threshold: f64,
    min_size: usize,
) -> PyResult<PyCliqueResult> {
    let token_objs: Vec<crate::Token> = tokens.iter().map(|s| crate::Token::new(s)).collect();
    let matrix = crate::DistanceMatrix::new(&token_objs);
    let result = crate::clique::find_maximal_cliques(&matrix, threshold, min_size, None);

    Ok(PyCliqueResult {
        count: result.count,
        largest_size: result.largest_size,
        cliques: result.cliques.iter().map(|c| c.members.clone()).collect(),
        coherences: result.cliques.iter().map(|c| c.coherence).collect(),
    })
}

/// Compute semantic distance between two tokens.
///
/// Args:
///     a: First token string.
///     b: Second token string.
///
/// Returns:
///     Float in [0, 1] where 0 = identical, 1 = maximally different.
#[pyfunction]
fn py_semantic_distance(a: String, b: String) -> f64 {
    let ta = crate::Token::new(a);
    let tb = crate::Token::new(b);
    ta.distance_to(&tb)
}

/// Return the version string of this plato_consensus build.
#[pyfunction]
fn py_version() -> String {
    crate::PLATO_CONSENSUS_VERSION.to_string()
}
