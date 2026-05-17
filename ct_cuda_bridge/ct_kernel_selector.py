#!/usr/bin/env python3
"""
CT Kernel Selector: Choose the Right CUDA Kernel Based on CT Properties

This module bridges constraint theory mathematics to GPU kernel selection.
It analyzes graph properties (H1 cohomology, Laman condition, spread) and
returns the optimal CUDA kernel for memory access patterns.

Key CT properties mapped to kernels:
- H1 dimension (β₁): Determines rigidity vs flexibility
- Laman condition (E = 2V - 3): Checks if graph is minimally rigid
- Spread threshold: Measures vertex distribution in space
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class KernelType(Enum):
    """CUDA kernel variants for different memory access patterns"""
    COALESCED_DISTANCE = "coalesced_distance_kernel"
    MIXED_ACCESS = "mixed_kernel"
    SCATTERED_GATHER = "scattered_gather_kernel"
    REGISTER_ONLY = "register_only_kernel"
    SHARED_MEMORY = "shared_memory_kernel"


@dataclass
class GraphProperties:
    """Constraint theory properties of a graph"""
    num_vertices: int
    num_edges: int
    h1_dimension: int  # β₁ = dim(H¹)
    laman_satisfied: bool  # E = 2V - 3
    spread: float  # Spatial distribution metric
    connected_components: int
    density: float  # E / V


@dataclass
class KernelRecommendation:
    """Recommended CUDA kernel with reasoning"""
    kernel_type: KernelType
    confidence: float  # 0.0 to 1.0
    reasoning: str
    expected_performance: str  # "high", "medium", "low"
    memory_access_pattern: str


def compute_h1_cohomology(vertices: np.ndarray, edges: np.ndarray) -> int:
    """
    Compute H1 cohomology dimension (β₁)

    For a graph G = (V, E):
    - β₁ = dim(H¹) = dim(ker(∂₁^T)) / dim(im(∂₂^T))
    - For connected graphs: β₁ = E - V + 1
    - β₁ = 0 → rigid (no independent cycles)
    - β₁ > 0 → flexible (has independent cycles)

    Args:
        vertices: Array of vertex IDs [V]
        edges: Edge list as array [E × 2]

    Returns:
        H1 dimension (β₁)
    """
    num_vertices = len(vertices)
    num_edges = len(edges)

    # Use Union-Find to count connected components
    parent = {v: v for v in vertices}

    def find(v):
        if parent[v] != v:
            parent[v] = find(parent[v])
        return parent[v]

    def union(v1, v2):
        root1, root2 = find(v1), find(v2)
        if root1 != root2:
            parent[root1] = root2

    # Union all edges
    for edge in edges:
        union(edge[0], edge[1])

    # Count connected components
    unique_roots = set(find(v) for v in vertices)
    num_components = len(unique_roots)

    # Compute H1 dimension: β₁ = E - V + C
    # where C = number of connected components
    beta_1 = num_edges - num_vertices + num_components

    return beta_1


def check_laman_condition(num_vertices: int, num_edges: int) -> bool:
    """
    Check Laman's theorem for rigidity in 2D

    Laman's theorem: A graph is minimally rigid iff:
    1. E = 2V - 3
    2. Every subgraph with V' vertices has at most 2V' - 3 edges

    Args:
        num_vertices: Number of vertices (V)
        num_edges: Number of edges (E)

    Returns:
        True if Laman condition is satisfied
    """
    # Check basic edge count condition
    if num_edges != 2 * num_vertices - 3:
        return False

    # TODO: Implement full Laman check (all subgraphs)
    # For now, just check the basic condition
    return True


def compute_spread_threshold(vertices: np.ndarray, embeddings: np.ndarray) -> float:
    """
    Compute spatial spread of vertices

    Spread measures how distributed vertices are in embedding space.
    High spread → vertices are well-distributed → better cache locality
    Low spread → vertices are clustered → potential cache thrashing

    Args:
        vertices: Vertex indices [V]
        embeddings: Embedding matrix [V × dim]

    Returns:
        Spread metric (0.0 to 1.0)
    """
    if len(vertices) == 0:
        return 0.0

    vertex_embeddings = embeddings[vertices]

    # Compute centroid
    centroid = np.mean(vertex_embeddings, axis=0)

    # Compute average distance from centroid
    distances = np.linalg.norm(vertex_embeddings - centroid, axis=1)
    spread = np.mean(distances)

    # Normalize by maximum possible distance (diagonal of unit hypercube)
    max_distance = np.sqrt(len(centroid))
    normalized_spread = spread / max_distance

    return min(normalized_spread, 1.0)


def compute_graph_properties(
    vertices: np.ndarray,
    edges: np.ndarray,
    embeddings: Optional[np.ndarray] = None
) -> GraphProperties:
    """
    Compute all relevant CT properties of a graph

    Args:
        vertices: Array of vertex IDs [V]
        edges: Edge list [E × 2]
        embeddings: Optional embedding matrix [V × dim]

    Returns:
        GraphProperties object with all CT metrics
    """
    num_vertices = len(vertices)
    num_edges = len(edges)

    # Compute H1 cohomology
    h1_dim = compute_h1_cohomology(vertices, edges)

    # Check Laman condition
    laman = check_laman_condition(num_vertices, num_edges)

    # Compute spread (if embeddings provided)
    if embeddings is not None:
        spread = compute_spread_threshold(vertices, embeddings)
    else:
        spread = 0.5  # Default

    # Count connected components
    parent = {v: v for v in vertices}

    def find(v):
        if parent[v] != v:
            parent[v] = find(parent[v])
        return parent[v]

    def union(v1, v2):
        root1, root2 = find(v1), find(v2)
        if root1 != root2:
            parent[root1] = root2

    for edge in edges:
        union(edge[0], edge[1])

    unique_roots = set(find(v) for v in vertices)
    num_components = len(unique_roots)

    # Compute density
    density = num_edges / num_vertices if num_vertices > 0 else 0.0

    return GraphProperties(
        num_vertices=num_vertices,
        num_edges=num_edges,
        h1_dimension=h1_dim,
        laman_satisfied=laman,
        spread=spread,
        connected_components=num_components,
        density=density
    )


def select_kernel(partials: List[np.ndarray],
                  embeddings: Optional[np.ndarray] = None) -> KernelRecommendation:
    """
    Select optimal CUDA kernel based on CT properties

    Decision tree:
    1. Compute H1 dimension (β₁)
    2. Check Laman condition
    3. Evaluate spread threshold
    4. Branch to appropriate kernel

    Args:
        partials: List of vertex sets (partial answers)
        embeddings: Optional embedding matrix

    Returns:
        KernelRecommendation with kernel type and reasoning
    """
    if not partials:
        return KernelRecommendation(
            kernel_type=KernelType.COALESCED_DISTANCE,
            confidence=0.0,
            reasoning="No partials provided",
            expected_performance="low",
            memory_access_pattern="unknown"
        )

    # Flatten all vertices from partials
    all_vertices = np.concatenate(partials)
    unique_vertices = np.unique(all_vertices)

    # Build edge set from partials (simplified)
    # In practice, this would come from the actual graph structure
    edges = []
    for partial in partials:
        if len(partial) > 1:
            # Create edges between consecutive vertices in partial
            for i in range(len(partial) - 1):
                edges.append([partial[i], partial[i + 1]])

    edges = np.array(edges) if edges else np.empty((0, 2), dtype=int)

    # Compute graph properties
    props = compute_graph_properties(unique_vertices, edges, embeddings)

    # Decision tree
    if props.h1_dimension == 0 and props.laman_satisfied:
        # Rigid graph: predictable memory access
        if props.spread > 0.7:
            # High spread: excellent cache locality
            return KernelRecommendation(
                kernel_type=KernelType.REGISTER_ONLY,
                confidence=0.95,
                reasoning=f"Rigid graph (β₁=0) with high spread ({props.spread:.2f}). "
                          f"All data fits in registers with excellent locality.",
                expected_performance="high",
                memory_access_pattern="coalesced, register-only"
            )
        else:
            # Moderate spread: use coalesced kernel
            return KernelRecommendation(
                kernel_type=KernelType.COALESCED_DISTANCE,
                confidence=0.90,
                reasoning=f"Rigid graph (β₁=0) with moderate spread ({props.spread:.2f}). "
                          f"Coalesced memory access maximizes bandwidth.",
                expected_performance="high",
                memory_access_pattern="coalesced global loads"
            )

    elif 0 < props.h1_dimension < props.num_vertices - 2:
        # Partial flexibility: mixed access patterns
        return KernelRecommendation(
            kernel_type=KernelType.MIXED_ACCESS,
            confidence=0.75,
            reasoning=f"Partially flexible graph (β₁={props.h1_dimension}). "
                      f"Mixed kernel handles both rigid and flexible regions.",
            expected_performance="medium",
            memory_access_pattern="mixed coalesced/scattered"
        )

    else:
        # Highly flexible: irregular access patterns
        if props.density < 0.1:
            # Sparse graph: use scattered kernel
            return KernelRecommendation(
                kernel_type=KernelType.SCATTERED_GATHER,
                confidence=0.70,
                reasoning=f"Highly flexible graph (β₁={props.h1_dimension}). "
                          f"Low density ({props.density:.2f}) requires gather/scatter.",
                expected_performance="low",
                memory_access_pattern="scattered gather/scatter"
            )
        else:
            # Dense but flexible: use shared memory to mitigate
            return KernelRecommendation(
                kernel_type=KernelType.SHARED_MEMORY,
                confidence=0.65,
                reasoning=f"Flexible graph (β₁={props.h1_dimension}) with moderate density. "
                          f"Shared memory reduces global memory access.",
                expected_performance="medium",
                memory_access_pattern="shared memory with some coalescing"
            )


def benchmark_kernel_selection(
    num_vertices_list: List[int],
    edge_probabilities: List[float]
) -> None:
    """
    Benchmark kernel selection across various graph configurations

    Args:
        num_vertices_list: List of vertex counts to test
        edge_probabilities: List of edge probabilities for random graphs
    """
    print("=== CT Kernel Selection Benchmark ===\n")

    for num_vertices in num_vertices_list:
        print(f"Testing {num_vertices} vertices:")

        for edge_prob in edge_probabilities:
            # Generate random graph
            vertices = np.arange(num_vertices)
            edges = []
            for i in range(num_vertices):
                for j in range(i + 1, num_vertices):
                    if np.random.random() < edge_prob:
                        edges.append([i, j])

            edges = np.array(edges) if edges else np.empty((0, 2), dtype=int)

            # Select kernel
            recommendation = select_kernel([vertices], None)

            # Compute metrics
            props = compute_graph_properties(vertices, edges)

            print(f"  Edge prob={edge_prob:.2f}: β₁={props.h1_dimension}, "
                  f"Laman={props.laman_satisfied}, Spread={props.spread:.2f}")
            print(f"    → {recommendation.kernel_type.value}")
            print(f"    Confidence: {recommendation.confidence:.2f}")
            print(f"    Reasoning: {recommendation.reasoning}")
            print(f"    Expected performance: {recommendation.expected_performance}\n")


def adaptive_kernel_dispatcher(
    partials: List[np.ndarray],
    embeddings: Optional[np.ndarray] = None,
    enable_fallback: bool = True
) -> str:
    """
    Adaptive kernel dispatcher with fallback support

    This is the main entry point for runtime kernel selection.
    It can dynamically switch kernels if performance is poor.

    Args:
        partials: List of vertex sets (partial answers)
        embeddings: Optional embedding matrix
        enable_fallback: Whether to enable fallback to slower kernels

    Returns:
        CUDA kernel name to launch
    """
    recommendation = select_kernel(partials, embeddings)

    print(f"Adaptive dispatcher selected: {recommendation.kernel_type.value}")
    print(f"  Reasoning: {recommendation.reasoning}")
    print(f"  Expected performance: {recommendation.expected_performance}")

    # TODO: Implement runtime fallback mechanism
    # - Launch kernel
    # - Monitor performance
    # - If below threshold, switch to fallback kernel

    return recommendation.kernel_type.value


if __name__ == "__main__":
    # Run benchmark
    benchmark_kernel_selection(
        num_vertices_list=[10, 50, 100, 500],
        edge_probabilities=[0.1, 0.3, 0.5, 0.9]
    )

    # Test adaptive dispatcher
    print("\n=== Adaptive Dispatcher Test ===")
    test_partials = [
        np.array([0, 1, 2, 3, 4]),
        np.array([5, 6, 7, 8, 9])
    ]
    kernel = adaptive_kernel_dispatcher(test_partials)
    print(f"\nSelected kernel: {kernel}")
