# plato-tour-guide/consensus.py
"""
Consensus Snap — the core mechanism for converging partial answers.

Mathematical foundation (from deepseek-pro model swarm):
- Partial answers live in embedding space E
- Compute pairwise semantic distances d(p_i, p_j)
- spread = max(d) across all pairs
- T = 0.3 (tunable threshold)
- If spread < T: full snap to Fréchet mean
- If T <= spread < 2T: partial snap to best partial (maximal clique)
- If spread >= 2T: no snap, escalate to expert

The "Čech nerve" is the simplicial complex where k-simplices are
(k+1)-tuples of partials with all pairwise distances < ε.
The spread is the diameter of this nerve.

Related to H1 cohomology: if partials have a "hole" in their coverage
(agree on 80%, diverge on 20%), the 20% is the boundary condition.
The consensus tile fills the hole — it's the cycle that closes the gap.
"""

import math
from typing import Optional
from .tile import PartialAnswer, Tile


# ── Semantic Distance ────────────────────────────────────────────────────────

def semantic_distance(answer_a: str, answer_b: str) -> float:
    """
    Compute semantic distance between two partial answers.
    
    In the full implementation, this uses embedding models
    (sentence-transformers or OpenAI embeddings) to compute
    cosine distance in semantic space.
    
    For the prototype, we use an n-gram overlap heuristic
    that handles word variations better than raw Jaccard.
    
    Returns: 0.0 (identical) to 1.0 (maximally different)
    """
    if answer_a.lower() == answer_b.lower():
        return 0.0
    
    def normalize(text: str) -> list[str]:
        """Normalize text: lowercase, extract alphanumeric tokens, sort."""
        tokens = []
        for word in text.lower().split():
            # Strip punctuation, extract alphanumeric core
            clean = ''.join(c for c in word if c.isalnum())
            if clean:
                # Normalize common variants
                clean = clean.replace('¹', '1').replace('²', '2')
                clean = clean.replace('zero', '0').replace('one', '1')
                tokens.append(clean)
        return tokens
    
    # Normalize both answers
    a_tokens = normalize(answer_a)
    b_tokens = normalize(answer_b)
    
    if not a_tokens or not b_tokens:
        return 0.5  # neutral distance for empty answers
    
    # Compute overlap coefficient (not Jaccard)
    # Overlap = |A ∩ B| / min(|A|, |B|)
    # This is less strict than Jaccard — shared words count more
    intersection = len(set(a_tokens) & set(b_tokens))
    min_size = min(len(a_tokens), len(b_tokens))
    
    if min_size == 0:
        return 0.5
    
    overlap = intersection / min_size
    
    # Also check for substring containment (answers that embed in each other)
    a_text = answer_a.lower()
    b_text = answer_b.lower()
    substring_overlap = 0.0
    if len(a_text) > 10 and len(b_text) > 10:
        # If one is a substring of the other, high similarity
        if a_text in b_text or b_text in a_text:
            substring_overlap = 0.5
    
    # Combine overlap with substring bonus
    distance = 1.0 - overlap - substring_overlap
    
    # Also check bigram overlap (captures word order)
    def bigrams(tokens: list[str]) -> set[tuple[str, str]]:
        return {(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)}
    
    a_bigrams = bigrams(a_tokens)
    b_bigrams = bigrams(b_tokens)
    if a_bigrams and b_bigrams:
        bg_overlap = len(a_bigrams & b_bigrams) / max(len(a_bigrams), len(b_bigrams))
        distance = min(distance, 1.0 - bg_overlap)
    
    return max(0.0, min(1.0, distance))


def embedding_distance(answer_a: str, answer_b: str) -> float:
    """
    Full implementation using embedding models.
    
    Requires: sentence-transformers or OpenAI embeddings API.
    Returns cosine distance in embedding space.
    
    def semantic_distance(a, b):
        import requests
        emb_a = requests.post("http://embedding-service/encode",
                              json={"text": a}).json()["embedding"]
        emb_b = requests.post("http://embedding-service/encode",
                              json={"text": b}).json()["embedding"]
        return cosine_distance(emb_a, emb_b)
    """
    # Fallback to keyword heuristic
    return semantic_distance(answer_a, answer_b)


# ── Consensus Snap ─────────────────────────────────────────────────────────────

def compute_spread(partials: list[PartialAnswer]) -> float:
    """
    Compute the semantic spread of partial answers.
    
    spread = max_{i,j} d(partial_i, partial_j)
    
    This is the diameter of the Čech nerve — the largest
    pairwise distance between any two partial answers.
    """
    if len(partials) < 2:
        return 0.0
    
    max_d = 0.0
    for i in range(len(partials)):
        for j in range(i + 1, len(partials)):
            d = semantic_distance(partials[i].answer, partials[j].answer)
            if d > max_d:
                max_d = d
            partials[i].distance = d
            partials[j].distance = d
    
    return max_d


def compute_pairwise_distances(partials: list[PartialAnswer]) -> list[tuple[int, int, float]]:
    """
    Compute all pairwise distances.
    Returns list of (i, j, distance) tuples.
    """
    distances = []
    for i in range(len(partials)):
        for j in range(i + 1, len(partials)):
            d = semantic_distance(partials[i].answer, partials[j].answer)
            distances.append((i, j, d))
    return distances


def find_maximal_clique(partials: list[PartialAnswer], threshold: float = 0.2) -> list[int]:
    """
    Find the largest subset of partials where all pairwise distances < threshold.
    
    This is finding the maximal clique in the agreement graph,
    where edges connect partials with distance < threshold.
    
    For small N (N <= 10), we use brute force.
    For larger N, would use a proper clique-finding algorithm.
    
    Returns: list of indices of partials in the maximal clique.
    """
    n = len(partials)
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    # Build adjacency matrix (True if within threshold)
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = semantic_distance(partials[i].answer, partials[j].answer)
            if d < threshold:
                adj[i][j] = adj[j][i] = True
    
    # Find largest clique via brute force (N <= 10 for prototype)
    best_clique = []
    for size in range(n, 0, -1):
        from itertools import combinations
        for combo in combinations(range(n), size):
            # Check if all pairs in combo are connected
            valid = True
            for i in range(size):
                for j in range(i + 1, size):
                    if not adj[combo[i]][combo[j]]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return list(combo)
    
    return best_clique


def center_of_mass(answers: list[str]) -> str:
    """
    Compute the "center of mass" of a list of answers.
    
    For text, we use the answer with minimal total distance to all others
    (the medoid) as a proxy for the center of mass.
    
    In a full embedding implementation, this would be the
    Fréchet mean: argmin_{p} sum_i w_i * d(p, p_i)^2
    """
    if len(answers) == 1:
        return answers[0]
    
    # Find the answer with minimal total distance to all others
    best_answer = answers[0]
    best_total_distance = float('inf')
    
    for candidate in answers:
        total_d = sum(semantic_distance(candidate, other) for other in answers)
        if total_d < best_total_distance:
            best_total_distance = total_d
            best_answer = candidate
    
    return best_answer


def consensus_snap(
    partials: list[PartialAnswer],
    question: str,
    T: float = 0.3
) -> Optional[Tile]:
    """
    Main consensus snap function.
    
    Takes partial answers from a swarm, computes spread,
    and either snaps to a consensus tile or returns None (no snap).
    
    Args:
        partials: List of PartialAnswer from swarm agents
        question: The original question
        T: Snap threshold. Default 0.3.
            spread < T: full snap
            T <= spread < 2T: partial snap
            spread >= 2T: no snap
    
    Returns:
        Tile with consensus answer, or None if no snap possible.
    """
    if not partials:
        return None
    
    if len(partials) == 1:
        # Single partial — no consensus to compute, just use it
        return Tile(
            question=question,
            answer=partials[0].answer,
            confidence=0.7,
            source="swarm",
            swarm_flag=True,
            partials_count=1,
            spread=0.0,
            notes="single_partial_no_consensus"
        )
    
    # Compute spread
    spread = compute_spread(partials)
    
    # Snap decision
    if spread < T:
        # Full snap: center of mass
        consensus_answer = center_of_mass([p.answer for p in partials])
        return Tile(
            question=question,
            answer=consensus_answer,
            confidence=0.7,  # inferred, not verified
            source="consensus_snap",
            swarm_flag=True,
            partials_count=len(partials),
            spread=spread,
            notes=f"snap_type: full; T={T}; spread={spread:.3f}"
        )
    
    elif spread < 2 * T:
        # Partial snap: best partial (highest confidence)
        # Or find maximal clique and use its center of mass
        clique = find_maximal_clique(partials, threshold=0.2)
        
        if len(clique) >= 2:
            # Use the clique
            clique_answers = [partials[i].answer for i in clique]
            consensus_answer = center_of_mass(clique_answers)
            return Tile(
                question=question,
                answer=consensus_answer,
                confidence=0.6,  # partial consensus
                source="consensus_snap",
                swarm_flag=True,
                partials_count=len(partials),
                spread=spread,
                notes=f"snap_type: partial; clique_size={len(clique)}/{len(partials)}; spread={spread:.3f}"
            )
        else:
            # No strong clique — use best partial
            best_partial = max(partials, key=lambda p: p.confidence)
            return Tile(
                question=question,
                answer=best_partial.answer,
                confidence=0.6,
                source="consensus_snap",
                swarm_flag=True,
                partials_count=len(partials),
                spread=spread,
                notes=f"snap_type: partial; best_from={best_partial.room}; spread={spread:.3f}"
            )
    
    else:
        # No snap — spread too large, escalate to expert
        return None


def snap_decision_info(partials: list[PartialAnswer], T: float = 0.3) -> dict:
    """
    Return detailed information about what the snap decision would be.
    Useful for debugging and for expert escalation.
    """
    if not partials:
        return {"decision": "no_partials", "spread": 0.0, "clique": []}
    
    spread = compute_spread(partials)
    clique = find_maximal_clique(partials, threshold=0.2)
    
    if spread < T:
        decision = "full_snap"
    elif spread < 2 * T:
        decision = "partial_snap"
    else:
        decision = "no_snap"
    
    return {
        "decision": decision,
        "spread": spread,
        "threshold_T": T,
        "2T": 2 * T,
        "clique_size": len(clique),
        "clique": clique,
        "partials_summary": [
            {"room": p.room, "confidence": p.confidence, "distance": p.distance}
            for p in partials
        ]
    }