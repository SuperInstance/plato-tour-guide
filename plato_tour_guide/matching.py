# plato-tour-guide/matching.py
"""
Penrose Scope Matching Rules — how a room knows its scope.

Each room has local matching rules that define what questions it can
answer correctly. A question outside the matching rules escalates
to a neighbor room.

The Penrose analogy:
- Local matching rules: which tiles can touch which tiles
- Global non-periodicity: pattern never exactly repeats
- Entanglement at boundaries: rooms connect at edges with matching rules

Example: the fleet-math room
"""

from typing import Optional


# ── Scope Configuration for fleet-math Room ─────────────────────────────────

FLEET_MATH_CONFIG = {
    "name": "fleet-math",
    
    # Keywords that strongly indicate IN-scope
    "hard_positive_keywords": [
        "h1", "cohomology", "betti", "emergence",
        "zero holonomy", "holonomy", "consensus",
        "laman", "rigidity", "v=2e-3", "2e-3",
        "pythagorean", "48 direction", "48-dir",
        "trust vector", "fleet-coordinate", "fleet-homology",
        "fleet-topology", "holonomy-48-bridge",
        "byzantine", "tolerance", "latency", "38ms",
        "beam search", "tile coordinate",
        "sheaf", "čech", "nerve", "H1",
        "self-coordinating", "over-constrained",
    ],
    
    # Keywords that strongly indicate OUT-of-scope
    "hard_negative_keywords": [
        "midi", "synthesis", "audio", "sound",
        "cloudflare", "cf workers", "wrangler", "pages",
        "fishing", "log", "museum", "exhibit",
        "cruise", "ship", "tour guide",
        "react", "frontend", "css", "html",
        "python", "3.12", "numpy", "pandas",
    ],
    
    # Soft positive tags (medium signal)
    "positive_tags": [
        "constraint-theory", "mathematical", "formal",
        "consensus-protocol", "distributed", "topology",
        "graph-theory", "rigidity",
    ],
    
    # Tags that trigger escalation
    "escalate_tags": [
        "deployment", "infrastructure", "frontend",
        "audio", "video", "image", "ui",
        "cf-workers", "pages", "wrangler",
    ],
    
    # Neighbor rooms (entanglement at boundaries)
    "neighbors": [
        "fleet-coordinate",      # ZHC + Laman + Pythagorean48
        "fleet-homology",       # H1 algebraic cycle space
        "holonomy-48-bridge",   # ZHC + Pythagorean48 algebra
        "constraint-theory-core", # formal verification, Coq
        "flux-lucid",           # consensus engines
    ],
    
    # Boundary description for this room's scope
    "scope_description": (
        "Mathematical fleet coordination: H1 cohomology for emergence detection, "
        "Zero Holonomy Consensus for Byzantine tolerance, Laman's theorem for "
        "rigidity threshold, Pythagorean48 for 6-bit trust encoding. Interfaces "
        "with fleet-coordinate (tile coordinates), fleet-homology (cycle space), "
        "holonomy-48-bridge (algebra), constraint-theory (formal verification)."
    ),
}


def matching_rules(
    question: str,
    config: dict = FLEET_MATH_CONFIG,
    embedding_centroid: Optional[list[float]] = None,
) -> tuple[str, str]:
    """
    Determine if a question is in-scope for a room.
    
    Returns: (decision, direction)
        decision: "ANSWER" | "ESCALATE" | "BOUNDARY"
        direction: None | "north" | "east" | "south" | "west" | "neighbor_name"
    
    Matching hierarchy:
    1. Hard negative keyword → ESCALATE immediately
    2. Hard positive keyword → ANSWER immediately  
    3. Embedding distance check (if centroid available)
    4. Tag matching
    5. Boundary resolution (Penrose edge)
    
    Example boundary resolution:
    - Query has 48-direction keywords AND museum context
    - fleet-math scores 0.31, tile-rendering-engine scores 0.29
    - Since adjacent room has better match, ESCALATE south
    """
    q_lower = question.lower()
    tokens = set(q_lower.split())
    
    # Step 1: Hard negative keywords → immediate escalation
    for kw in config.get("hard_negative_keywords", []):
        if kw.lower() in q_lower:
            return ("ESCALATE", "hard_negative")
    
    # Step 2: Hard positive keywords → immediate answer
    for kw in config.get("hard_positive_keywords", []):
        if kw.lower() in q_lower:
            return ("ANSWER", None)
    
    # Step 3: Escalate tags → escalate
    for tag in config.get("escalate_tags", []):
        if tag.lower() in q_lower:
            return ("ESCALATE", f"tag:{tag}")
    
    # Step 4: Positive tags → in scope (with lower confidence)
    for tag in config.get("positive_tags", []):
        if tag.lower() in q_lower:
            return ("ANSWER", "soft_tag")
    
    # Step 5: Embedding distance (if centroid available)
    # For prototype, skip embedding check — use keyword fallback
    # In production: cosine_distance(query_embedding, room_centroid)
    
    # Step 6: Boundary resolution
    # This is where Penrose edge logic applies — queries that fall
    # on the boundary between two rooms get resolved based on
    # which room has better match AND user context
    
    # Default: can't decide, escalate
    return ("ESCALATE", "unclassifiable")


def matching_score(question: str, config: dict) -> float:
    """
    Compute a matching score (0.0-1.0) for a question in a room.
    
    This is used for cross-room queries — we route to the room
    with the highest matching score.
    
    Score components:
    - Hard positive keywords: +0.4 per match
    - Positive tags: +0.3 per match
    - Embedding similarity: +0.3 if available
    """
    q_lower = question.lower()
    
    score = 0.0
    
    # Hard positive keywords
    for kw in config.get("hard_positive_keywords", []):
        if kw.lower() in q_lower:
            score += 0.4
    
    # Positive tags
    for tag in config.get("positive_tags", []):
        if tag.lower() in q_lower:
            score += 0.3
    
    # Cap at 1.0
    return min(score, 1.0)


def get_scope_description(config: dict) -> str:
    """Return human-readable scope description for a room."""
    return config.get("scope_description", "")


def get_neighbors(config: dict) -> list[str]:
    """Return list of neighbor room names."""
    return config.get("neighbors", [])


# ── Cross-Room Routing ──────────────────────────────────────────────────────────

def select_rooms_for_swarm(question: str, all_rooms: list[dict]) -> list[str]:
    """
    Select which rooms to query for a swarm.
    
    Uses matching scores to pick the top N rooms that might
    have relevant tiles. Excludes rooms with hard negative hits.
    
    Args:
        question: The question to route
        all_rooms: List of room configs (dicts with name, matching_rules)
    
    Returns:
        List of room names to query (max 5)
    """
    scored = []
    for room in all_rooms:
        if room.get("name") == "tour-guide":
            continue  # don't query the tour guide room itself
        
        decision, _ = matching_rules(question, room)
        if decision == "ESCALATE":
            continue  # skip rooms with hard negative
        
        score = matching_score(question, room)
        scored.append((room["name"], score))
    
    # Sort by score descending, take top 5
    scored.sort(key=lambda x: x[1], reverse=True)
    return [name for name, score in scored[:5]]