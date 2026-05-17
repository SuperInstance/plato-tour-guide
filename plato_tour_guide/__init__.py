"""
Plato Tour Guide — Wayfinding Science as Technology.

A PLATO room that functions as a tour guide for developer knowledge.
Uses wayfinding science (Kevin Lynch, Paul Mijksenaar, Mollerup, Arthur/Passini)
to design the escalation cascade, scoping, and documentation generation.

Usage:
    from plato_tour_guide import TourGuideAgent
    
    guide = TourGuideAgent(room_name="fleet-math")
    answer, level, confidence = guide.handle(
        "How does H1 cohomology detect emergence?",
        user_mode="morning"
    )
"""

from .tile import Tile, PartialAnswer, EscalationContext
from .guide import TourGuideAgent
from .consensus import consensus_snap, compute_spread, snap_decision_info
from .matching import matching_rules, FLEET_MATH_CONFIG, select_rooms_for_swarm
from .plato_client import PlatoClient

__version__ = "0.1.0"
__all__ = [
    "Tile",
    "PartialAnswer", 
    "EscalationContext",
    "TourGuideAgent",
    "consensus_snap",
    "compute_spread",
    "snap_decision_info",
    "matching_rules",
    "FLEET_MATH_CONFIG",
    "select_rooms_for_swarm",
    "PlatoClient",
]