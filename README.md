# Plato Tour Guide
> Wayfinding science as technology — the PLATO room that guides developers through the fleet.

## The Problem

Wayfinding — how humans orient and navigate through spaces — has been studied for over 60 years. Kevin Lynch documented the five elements of city mental maps in 1960. Paul Mijksenaar designed the yellow line at Schiphol Airport so passengers could follow it without thinking. City engineers spent 20 years trying signs before moving the crosswalk.

The same principles apply to navigating a codebase, an API, or a fleet of agents. But most developer documentation is designed like a wall of signs in a city with no map.

## The Solution

The tour guide room is a wayfinding system for developer knowledge. It:

1. **Orients before navigating** — serves the map before asking for a destination
2. **Answers direct questions** — high-confidence tile lookup
3. **Escalates across rooms** — asks neighbors when it doesn't know
4. **Spawns swarms for complexity** — calls in specialists when needed
5. **Snaps to consensus** — converges when partials are close enough
6. **Files everything** — generates documentation through use

## Architecture

```
Level 0: Orientation  → Map at museum entrance
Level 1: Direct Tile → Sign at the junction
Level 2: Cross-Room  → Asking the next exhibit
Level 3: Agent Swarm → Calling in specialists
Level 4: Consensus Snap → Pedestrians converge
Level 5: Expert Escalation → Void, no tiling exists
```

## Penrose + Mendelbrot

Each room has scope defined by **local matching rules** (Penrose lattice). Rooms are entangled at boundaries. The whole fleet is self-similar at every scale (Mendelbrot): tile/room/fleet all have the same structure — orientation + detail + cascade + swarm + expert.

## Quick Start

```python
from plato_tour_guide import TourGuideAgent

guide = TourGuideAgent(room_name="fleet-math")
answer, level, confidence = guide.handle(
    "How does H1 cohomology detect emergence?",
    user_mode="morning"  # or "afternoon"
)
print(f"Level {level}, confidence {confidence}: {answer}")
```

## Key Files

- `tile.py` — Tile data structure (question, answer, confidence, tags, swarm_flag)
- `room.py` — Room class with matching rules and neighbors
- `guide.py` — TourGuideAgent (main entry point, escalation cascade)
- `cascade.py` — Escalation level logic
- `consensus.py` — Snap-to-consensus algorithm (Čech nerve, Fréchet mean)
- `matching.py` — Penrose scope matching rules
- `models.py` — Fleet model router (DeepSeek, DeepInfra models)
- `plato_client.py` — Connection to PLATO Room Server HTTP API

## Research

See `research/plato-tour-guide/IMPLEMENTATION-DESIGN.md` for the full design document, including:
- Wayfinding science foundations (Lynch, Mollerup, Arthur/Passini, Gibson)
- Escalation cascade with wayfinding principles
- Mathematical formalization of consensus snap (sheaf theory, H¹ cohomology)
- Morning/afternoon split (priming vs diagnosis)
- Directory structure, API design, data structures