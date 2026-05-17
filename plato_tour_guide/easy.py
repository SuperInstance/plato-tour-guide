"""
plato_tour_guide.easy — Tourist-grade simplified API.

The full PLATO Tour Guide system has cascading, Penrose matching, swarms,
and consensus snap. That's powerful but intimidating.

This module gives you the 80/20: the things most developers actually need,
with zero ceremony.

Quick start:
    >>> from plato_tour_guide.easy import TourGuide, ConsensusSnap
    >>> 
    >>> # Ask a question
    >>> guide = TourGuide("fleet-math")
    >>> answer = guide.ask("How does H1 work?")
    >>> 
    >>> # Merge some answers without a room
    >>> merger = ConsensusSnap(threshold=0.3)
    >>> result = merger.merge([
    ...     "H1 is the first Betti number",
    ...     "H1 measures holes in shapes",
    ...     "H1 is homology group"
    ... ])
    >>> print(result)  # Returns best consensus answer
    >>> 
    >>> # Add knowledge
    >>> guide.teach("what is H1?", "H1 is the first Betti number")
    """

from typing import Optional, Union
import time

from .tile import Tile, PartialAnswer
from .consensus import (
    consensus_snap,
    semantic_distance,
    compute_spread,
    snap_decision_info,
    center_of_mass,
)
from .guide import TourGuideAgent
from .matching import (
    matching_rules,
    FLEET_MATH_CONFIG,
    matching_score,
    get_scope_description,
    get_neighbors,
)
from .plato_client import PlatoClient


# ── EasyTile ─────────────────────────────────────────────────────────────────

class EasyTile:
    """
    A simplified tile with just the essentials.

    What you need to know:
        question    — what was asked
        answer      — what was answered
        confidence  — how sure we are (0.0–1.0)
        tags        — optional keywords for search

    Full-fidelity dataclass: plato_tour_guide.tile.Tile

    Usage:
        >>> t = EasyTile("what is H1?", "H1 is the first Betti number", 1.0)
        >>> t.question
        'what is H1?'
        >>> t.confidence
        1.0
    """

    def __init__(
        self,
        question: str,
        answer: str,
        confidence: float = 1.0,
        tags: Optional[list[str]] = None,
    ):
        self.question = question.strip()
        self.answer = answer.strip()
        self.confidence = confidence
        self.tags = tags or []

    def __repr__(self) -> str:
        return (
            f"EasyTile(q=\"{self.question[:40]}\", "
            f"a=\"{self.answer[:40]}\", "
            f"conf={self.confidence})"
        )

    def is_verified(self) -> bool:
        """True if this tile has full confidence (human-verified)."""
        return self.confidence >= 1.0

    def to_tile(self) -> Tile:
        """Convert to a full Tile dataclass for the PLATO system."""
        return Tile(
            question=self.question,
            answer=self.answer,
            confidence=self.confidence,
            tags=self.tags,
            source="human" if self.is_verified() else "swarm",
        )

    @classmethod
    def from_tile(cls, tile: Tile) -> "EasyTile":
        """Create an EasyTile from a full Tile dataclass."""
        return cls(
            question=tile.question,
            answer=tile.answer,
            confidence=tile.confidence,
            tags=tile.tags,
        )


# ── Mode Detection ───────────────────────────────────────────────────────────

def _detect_mode(question: str) -> str:
    """
    Automatically detect if this is a "morning" or "afternoon" question.

    Morning questions are broad, uncertain, exploratory.
    Afternoon questions are specific, point-at-a-failure.

    Heuristic: short + many question words = morning.
    Specific + technical = afternoon.
    """
    q = question.lower().strip()

    # Morning signals
    morning_signals = [
        "what is", "what are", "how do i start", "getting started",
        "overview", "introduction", "basics", "fundamentals",
        "where do i", "how does this work", "explain",
        "tour", "guide me", "tell me about",
    ]
    for sig in morning_signals:
        if q.startswith(sig) or sig in q:
            return "morning"

    # Short, broad questions → morning
    if len(q.split()) <= 5 and ("?" in q or "how" in q[:10]):
        return "morning"

    # Default: afternoon (specific, technical)
    return "afternoon"


def _merge_partials(partials: list[PartialAnswer], question: str,
                    threshold: float = 0.3) -> Optional[EasyTile]:
    """
    Run consensus snap on partial answers and return an EasyTile.
    Returns None if no consensus could be reached.
    """
    tile = consensus_snap(partials, question, T=threshold)
    if tile is None:
        return None
    return EasyTile(question=tile.question, answer=tile.answer,
                    confidence=tile.confidence)


# ── TourGuide ────────────────────────────────────────────────────────────────

class TourGuide:
    """
    A dead-simple PLATO tour guide.

    You give it a room name and ask questions. It handles the rest.

    Usage:
        >>> guide = TourGuide("fleet-math")
        >>> answer = guide.ask("How does H1 work?")
        >>> guide.add_tile("what is H1?", "H1 is the first Betti number")
        >>> guide.help()

    What happens internally:
        1. Mode detection (morning vs afternoon)
        2. Orientation tiles for morning mode
        3. Direct tile lookup in your room
        4. Cross-room query to neighbors
        5. Agent swarm + consensus snap (auto)
        6. Expert escalation if nothing found
    """

    def __init__(
        self,
        room_name: str,
        plato_url: str = "http://localhost:8847",
        config: Optional[dict] = None,
    ):
        self.room_name = room_name
        self._agent = TourGuideAgent(
            room_name=room_name,
            plato_url=plato_url,
            config=config or FLEET_MATH_CONFIG,
        )
        self._local_tiles: list[EasyTile] = []
        self._plato = PlatoClient(plato_url)

    # ── Main API ──

    def ask(self, question: str, mode: Optional[str] = None) -> str:
        """
        Ask a question. Returns the best answer we can find.

        Args:
            question: What you want to know.
            mode:     "morning" (exploratory) or "afternoon" (specific).
                      If None, detected automatically.

        Returns:
            Answer string. May include a note about confidence/cascade level.

        Examples:
            >>> guide.ask("How does H1 detect emergence?")
            'H1 cohomology detects emergence through Betti numbers...'
            >>> guide.ask("What is this room?", mode="morning")
            'Fleet Math Room — mathematical fleet coordination...'
        """
        if mode is None:
            mode = _detect_mode(question)

        answer, level, confidence = self._agent.handle(question, mode)

        return answer

    def add_tile(
        self,
        question: str,
        answer: str,
        confidence: float = 1.0,
        tags: Optional[list[str]] = None,
    ) -> EasyTile:
        """
        Add a tile to the local knowledge.

        Args:
            question:   The question this tile answers.
            answer:     The answer.
            confidence: 1.0 = human-verified, 0.7 = swarm-generated, etc.
            tags:       Optional keywords for search.

        Returns:
            The EasyTile that was added.

        Usage:
            >>> guide.add_tile("what is H1?", "H1 is the first Betti number")
            >>> guide.add_tile("what is ZHC?", "Zero Holonomy Consensus",
            ...                confidence=0.9, tags=["consensus", "ZHC"])
        """
        tile = EasyTile(question, answer, confidence, tags)
        self._local_tiles.append(tile)

        # Also try to write to PLATO (silent on failure — local is enough)
        try:
            self._agent._write_tile(tile.to_tile())
        except Exception:
            pass

        return tile

    def teach(
        self,
        question: str,
        answer: Optional[str] = None,
        confidence: float = 1.0,
    ) -> EasyTile:
        """
        Add a tile by answering a question.

        This is the high-level way to add knowledge. You can provide
        the answer directly, or leave it blank to compose one.

        Args:
            question:   The question you're answering.
            answer:     The answer. If None, raises ValueError.
            confidence: How sure you are (0.0–1.0).

        Returns:
            The EasyTile that was created.

        Usage:
            >>> tile = guide.teach("what is emergence?", "Emergence is...")
            >>> tile.confidence
            1.0
        """
        if answer is None:
            raise ValueError(
                "teach() requires an answer. "
                "Use guide.add_tile(question, answer) to add existing answers."
            )

        return self.add_tile(question, answer, confidence)

    def find(self, keyword: str) -> list[EasyTile]:
        """
        Simple keyword search across all tiles.

        Searches both local tiles and PLATO tiles in the room.

        Args:
            keyword: A word or phrase to search for (case-insensitive).

        Returns:
            List of matching EasyTiles, sorted by confidence (highest first).

        Usage:
            >>> matches = guide.find("betti")
            >>> for m in matches:
            ...     print(f"{m.question}: {m.answer[:50]}")
        """
        kw = keyword.lower()
        results = []

        # Search local tiles
        for t in self._local_tiles:
            if kw in t.question.lower() or kw in t.answer.lower():
                results.append(t)

        # Search PLATO tiles
        try:
            raw_tiles = self._plato.read_tiles(self.room_name)
            seen_questions = {t.question for t in self._local_tiles}
            for raw in raw_tiles:
                q = raw.get("question", "").lower()
                a = raw.get("answer", "").lower()
                if kw in q or kw in a:
                    question_raw = raw.get("question", "")
                    if question_raw not in seen_questions:
                        results.append(EasyTile(
                            question=raw.get("question", ""),
                            answer=raw.get("answer", ""),
                            confidence=raw.get("confidence", 0.5),
                        ))
                        seen_questions.add(question_raw)
        except Exception:
            pass  # PLATO might not be available — that's fine

        # Sort by confidence descending, then alphabetically
        results.sort(key=lambda t: (-t.confidence, t.question))
        return results

    def quick_start(self) -> list[EasyTile]:
        """
        Show the 3 most important tiles in the room.

        Returns the highest-confidence tiles: orientation first,
        then highest-confidence verified tiles.

        Usage:
            >>> top3 = guide.quick_start()
            >>> for t in top3:
            ...     print(f"  {t.question} → {t.answer[:60]}")
        """
        all_tiles = self.find("")  # returns everything sorted by confidence

        # Collect: orientation tiles (source="orientation") first
        # Then highest confidence non-orientation tiles
        orientation = [t for t in all_tiles if "orientation" in t.tags]
        verified = [t for t in all_tiles if t.is_verified()
                    and "orientation" not in t.tags]
        others = [t for t in all_tiles
                  if t not in orientation and t not in verified]

        # Take top orientation, then fill from verified, then others
        result = orientation[:1]
        result.extend(verified[:3 - len(result)])
        result.extend(others[:3 - len(result)])

        # If we still have nothing, return local tiles
        if not result:
            result = sorted(
                self._local_tiles,
                key=lambda t: -t.confidence
            )[:3]

        return result

    def help(self) -> str:
        """
        Print a human-readable orientation to this room.

        Shows room name, scope description, neighbor rooms,
        and how many tiles are available.

        Usage:
            >>> print(guide.help())
        """
        config = self._agent.config
        scope = get_scope_description(config)
        neighbors = get_neighbors(config)

        lines = [
            f"╔══ PLATO Tour Guide ── {self.room_name}",
            f"║",
        ]

        if scope:
            lines.append(f"║  {scope}")

        if neighbors:
            lines.append(f"║")
            lines.append(f"║  Neighbor rooms: {', '.join(neighbors)}")

        # Tile counts
        local_count = len(self._local_tiles)
        try:
            raw_tiles = self._plato.read_tiles(self.room_name)
            plato_count = len(raw_tiles)
        except Exception:
            plato_count = 0

        lines.append(f"║")
        lines.append(f"║  Tiles: {local_count} local, {plato_count} on PLATO")
        lines.append(f"║")
        lines.append(f"║  Try:  guide.ask(\"your question\")")
        lines.append(f"║  Try:  guide.find(\"keyword\")")
        lines.append(f"║  Try:  guide.teach(\"q\", \"a\")")
        lines.append(f"╚══════════════════════")

        return "\n".join(lines)

    def status(self) -> dict:
        """
        Get a status summary as a dictionary.

        Usage:
            >>> guide.status()
            {'room': 'fleet-math', 'local_tiles': 3, 'plato_tiles': 12, ...}
        """
        try:
            agent_status = self._agent.get_status()
        except Exception:
            agent_status = {}

        return {
            "room": self.room_name,
            "local_tiles": len(self._local_tiles),
            "plato_tiles": agent_status.get("total_tiles", 0),
            "neighbors": get_neighbors(self._agent.config),
            "scope": get_scope_description(self._agent.config)[:80],
        }

    def get_tile_count(self) -> int:
        """Total number of tiles (local + PLATO)."""
        plato_count = 0
        try:
            raw_tiles = self._plato.read_tiles(self.room_name)
            plato_count = len(raw_tiles)
        except Exception:
            pass
        return len(self._local_tiles) + plato_count


# ── ConsensusSnap ───────────────────────────────────────────────────────────

class ConsensusSnap:
    """
    Merge a list of answer strings into a single consensus answer.

    No room, no PLATO — just pure semantic consensus.

    How it works:
        1. Compute pairwise semantic distances between all answers
        2. If the spread (max distance) is below the threshold: "full snap"
           — returns the center-of-mass answer
        3. If spread is moderate: "partial snap"
           — finds the largest agreeing subset and returns its best answer
        4. If spread is too wide: returns None (no consensus)

    Usage:
        >>> merger = ConsensusSnap(threshold=0.3)
        >>> result = merger.merge([
        ...     "H1 is the first Betti number",
        ...     "H1 measures holes in shapes",
        ...     "H1 is homology group"
        ... ])

    Full snap (threshold=0.5):
        >>> result  # doctest: +SKIP
        'H1 is the first Betti number'  # center of mass

    No consensus (threshold=0.1, too strict):
        >>> merger_no = ConsensusSnap(threshold=0.1)
        >>> merger_no.merge(["H1 is Betti", "cookies are good"]) is None
        True
    """

    def __init__(self, threshold: float = 0.3):
        """
        Args:
            threshold: Snap threshold (0.0–1.0).
                       Lower = stricter (rarely snaps).
                       Higher = looser (snaps more aggressively).
                       Default 0.3 works well for most use cases.
        """
        self.threshold = threshold

    def merge(self, answers: list[str]) -> Optional[str]:
        """
        Try to reach consensus on a list of answers.

        Args:
            answers: List of answer strings to merge.

        Returns:
            Consensus answer string, or None if no consensus possible.

        Usage:
            >>> merger = ConsensusSnap()
            >>> result = merger.merge([
            ...     "H1 is the first Betti number",
            ...     "H1 measures holes in shapes",
            ... ])
        """
        if not answers:
            return None

        if len(answers) == 1:
            return answers[0]

        # Build PartialAnswer wrappers
        partials = [
            PartialAnswer(
                room=f"answer-{i}",
                answer=a,
                confidence=0.7,
                reasoning="user_provided",
            )
            for i, a in enumerate(answers)
        ]

        # Try consensus snap
        snap = consensus_snap(partials, "user query", T=self.threshold)

        if snap is None:
            return None

        return snap.answer

    def merge_with_info(self, answers: list[str]) -> dict:
        """
        Like merge(), but returns a detailed info dict.

        Returns:
            {
                "decision": "full_snap" | "partial_snap" | "no_snap",
                "spread": 0.0,
                "answer": "...",  # or None if no snap
                "details": {...}  # full snap_decision_info
            }

        Usage:
            >>> info = merger.merge_with_info([
            ...     "H1 is the first Betti number",
            ...     "H1 measures holes in shapes",
            ... ])
            >>> info["decision"]
            'full_snap'
        """
        if not answers:
            return {"decision": "empty", "spread": 0.0, "answer": None,
                    "details": {}}

        partials = [
            PartialAnswer(
                room=f"answer-{i}",
                answer=a,
                confidence=0.7,
                reasoning="user_provided",
            )
            for i, a in enumerate(answers)
        ]

        info = snap_decision_info(partials, T=self.threshold)
        snap = consensus_snap(partials, "user query", T=self.threshold)

        return {
            "decision": info["decision"],
            "spread": info["spread"],
            "answer": snap.answer if snap else None,
            "details": info,
        }


# ── Functional Interface ─────────────────────────────────────────────────────

def quick_start(room_name: str = "fleet-math") -> list[EasyTile]:
    """
    Quick one-shot: show the 3 most important things in a room.

    Usage:
        >>> from plato_tour_guide.easy import quick_start
        >>> top = quick_start("fleet-math")
        >>> for t in top:
        ...     print(t.question)
    """
    guide = TourGuide(room_name)
    return guide.quick_start()


def find(room_name: str, keyword: str) -> list[EasyTile]:
    """
    Quick one-shot: keyword search across all tiles in a room.

    Usage:
        >>> from plato_tour_guide.easy import find
        >>> results = find("fleet-math", "betti")
        >>> len(results)
        1
    """
    guide = TourGuide(room_name)
    return guide.find(keyword)


def teach(room_name: str, question: str, answer: str,
          confidence: float = 1.0) -> EasyTile:
    """
    Quick one-shot: add a tile to a room.

    Usage:
        >>> from plato_tour_guide.easy import teach
        >>> tile = teach("fleet-math", "what is H1?",
        ...              "H1 is the first Betti number")
    """
    guide = TourGuide(room_name)
    return guide.teach(question, answer, confidence)


# ── Module Helpers ──────────────────────────────────────────────────────────

def _demo() -> None:
    """
    Run a quick demo of all the easy-mode features.

    Usage:
        >>> from plato_tour_guide.easy import _demo
        >>> _demo()  # doctest: +SKIP
    """
    print("==> plato_tour_guide.easy demo")
    print()
    print("==> 1. ConsensusSnap — merge similar answers")
    merger = ConsensusSnap(threshold=0.3)
    result = merger.merge([
        "H1 cohomology detects emergence via Betti numbers",
        "H1 cohomology detects emergence through topology",
        "H1 cohomology measures emergence via Betti numbers",
    ])
    print(f"    Consensus: {result}")
    print()

    print("==> 2. ConsensusSnap — different answers (no consensus)")
    result2 = merger.merge([
        "H1 cohomology detects emergence",
        "Deploy on Cloudflare Workers",
    ])
    print(f"    No consensus: {result2}")
    print()

    print("==> 3. ConsensusSnap — single input (bypasses consensus)")
    result3 = merger.merge(["H1 is the first Betti number"])
    print(f"    Single: {result3}")
    print()

    print("==> 4. ConsensusSnap — merge_with_info()")
    info = merger.merge_with_info([
        "H1 detects emergence via Betti numbers",
        "H1 cohomology detects emergence via Betti numbers",
    ])
    print(f"    Decision: {info['decision']}")
    print(f"    Spread: {info['spread']:.3f}")
    print(f"    Answer: {info['answer']}")
    print()

    print("==> 5. EasyTile — simplified tile")
    t = EasyTile("what is H1?", "H1 is the first Betti number", 1.0)
    print(f"    {repr(t)}")
    print(f"    Verified: {t.is_verified()}")
    print()

    print("==> 6. TourGuide — create with local tiles")
    guide = TourGuide("fleet-math")
    guide.add_tile("what is H1?", "H1 is the first Betti number", 1.0,
                   tags=["topology", "homology"])
    guide.add_tile("what is ZHC?", "Zero Holonomy Consensus", 0.9)
    guide.add_tile("what is fleet-math?",
                   "Mathematical fleet coordination room", 1.0,
                   tags=["orientation"])
    print(f"    Local tiles: {guide.get_tile_count()}")
    print()

    print("==> 7. TourGuide.help() — orientation")
    print(guide.help())
    print()

    print("==> 8. TourGuide.find() — keyword search")
    results = guide.find("H1")
    for r in results:
        print(f"    -> {r.question}: {r.answer[:50]}")
    print()

    print("==> 9. TourGuide.quick_start() — top 3 tiles")
    top3 = guide.quick_start()
    for t in top3:
        print(f"    [{t.confidence:.1f}] {t.question}")
    print()

    print("==> Done!")
