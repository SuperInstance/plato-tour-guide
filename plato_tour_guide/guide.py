# plato-tour-guide/guide.py
"""
TourGuideAgent — the main entry point for the PLATO tour guide room.

This is the heart of the system: it handles questions through the
escalation cascade, files every interaction, and builds documentation
through use.
"""

import time
from typing import Optional
from .tile import Tile, PartialAnswer, EscalationContext
from .consensus import consensus_snap, snap_decision_info
from .matching import matching_rules, FLEET_MATH_CONFIG, select_rooms_for_swarm
from .plato_client import PlatoClient


class TourGuideAgent:
    """
    A PLATO room tour guide agent.
    
    Handles developer questions through the escalation cascade:
    Level 0: Orientation (morning mode or pre-programmed)
    Level 1: Direct tile lookup (high confidence)
    Level 2: Cross-room query (medium confidence)
    Level 3: Agent swarm (no tile found)
    Level 4: Consensus snap (partials converge)
    Level 5: Expert escalation (no convergence)
    
    Usage:
        guide = TourGuideAgent(room_name="fleet-math")
        answer, level, confidence = guide.handle(
            "How does H1 cohomology detect emergence?",
            user_mode="morning"
        )
    """
    
    def __init__(
        self,
        room_name: str,
        plato_url: str = "http://localhost:8847",
        config: Optional[dict] = None,
    ):
        self.room_name = room_name
        self.plato = PlatoClient(plato_url)
        self.config = config or FLEET_MATH_CONFIG
        
        # Cache tiles from PLATO on init
        self._tiles_cache = None
        self._last_cache_time = 0
        self._cache_ttl = 60  # seconds
    
    def _get_tiles(self, force_refresh: bool = False) -> list[Tile]:
        """Get tiles from PLATO, with caching."""
        now = time.time()
        if force_refresh or (now - self._last_cache_time > self._cache_ttl):
            raw_tiles = self.plato.read_tiles(self.room_name)
            self._tiles_cache = [Tile.from_dict(t) for t in raw_tiles]
            self._last_cache_time = now
        return self._tiles_cache or []
    
    def handle(
        self,
        question: str,
        user_mode: str = "unknown",
    ) -> tuple[str, int, float]:
        """
        Handle a question through the escalation cascade.
        
        Args:
            question: The developer's question
            user_mode: "morning" | "afternoon" | "unknown"
                morning = fresh developer, needs orientation first
                afternoon = post-experience, has specific failures
        
        Returns:
            tuple of (answer, cascade_level, confidence)
            
        cascade_level:
            0 = orientation (morning mode served)
            1 = direct tile (found with high confidence)
            2 = cross-room (fetched from neighbor)
            3 = swarm spawned (no direct answer)
            4 = consensus snap (partials converged)
            5 = expert escalation (no convergence)
        """
        ctx = EscalationContext(
            question=question,
            user_mode=user_mode,
            originating_room=self.room_name,
        )
        
        # ── Level 0: Orientation ─────────────────────────────────────
        if user_mode == "morning" or self._is_orientation_question(question):
            answer, conf = self._serve_orientation(ctx)
            ctx.answer = answer
            ctx.confidence = conf
            ctx.cascade_level = 0
            self._file_interaction(ctx)
            return answer, 0, conf
        
        # ── Level 1: Direct Tile Lookup ────────────────────────────
        tile, conf = self._lookup(question)
        if tile and conf >= 0.9:
            ctx.answer = tile.answer
            ctx.confidence = conf
            ctx.cascade_level = 1
            self._file_interaction(ctx)
            return tile.answer, 1, conf
        
        # ── Level 2: Cross-Room Query ───────────────────────────────
        tile, conf, neighbor = self._cross_room_query(question)
        if tile:
            ctx.answer = tile.answer
            ctx.confidence = conf
            ctx.cascade_level = 2
            ctx.tiles_checked.append(neighbor)
            self._file_interaction(ctx)
            return tile.answer, 2, conf
        
        # ── Level 3: Agent Swarm ────────────────────────────────────
        partials = self._spawn_swarm(question)
        if not partials:
            # No partials — escalate to expert
            ctx.cascade_level = 5
            self._file_interaction(ctx)
            return self._expert_escalation(question, ctx), 5, 0.0
        
        ctx.partials = partials
        
        # ── Level 4: Consensus Snap ─────────────────────────────────
        consensus_tile = consensus_snap(partials, question)
        if consensus_tile:
            # Write the consensus tile to PLATO
            self._write_tile(consensus_tile)
            ctx.answer = consensus_tile.answer
            ctx.confidence = consensus_tile.confidence
            ctx.cascade_level = 4
            self._file_interaction(ctx)
            return consensus_tile.answer, 4, consensus_tile.confidence
        
        # ── Level 5: Expert Escalation ─────────────────────────────
        ctx.cascade_level = 5
        self._file_interaction(ctx)
        return self._expert_escalation(question, ctx), 5, 0.0
    
    def _is_orientation_question(self, question: str) -> bool:
        """Check if this is an orientation question (morning-type)."""
        orientation_keywords = [
            "where do i start", "how do i start", "what is the big picture",
            "what can i do", "getting started", "overview", "introduction",
            "tutorial", "basics", "fundamentals",
        ]
        q_lower = question.lower()
        return any(kw in q_lower for kw in orientation_keywords)
    
    def _serve_orientation(self, ctx: EscalationContext) -> tuple[str, float]:
        """Serve orientation tiles (Level 0)."""
        tiles = self._get_tiles()
        orientation_tiles = [t for t in tiles if t.is_orientation()]
        
        if orientation_tiles:
            # Return the most relevant orientation tile
            best = orientation_tiles[0]
            return f"{best.answer}\n\n(Fleet Math orientation tile)", 1.0
        
        # Fallback: serve the scope description
        scope = self.config.get("scope_description", "A PLATO room for fleet mathematics.")
        return f"Fleet Math Room\n\n{scope}", 0.8
    
    def _lookup(self, question: str) -> tuple[Optional[Tile], float]:
        """Direct tile lookup in this room (Level 1)."""
        tiles = self._get_tiles()
        
        best_tile = None
        best_conf = 0.0
        
        for tile in tiles:
            if tile.is_orientation():
                continue  # skip orientation in direct lookup
            
            # Simple keyword matching for prototype
            # In production: semantic similarity via embeddings
            q_lower = question.lower()
            t_lower = tile.question.lower()
            
            # Check for keyword overlap
            q_tokens = set(q_lower.split())
            t_tokens = set(t_lower.split())
            overlap = len(q_tokens & t_tokens)
            
            if overlap >= 2:  # at least 2 common keywords
                conf = min(0.9, 0.5 + (overlap * 0.1))
                if conf > best_conf:
                    best_conf = conf
                    best_tile = tile
        
        return best_tile, best_conf
    
    def _cross_room_query(self, question: str) -> tuple[Optional[Tile], float, str]:
        """Query neighbor rooms (Level 2)."""
        neighbors = self.config.get("neighbors", [])
        
        for neighbor in neighbors:
            tiles = self.plato.read_tiles(neighbor)
            for raw_tile in tiles:
                tile = Tile.from_dict(raw_tile)
                
                # Check keyword overlap
                q_lower = question.lower()
                t_lower = tile.question.lower()
                q_tokens = set(q_lower.split())
                t_tokens = set(t_lower.split())
                overlap = len(q_tokens & t_tokens)
                
                if overlap >= 2:
                    return tile, 0.8, neighbor
        
        return None, 0.0, ""
    
    def _spawn_swarm(self, question: str) -> list[PartialAnswer]:
        """
        Spawn agents to multiple rooms (Level 3).
        
        For the prototype, we simulate the swarm by querying
        the fleet model router for each target room.
        
        In production, this would:
        1. Select target rooms using matching scores
        2. Spawn subagents in each room via the Plato agent API
        3. Collect partial answers with confidence
        """
        # Select target rooms
        all_rooms = [self.config]  # In production: all room configs
        target_rooms = select_rooms_for_swarm(question, all_rooms)
        
        # For prototype: simulate partials from model router
        # In production: spawn actual agents in each room
        partials = []
        
        # Simulate 3 partials from different angles
        from .models import ask
        
        prompt = f"""Answer this question from your room's perspective.
Be specific and technical. If you don't have a direct answer, say 'no_tile'.

Question: {question}

Your answer (or 'no_tile'):"""
        
        # Use different models to simulate different "angles"
        try:
            r1 = ask(prompt, mode="reason")
            if r1 and r1 != "no_tile" and len(r1) > 5:
                partials.append(PartialAnswer(
                    room="room-1",
                    answer=r1,
                    confidence=0.7,
                    reasoning="nemotron-nano-reasoning"
                ))
        except:
            pass
        
        try:
            r2 = ask(prompt, mode="code")
            if r2 and r2 != "no_tile" and len(r2) > 5:
                partials.append(PartialAnswer(
                    room="room-2", 
                    answer=r2,
                    confidence=0.65,
                    reasoning="seed-code-generation"
                ))
        except:
            pass
        
        try:
            r3 = ask(prompt, mode="fast")
            if r3 and r3 != "no_tile" and len(r3) > 5:
                partials.append(PartialAnswer(
                    room="room-3",
                    answer=r3,
                    confidence=0.6,
                    reasoning="deepseek-flash-fast"
                ))
        except:
            pass
        
        return partials
    
    def _expert_escalation(self, question: str, ctx: EscalationContext) -> str:
        """Handle expert escalation (Level 5)."""
        # File the unresolved question for human review
        self.plato.write_expert_queue({
            "question": question,
            "cascade_level": ctx.cascade_level,
            "partials": [p.to_dict() for p in ctx.partials],
            "tiles_checked": ctx.tiles_checked,
            "snap_info": snap_decision_info(ctx.partials) if ctx.partials else {},
        })
        
        return (
            "I've escalated your question to the expert queue. "
            "A human will review it and add a tile when there's an answer. "
            "Thank you for helping build the documentation."
        )
    
    def _write_tile(self, tile: Tile) -> bool:
        """Write a tile to PLATO."""
        return self.plato.write_tile(tile, self.room_name)
    
    def _file_interaction(self, ctx: EscalationContext) -> None:
        """File the interaction for learning."""
        # In production: write to the interaction log
        # For prototype: just log to stdout
        print(f"[TourGuide/{self.room_name}] "
              f"L{ctx.cascade_level} | mode={ctx.user_mode} | "
              f"q='{ctx.question[:50]}...' | "
              f"conf={ctx.confidence:.2f}")
    
    def get_status(self) -> dict:
        """Get current status of the tour guide room."""
        tiles = self._get_tiles(force_refresh=True)
        orientation_count = len([t for t in tiles if t.is_orientation()])
        direct_count = len([t for t in tiles if not t.is_orientation()])
        
        return {
            "room": self.room_name,
            "total_tiles": len(tiles),
            "orientation_tiles": orientation_count,
            "direct_tiles": direct_count,
            "neighbors": self.config.get("neighbors", []),
            "cache_age_seconds": time.time() - self._last_cache_time,
        }