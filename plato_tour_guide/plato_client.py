# plato-tour-guide/plato_client.py
"""Connection to PLATO Room Server HTTP API."""

import json
import urllib.request
from typing import Optional
from .tile import Tile


class PlatoClient:
    """
    Connection to the PLATO Room Server HTTP API.
    
    Handles reading tiles, writing tiles, and expert queue management.
    
    PLATO server runs at http://localhost:8847 by default.
    
    Key endpoints:
    - GET  /status                         — server status
    - GET  /rooms                          — list all rooms
    - GET  /room/{name}                    — room info (not the same as history)
    - GET  /room/{name}/history            — room tile history (BROKEN as of 2026-05-17)
    - POST /submit                          — submit a tile (requires 'answer' field)
    - POST /room/{name}/submit             — direct room submission (BROKEN as of 2026-05-17)
    """
    
    def __init__(self, url: str = "http://localhost:8847"):
        self.url = url.rstrip("/")
    
    def _get(self, path: str) -> dict:
        """Make a GET request to PLATO."""
        req = urllib.request.Request(f"{self.url}{path}")
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    
    def _post(self, path: str, data: dict) -> dict:
        """Make a POST request to PLATO."""
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            f"{self.url}{path}",
            data=body,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    
    def get_status(self) -> dict:
        """Get server status."""
        return self._get("/status")
    
    def list_rooms(self) -> list[dict]:
        """List all rooms."""
        return self._get("/rooms")["rooms"]
    
    def read_tiles(self, room: str) -> list[dict]:
        """
        Read all tiles in a room.
        
        Note: /room/{room}/history returns 404 in current PLATO build.
        Fallback: use /tiles/recent and filter by domain.
        """
        try:
            # Try direct room history endpoint
            return self._get(f"/room/{room}/history")["tiles"]
        except Exception:
            # Fallback: use recent tiles and filter
            try:
                recent = self._get("/tiles/recent")["tiles"]
                return [t for t in recent if t.get("domain") == room]
            except Exception:
                return []
    
    def write_tile(self, tile: Tile, room: str) -> bool:
        """
        Write a tile to PLATO.
        
        Requires: question, answer, domain (room name)
        Optional: confidence, tags, source, swarm_flag
        
        Returns True on success, False on failure.
        """
        data = tile.to_dict()
        data["domain"] = room  # PLATO uses domain as room name
        
        try:
            resp = self._post("/submit", data)
            return resp.get("status") == "accepted"
        except Exception as e:
            print(f"PLATO write_tile failed: {e}")
            return False
    
    def query_room(self, room: str, question: str) -> Optional[Tile]:
        """
        Query a room for a tile matching the question.
        
        Returns Tile or None.
        """
        tiles = self.read_tiles(room)
        q_lower = question.lower()
        q_tokens = set(q_lower.split())
        
        best_tile = None
        best_score = 0
        
        for raw_tile in tiles:
            t_lower = raw_tile.get("question", "").lower()
            t_tokens = set(t_lower.split())
            overlap = len(q_tokens & t_tokens)
            
            if overlap >= 2 and overlap > best_score:
                best_score = overlap
                best_tile = Tile.from_dict(raw_tile)
        
        return best_tile
    
    def write_expert_queue(self, item: dict) -> bool:
        """
        Write to the expert escalation queue.
        
        For now, we write to a special "expert-queue" room in PLATO.
        In production, this would be a separate queue system.
        """
        tile_data = {
            "domain": "expert-queue",
            "question": item.get("question", ""),
            "answer": json.dumps({
                "type": "expert_escalation",
                "cascade_level": item.get("cascade_level", 5),
                "partials": item.get("partials", []),
                "tiles_checked": item.get("tiles_checked", []),
                "snap_info": item.get("snap_info", {}),
            }),
            "confidence": 0.0,
            "tags": ["expert-escalation"],
            "source": "tour-guide",
        }
        
        try:
            resp = self._post("/submit", tile_data)
            return resp.get("status") == "accepted"
        except Exception as e:
            print(f"PLATO expert queue write failed: {e}")
            return False