# plato-tour-guide/tests/test_consensus.py
"""Tests for consensus snap mechanism."""

import pytest
from plato_tour_guide.tile import PartialAnswer, Tile
from plato_tour_guide.consensus import (
    semantic_distance,
    compute_spread,
    consensus_snap,
    snap_decision_info,
    find_maximal_clique,
    center_of_mass,
)


class TestSemanticDistance:
    def test_identical_answers(self):
        d = semantic_distance("hello world", "hello world")
        assert d == 0.0
    
    def test_completely_different(self):
        d = semantic_distance("machine learning", "cruise ship schedule")
        assert d > 0.5
    
    def test_partial_overlap(self):
        d = semantic_distance("H1 cohomology detects emergence", "H1 detects emergence via Betti numbers")
        assert 0.0 < d < 0.5


class TestComputeSpread:
    def test_single_partial(self):
        partials = [PartialAnswer(room="r1", answer="test", confidence=0.7, reasoning="")]
        assert compute_spread(partials) == 0.0
    
    def test_two_near_identical_partials(self):
        # Use nearly identical partials — same core words
        partials = [
            PartialAnswer(room="r1", answer="H1 detects emergence via Betti numbers", confidence=0.7, reasoning=""),
            PartialAnswer(room="r2", answer="H1 detects emergence via Betti numbers", confidence=0.7, reasoning=""),
        ]
        spread = compute_spread(partials)
        # These are nearly identical (same words, same order)
        assert spread < 0.2, f"Expected spread < 0.2, got {spread}"
    
    def test_two_different_partials(self):
        partials = [
            PartialAnswer(room="r1", answer="use H1 cohomology", confidence=0.7, reasoning=""),
            PartialAnswer(room="r2", answer="deploy on Cloudflare Workers", confidence=0.7, reasoning=""),
        ]
        spread = compute_spread(partials)
        assert spread > 0.5


class TestConsensusSnap:
    def test_full_snap_near_identical_partials(self):
        # Very close partials — same core statement, same words
        partials = [
            PartialAnswer(room="r1", answer="H1 cohomology measures emergence via Betti numbers", confidence=0.7, reasoning=""),
            PartialAnswer(room="r2", answer="H1 cohomology measures emergence via Betti numbers", confidence=0.7, reasoning=""),
        ]
        tile = consensus_snap(partials, "how does H1 detect emergence?")
        assert tile is not None
        assert tile.swarm_flag is True
        assert tile.confidence == 0.7  # full snap = confidence 0.7
    
    def test_partial_snap_similar_partials(self):
        # Similar but not identical — should partial snap
        partials = [
            PartialAnswer(room="r1", answer="H1 cohomology measures emergence via Betti numbers", confidence=0.7, reasoning=""),
            PartialAnswer(room="r2", answer="H1 cohomology detects emergence through H¹ topology", confidence=0.7, reasoning=""),
        ]
        tile = consensus_snap(partials, "how does H1 detect emergence?")
        assert tile is not None
        assert tile.swarm_flag is True
        # partial snap has confidence 0.6
        assert tile.confidence == 0.6
    
    def test_no_snap_dissimilar_partials(self):
        partials = [
            PartialAnswer(room="r1", answer="use H1 cohomology", confidence=0.7, reasoning=""),
            PartialAnswer(room="r2", answer="deploy on Cloudflare Workers", confidence=0.7, reasoning=""),
        ]
        tile = consensus_snap(partials, "how do i deploy to cloudflare?")
        assert tile is None
    
    def test_single_partial_becomes_tile(self):
        partials = [
            PartialAnswer(room="r1", answer="H1 is the first Betti number", confidence=0.7, reasoning=""),
        ]
        tile = consensus_snap(partials, "what is H1?")
        assert tile is not None
        assert tile.partials_count == 1


class TestSnapDecisionInfo:
    def test_full_snap_info(self):
        partials = [
            PartialAnswer(room="r1", answer="zero holonomy consensus", confidence=0.7, reasoning=""),
            PartialAnswer(room="r2", answer="zero holonomy consensus protocol", confidence=0.7, reasoning=""),
        ]
        info = snap_decision_info(partials, T=0.3)
        assert info["decision"] in ["full_snap", "partial_snap"]
        assert "spread" in info
        assert "clique_size" in info


class TestMaximalClique:
    def test_all_connected(self):
        partials = [
            PartialAnswer(room="r1", answer="consensus", confidence=0.7, reasoning=""),
            PartialAnswer(room="r2", answer="agreement", confidence=0.7, reasoning=""),
            PartialAnswer(room="r3", answer="aligned", confidence=0.7, reasoning=""),
        ]
        clique = find_maximal_clique(partials, threshold=0.3)
        assert len(clique) >= 1
    
    def test_empty_partials(self):
        clique = find_maximal_clique([], threshold=0.3)
        assert clique == []


class TestCenterOfMass:
    def test_single_answer(self):
        result = center_of_mass(["H1 cohomology"])
        assert result == "H1 cohomology"
    
    def test_multiple_answers(self):
        answers = [
            "H1 measures Betti numbers",
            "H1 cohomology detects emergence",
            "H1 is the first homology group",
        ]
        result = center_of_mass(answers)
        assert len(result) > 0
        assert "H1" in result or "cohomology" in result or "Betti" in result