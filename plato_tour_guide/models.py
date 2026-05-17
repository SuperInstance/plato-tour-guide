# plato-tour-guide/models.py
"""
Fleet Model Router — wrapper for the workspace fleet_router.py.

This module re-exports the fleet_router functionality for use
by the tour guide agent's swarm simulation.
"""

import sys
import os
from pathlib import Path

# Import from workspace scripts
workspace = Path("/home/ubuntu/.openclaw/workspace")
sys.path.insert(0, str(workspace / "scripts"))

try:
    from fleet_router import ask as _ask, swarm as _swarm, ask_batch as _ask_batch
except ImportError:
    # Fallback if fleet_router isn't available
    def _ask(prompt, mode="fast", **kwargs):
        return f"[{mode}] placeholder response to: {prompt[:50]}..."
    
    def _swarm(prompts, modes):
        return [f"swarm_{i}" for i in range(len(prompts))]
    
    def _ask_batch(prompts, mode="fast", n=3):
        return [f"batch_{i}" for i in range(len(prompts))]


def ask(prompt: str, mode: str = "fast", **kwargs) -> str:
    """
    Route to the appropriate model based on task type.
    
    Modes:
      fast     — deepseek-v4-flash (iteration, analysis, quick answers)
      deep     — deepseek-v4-pro (mathematical proofs, architecture)
      creative — seed-2.0-mini (3-5 divergent options, temp 0.85)
      code     — seed-2.0-code (implementation, Rust, Python)
      reason   — nemotron-3-nano-30b (fast reasoning chain)
      reasoning-deep — nemotron-super-120b (deep multi-step reasoning)
      generation — gemma-4-26b (document generation, descriptions)
      synthesis — kimi-k2.6 (creative + reasoning synthesis)
      sparse   — hermes-3-llama-3.1-405b (unique perspective when needed)
      swiss-army — qwen3.6-35b (reliable all-rounder, fast)
    """
    return _ask(prompt, mode, **kwargs)


def swarm(prompts: list, modes: list) -> list:
    """
    Fire multiple prompts simultaneously with different models.
    Returns results in same order as prompts.
    """
    return _swarm(prompts, modes)


def ask_batch(prompts: list, mode: str = "fast", n: int = 3) -> list:
    """
    Take a list of prompts and return n results from different angles.
    Best for creative exploration or getting multiple perspectives.
    """
    return _ask_batch(prompts, mode, n)