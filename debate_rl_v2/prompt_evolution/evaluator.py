"""PromptEvaluator — prompt fitness evaluation via debate quality metrics.

Evaluates prompt candidates by tracking the quality of debates
where each prompt was used, providing fitness scores for evolution.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from debate_rl_v2.logging_config import get_logger

logger = get_logger("prompt_evolution.evaluator")


class PromptEvaluator:
    """Evaluates prompt fitness from debate quality metrics.

    Tracks per-prompt quality observations and computes fitness
    as a weighted average favoring recent observations.
    """

    def __init__(self, decay: float = 0.95, min_samples: int = 3) -> None:
        self._decay = decay
        self._min_samples = min_samples
        # prompt_id -> list[(quality, weight)]
        self._observations: dict[str, list[tuple[float, float]]] = defaultdict(list)

    def record(self, prompt_id: str, quality: float, weight: float = 1.0) -> None:
        """Record a quality observation for a prompt."""
        self._observations[prompt_id].append((quality, weight))

    def get_fitness(self, prompt_id: str) -> float:
        """Compute fitness for a prompt (exponentially weighted average).

        Returns 0.0 if insufficient samples.
        """
        obs = self._observations.get(prompt_id, [])
        if len(obs) < self._min_samples:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0
        for i, (q, w) in enumerate(reversed(obs)):
            decay_w = w * (self._decay ** i)
            weighted_sum += q * decay_w
            total_weight += decay_w

        return weighted_sum / max(total_weight, 1e-8)

    def get_all_fitness(self) -> dict[str, float]:
        """Get fitness scores for all tracked prompts."""
        return {pid: self.get_fitness(pid) for pid in self._observations}

    def get_stats(self, prompt_id: str) -> dict[str, Any]:
        """Get detailed stats for a prompt."""
        obs = self._observations.get(prompt_id, [])
        if not obs:
            return {"samples": 0, "fitness": 0.0}
        qualities = [q for q, _ in obs]
        return {
            "samples": len(obs),
            "fitness": self.get_fitness(prompt_id),
            "avg_quality": sum(qualities) / len(qualities),
            "max_quality": max(qualities),
            "min_quality": min(qualities),
        }

    def clear(self, prompt_id: str | None = None) -> None:
        if prompt_id:
            self._observations.pop(prompt_id, None)
        else:
            self._observations.clear()
