"""PromptTemplateBank — manages prompt template populations per role.

Thin wrapper over PromptEvolver for template CRUD and retrieval.
"""

from __future__ import annotations

from typing import Any

from debate_rl_v2.prompt_evolution.evolver import PromptCandidate, PromptEvolver
from debate_rl_v2.logging_config import get_logger

logger = get_logger("prompt_evolution.template_bank")


class PromptTemplateBank:
    """Template population manager — delegates to PromptEvolver.

    Provides a simpler interface for template retrieval and management
    without exposing evolution internals.
    """

    def __init__(self, evolver: PromptEvolver) -> None:
        self._evolver = evolver

    def get_best_template(self, role: str) -> str:
        """Get the best-performing template for a role."""
        candidate = self._evolver.get_best(role)
        return candidate.template

    def get_template_by_tournament(self, role: str) -> tuple[str, str]:
        """Get a template via tournament selection. Returns (prompt_id, template)."""
        candidate = self._evolver.select(role)
        return candidate.prompt_id, candidate.template

    def record_performance(self, prompt_id: str, quality: float) -> None:
        """Record template performance for fitness tracking."""
        self._evolver.record_fitness(prompt_id, quality)

    def get_population_stats(self, role: str) -> dict[str, Any]:
        """Get population statistics for a role."""
        pop = self._evolver.get_population(role)
        if not pop:
            return {"size": 0, "avg_fitness": 0.0, "best_fitness": 0.0}
        fitnesses = [c.fitness for c in pop]
        return {
            "size": len(pop),
            "avg_fitness": sum(fitnesses) / len(fitnesses),
            "best_fitness": max(fitnesses),
            "min_fitness": min(fitnesses),
            "total_usage": sum(c.usage_count for c in pop),
            "max_generation": max(c.generation for c in pop),
        }

    def initialize_role(self, role: str, base_prompt: str) -> None:
        """Initialize population for a role if not already done."""
        if not self._evolver.get_population(role):
            self._evolver.initialize_population(role, base_prompt)
