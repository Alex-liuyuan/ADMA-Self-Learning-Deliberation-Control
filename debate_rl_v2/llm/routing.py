"""Smart Model Routing — hermes-agent inspired cost-aware routing.

Routes LLM calls to cheap or strong models based on debate round context:
  - Early exploration rounds → cheap model (lower cost)
  - Critical rounds (consensus forming, devil's advocate) → strong model
  - Simple coordinator decisions → cheap model
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from debate_rl_v2.logging_config import get_logger

logger = get_logger("llm.routing")


@dataclass
class ModelSpec:
    """Specification for an LLM model."""
    provider: str
    model: str
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    max_tokens: int = 4096
    supports_json_mode: bool = True


class SmartModelRouter:
    """Routes LLM calls to appropriate models based on context.

    Strategies:
      - "fixed": Always use the primary model.
      - "smart": Route based on round importance and role.

    Parameters
    ----------
    primary : ModelSpec
        Default strong model.
    cheap : ModelSpec | None
        Optional cheap model for simple rounds.
    strategy : str
        Routing strategy ("fixed" or "smart").
    critical_round_threshold : float
        Quality threshold below which rounds are considered critical.
    """

    def __init__(
        self,
        primary: ModelSpec,
        cheap: ModelSpec | None = None,
        strategy: str = "fixed",
        critical_round_threshold: float = 0.7,
    ) -> None:
        self.primary = primary
        self.cheap = cheap
        self.strategy = strategy
        self.critical_round_threshold = critical_round_threshold
        self._call_count = {"primary": 0, "cheap": 0}

    def route(
        self,
        role: str,
        round_num: int,
        max_rounds: int,
        quality: float = 0.5,
        disagreement: float = 0.5,
        da_active: bool = False,
        is_consensus_round: bool = False,
    ) -> ModelSpec:
        """Select the appropriate model for this call.

        Parameters
        ----------
        role : str
            Agent role (proposer, challenger, arbiter, coordinator).
        round_num : int
            Current round number.
        max_rounds : int
            Maximum rounds.
        quality : float
            Current quality score.
        disagreement : float
            Current disagreement level.
        da_active : bool
            Devil's advocate is active.
        is_consensus_round : bool
            This round might reach consensus.

        Returns
        -------
        model : ModelSpec
            Selected model specification.
        """
        if self.strategy == "fixed" or self.cheap is None:
            self._call_count["primary"] += 1
            return self.primary

        # Smart routing logic
        use_strong = False

        # Always use strong model for critical situations
        if da_active:
            use_strong = True
        elif is_consensus_round:
            use_strong = True
        elif quality < self.critical_round_threshold and round_num > max_rounds * 0.5:
            # Late rounds with low quality need strong model
            use_strong = True
        elif role == "arbiter":
            # Arbiter always needs precision
            use_strong = True
        elif round_num <= 2:
            # First rounds: cheap model for exploration
            use_strong = False
        elif disagreement > 0.7:
            # High disagreement: strong model to resolve
            use_strong = True
        else:
            # Default: cheap model for routine rounds
            use_strong = False

        if use_strong:
            self._call_count["primary"] += 1
            logger.debug(
                "Routing to primary model: role=%s round=%d",
                role, round_num,
            )
            return self.primary
        else:
            self._call_count["cheap"] += 1
            logger.debug(
                "Routing to cheap model: role=%s round=%d",
                role, round_num,
            )
            return self.cheap

    @property
    def stats(self) -> dict[str, Any]:
        total = sum(self._call_count.values())
        return {
            "total_calls": total,
            "primary_calls": self._call_count["primary"],
            "cheap_calls": self._call_count["cheap"],
            "cheap_ratio": self._call_count["cheap"] / max(total, 1),
        }

    def reset_stats(self) -> None:
        self._call_count = {"primary": 0, "cheap": 0}
