"""Base Strategy Bridge — domain-agnostic RL→LLM action translation.

Separates concerns that were mixed in the old StrategyBridge:
  - Action translation: BaseStrategyBridge (this file)
  - Style composition: BaseStyleComposer (this file)
  - Reward computation: framework/reward.py
  - Compliance verification: framework/compliance.py

Scenario-specific bridges subclass and override translate().
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from debate_rl_v2.framework.types import StrategySignals, ComplianceResult
from debate_rl_v2.framework.compliance import BaseComplianceVerifier
from debate_rl_v2.logging_config import get_logger

logger = get_logger("framework.strategy")


class BaseStrategyBridge(ABC):
    """Domain-agnostic RL→LLM strategy translation.

    Subclasses implement translate() to map continuous RL actions
    to scenario-specific strategy signals.

    Usage::

        class DebateStrategyBridge(BaseStrategyBridge):
            def translate(self, actions):
                signals = StrategySignals()
                signals.temperatures["proposer"] = self._map_temp(actions["proposer"][0])
                signals.style_dimensions["proposer"] = {
                    "assertiveness": self._to_unit(actions["proposer"][1]),
                }
                return signals
    """

    def __init__(
        self,
        temp_range: tuple[float, float] = (0.3, 1.2),
        compliance_verifier: BaseComplianceVerifier | None = None,
    ) -> None:
        self.temp_range = temp_range
        self.compliance_verifier = compliance_verifier
        self._last_compliance: dict[str, ComplianceResult] = {}

    @abstractmethod
    def translate(self, actions: dict[str, np.ndarray]) -> StrategySignals:
        """Translate RL actions to strategy signals. Scenario-specific."""
        ...

    def verify_compliance(
        self,
        signals: StrategySignals,
        responses: dict[str, str],
    ) -> dict[str, ComplianceResult]:
        """Verify LLM responses against strategy signals."""
        if self.compliance_verifier is None:
            return {}

        results: dict[str, ComplianceResult] = {}
        for role, response in responses.items():
            target = signals.get_style(role)
            if target:
                results[role] = self.compliance_verifier.verify(response, target)

        self._last_compliance = results
        return results

    def get_compliance_rewards(self, weight: float = 0.15) -> dict[str, float]:
        if not self._last_compliance or self.compliance_verifier is None:
            return {}
        return self.compliance_verifier.compute_reward(self._last_compliance, weight)

    def reset(self) -> None:
        self._last_compliance.clear()

    # ── Shared helpers ──

    @staticmethod
    def _to_unit(val: float | np.floating) -> float:
        """Map [-1, 1] → [0, 1]."""
        return float(np.clip((val + 1.0) / 2.0, 0.0, 1.0))

    @staticmethod
    def _to_unit_array(arr: np.ndarray) -> np.ndarray:
        """Map [-1, 1]^n → [0, 1]^n."""
        return np.clip((arr + 1.0) / 2.0, 0.0, 1.0)

    def _map_temp(self, unit_val: float) -> float:
        """Map [0, 1] → temperature range."""
        lo, hi = self.temp_range
        return lo + float(np.clip(unit_val, 0.0, 1.0)) * (hi - lo)


class BaseStyleComposer(ABC):
    """Domain-agnostic style directive composer.

    Translates numeric style dimensions into natural language directives
    that are injected into LLM system prompts.

    Subclasses implement compose() for each role.
    """

    @abstractmethod
    def compose(self, role: str, style: dict[str, float]) -> str:
        """Generate a style directive string for the given role.

        Parameters
        ----------
        role : str
            Role name.
        style : dict
            {dimension_name: value} where value is in [0, 1].

        Returns
        -------
        str
            Natural language style directive (empty string if no directive).
        """
        ...
