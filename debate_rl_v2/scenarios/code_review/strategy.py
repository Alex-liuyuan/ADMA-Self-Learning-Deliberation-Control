"""Code Review Strategy Bridge — RL→LLM translation for code review."""

from __future__ import annotations

import numpy as np

from debate_rl_v2.framework.strategy import BaseStrategyBridge
from debate_rl_v2.framework.types import StrategySignals


class CodeReviewStrategyBridge(BaseStrategyBridge):
    """Translates RL continuous actions to code review strategy signals.

    Roles and their action dimensions:
      - author_ctrl (3D): defensiveness, detail_level, refactor_willingness
      - reviewer_ctrl (3D): thoroughness, constructiveness, strictness
      - maintainer (3D): strictness, detail_feedback, merge_threshold
    """

    def translate(self, actions: dict[str, np.ndarray]) -> StrategySignals:
        signals = StrategySignals()

        # Author
        a = actions.get("author_ctrl", np.zeros(3))
        a_unit = self._to_unit_array(a)
        signals.temperatures["author"] = self._map_temp(a_unit[0])
        signals.style_dimensions["author"] = {
            "defensiveness": float(a_unit[0]),
            "detail_level": float(a_unit[1]),
            "refactor_willingness": float(a_unit[2]),
        }

        # Reviewer
        r = actions.get("reviewer_ctrl", np.zeros(3))
        r_unit = self._to_unit_array(r)
        signals.temperatures["reviewer"] = self._map_temp(r_unit[0])
        signals.style_dimensions["reviewer"] = {
            "thoroughness": float(r_unit[0]),
            "constructiveness": float(r_unit[1]),
            "strictness": float(r_unit[2]),
        }

        # Maintainer (evaluator)
        m = actions.get("maintainer", np.zeros(3))
        m_unit = self._to_unit_array(m)
        signals.temperatures["maintainer"] = self._map_temp(m_unit[0])
        signals.style_dimensions["maintainer"] = {
            "strictness": float(m_unit[0]),
            "detail_feedback": float(m_unit[1]),
        }

        return signals
