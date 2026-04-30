"""Code Review Observation Encoder — encodes state for RL agents."""

from __future__ import annotations

import numpy as np

from debate_rl_v2.framework.observation import BaseObservationEncoder
from debate_rl_v2.framework.types import CollaborationState


class CodeReviewObservationEncoder(BaseObservationEncoder):
    """Observation encoder for the code review scenario.

    Shared observation (8D):
      [0] round_progress (round_num / max_rounds)
      [1] quality_score
      [2] agreement_level
      [3] compliance
      [4] issues_found_norm (capped at 10)
      [5] issues_resolved_ratio
      [6] quality_trend
      [7] decision_encoding (0=pending, 0.5=changes, 1.0=merge/reject)

    Role-specific (6D): zero-padded, filled by GenericRoleObservationTracker.
    """

    _SHARED_DIM = 8
    _ROLE_DIM = 6

    def __init__(self) -> None:
        self._prev_quality: float = 0.5

    def shared_obs_dim(self) -> int:
        return self._SHARED_DIM

    def role_obs_dim(self) -> int:
        return self._ROLE_DIM

    def encode_shared(self, state: object, round_num: int, max_rounds: int) -> np.ndarray:
        if not isinstance(state, CollaborationState):
            return np.zeros(self._SHARED_DIM, dtype=np.float32)

        meta = state.metadata
        issues_found = meta.get("issues_found", 0)
        issues_resolved = meta.get("issues_resolved", 0)
        decision = meta.get("decision", "pending")

        decision_map = {"pending": 0.0, "changes": 0.5, "merge": 1.0, "reject": 1.0}
        decision_enc = decision_map.get(decision, 0.0)

        quality_trend = state.quality_score - self._prev_quality
        self._prev_quality = state.quality_score

        issues_ratio = issues_resolved / max(issues_found, 1)

        obs = np.array([
            round_num / max(max_rounds, 1),
            state.quality_score,
            state.agreement_level,
            state.compliance,
            min(issues_found, 10) / 10.0,
            issues_ratio,
            float(np.clip(quality_trend, -1, 1)),
            decision_enc,
        ], dtype=np.float32)

        return obs

    def encode_role(self, shared_obs: np.ndarray, role: str) -> np.ndarray:
        role_ext = np.zeros(self._ROLE_DIM, dtype=np.float32)
        return np.concatenate([shared_obs, role_ext])

    def reset(self) -> None:
        self._prev_quality = 0.5
