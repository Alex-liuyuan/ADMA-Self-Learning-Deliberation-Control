"""辩论场景数据类型 — 权威来源。

从 envs/types.py 迁移至此，修复 scenarios/debate/ 反向依赖 envs/ 的层级违反。
envs/types.py 保留 re-export shim 以向后兼容。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class TextDebateState:
    """Snapshot of the current debate state."""
    round_num: int = 0
    proposal: str = ""
    challenge: str = ""
    verdict: str = ""
    compliance: float = 1.0
    disagreement: float = 0.5
    lambda_adv: float = 0.5
    mode: str = "standard"
    quality_score: float = 0.5
    da_active: bool = False
    consensus_reached: bool = False


@dataclass
class DebateTurn:
    """Record of one complete debate round."""
    round_num: int
    proposal: str
    proposal_confidence: float
    challenge: str
    challenge_confidence: float
    verdict: str
    arbiter_scores: Dict[str, float]
    coordinator_action: int
    state: TextDebateState
    reasoning: Dict[str, str] = field(default_factory=dict)


@dataclass
class FusionRoundRecord:
    """Record of one fusion debate round with RL + LLM data."""
    round_num: int
    observation: np.ndarray
    rl_actions: Dict[str, np.ndarray]
    strategy_signals: Dict[str, float]
    rewards: Dict[str, float]
    # LLM side
    proposal: str = ""
    challenge: str = ""
    verdict: str = ""
    quality_score: float = 0.5
    disagreement: float = 0.5
    compliance: float = 0.5
    lambda_adv: float = 0.5
    mode: str = "standard"
    # Compliance scores
    compliance_scores: Dict[str, float] = field(default_factory=dict)
