"""Shared data types for the multi-agent collaboration framework.

All domain-agnostic types live here. Scenario-specific types
(e.g. DebateTurn) extend these base types.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class InteractionPhase(str, Enum):
    """Generic phases of a multi-agent collaboration round."""
    PROPOSE = "propose"
    CHALLENGE = "challenge"
    EVALUATE = "evaluate"
    COORDINATE = "coordinate"
    CUSTOM = "custom"


@dataclass
class CollaborationState:
    """Domain-agnostic snapshot of a collaboration session.

    Subclass this for scenario-specific state (e.g. DebateState adds
    `proposal`, `challenge`, `verdict` fields).
    """
    round_num: int = 0
    quality_score: float = 0.5
    agreement_level: float = 0.5      # 1.0 = full agreement, 0.0 = full disagreement
    compliance: float = 1.0
    intensity: float = 0.5            # adversarial / competitive intensity
    mode: str = "standard"
    is_terminal: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def disagreement(self) -> float:
        return 1.0 - self.agreement_level


@dataclass
class RoundRecord:
    """Domain-agnostic record of one collaboration round."""
    round_num: int
    role_outputs: dict[str, str] = field(default_factory=dict)
    role_confidences: dict[str, float] = field(default_factory=dict)
    evaluator_scores: dict[str, float] = field(default_factory=dict)
    coordinator_action: int = 0
    state: CollaborationState = field(default_factory=CollaborationState)
    reasoning: dict[str, str] = field(default_factory=dict)


@dataclass
class AgentMessage:
    """Typed message for inter-agent communication."""
    msg_type: str
    sender: str
    content: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    round_num: int = 0
    msg_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategySignals:
    """Domain-agnostic RL strategy signals for LLM agents.

    Each role gets: temperature, and a dict of style dimensions.
    Scenario-specific bridges populate the style dimensions.
    """
    temperatures: dict[str, float] = field(default_factory=dict)
    style_dimensions: dict[str, dict[str, float]] = field(default_factory=dict)
    mechanism_deltas: dict[str, float] = field(default_factory=dict)
    exploration_rate: float = 0.3

    def get_temperature(self, role: str, default: float = 0.7) -> float:
        return self.temperatures.get(role, default)

    def get_style(self, role: str) -> dict[str, float]:
        return self.style_dimensions.get(role, {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperatures": self.temperatures,
            "style_dimensions": self.style_dimensions,
            "mechanism_deltas": self.mechanism_deltas,
            "exploration_rate": self.exploration_rate,
        }


@dataclass
class ComplianceResult:
    """Result of verifying agent compliance with strategy signals."""
    overall_score: float = 0.5
    dimension_scores: dict[str, float] = field(default_factory=dict)
    details: str = ""
