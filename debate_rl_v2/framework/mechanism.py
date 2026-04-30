"""Framework-level mechanism orchestration abstractions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from debate_rl_v2.framework.types import CollaborationState


@dataclass
class MechanismSnapshot:
    """Generic mechanism state emitted after a round update."""

    values: dict[str, Any] = field(default_factory=dict)

    def apply_to_state(self, state: CollaborationState) -> None:
        agreement = self.values.get("agreement_level")
        if agreement is not None:
            state.agreement_level = float(agreement)

        disagreement = self.values.get("disagreement")
        if disagreement is not None:
            state.agreement_level = max(0.0, min(1.0, 1.0 - float(disagreement)))

        intensity = self.values.get("intensity", self.values.get("lambda_adv"))
        if intensity is not None:
            state.intensity = float(intensity)

        mode = self.values.get("mode")
        if mode is not None:
            state.mode = str(mode)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)


class BaseMechanismOrchestrator:
    """Optional mechanism layer that reacts to round outputs."""

    def reset(self) -> None:
        """Reset internal mechanism state for a new episode."""
        return None

    def update(
        self,
        *,
        state: CollaborationState,
        round_num: int,
        role_outputs: dict[str, dict[str, Any]],
        history: list[dict[str, Any]],
    ) -> MechanismSnapshot | dict[str, Any] | None:
        """Update mechanism state and optionally return a snapshot."""
        return None
