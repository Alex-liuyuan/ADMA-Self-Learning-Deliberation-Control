"""Framework-level knowledge adapter interface.

This keeps symbolic/neural-symbolic knowledge signals visible in the normal
GameScenario/GameEngine path without forcing every scenario to use the same
torch-backed KnowledgeEngine implementation.
"""

from __future__ import annotations

from typing import Any

from debate_rl_v2.framework.types import CollaborationState


class BaseKnowledgeAdapter:
    """Optional adapter that projects state/output into knowledge signals."""

    def reset(self) -> None:
        """Reset any episode-local buffers."""
        return None

    def before_round(
        self,
        *,
        state: CollaborationState,
        round_num: int,
    ) -> dict[str, Any]:
        """Return knowledge hints available before agents act."""
        return {}

    def after_round(
        self,
        *,
        state: CollaborationState,
        round_num: int,
        role_outputs: dict[str, dict[str, Any]],
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return knowledge/compliance signals after a round."""
        return {}
