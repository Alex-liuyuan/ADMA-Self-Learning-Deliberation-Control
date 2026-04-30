"""Observer hooks for GameEngine execution.

Keeps instrumentation, dashboards, and side effects out of the core loop.
"""

from __future__ import annotations

from typing import Any

from debate_rl_v2.framework.types import CollaborationState


class BaseGameObserver:
    """No-op observer that can subscribe to GameEngine lifecycle events."""

    def should_stop(
        self,
        round_num: int,
        state: CollaborationState,
    ) -> bool:
        return False

    def on_episode_start(
        self,
        episode_context: dict[str, Any],
        state: CollaborationState,
    ) -> None:
        pass

    def on_round_start(
        self,
        round_num: int,
        max_rounds: int,
        state: CollaborationState,
    ) -> None:
        pass

    def on_role_output(
        self,
        role_name: str,
        output: dict[str, Any],
        round_num: int,
        stage: str = "",
        meta: dict[str, Any] | None = None,
    ) -> None:
        pass

    def on_knowledge_updated(
        self,
        state: CollaborationState,
        knowledge_state: dict[str, Any],
        round_num: int,
        role_outputs: dict[str, dict[str, Any]],
    ) -> None:
        pass

    def on_state_updated(
        self,
        state: CollaborationState,
        round_num: int,
        role_outputs: dict[str, dict[str, Any]],
    ) -> None:
        pass

    def on_mechanism_updated(
        self,
        state: CollaborationState,
        mechanism_state: Any,
        round_num: int,
        role_outputs: dict[str, dict[str, Any]],
    ) -> None:
        pass

    def on_round_end(
        self,
        round_num: int,
        state: CollaborationState,
        role_outputs: dict[str, dict[str, Any]],
    ) -> None:
        pass

    def on_episode_end(
        self,
        result: dict[str, Any],
        state: CollaborationState,
    ) -> None:
        pass
