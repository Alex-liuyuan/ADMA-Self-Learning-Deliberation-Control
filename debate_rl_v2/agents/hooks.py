"""Lifecycle Hooks — v2 rewrite with structured logging, no bare except.

Migrated from debate_rl/agents/hooks.py with improvements:
  - Uses debate_rl_v2.agents.protocol (no v1 import)
  - Structured logging replaces print()
  - Hook failures are logged, never silently swallowed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from debate_rl_v2.agents.protocol import AgentMessage
from debate_rl_v2.logging_config import get_logger

logger = get_logger("agents.hooks")


class HookPoint(str, Enum):
    """All available lifecycle hook points."""
    DEBATE_START = "debate_start"
    DEBATE_END = "debate_end"
    ROUND_START = "round_start"
    ROUND_END = "round_end"
    BEFORE_AGENT_ACT = "before_agent_act"
    AFTER_AGENT_ACT = "after_agent_act"
    BEFORE_LLM_CALL = "before_llm_call"
    AFTER_LLM_CALL = "after_llm_call"
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
    ADVERSARIAL_UPDATE = "adversarial_update"
    DEVIL_ADVOCATE_TRIGGER = "devil_advocate_trigger"
    CONSENSUS_CHECK = "consensus_check"
    HUMAN_REVIEW = "human_review"


@dataclass
class HookContext:
    """Context passed to hook callbacks."""
    hook_point: HookPoint
    round_num: int = 0
    role: str = ""
    message: Optional[AgentMessage] = None
    state: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    skip_action: bool = False
    override_result: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


HookCallback = Callable[[HookContext], None]


class HookManager:
    """Manages lifecycle hooks for debate environments.

    Usage::

        hooks = HookManager()
        hooks.register(HookPoint.BEFORE_AGENT_ACT, my_callback)
        ctx = hooks.trigger(HookPoint.BEFORE_AGENT_ACT, round_num=3, role="proposer")
        if ctx.skip_action:
            # Hook requested to skip this action
            pass
    """

    def __init__(self) -> None:
        self._hooks: Dict[HookPoint, List[HookCallback]] = {}

    def register(self, point: HookPoint, callback: HookCallback) -> None:
        self._hooks.setdefault(point, []).append(callback)

    def unregister(self, point: HookPoint, callback: HookCallback) -> None:
        callbacks = self._hooks.get(point, [])
        if callback in callbacks:
            callbacks.remove(callback)

    def trigger(self, point: HookPoint, **kwargs: Any) -> HookContext:
        """Trigger all hooks at a given point.

        Hook failures are logged as warnings, never silently swallowed.
        """
        ctx = HookContext(hook_point=point, **kwargs)
        for callback in self._hooks.get(point, []):
            try:
                callback(ctx)
            except Exception as e:
                logger.warning(
                    "Hook callback failed at %s: %s",
                    point.value, e,
                    exc_info=True,
                )
        return ctx

    def clear(self) -> None:
        self._hooks.clear()

    def has_hooks(self, point: HookPoint) -> bool:
        return bool(self._hooks.get(point))
