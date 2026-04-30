"""Debate Event Emitter — Observer pattern replacing try/except Exception: pass.

All dashboard/visualization updates go through this emitter.
Listeners register for specific event types and receive structured data.
Failures in listeners are logged, never silently swallowed.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable

from debate_rl_v2.logging_config import get_logger

logger = get_logger("envs.events")


class DebateEventType(enum.Enum):
    """All possible debate events."""
    ROUND_START = "round_start"
    ROUND_END = "round_end"
    PROPOSAL = "proposal"
    CHALLENGE = "challenge"
    VERDICT = "verdict"
    COORDINATOR_ACTION = "coordinator_action"
    MODE_SWITCH = "mode_switch"
    DEVIL_ADVOCATE = "devil_advocate"
    CONSENSUS_CHECK = "consensus_check"
    CONSENSUS_REACHED = "consensus_reached"
    DEBATE_START = "debate_start"
    DEBATE_END = "debate_end"
    RL_SIGNALS = "rl_signals"
    METRICS_UPDATE = "metrics_update"


@dataclass
class DebateEvent:
    """A structured debate event."""
    event_type: DebateEventType
    round_num: int = 0
    data: dict[str, Any] = field(default_factory=dict)


# Listener type: (event) -> None
EventListener = Callable[[DebateEvent], None]


class DebateEventEmitter:
    """Central event bus for debate environments.

    Replaces scattered try/except Exception: pass blocks with
    a clean observer pattern. Dashboard, tracing, and hooks
    all register as listeners.

    Usage::

        emitter = DebateEventEmitter()
        emitter.on(DebateEventType.PROPOSAL, dashboard.on_proposal)
        emitter.on(DebateEventType.VERDICT, tracer.on_verdict)

        # In environment:
        emitter.emit(DebateEvent(
            event_type=DebateEventType.PROPOSAL,
            round_num=3,
            data={"content": "...", "confidence": 0.85},
        ))
    """

    def __init__(self) -> None:
        self._listeners: dict[DebateEventType, list[EventListener]] = {}
        self._global_listeners: list[EventListener] = []

    def on(self, event_type: DebateEventType, listener: EventListener) -> None:
        """Register a listener for a specific event type."""
        self._listeners.setdefault(event_type, []).append(listener)

    def on_all(self, listener: EventListener) -> None:
        """Register a listener for all events."""
        self._global_listeners.append(listener)

    def off(self, event_type: DebateEventType, listener: EventListener) -> None:
        """Remove a listener."""
        listeners = self._listeners.get(event_type, [])
        if listener in listeners:
            listeners.remove(listener)

    def emit(self, event: DebateEvent) -> None:
        """Emit an event to all registered listeners.

        Listener failures are logged as warnings, never silently swallowed.
        """
        # Type-specific listeners
        for listener in self._listeners.get(event.event_type, []):
            try:
                listener(event)
            except Exception as e:
                logger.warning(
                    "Event listener failed for %s: %s",
                    event.event_type.value, e,
                    exc_info=True,
                )

        # Global listeners
        for listener in self._global_listeners:
            try:
                listener(event)
            except Exception as e:
                logger.warning(
                    "Global event listener failed for %s: %s",
                    event.event_type.value, e,
                    exc_info=True,
                )

    def clear(self) -> None:
        """Remove all listeners."""
        self._listeners.clear()
        self._global_listeners.clear()


class DashboardAdapter:
    """Adapts the existing dashboard interface to the event emitter pattern.

    Wraps a dashboard object and translates events into dashboard method calls.
    """

    def __init__(self, dashboard: Any) -> None:
        self._dashboard = dashboard

    def connect(self, emitter: DebateEventEmitter) -> None:
        """Register all dashboard event handlers."""
        emitter.on(DebateEventType.ROUND_START, self._on_round_start)
        emitter.on(DebateEventType.PROPOSAL, self._on_dialogue)
        emitter.on(DebateEventType.CHALLENGE, self._on_dialogue)
        emitter.on(DebateEventType.VERDICT, self._on_dialogue)
        emitter.on(DebateEventType.COORDINATOR_ACTION, self._on_dialogue)
        emitter.on(DebateEventType.MODE_SWITCH, self._on_dialogue)
        emitter.on(DebateEventType.ROUND_END, self._on_round_end)

    def _on_round_start(self, event: DebateEvent) -> None:
        if hasattr(self._dashboard, "append_round_header"):
            self._dashboard.append_round_header(
                event.round_num,
                event.data.get("max_rounds", 10),
            )

    def _on_dialogue(self, event: DebateEvent) -> None:
        if hasattr(self._dashboard, "append_dialogue"):
            self._dashboard.append_dialogue(
                event.data.get("role", "system"),
                event.data.get("content", ""),
                meta=event.data.get("meta", ""),
            )

    def _on_round_end(self, event: DebateEvent) -> None:
        if hasattr(self._dashboard, "record_round"):
            self._dashboard.record_round(**event.data)
        if hasattr(self._dashboard, "update"):
            self._dashboard.update()
