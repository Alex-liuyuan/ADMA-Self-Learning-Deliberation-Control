"""Message Bus & Blackboard — inspired by MetaGPT's shared environment.

Implements a publish/subscribe message routing system that decouples
agents from each other.  Each agent subscribes to message types
it cares about and publishes its outputs to the bus.

The **Blackboard** provides a shared state space where agents can
read/write structured data (MetaGPT's "Environment" pattern).

Key ideas borrowed:
  - **MetaGPT**: Environment as shared workspace + subscription routing
  - **AutoGen**: GroupChat with speaker selection
  - **Blackboard architecture**: AI classic pattern for shared problem solving
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from debate_rl_v2.agents.protocol import AgentMessage, MessageType


# ======================================================================
# Message Bus (Pub/Sub)
# ======================================================================


class MessageBus:
    """Publish/subscribe message bus for inter-agent communication.

    Agents subscribe to specific ``MessageType``s.  When a message
    is published, all matching subscribers receive it.
    """

    def __bool__(self) -> bool:
        return True

    def __init__(self) -> None:
        self._subscriptions: Dict[MessageType, List[Callable]] = defaultdict(list)
        self._global_subscribers: List[Callable] = []
        self._history: List[AgentMessage] = []
        self._filters: List[Callable[[AgentMessage], bool]] = []

    def subscribe(
        self,
        msg_type: MessageType,
        handler: Callable[[AgentMessage], None],
    ) -> None:
        """Subscribe to a specific message type."""
        self._subscriptions[msg_type].append(handler)

    def subscribe_all(self, handler: Callable[[AgentMessage], None]) -> None:
        """Subscribe to ALL message types (e.g. for logging)."""
        self._global_subscribers.append(handler)

    def add_filter(self, predicate: Callable[[AgentMessage], bool]) -> None:
        """Add a message filter.  Returning False blocks the message."""
        self._filters.append(predicate)

    def publish(self, msg: AgentMessage) -> bool:
        """Publish a message to the bus.

        Returns False if blocked by a filter.
        """
        # Run filters
        for f in self._filters:
            if not f(msg):
                return False

        self._history.append(msg)

        # Notify type-specific subscribers
        for handler in self._subscriptions.get(msg.msg_type, []):
            handler(msg)

        # Notify global subscribers
        for handler in self._global_subscribers:
            handler(msg)

        return True

    def get_history(
        self,
        msg_type: Optional[MessageType] = None,
        sender: Optional[str] = None,
        last_n: Optional[int] = None,
    ) -> List[AgentMessage]:
        result = self._history
        if msg_type:
            result = [m for m in result if m.msg_type == msg_type]
        if sender:
            result = [m for m in result if m.sender == sender]
        if last_n:
            result = result[-last_n:]
        return result

    def clear(self) -> None:
        self._history.clear()

    def __len__(self) -> int:
        return len(self._history)


# ======================================================================
# Blackboard (Shared State)
# ======================================================================


class Blackboard:
    """Shared workspace for multi-agent collaboration.

    A structured key-value store where agents can read/write
    debate state, intermediate results, and coordination signals.

    Inspired by MetaGPT's Environment and classic AI blackboard
    architecture.
    """

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}
        self._history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._watchers: Dict[str, List[Callable]] = defaultdict(list)

    def write(self, key: str, value: Any, writer: str = "") -> None:
        """Write a value, notifying any watchers."""
        old_value = self._data.get(key)
        self._data[key] = value
        self._history[key].append({
            "value": value,
            "writer": writer,
            "timestamp": time.time(),
        })
        # Notify watchers
        for handler in self._watchers.get(key, []):
            handler(key, value, old_value)

    def read(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def watch(self, key: str, handler: Callable[[str, Any, Any], None]) -> None:
        """Watch for changes to a key.

        Handler receives (key, new_value, old_value).
        """
        self._watchers[key].append(handler)

    def get_history(self, key: str) -> List[Dict[str, Any]]:
        """Get full write history for a key."""
        return self._history.get(key, [])

    def keys(self) -> List[str]:
        return list(self._data.keys())

    def snapshot(self) -> Dict[str, Any]:
        """Return a copy of all current values."""
        return dict(self._data)

    def to_context_string(self, keys: Optional[List[str]] = None) -> str:
        """Format blackboard state for LLM prompts."""
        target_keys = keys or list(self._data.keys())
        if not target_keys:
            return ""
        lines = ["## 共享状态"]
        for k in target_keys:
            v = self._data.get(k)
            if v is not None:
                # Truncate long values
                v_str = str(v)
                if len(v_str) > 200:
                    v_str = v_str[:200] + "..."
                lines.append(f"  - {k}: {v_str}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._data.clear()
        self._history.clear()

    def __bool__(self) -> bool:
        return True

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)


# ======================================================================
# Debate Workspace (Bus + Blackboard combined)
# ======================================================================


class DebateWorkspace:
    """Combined message bus + blackboard for debates.

    This is the central coordination point that replaces direct
    agent-to-agent communication.  All interaction flows through
    this workspace.
    """

    def __init__(self) -> None:
        self.bus = MessageBus()
        self.board = Blackboard()

        # Auto-update blackboard from certain message types
        self.bus.subscribe(MessageType.PROPOSAL, self._on_proposal)
        self.bus.subscribe(MessageType.CHALLENGE, self._on_challenge)
        self.bus.subscribe(MessageType.VERDICT, self._on_verdict)

    def _on_proposal(self, msg: AgentMessage) -> None:
        self.board.write("current_proposal", msg.content, writer=msg.sender)
        self.board.write("proposal_confidence",
                         msg.data.get("confidence", 0.5), writer=msg.sender)

    def _on_challenge(self, msg: AgentMessage) -> None:
        self.board.write("current_challenge", msg.content, writer=msg.sender)
        self.board.write("challenge_confidence",
                         msg.data.get("confidence", 0.5), writer=msg.sender)

    def _on_verdict(self, msg: AgentMessage) -> None:
        self.board.write("current_verdict", msg.content, writer=msg.sender)
        self.board.write("quality_score",
                         msg.data.get("quality_score", 0.5), writer=msg.sender)

    def reset(self) -> None:
        self.bus.clear()
        self.board.clear()
