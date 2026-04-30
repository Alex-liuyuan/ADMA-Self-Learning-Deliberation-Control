"""Structured Message Protocol — inspired by MetaGPT's typed message system.

Provides strongly-typed, validated message passing between agents.
All inter-agent communication goes through ``AgentMessage`` objects
with schema enforcement, routing metadata, and conversation threading.

Key ideas borrowed:
  - **MetaGPT**: Typed messages with cause_by / send_to routing
  - **AutoGen**: Conversation thread management
  - **CAMEL**: Role-tagged inception prompting
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MessageType(str, Enum):
    """Enumeration of all structured message types."""

    # Core debate actions
    PROPOSAL = "proposal"
    CHALLENGE = "challenge"
    VERDICT = "verdict"
    META_ACTION = "meta_action"

    # System control
    SYSTEM = "system"
    DEVIL_ADVOCATE = "devil_advocate"
    RULE_UPDATE = "rule_update"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"

    # Human-in-the-loop
    HUMAN_INPUT = "human_input"
    HUMAN_OVERRIDE = "human_override"

    # Lifecycle
    RESET = "reset"
    TERMINATE = "terminate"


@dataclass
class AgentMessage:
    """Typed, validated message for inter-agent communication.

    Attributes
    ----------
    msg_type : MessageType
        Category of the message (proposal, challenge, verdict, etc.).
    sender : str
        Role name of the sending agent.
    content : str
        Free-text content of the message.
    data : dict
        Structured payload (role-specific parsed fields).
    round_num : int
        Debate round this message belongs to.
    msg_id : str
        Unique message identifier (auto-generated).
    parent_id : str
        ID of the message this is responding to (for threading).
    timestamp : float
        Unix timestamp of creation.
    metadata : dict
        Arbitrary extra metadata (tool results, confidence, etc.).
    """

    msg_type: MessageType
    sender: str
    content: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    round_num: int = 0
    msg_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def proposal(cls, sender: str, content: str, confidence: float = 0.5,
                 round_num: int = 0, **data: Any) -> "AgentMessage":
        return cls(
            msg_type=MessageType.PROPOSAL,
            sender=sender,
            content=content,
            data={"confidence": confidence, **data},
            round_num=round_num,
        )

    @classmethod
    def challenge(cls, sender: str, content: str, confidence: float = 0.5,
                  round_num: int = 0, **data: Any) -> "AgentMessage":
        return cls(
            msg_type=MessageType.CHALLENGE,
            sender=sender,
            content=content,
            data={"confidence": confidence, **data},
            round_num=round_num,
        )

    @classmethod
    def verdict(cls, sender: str, content: str,
                quality_score: float = 0.5, round_num: int = 0,
                **data: Any) -> "AgentMessage":
        return cls(
            msg_type=MessageType.VERDICT,
            sender=sender,
            content=content,
            data={"quality_score": quality_score, **data},
            round_num=round_num,
        )

    @classmethod
    def meta_action(cls, sender: str, action: int, reasoning: str = "",
                    round_num: int = 0) -> "AgentMessage":
        return cls(
            msg_type=MessageType.META_ACTION,
            sender=sender,
            content=reasoning,
            data={"action": action},
            round_num=round_num,
        )

    @classmethod
    def tool_call(cls, sender: str, tool_name: str, tool_input: Dict,
                  round_num: int = 0) -> "AgentMessage":
        return cls(
            msg_type=MessageType.TOOL_CALL,
            sender=sender,
            content=f"调用工具: {tool_name}",
            data={"tool_name": tool_name, "tool_input": tool_input},
            round_num=round_num,
        )

    @classmethod
    def tool_result(cls, tool_name: str, result: Any,
                    parent_id: str = "", round_num: int = 0) -> "AgentMessage":
        return cls(
            msg_type=MessageType.TOOL_RESULT,
            sender="system",
            content=str(result),
            data={"tool_name": tool_name, "result": result},
            parent_id=parent_id,
            round_num=round_num,
        )

    @classmethod
    def human_input(cls, content: str, round_num: int = 0) -> "AgentMessage":
        return cls(
            msg_type=MessageType.HUMAN_INPUT,
            sender="human",
            content=content,
            round_num=round_num,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "msg_type": self.msg_type.value,
            "sender": self.sender,
            "content": self.content,
            "data": self.data,
            "round_num": self.round_num,
            "msg_id": self.msg_id,
            "parent_id": self.parent_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentMessage":
        return cls(
            msg_type=MessageType(d["msg_type"]),
            sender=d["sender"],
            content=d.get("content", ""),
            data=d.get("data", {}),
            round_num=d.get("round_num", 0),
            msg_id=d.get("msg_id", uuid.uuid4().hex[:12]),
            parent_id=d.get("parent_id", ""),
            timestamp=d.get("timestamp", time.time()),
            metadata=d.get("metadata", {}),
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Return list of validation errors (empty = valid)."""
        errors: List[str] = []

        if self.msg_type == MessageType.PROPOSAL:
            if not self.content:
                errors.append("Proposal content cannot be empty")

        elif self.msg_type == MessageType.CHALLENGE:
            if not self.content:
                errors.append("Challenge content cannot be empty")

        elif self.msg_type == MessageType.VERDICT:
            qs = self.data.get("quality_score")
            if qs is not None and not (0 <= qs <= 1):
                errors.append(f"quality_score must be in [0,1], got {qs}")

        elif self.msg_type == MessageType.META_ACTION:
            action = self.data.get("action")
            if action is not None and not (0 <= action <= 9):
                errors.append(f"Meta action must be in [0,9], got {action}")

        elif self.msg_type == MessageType.TOOL_CALL:
            if "tool_name" not in self.data:
                errors.append("Tool call must include tool_name")

        return errors

    def __repr__(self) -> str:
        return (
            f"AgentMessage({self.msg_type.value}, "
            f"sender={self.sender!r}, "
            f"round={self.round_num}, "
            f"content={self.content[:50]!r}...)"
        )


# ======================================================================
# Conversation Thread
# ======================================================================


class ConversationThread:
    """Thread-safe conversation history with filtering and search.

    Inspired by AutoGen's conversation management — supports
    filtering by role, type, round, and parent threading.
    """

    def __init__(self) -> None:
        self._messages: List[AgentMessage] = []
        self._index: Dict[str, AgentMessage] = {}  # msg_id → message

    def add(self, msg: AgentMessage) -> None:
        self._messages.append(msg)
        self._index[msg.msg_id] = msg

    def get_by_id(self, msg_id: str) -> Optional[AgentMessage]:
        return self._index.get(msg_id)

    def get_thread(self, msg_id: str) -> List[AgentMessage]:
        """Get all messages in a reply chain starting from msg_id."""
        chain = []
        current = self.get_by_id(msg_id)
        while current:
            chain.append(current)
            current = self.get_by_id(current.parent_id) if current.parent_id else None
        return list(reversed(chain))

    def filter(
        self,
        sender: Optional[str] = None,
        msg_type: Optional[MessageType] = None,
        round_num: Optional[int] = None,
        last_n: Optional[int] = None,
    ) -> List[AgentMessage]:
        """Filter messages by criteria."""
        result = self._messages
        if sender:
            result = [m for m in result if m.sender == sender]
        if msg_type:
            result = [m for m in result if m.msg_type == msg_type]
        if round_num is not None:
            result = [m for m in result if m.round_num == round_num]
        if last_n:
            result = result[-last_n:]
        return result

    def summary(self, last_n: int = 5) -> str:
        """Generate a text summary of recent conversation."""
        recent = self._messages[-last_n:]
        lines = []
        for m in recent:
            lines.append(
                f"[R{m.round_num}|{m.sender}|{m.msg_type.value}] "
                f"{m.content[:80]}"
            )
        return "\n".join(lines)

    def clear(self) -> None:
        self._messages.clear()
        self._index.clear()

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self):
        return iter(self._messages)
