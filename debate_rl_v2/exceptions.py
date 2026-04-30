"""Structured exception hierarchy for debate_rl_v2.

All domain exceptions inherit from DebateError, enabling
fine-grained catch blocks and eliminating `except Exception: pass`.
"""

from __future__ import annotations


class DebateError(Exception):
    """Base exception for all debate_rl_v2 errors."""


# ── LLM layer ──

class LLMError(DebateError):
    """LLM call failed after all retries."""


class LLMParseError(LLMError):
    """LLM returned unparseable JSON / unexpected format."""


class LLMTimeoutError(LLMError):
    """LLM call timed out."""


class LLMRateLimitError(LLMError):
    """LLM provider rate-limited the request."""


# ── Tool layer ──

class ToolError(DebateError):
    """Tool execution failed."""


class ToolNotFoundError(ToolError):
    """Requested tool does not exist in registry."""

    def __init__(self, name: str, available: list[str] | None = None) -> None:
        self.name = name
        self.available = available or []
        msg = f"Unknown tool '{name}'"
        if self.available:
            msg += f". Available: {self.available}"
        super().__init__(msg)


class ToolValidationError(ToolError):
    """Tool input failed schema validation."""


# ── Reward / RL layer ──

class RewardError(DebateError):
    """Reward computation produced invalid values (NaN, Inf)."""


class TrainingError(DebateError):
    """RL training loop encountered an unrecoverable error."""


class DomainGapError(TrainingError):
    """Observation space mismatch between pretrain and fusion."""


# ── Config layer ──

class ConfigError(DebateError):
    """Configuration validation failed."""

    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"Config error in '{field}': {reason}")


# ── Environment layer ──

class GameEnvironmentError(DebateError):
    """Game environment encountered an invalid state."""


class ConsensusBlockedError(GameEnvironmentError):
    """Consensus was blocked by a hook or quality gate."""


# ── Memory / Skills layer ──

class GameMemoryError(DebateError):
    """Memory system I/O or capacity error."""


class SkillError(DebateError):
    """Skill loading or extraction failed."""
