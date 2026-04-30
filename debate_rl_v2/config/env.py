"""Environment configuration with validation."""

from __future__ import annotations

from dataclasses import dataclass

from debate_rl_v2.exceptions import ConfigError


@dataclass
class EnvConfig:
    name: str = "base"
    context_dim: int = 128
    proposal_dim: int = 32
    proposal_values: int = 20
    embed_dim: int = 512
    rule_count: int = 32
    max_steps: int = 50
    meta_interval: int = 5
    action_mode: str = "discrete"  # "discrete" | "hybrid"

    def __post_init__(self) -> None:
        if self.context_dim <= 0:
            raise ConfigError("env.context_dim", "must be positive")
        if self.proposal_dim <= 0:
            raise ConfigError("env.proposal_dim", "must be positive")
        if self.proposal_values <= 0:
            raise ConfigError("env.proposal_values", "must be positive")
        if self.max_steps <= 0:
            raise ConfigError("env.max_steps", "must be positive")
        if self.meta_interval <= 0:
            raise ConfigError("env.meta_interval", "must be positive")
        if self.action_mode not in ("discrete", "hybrid"):
            raise ConfigError("env.action_mode", f"must be 'discrete' or 'hybrid', got '{self.action_mode}'")
