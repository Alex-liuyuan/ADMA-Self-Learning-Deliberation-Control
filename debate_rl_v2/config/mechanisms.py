"""Core mechanism configurations with validation."""

from __future__ import annotations

from dataclasses import dataclass

from debate_rl_v2.exceptions import ConfigError


@dataclass
class AdversarialConfig:
    eta: float = 0.2
    alpha: float = 0.6
    omega: float = 0.7
    steepness: float = 10.0

    def __post_init__(self) -> None:
        if not 0 < self.eta <= 1:
            raise ConfigError("adversarial.eta", "must be in (0, 1]")
        if not 0 < self.alpha <= 1:
            raise ConfigError("adversarial.alpha", "must be in (0, 1]")
        if not 0 < self.omega <= 1:
            raise ConfigError("adversarial.omega", "must be in (0, 1]")


@dataclass
class KnowledgeConfig:
    initial_threshold: float = 0.0
    confidence_lr: float = 0.01
    mine_interval: int = 50
    max_mined_rules: int = 8
    ilp_min_samples: int = 100

    def __post_init__(self) -> None:
        if self.mine_interval <= 0:
            raise ConfigError("knowledge.mine_interval", "must be positive")
        if self.max_mined_rules <= 0:
            raise ConfigError("knowledge.max_mined_rules", "must be positive")


@dataclass
class SoftSwitchConfig:
    tau_low: float = 0.3
    tau_high: float = 0.7
    steepness: float = 10.0

    def __post_init__(self) -> None:
        if self.tau_low >= self.tau_high:
            raise ConfigError("soft_switch", "tau_low must be < tau_high")
        if self.steepness <= 0:
            raise ConfigError("soft_switch.steepness", "must be positive")


@dataclass
class DevilAdvocateConfig:
    disagreement_threshold: float = 0.2
    update_threshold: float = 0.1
    stability_window: int = 3
    reactivation_threshold: float = 0.35
    max_challenges: int = 3

    def __post_init__(self) -> None:
        if self.stability_window <= 0:
            raise ConfigError("devil_advocate.stability_window", "must be positive")
        if self.max_challenges <= 0:
            raise ConfigError("devil_advocate.max_challenges", "must be positive")
