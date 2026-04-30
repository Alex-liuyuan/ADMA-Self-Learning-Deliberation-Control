"""Configuration public exports with lazy loading for optional heavy modules."""

from __future__ import annotations

from debate_rl_v2.config.env import EnvConfig
from debate_rl_v2.config.rl import (
    PPOConfig,
    MADDPGConfig,
    ContinuousAgentConfig,
    NetworkConfig,
    HierarchicalConfig,
    CreditConfig,
    RewardConfig,
    EnhancedRewardTrainingConfig,
    SelfPlayTrainingConfig,
    CurriculumTrainingConfig,
    TrainingConfig,
    StrategyBridgeConfig,
    ModeConfig,
    OnlineUpdateConfig,
    CausalConfig,
    PromptEvolutionConfig,
)
from debate_rl_v2.config.llm import LLMConfig, TextDebateConfig
from debate_rl_v2.config.mechanisms import (
    AdversarialConfig,
    KnowledgeConfig,
    SoftSwitchConfig,
    DevilAdvocateConfig,
)

__all__ = [
    "Config",
    "load_config",
    "EnvConfig",
    "PPOConfig",
    "MADDPGConfig",
    "ContinuousAgentConfig",
    "NetworkConfig",
    "HierarchicalConfig",
    "CreditConfig",
    "RewardConfig",
    "EnhancedRewardTrainingConfig",
    "SelfPlayTrainingConfig",
    "CurriculumTrainingConfig",
    "TrainingConfig",
    "StrategyBridgeConfig",
    "ModeConfig",
    "OnlineUpdateConfig",
    "CausalConfig",
    "PromptEvolutionConfig",
    "LLMConfig",
    "TextDebateConfig",
    "AdversarialConfig",
    "KnowledgeConfig",
    "SoftSwitchConfig",
    "DevilAdvocateConfig",
]


def __getattr__(name: str):
    if name in {"Config", "load_config"}:
        from debate_rl_v2.config.master import Config, load_config

        exports = {
            "Config": Config,
            "load_config": load_config,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
