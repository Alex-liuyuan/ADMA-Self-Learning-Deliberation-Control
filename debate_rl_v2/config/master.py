"""Master configuration — aggregates all sub-configs with YAML loading."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import yaml
try:
    import torch
except ImportError:  # pragma: no cover - optional for non-training config use
    torch = None

from debate_rl_v2.config.env import EnvConfig
from debate_rl_v2.config.rl import (
    CausalConfig,
    ContinuousAgentConfig,
    CreditConfig,
    CurriculumTrainingConfig,
    EnhancedRewardTrainingConfig,
    HierarchicalConfig,
    MADDPGConfig,
    ModeConfig,
    NetworkConfig,
    OnlineUpdateConfig,
    PPOConfig,
    PromptEvolutionConfig,
    RewardConfig,
    SelfPlayTrainingConfig,
    StrategyBridgeConfig,
    TrainingConfig,
)
from debate_rl_v2.config.llm import LLMConfig, TextDebateConfig
from debate_rl_v2.config.mechanisms import (
    AdversarialConfig,
    DevilAdvocateConfig,
    KnowledgeConfig,
    SoftSwitchConfig,
)


# Checkpoint version for migration safety
CHECKPOINT_VERSION = "2.0.0"


@dataclass
class Config:
    seed: int = 42
    device: str = "auto"

    env: EnvConfig = field(default_factory=EnvConfig)
    adversarial: AdversarialConfig = field(default_factory=AdversarialConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    soft_switch: SoftSwitchConfig = field(default_factory=SoftSwitchConfig)
    devil_advocate: DevilAdvocateConfig = field(default_factory=DevilAdvocateConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    continuous_agent: ContinuousAgentConfig = field(default_factory=ContinuousAgentConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    hierarchical: HierarchicalConfig = field(default_factory=HierarchicalConfig)
    credit: CreditConfig = field(default_factory=CreditConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    text_debate: TextDebateConfig = field(default_factory=TextDebateConfig)
    maddpg: MADDPGConfig = field(default_factory=MADDPGConfig)
    strategy_bridge: StrategyBridgeConfig = field(default_factory=StrategyBridgeConfig)
    self_play: SelfPlayTrainingConfig = field(default_factory=SelfPlayTrainingConfig)
    curriculum: CurriculumTrainingConfig = field(default_factory=CurriculumTrainingConfig)
    enhanced_reward: EnhancedRewardTrainingConfig = field(default_factory=EnhancedRewardTrainingConfig)
    mode: ModeConfig = field(default_factory=ModeConfig)
    online_update: OnlineUpdateConfig = field(default_factory=OnlineUpdateConfig)
    causal: CausalConfig = field(default_factory=CausalConfig)
    prompt_evolution: PromptEvolutionConfig = field(default_factory=PromptEvolutionConfig)

    def resolve_device(self):
        if torch is None:
            if self.device in {"auto", "cpu"}:
                return "cpu"
            raise ImportError("resolve_device requires torch for non-CPU devices")
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def architecture_hash(self) -> str:
        """Compute a hash of architecture-relevant config for checkpoint compatibility."""
        arch_keys = {
            "network": asdict(self.network),
            "env": {"context_dim": self.env.context_dim, "proposal_dim": self.env.proposal_dim},
            "maddpg": {"actor_hidden_dim": self.maddpg.actor_hidden_dim,
                       "critic_hidden_dim": self.maddpg.critic_hidden_dim},
        }
        raw = json.dumps(arch_keys, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _update_dataclass(dc: Any, data: Dict[str, Any]) -> None:
    """Recursively update a dataclass from a dict."""
    for key, val in data.items():
        if hasattr(dc, key):
            attr = getattr(dc, key)
            if hasattr(attr, "__dataclass_fields__") and isinstance(val, dict):
                _update_dataclass(attr, val)
            else:
                setattr(dc, key, val)


def load_config(path: Optional[str] = None) -> Config:
    """Load config from YAML file, falling back to defaults.

    Supports ``${VAR}`` variable substitution from environment variables.
    Automatically loads ``.env`` file (via python-dotenv) if present.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    cfg = Config()
    if path is not None and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        def _resolve_env(m: re.Match) -> str:
            var_name = m.group(1)
            return os.environ.get(var_name, m.group(0))

        raw_text = re.sub(r"\$\{(\w+)\}", _resolve_env, raw_text)
        raw = yaml.safe_load(raw_text)
        if raw:
            _update_dataclass(cfg, raw)
    return cfg
