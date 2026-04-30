"""Dynamic Rule Environment — Section 6.1 (Environment 3).

.. deprecated::
    Use GameEngine + GameScenario instead.
    See debate_rl_v2.framework.game_scenario for the base protocol.

An environment where rules change every N episodes, testing system
adaptability. This stresses the knowledge engine's online learning
and rule mining capabilities.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from debate_rl_v2.config import (
    AdversarialConfig,
    DevilAdvocateConfig,
    EnvConfig,
    KnowledgeConfig,
    RewardConfig,
    SoftSwitchConfig,
)
from debate_rl_v2.envs.base_env import DebateEnv


class DynamicRuleDebateEnv(DebateEnv):
    """Debate environment with periodically changing rules.

    Every `rule_change_interval` episodes, a fraction of rules are randomly
    modified, simulating evolving domain constraints.

    Parameters
    ----------
    rule_change_interval : int
        Episodes between rule changes.
    change_fraction : float
        Fraction of rules modified at each change (0, 1].
    """

    def __init__(
        self,
        env_cfg: Optional[EnvConfig] = None,
        adv_cfg: Optional[AdversarialConfig] = None,
        know_cfg: Optional[KnowledgeConfig] = None,
        ss_cfg: Optional[SoftSwitchConfig] = None,
        da_cfg: Optional[DevilAdvocateConfig] = None,
        rew_cfg: Optional[RewardConfig] = None,
        seed: int = 0,
        rule_change_interval: int = 100,
        change_fraction: float = 0.3,
    ) -> None:
        self.rule_change_interval = rule_change_interval
        self.change_fraction = change_fraction
        self._episode_counter = 0
        self._rules_changed = False
        super().__init__(env_cfg, adv_cfg, know_cfg, ss_cfg, da_cfg, rew_cfg, seed)

    def reset(self) -> Dict[str, np.ndarray]:
        self._episode_counter += 1
        self._rules_changed = False

        # Periodically perturb rules
        if self._episode_counter % self.rule_change_interval == 0:
            self._perturb_rules()
            self._rules_changed = True

        return super().reset()

    def _perturb_rules(self) -> None:
        """Randomly modify a fraction of rules in the knowledge engine."""
        ke = self.knowledge_engine
        num_to_change = max(1, int(len(ke.rules) * self.change_fraction))
        indices = self.rng.choice(len(ke.rules), size=num_to_change, replace=False)

        for idx in indices:
            rule = ke.rules[idx]
            with torch.no_grad():
                # Reinitialize the predicate network weights
                for param in rule.predicate.parameters():
                    torch.nn.init.normal_(param, mean=0.0, std=0.3)
                # Reset confidence to 0.5 (logit = 0)
                rule._confidence_logit.data.fill_(0.0)

    @property
    def rules_changed_this_episode(self) -> bool:
        return self._rules_changed
