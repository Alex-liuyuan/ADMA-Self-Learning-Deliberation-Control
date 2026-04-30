"""Curriculum Learning — Adaptive Difficulty Progression.

Implements a curriculum learning system that gradually increases
debate difficulty based on agent performance. Key components:

  - **DifficultyLevel**: Defines what changes at each difficulty
  - **CurriculumScheduler**: Auto-progression and regression logic
  - **AdaptiveCurriculum**: Dynamic difficulty based on recent performance

Difficulty Dimensions::

    Level 1 (Beginner):
      - 2-3 simple, non-conflicting rules
      - 5 rounds max
      - Low adversarial intensity (λ=0.2)
      - No dynamic rules

    Level 2 (Intermediate):
      - 5-6 rules, some conflicting
      - 8 rounds max
      - Medium adversarial intensity (λ=0.4)
      - Mild rule dynamics

    Level 3 (Advanced):
      - 10+ rules, frequent conflicts
      - 12 rounds max
      - High adversarial intensity (λ=0.6)
      - Dynamic rule changes

    Level 4 (Expert):
      - 15+ complex rules
      - 15 rounds max
      - Adversarial + Devil's advocate
      - Full dynamic rules + rule mutations

    Level 5 (Adversarial):
      - All mechanisms at maximum
      - Opponent from self-play pool
      - Time pressure + token budget constraints
"""

from __future__ import annotations
from debate_rl_v2.logging_config import get_logger
logger = get_logger("algorithms.curriculum")

import json
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════
# Difficulty Level Definition
# ══════════════════════════════════════════════════════════


@dataclass
class DifficultyLevel:
    """Definition of a difficulty level for curriculum learning.

    Each field controls an aspect of the debate environment
    that changes with difficulty.
    """

    level: int
    name: str
    description: str

    # Environment parameters
    max_rounds: int = 5
    rule_count: int = 3
    max_steps: int = 20
    allow_conflicting_rules: bool = False
    dynamic_rules: bool = False
    rule_mutation_rate: float = 0.0

    # Adversarial parameters
    initial_lambda: float = 0.2
    max_lambda: float = 0.5
    enable_devil_advocate: bool = False
    devil_advocate_threshold: float = 0.2

    # Soft switch
    tau_low: float = 0.3
    tau_high: float = 0.7

    # Constraints
    token_budget: int = 0           # 0 = unlimited
    quality_gate: float = 0.0       # 0 = disabled

    # Self-play
    use_self_play: bool = False
    opponent_strength: str = "any"  # any|weak|matched|strong

    # Reward scaling
    reward_scale: float = 1.0
    bonus_for_efficiency: float = 0.0

    def to_env_params(self) -> Dict[str, Any]:
        """Convert to environment initialization parameters."""
        return {
            "max_rounds": self.max_rounds,
            "rule_count": self.rule_count,
            "max_steps": self.max_steps,
            "initial_lambda": self.initial_lambda,
            "enable_devil_advocate": self.enable_devil_advocate,
            "token_budget": self.token_budget,
            "quality_gate": self.quality_gate,
        }

    def to_mechanism_params(self) -> Dict[str, Any]:
        """Convert to mechanism-specific parameters."""
        return {
            "eta": 0.1 + self.level * 0.05,
            "alpha": 0.4 + self.level * 0.1,
            "tau_low": self.tau_low,
            "tau_high": self.tau_high,
            "devil_advocate_threshold": self.devil_advocate_threshold,
        }


# Pre-defined difficulty levels
DIFFICULTY_LEVELS: Dict[int, DifficultyLevel] = {
    1: DifficultyLevel(
        level=1,
        name="入门",
        description="基础辩论：简单规则，少轮次，低对抗",
        max_rounds=5,
        rule_count=3,
        max_steps=20,
        allow_conflicting_rules=False,
        dynamic_rules=False,
        initial_lambda=0.2,
        max_lambda=0.4,
        enable_devil_advocate=False,
        reward_scale=1.0,
    ),
    2: DifficultyLevel(
        level=2,
        name="进阶",
        description="中等复杂度：规则冲突，8轮辩论",
        max_rounds=8,
        rule_count=6,
        max_steps=30,
        allow_conflicting_rules=True,
        dynamic_rules=False,
        initial_lambda=0.3,
        max_lambda=0.6,
        enable_devil_advocate=False,
        tau_low=0.25,
        tau_high=0.65,
        reward_scale=1.2,
    ),
    3: DifficultyLevel(
        level=3,
        name="高级",
        description="高复杂度：动态规则，强对抗，12轮辩论",
        max_rounds=12,
        rule_count=10,
        max_steps=40,
        allow_conflicting_rules=True,
        dynamic_rules=True,
        rule_mutation_rate=0.1,
        initial_lambda=0.4,
        max_lambda=0.7,
        enable_devil_advocate=True,
        devil_advocate_threshold=0.25,
        tau_low=0.2,
        tau_high=0.6,
        reward_scale=1.5,
        bonus_for_efficiency=0.1,
    ),
    4: DifficultyLevel(
        level=4,
        name="专家",
        description="全机制：15轮，规则突变，时间压力",
        max_rounds=15,
        rule_count=15,
        max_steps=50,
        allow_conflicting_rules=True,
        dynamic_rules=True,
        rule_mutation_rate=0.2,
        initial_lambda=0.5,
        max_lambda=0.8,
        enable_devil_advocate=True,
        devil_advocate_threshold=0.2,
        tau_low=0.15,
        tau_high=0.55,
        token_budget=100000,
        quality_gate=0.6,
        reward_scale=2.0,
        bonus_for_efficiency=0.2,
    ),
    5: DifficultyLevel(
        level=5,
        name="对抗",
        description="极限挑战：自博弈对手，全部约束启用",
        max_rounds=15,
        rule_count=20,
        max_steps=60,
        allow_conflicting_rules=True,
        dynamic_rules=True,
        rule_mutation_rate=0.3,
        initial_lambda=0.6,
        max_lambda=0.9,
        enable_devil_advocate=True,
        devil_advocate_threshold=0.15,
        tau_low=0.1,
        tau_high=0.5,
        token_budget=80000,
        quality_gate=0.7,
        use_self_play=True,
        opponent_strength="matched",
        reward_scale=2.5,
        bonus_for_efficiency=0.3,
    ),
}


# ══════════════════════════════════════════════════════════
# Curriculum Scheduler
# ══════════════════════════════════════════════════════════


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""

    enabled: bool = True
    # Level management
    initial_level: int = 1
    max_level: int = 5
    # Promotion criteria
    promotion_threshold: float = 0.80     # quality above this → promote
    promotion_window: int = 20            # evaluate over last N episodes
    min_episodes_per_level: int = 100     # minimum episodes before promotion
    # Demotion criteria
    demotion_threshold: float = 0.50      # quality below this → demote
    demotion_window: int = 10             # evaluate over last N episodes
    allow_demotion: bool = True           # whether to allow level decrease
    # Mixed difficulty
    exploration_rate: float = 0.1         # fraction of episodes at random level
    # Persistence
    save_path: str = ""


class CurriculumScheduler:
    """Manages curriculum progression based on agent performance.

    Automatically promotes agents to harder difficulty when they
    consistently perform well, and optionally demotes when struggling.

    Usage::

        scheduler = CurriculumScheduler(config)

        for episode in range(total):
            level = scheduler.get_current_level()
            difficulty = scheduler.get_difficulty()
            # ... run episode at this difficulty ...
            scheduler.record_result(quality=0.85, consensus=True)

            if scheduler.should_promote():
                scheduler.promote()
                logger.info("Promoted to level %d!", scheduler.current_level)
    """

    def __init__(self, cfg: CurriculumConfig):
        self.cfg = cfg
        self._current_level = cfg.initial_level
        self._episodes_at_level = 0
        self._total_episodes = 0
        self._quality_history: deque = deque(maxlen=max(
            cfg.promotion_window, cfg.demotion_window
        ))
        self._consensus_history: deque = deque(maxlen=cfg.promotion_window)
        self._level_history: List[Tuple[int, int]] = []  # (episode, level)
        self._promotion_count = 0
        self._demotion_count = 0

    @property
    def current_level(self) -> int:
        return self._current_level

    def get_difficulty(self) -> DifficultyLevel:
        """Get the DifficultyLevel for the current level."""
        # Exploration: occasionally try a random level
        if (self.cfg.exploration_rate > 0
                and random.random() < self.cfg.exploration_rate):
            rand_level = random.randint(
                max(1, self._current_level - 1),
                min(self.cfg.max_level, self._current_level + 1),
            )
            return DIFFICULTY_LEVELS.get(
                rand_level,
                DIFFICULTY_LEVELS[self._current_level],
            )

        return DIFFICULTY_LEVELS.get(
            self._current_level,
            DIFFICULTY_LEVELS[1],
        )

    def record_result(
        self,
        quality: float,
        consensus: bool = False,
        extra: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record episode result for curriculum tracking."""
        self._quality_history.append(quality)
        self._consensus_history.append(float(consensus))
        self._episodes_at_level += 1
        self._total_episodes += 1

    def should_promote(self) -> bool:
        """Check if agent should advance to next difficulty level."""
        if self._current_level >= self.cfg.max_level:
            return False
        if self._episodes_at_level < self.cfg.min_episodes_per_level:
            return False
        if len(self._quality_history) < self.cfg.promotion_window:
            return False

        recent = list(self._quality_history)[-self.cfg.promotion_window:]
        avg_quality = np.mean(recent)
        consensus_rate = np.mean(
            list(self._consensus_history)[-self.cfg.promotion_window:]
        )

        return (avg_quality >= self.cfg.promotion_threshold
                and consensus_rate >= 0.5)

    def should_demote(self) -> bool:
        """Check if agent should fall back to easier difficulty."""
        if not self.cfg.allow_demotion:
            return False
        if self._current_level <= 1:
            return False
        if self._episodes_at_level < 20:
            return False
        if len(self._quality_history) < self.cfg.demotion_window:
            return False

        recent = list(self._quality_history)[-self.cfg.demotion_window:]
        avg_quality = np.mean(recent)
        return avg_quality < self.cfg.demotion_threshold

    def promote(self) -> int:
        """Advance to next difficulty level."""
        if self._current_level < self.cfg.max_level:
            self._current_level += 1
            self._episodes_at_level = 0
            self._promotion_count += 1
            self._level_history.append((self._total_episodes, self._current_level))
        return self._current_level

    def demote(self) -> int:
        """Fall back to previous difficulty level."""
        if self._current_level > 1:
            self._current_level -= 1
            self._episodes_at_level = 0
            self._demotion_count += 1
            self._level_history.append((self._total_episodes, self._current_level))
        return self._current_level

    def auto_adjust(self) -> Optional[str]:
        """Automatically adjust level based on performance.

        Returns
        -------
        action : str or None
            'promote', 'demote', or None.
        """
        if self.should_promote():
            self.promote()
            return "promote"
        elif self.should_demote():
            self.demote()
            return "demote"
        return None

    def summary(self) -> Dict[str, Any]:
        """Get curriculum training summary."""
        recent_quality = (
            float(np.mean(self._quality_history))
            if self._quality_history else 0.0
        )
        return {
            "current_level": self._current_level,
            "level_name": DIFFICULTY_LEVELS.get(
                self._current_level, DIFFICULTY_LEVELS[1]
            ).name,
            "episodes_at_level": self._episodes_at_level,
            "total_episodes": self._total_episodes,
            "recent_avg_quality": recent_quality,
            "promotions": self._promotion_count,
            "demotions": self._demotion_count,
            "level_history": self._level_history[-10:],
        }

    def save(self, path: str) -> None:
        """Persist curriculum state."""
        data = {
            "current_level": self._current_level,
            "episodes_at_level": self._episodes_at_level,
            "total_episodes": self._total_episodes,
            "promotion_count": self._promotion_count,
            "demotion_count": self._demotion_count,
            "level_history": self._level_history,
            "quality_history": list(self._quality_history),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load curriculum state."""
        if not Path(path).exists():
            return
        with open(path, "r") as f:
            data = json.load(f)
        self._current_level = data.get("current_level", self.cfg.initial_level)
        self._episodes_at_level = data.get("episodes_at_level", 0)
        self._total_episodes = data.get("total_episodes", 0)
        self._promotion_count = data.get("promotion_count", 0)
        self._demotion_count = data.get("demotion_count", 0)
        self._level_history = data.get("level_history", [])
        for q in data.get("quality_history", []):
            self._quality_history.append(q)


# ══════════════════════════════════════════════════════════
# Rule Generators for Each Difficulty
# ══════════════════════════════════════════════════════════


# Pre-built rule sets for MDT debate scenarios
RULE_TEMPLATES: Dict[int, List[str]] = {
    1: [
        "方案必须包含具体的实施时间表",
        "方案必须考虑成本效益",
        "方案必须符合基本安全标准",
    ],
    2: [
        "方案必须包含具体的实施时间表",
        "方案必须考虑成本效益",
        "方案必须符合基本安全标准",
        "方案必须有明确的风险评估",
        "方案必须考虑利益相关方意见",
        "方案在紧急情况下可适当简化流程（与时间表冲突）",
    ],
    3: [
        "方案必须包含具体的实施时间表",
        "方案必须考虑成本效益",
        "方案必须符合基本安全标准",
        "方案必须有明确的风险评估",
        "方案必须考虑利益相关方意见",
        "方案在紧急情况下可适当简化流程",
        "方案必须有量化的预期效果指标",
        "方案必须包含备选方案",
        "所有数据引用必须有明确来源",
        "方案必须通过伦理审查",
    ],
    4: [
        "方案必须包含具体的实施时间表",
        "方案必须考虑成本效益",
        "方案必须符合基本安全标准",
        "方案必须有明确的风险评估",
        "方案必须考虑利益相关方意见",
        "方案在紧急情况下可适当简化流程",
        "方案必须有量化的预期效果指标",
        "方案必须包含备选方案",
        "所有数据引用必须有明确来源",
        "方案必须通过伦理审查",
        "成本不得超过预算上限的120%",
        "实施周期不得超过既定时间的150%",
        "风险指标必须低于行业平均水平",
        "必须有至少2个独立的验证数据源",
        "方案修改不得降低已达标的指标",
    ],
    5: [
        "方案必须包含具体的实施时间表",
        "方案必须考虑成本效益",
        "方案必须符合基本安全标准",
        "方案必须有明确的风险评估",
        "方案必须考虑利益相关方意见",
        "方案在紧急情况下可适当简化流程",
        "方案必须有量化的预期效果指标",
        "方案必须包含备选方案",
        "所有数据引用必须有明确来源",
        "方案必须通过伦理审查",
        "成本不得超过预算上限的120%",
        "实施周期不得超过既定时间的150%",
        "风险指标必须低于行业平均水平",
        "必须有至少2个独立的验证数据源",
        "方案修改不得降低已达标的指标",
        "创新性指标必须达到同类方案前30%",
        "长期影响评估周期不少于5年",
        "必须考虑极端情况下的应急方案",
        "关键指标的置信区间需达到95%",
        "方案需获得至少3位专家的独立认可",
    ],
}


def get_rules_for_level(level: int, dynamic: bool = False) -> List[str]:
    """Get rule set appropriate for a difficulty level.

    Parameters
    ----------
    level : int
        Difficulty level (1-5).
    dynamic : bool
        If True, randomly shuffle and subset rules for variety.

    Returns
    -------
    rules : list of str
    """
    base_rules = RULE_TEMPLATES.get(level, RULE_TEMPLATES[1])
    if dynamic:
        # Randomly drop some rules and shuffle for variety
        n_keep = max(2, int(len(base_rules) * random.uniform(0.7, 1.0)))
        rules = random.sample(base_rules, n_keep)
        random.shuffle(rules)
        return rules
    return list(base_rules)
