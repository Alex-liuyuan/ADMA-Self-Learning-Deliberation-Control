"""ModeController — dual-mode switching core.

Configuration-driven decision point for training vs online mode.
No new abstractions — just a thin config wrapper that FusionDebateEnv
and TextDebateEnv query to decide behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from debate_rl_v2.logging_config import get_logger

if TYPE_CHECKING:
    from debate_rl_v2.config.rl import ModeConfig
    from debate_rl_v2.skills.skill_db import SkillDatabase

logger = get_logger("mode.controller")


class ModeController:
    """Dual-mode unified controller.

    Parameters
    ----------
    config : ModeConfig
        Mode configuration (mode, thresholds, intervals).
    skill_db : SkillDatabase | None
        SQLite skill database for online skill accumulation.
    """

    def __init__(
        self,
        config: ModeConfig,
        skill_db: SkillDatabase | None = None,
    ) -> None:
        self._config = config
        self._skill_db = skill_db
        self._episode_count = 0
        logger.info("ModeController initialized: mode=%s", config.mode)

    @property
    def mode(self) -> str:
        return self._config.mode

    @property
    def is_training(self) -> bool:
        return self._config.mode == "training"

    @property
    def is_online(self) -> bool:
        return self._config.mode == "online"

    def should_update_rl(self) -> bool:
        """Training mode updates RL weights; online mode freezes them."""
        return self.is_training

    def get_exploration_noise(self) -> float:
        """Training uses configured noise; online uses zero."""
        if self.is_online:
            return 0.0
        return self._config.exploration_noise

    def should_distill(self, quality: float) -> bool:
        """Whether to distill episode knowledge based on quality threshold."""
        return quality >= self._config.distill_quality_threshold

    def should_evolve_prompts(self, episode: int) -> bool:
        """Whether to run prompt evolution this episode."""
        if not self._config.enable_prompt_evolution:
            return False
        return episode % self._config.prompt_evolve_interval == 0

    def should_extract_skills(self) -> bool:
        """Whether skill extraction is enabled."""
        return self._config.enable_skill_extraction

    def should_extract_causal(self) -> bool:
        """Whether causal extraction is enabled."""
        return self._config.enable_causal_extraction

    def record_episode(self) -> None:
        self._episode_count += 1

    @property
    def episode_count(self) -> int:
        return self._episode_count
