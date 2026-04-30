"""Skill Extractor — automatically extracts debate strategies from trajectories.

After a complex debate (5+ rounds to consensus), analyzes the trajectory
to identify effective patterns and creates reusable DebateSkill objects.
"""

from __future__ import annotations

from typing import Any

from debate_rl_v2.logging_config import get_logger
from debate_rl_v2.skills.skill_manager import DebateSkill, SkillManager

logger = get_logger("skills.extractor")


class SkillExtractor:
    """Extracts debate skills from completed debate trajectories.

    Criteria for extraction:
    - Consensus reached with quality >= threshold
    - Debate lasted >= min_rounds (complex enough to learn from)
    """

    def __init__(
        self,
        skill_manager: SkillManager,
        min_rounds: int = 5,
        min_quality: float = 0.7,
    ) -> None:
        self.skill_manager = skill_manager
        self.min_rounds = min_rounds
        self.min_quality = min_quality

    def try_extract(
        self,
        result: dict[str, Any],
        transcript: list[dict[str, Any]],
        topic: str = "",
        tags: list[str] | None = None,
        causal_extractor: Any = None,
    ) -> DebateSkill | None:
        """Try to extract a skill from a completed debate.

        Returns None if the debate doesn't meet extraction criteria.

        Parameters
        ----------
        result : dict
            Debate result.
        transcript : list[dict]
            Debate transcript.
        topic : str
            Debate topic.
        tags : list[str] | None
            Scenario tags.
        causal_extractor : CausalExtractor | None
            If provided, also extracts causal chains from the trajectory.
        """
        consensus = result.get("consensus_reached", False)
        quality = result.get("final_quality", 0.0)
        total_rounds = result.get("total_rounds", len(transcript))

        if not consensus or quality < self.min_quality or total_rounds < self.min_rounds:
            return None

        # Analyze trajectory for strategy patterns
        strategies = self._analyze_trajectory(transcript)

        skill = DebateSkill(
            name=self._generate_name(topic, tags),
            description=f"从「{topic}」辩论中提取的策略 (质量={quality:.2f}, {total_rounds}轮)",
            scenario_tags=tags or [],
            proposer_strategy=strategies.get("proposer", ""),
            challenger_strategy=strategies.get("challenger", ""),
            arbiter_strategy=strategies.get("arbiter", ""),
            coordinator_strategy=strategies.get("coordinator", ""),
            source_topic=topic,
            success_count=1,
            total_uses=1,
            avg_quality=quality,
        )

        self.skill_manager.add_skill(skill)
        logger.info("Extracted skill: %s (quality=%.2f)", skill.name, quality)

        # Extract causal chains if extractor provided
        if causal_extractor is not None and transcript:
            try:
                chains = causal_extractor.extract_from_trajectory(transcript, topic)
                if chains:
                    logger.info("Extracted %d causal chains from trajectory", len(chains))
            except Exception as e:
                logger.warning("Causal extraction failed: %s", e)

        return skill

    def _analyze_trajectory(self, transcript: list[dict[str, Any]]) -> dict[str, str]:
        """Analyze debate trajectory to extract per-role strategies."""
        strategies: dict[str, str] = {}

        if not transcript:
            return strategies

        # Find the turning point (biggest quality jump)
        qualities = [r.get("quality", r.get("quality_score", 0.5)) for r in transcript]
        max_jump_idx = 0
        max_jump = 0.0
        for i in range(1, len(qualities)):
            jump = qualities[i] - qualities[i - 1]
            if jump > max_jump:
                max_jump = jump
                max_jump_idx = i

        # Extract strategy from the turning point round
        if max_jump > 0.05 and max_jump_idx < len(transcript):
            tp = transcript[max_jump_idx]
            proposal = tp.get("proposal", "")[:200]
            challenge = tp.get("challenge", "")[:200]

            if proposal:
                strategies["proposer"] = f"关键转折策略: 在第{max_jump_idx+1}轮通过调整方案实现质量跃升"
            if challenge:
                strategies["challenger"] = f"有效挑战模式: 在第{max_jump_idx+1}轮的挑战促进了方案改进"

        # Analyze convergence pattern
        if len(qualities) >= 3:
            early_avg = sum(qualities[:len(qualities)//3]) / max(len(qualities)//3, 1)
            late_avg = sum(qualities[-len(qualities)//3:]) / max(len(qualities)//3, 1)
            if late_avg - early_avg > 0.15:
                strategies["coordinator"] = "渐进式质量提升策略: 前期允许充分对抗，后期引导收敛"

        return strategies

    @staticmethod
    def _generate_name(topic: str, tags: list[str] | None) -> str:
        """Generate a skill name from topic and tags."""
        parts = []
        if tags:
            parts.extend(tags[:2])
        if topic:
            # Take first few meaningful words
            words = topic.split()[:4]
            parts.extend(words)
        name = "-".join(parts) if parts else "unnamed-skill"
        return name[:60]
