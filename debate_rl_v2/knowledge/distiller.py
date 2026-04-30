"""EpisodeDistiller — episode-level knowledge distillation.

After each debate episode, distills successful strategies, causal chains,
and insights into the persistent knowledge stores (SkillDB, CausalGraph, Memory).
"""

from __future__ import annotations

from typing import Any

from debate_rl_v2.logging_config import get_logger

logger = get_logger("knowledge.distiller")


class EpisodeDistiller:
    """Distill episode outcomes into persistent knowledge.

    Orchestrates:
    1. SkillExtractor → SkillDB (debate strategies)
    2. CausalExtractor → CausalGraph + SkillDB (causal chains)
    3. MemoryManager → long-term insights

    Parameters
    ----------
    skill_db : SkillDatabase | None
        SQLite skill storage.
    causal_graph : CausalGraph | None
        Causal relation graph.
    memory : MemoryManager | None
        Long-term memory manager.
    skill_extractor : SkillExtractor | None
        Debate skill extractor.
    causal_extractor : CausalExtractor | None
        LLM-based causal extractor.
    min_quality : float
        Minimum quality threshold for distillation.
    """

    def __init__(
        self,
        skill_db: Any = None,
        causal_graph: Any = None,
        memory: Any = None,
        skill_extractor: Any = None,
        causal_extractor: Any = None,
        min_quality: float = 0.6,
    ) -> None:
        self._skill_db = skill_db
        self._causal_graph = causal_graph
        self._memory = memory
        self._skill_extractor = skill_extractor
        self._causal_extractor = causal_extractor
        self._min_quality = min_quality

    def distill(
        self,
        result: dict[str, Any],
        transcript: list[dict[str, Any]],
        topic: str,
        tags: list[str] | None = None,
    ) -> int:
        """Distill an episode into persistent knowledge.

        Parameters
        ----------
        result : dict
            Debate result (consensus_reached, final_quality, etc.).
        transcript : list[dict]
            Debate transcript (per-round data).
        topic : str
            Debate topic.
        tags : list[str] | None
            Scenario tags.

        Returns
        -------
        count : int
            Number of knowledge items distilled.
        """
        quality = result.get("final_quality", 0.0)
        if quality < self._min_quality:
            return 0

        count = 0

        # 1. Extract debate skills
        count += self._distill_skills(result, transcript, topic, tags)

        # 2. Extract causal chains
        count += self._distill_causal(transcript, topic)

        # 3. Add memory insight
        count += self._distill_memory(result, topic, tags)

        if count > 0:
            logger.info(
                "Distilled %d items from episode: topic=%s quality=%.2f",
                count, topic[:30], quality,
            )
        return count

    def _distill_skills(
        self,
        result: dict[str, Any],
        transcript: list[dict[str, Any]],
        topic: str,
        tags: list[str] | None,
    ) -> int:
        """Extract and store debate skills."""
        if self._skill_extractor is None:
            return 0

        skill = self._skill_extractor.try_extract(
            result, transcript, topic=topic, tags=tags,
        )
        return 1 if skill is not None else 0

    def _distill_causal(
        self,
        transcript: list[dict[str, Any]],
        topic: str,
    ) -> int:
        """Extract causal chains from trajectory."""
        if self._causal_extractor is None or self._causal_graph is None:
            return 0

        chains = self._causal_extractor.extract_from_trajectory(transcript, topic)
        count = 0
        for chain in chains:
            self._causal_graph.add_chain(chain)
            count += 1

            # Also store as causal_chain skill in DB
            if self._skill_db is not None:
                from debate_rl_v2.skills.skill_db import SkillRecord
                record = SkillRecord(
                    name=f"causal-{topic[:20]}-{chain.chain_id[:6]}",
                    skill_type="causal_chain",
                    description=chain.to_text()[:200],
                    scenario_tags=[topic[:30]],
                    causal_chain=chain.to_dict(),
                    avg_quality=chain.total_confidence,
                    success_count=1,
                    total_uses=1,
                )
                self._skill_db.upsert_skill(record)

        return count

    def _distill_memory(
        self,
        result: dict[str, Any],
        topic: str,
        tags: list[str] | None,
    ) -> int:
        """Add episode summary to long-term memory."""
        if self._memory is None:
            return 0

        quality = result.get("final_quality", 0.0)
        consensus = result.get("consensus_reached", False)
        rounds = result.get("total_rounds", 0)

        summary = (
            f"辩论「{topic}」: "
            f"{'达成共识' if consensus else '未达成共识'}, "
            f"质量={quality:.2f}, {rounds}轮"
        )

        self._memory.add_insight(
            content=summary,
            source="episode_distillation",
            importance=min(1.0, quality),
            tags=tags,
        )
        return 1
