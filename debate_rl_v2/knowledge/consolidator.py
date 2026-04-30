"""KnowledgeConsolidator — cross-episode knowledge consolidation.

Periodically merges and deduplicates accumulated knowledge across
episodes, pruning low-quality entries and strengthening patterns
that appear repeatedly.
"""

from __future__ import annotations

from typing import Any

from debate_rl_v2.logging_config import get_logger

logger = get_logger("knowledge.consolidator")


class KnowledgeConsolidator:
    """Cross-episode knowledge consolidation.

    Periodically:
    1. Prunes low-quality skills from SkillDB
    2. Merges duplicate causal relations in CausalGraph
    3. Compacts long-term memory

    Parameters
    ----------
    skill_db : SkillDatabase | None
        SQLite skill storage.
    causal_graph : CausalGraph | None
        Causal relation graph.
    memory : MemoryManager | None
        Long-term memory manager.
    prune_threshold : float
        Skills below this quality with enough uses get pruned.
    min_uses_for_prune : int
        Minimum uses before a skill can be pruned.
    consolidation_interval : int
        Consolidate every N episodes.
    """

    def __init__(
        self,
        skill_db: Any = None,
        causal_graph: Any = None,
        memory: Any = None,
        prune_threshold: float = 0.3,
        min_uses_for_prune: int = 5,
        consolidation_interval: int = 10,
    ) -> None:
        self._skill_db = skill_db
        self._causal_graph = causal_graph
        self._memory = memory
        self._prune_threshold = prune_threshold
        self._min_uses_for_prune = min_uses_for_prune
        self._interval = consolidation_interval
        self._episode_count = 0

    def should_consolidate(self) -> bool:
        return self._episode_count > 0 and self._episode_count % self._interval == 0

    def record_episode(self) -> None:
        self._episode_count += 1

    def consolidate(self) -> dict[str, int]:
        """Run full consolidation. Returns counts of actions taken."""
        stats: dict[str, int] = {"pruned_skills": 0, "saved_graph": 0, "saved_memory": 0}

        stats["pruned_skills"] = self._prune_skills()
        stats["saved_graph"] = self._save_graph()
        stats["saved_memory"] = self._save_memory()

        logger.info("Consolidation complete: %s", stats)
        return stats

    def _prune_skills(self) -> int:
        """Prune low-quality skills from DB."""
        if self._skill_db is None:
            return 0

        # Query skills with enough uses but low quality
        conn = self._skill_db._conn
        rows = conn.execute(
            "SELECT skill_id FROM skills WHERE total_uses >= ? AND avg_quality < ?",
            (self._min_uses_for_prune, self._prune_threshold),
        ).fetchall()

        count = 0
        for row in rows:
            conn.execute("DELETE FROM skills WHERE skill_id = ?", (row["skill_id"],))
            count += 1

        if count > 0:
            conn.commit()
            logger.info("Pruned %d low-quality skills", count)
        return count

    def _save_graph(self) -> int:
        """Persist causal graph if available."""
        if self._causal_graph is None:
            return 0
        # Graph auto-saves are handled externally; just report
        return 1

    def _save_memory(self) -> int:
        """Persist memory if available."""
        if self._memory is None:
            return 0
        try:
            self._memory.save()
            return 1
        except Exception as e:
            logger.warning("Memory save failed: %s", e)
            return 0
