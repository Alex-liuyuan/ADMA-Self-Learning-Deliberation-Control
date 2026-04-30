"""Skill Manager — hermes-agent inspired progressive disclosure.

Extracts successful debate strategies as reusable skills.
Skills are stored as structured templates that can be loaded
when similar debate scenarios are encountered.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from debate_rl_v2.logging_config import get_logger

logger = get_logger("skills.manager")


@dataclass
class DebateSkill:
    """A reusable debate strategy extracted from successful debates."""
    name: str
    description: str
    scenario_tags: list[str] = field(default_factory=list)
    # Strategy templates per role
    proposer_strategy: str = ""
    challenger_strategy: str = ""
    arbiter_strategy: str = ""
    coordinator_strategy: str = ""
    # Metadata
    source_topic: str = ""
    success_count: int = 0
    total_uses: int = 0
    avg_quality: float = 0.0
    created_at: float = field(default_factory=time.time)
    last_used: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success_count / max(self.total_uses, 1)

    def record_use(self, success: bool, quality: float) -> None:
        self.total_uses += 1
        if success:
            self.success_count += 1
        alpha = 1.0 / self.total_uses
        self.avg_quality = (1 - alpha) * self.avg_quality + alpha * quality
        self.last_used = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "scenario_tags": self.scenario_tags,
            "proposer_strategy": self.proposer_strategy,
            "challenger_strategy": self.challenger_strategy,
            "arbiter_strategy": self.arbiter_strategy,
            "coordinator_strategy": self.coordinator_strategy,
            "source_topic": self.source_topic,
            "success_count": self.success_count,
            "total_uses": self.total_uses,
            "avg_quality": self.avg_quality,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DebateSkill:
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            scenario_tags=d.get("scenario_tags", []),
            proposer_strategy=d.get("proposer_strategy", ""),
            challenger_strategy=d.get("challenger_strategy", ""),
            arbiter_strategy=d.get("arbiter_strategy", ""),
            coordinator_strategy=d.get("coordinator_strategy", ""),
            source_topic=d.get("source_topic", ""),
            success_count=d.get("success_count", 0),
            total_uses=d.get("total_uses", 0),
            avg_quality=d.get("avg_quality", 0.0),
            created_at=d.get("created_at", time.time()),
        )


class SkillManager:
    """Manages debate skills — loading, matching, and extraction.

    Skills are stored in a directory structure (legacy JSON) or
    delegated to SkillDatabase (SQLite) when provided.

    Parameters
    ----------
    skills_dir : str | None
        Legacy JSON skills directory.
    db : SkillDatabase | None
        SQLite backend. When provided, all operations delegate to it.
    """

    def __init__(self, skills_dir: str | None = None, db: Any = None) -> None:
        self._skills: dict[str, DebateSkill] = {}
        self._skills_dir = skills_dir
        self._db = db  # SkillDatabase instance
        if skills_dir and db is None:
            self._load_all(skills_dir)

    def _load_all(self, skills_dir: str) -> None:
        path = Path(skills_dir)
        if not path.exists():
            return
        for f in path.rglob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                skill = DebateSkill.from_dict(data)
                self._skills[skill.name] = skill
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to load skill %s: %s", f, e)
        logger.info("Loaded %d skills from %s", len(self._skills), skills_dir)

    def find_relevant(
        self,
        topic: str,
        tags: list[str] | None = None,
        top_k: int = 3,
    ) -> list[DebateSkill]:
        """Find skills relevant to the given topic/tags."""
        # Delegate to DB if available
        if self._db is not None:
            records = self._db.find_relevant(topic, tags=tags, skill_type="debate_strategy", top_k=top_k)
            return [self._record_to_skill(r) for r in records]

        if not self._skills:
            return []

        scored: list[tuple[float, DebateSkill]] = []
        topic_lower = topic.lower()
        tag_set = set(t.lower() for t in (tags or []))

        for skill in self._skills.values():
            score = 0.0
            # Tag overlap
            skill_tags = set(t.lower() for t in skill.scenario_tags)
            tag_overlap = len(tag_set & skill_tags) if tag_set else 0
            score += tag_overlap * 2.0
            # Topic keyword overlap
            topic_words = set(topic_lower.split())
            name_words = set(skill.name.lower().split())
            desc_words = set(skill.description.lower().split())
            score += len(topic_words & name_words) * 1.5
            score += len(topic_words & desc_words) * 0.5
            # Success rate bonus
            score += skill.success_rate * 0.5
            if score > 0:
                scored.append((score, skill))

        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:top_k]]

    def get_strategy_for_role(self, skill: DebateSkill, role: str) -> str:
        """Get role-specific strategy from a skill."""
        mapping = {
            "proposer": skill.proposer_strategy,
            "challenger": skill.challenger_strategy,
            "arbiter": skill.arbiter_strategy,
            "coordinator": skill.coordinator_strategy,
        }
        return mapping.get(role, "")

    def build_skill_context(
        self,
        topic: str,
        role: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """Build skill-based context for prompt injection."""
        relevant = self.find_relevant(topic, tags=tags)
        if not relevant:
            return ""

        lines = ["## 相关辩论技能"]
        for skill in relevant:
            lines.append(f"\n### {skill.name}")
            lines.append(f"场景: {skill.description}")
            lines.append(f"成功率: {skill.success_rate:.0%} ({skill.total_uses} 次使用)")
            if role:
                strategy = self.get_strategy_for_role(skill, role)
                if strategy:
                    lines.append(f"推荐策略: {strategy}")
        return "\n".join(lines)

    def add_skill(self, skill: DebateSkill) -> None:
        self._skills[skill.name] = skill
        if self._db is not None:
            from debate_rl_v2.skills.skill_db import SkillRecord
            record = SkillRecord(
                name=skill.name,
                skill_type="debate_strategy",
                description=skill.description,
                scenario_tags=skill.scenario_tags,
                strategy_data={
                    "proposer": skill.proposer_strategy,
                    "challenger": skill.challenger_strategy,
                    "arbiter": skill.arbiter_strategy,
                    "coordinator": skill.coordinator_strategy,
                },
                success_count=skill.success_count,
                total_uses=skill.total_uses,
                avg_quality=skill.avg_quality,
                created_at=skill.created_at,
            )
            self._db.upsert_skill(record)
        elif self._skills_dir:
            self._save_skill(skill)

    def _save_skill(self, skill: DebateSkill) -> None:
        if not self._skills_dir:
            return
        path = Path(self._skills_dir)
        # Use first tag as subdirectory
        subdir = skill.scenario_tags[0] if skill.scenario_tags else "general"
        out_dir = path / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_name = skill.name.replace(" ", "-").replace("/", "-")
        out_path = out_dir / f"{safe_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(skill.to_dict(), f, ensure_ascii=False, indent=2)

    def __len__(self) -> int:
        if self._db is not None:
            stats = self._db.get_statistics()
            return stats.get("total_skills", 0)
        return len(self._skills)

    def migrate_to_db(self, db: Any) -> int:
        """Migrate all in-memory JSON skills to a SkillDatabase.

        Returns number of skills migrated.
        """
        if not self._skills:
            return 0
        count = 0
        from debate_rl_v2.skills.skill_db import SkillRecord
        for skill in self._skills.values():
            record = SkillRecord(
                name=skill.name,
                skill_type="debate_strategy",
                description=skill.description,
                scenario_tags=skill.scenario_tags,
                strategy_data={
                    "proposer": skill.proposer_strategy,
                    "challenger": skill.challenger_strategy,
                    "arbiter": skill.arbiter_strategy,
                    "coordinator": skill.coordinator_strategy,
                },
                success_count=skill.success_count,
                total_uses=skill.total_uses,
                avg_quality=skill.avg_quality,
                created_at=skill.created_at,
            )
            db.upsert_skill(record)
            count += 1
        self._db = db
        logger.info("Migrated %d skills to SkillDatabase", count)
        return count

    @staticmethod
    def _record_to_skill(record: Any) -> DebateSkill:
        """Convert a SkillRecord to a DebateSkill."""
        sd = record.strategy_data or {}
        return DebateSkill(
            name=record.name,
            description=record.description,
            scenario_tags=record.scenario_tags,
            proposer_strategy=sd.get("proposer", ""),
            challenger_strategy=sd.get("challenger", ""),
            arbiter_strategy=sd.get("arbiter", ""),
            coordinator_strategy=sd.get("coordinator", ""),
            success_count=record.success_count,
            total_uses=record.total_uses,
            avg_quality=record.avg_quality,
            created_at=record.created_at,
        )
