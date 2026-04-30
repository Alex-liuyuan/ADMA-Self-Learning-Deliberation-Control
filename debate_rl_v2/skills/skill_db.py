"""SkillDatabase — SQLite persistent skill storage with semantic retrieval.

Replaces JSON file-based storage with structured SQLite backend.
Supports skill CRUD, causal chain storage, prompt candidate management,
and optional embedding-based semantic search.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from debate_rl_v2.logging_config import get_logger

logger = get_logger("skills.skill_db")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS skills (
    skill_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    skill_type TEXT NOT NULL,
    description TEXT,
    scenario_tags TEXT,
    strategy_data TEXT,
    causal_chain TEXT,
    prompt_template TEXT,
    success_count INTEGER DEFAULT 0,
    total_uses INTEGER DEFAULT 0,
    avg_quality REAL DEFAULT 0.0,
    source_mode TEXT DEFAULT 'training',
    created_at REAL,
    last_used REAL,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_skills_type ON skills(skill_type);
CREATE INDEX IF NOT EXISTS idx_skills_quality ON skills(avg_quality DESC);

CREATE TABLE IF NOT EXISTS causal_relations (
    relation_id TEXT PRIMARY KEY,
    cause TEXT NOT NULL,
    effect TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    context TEXT,
    source TEXT,
    skill_id TEXT REFERENCES skills(skill_id),
    created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_causal_cause ON causal_relations(cause);
CREATE INDEX IF NOT EXISTS idx_causal_effect ON causal_relations(effect);

CREATE TABLE IF NOT EXISTS prompt_candidates (
    prompt_id TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    template TEXT NOT NULL,
    fitness REAL DEFAULT 0.0,
    usage_count INTEGER DEFAULT 0,
    generation INTEGER DEFAULT 0,
    parent_id TEXT,
    source_mode TEXT DEFAULT 'training',
    created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_prompt_fitness ON prompt_candidates(role, fitness DESC);
"""


@dataclass
class SkillRecord:
    """Unified skill record for DB operations."""
    skill_id: str = ""
    name: str = ""
    skill_type: str = "debate_strategy"
    description: str = ""
    scenario_tags: list[str] = field(default_factory=list)
    strategy_data: dict[str, Any] = field(default_factory=dict)
    causal_chain: dict[str, Any] | None = None
    prompt_template: str = ""
    success_count: int = 0
    total_uses: int = 0
    avg_quality: float = 0.0
    source_mode: str = "training"
    created_at: float = 0.0
    last_used: float = 0.0
    embedding: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.skill_id:
            self.skill_id = str(uuid.uuid4())[:12]
        if not self.created_at:
            self.created_at = time.time()


class SkillDatabase:
    """SQLite-backed skill storage with semantic retrieval.

    Parameters
    ----------
    db_path : str
        Path to SQLite database file. Use ":memory:" for testing.
    """

    def __init__(self, db_path: str = "skills.db") -> None:
        self._db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()
        logger.info("SkillDatabase opened: %s", db_path)

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ── Skill CRUD ──

    def upsert_skill(self, record: SkillRecord) -> None:
        """Insert or update a skill record."""
        embedding_blob = record.embedding.tobytes() if record.embedding is not None else None
        self._conn.execute(
            """INSERT INTO skills
               (skill_id, name, skill_type, description, scenario_tags,
                strategy_data, causal_chain, prompt_template,
                success_count, total_uses, avg_quality, source_mode,
                created_at, last_used, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(skill_id) DO UPDATE SET
                name=excluded.name, description=excluded.description,
                scenario_tags=excluded.scenario_tags,
                strategy_data=excluded.strategy_data,
                causal_chain=excluded.causal_chain,
                prompt_template=excluded.prompt_template,
                success_count=excluded.success_count,
                total_uses=excluded.total_uses,
                avg_quality=excluded.avg_quality,
                last_used=excluded.last_used,
                embedding=excluded.embedding""",
            (
                record.skill_id, record.name, record.skill_type,
                record.description, json.dumps(record.scenario_tags, ensure_ascii=False),
                json.dumps(record.strategy_data, ensure_ascii=False),
                json.dumps(record.causal_chain, ensure_ascii=False) if record.causal_chain else None,
                record.prompt_template,
                record.success_count, record.total_uses, record.avg_quality,
                record.source_mode, record.created_at, record.last_used,
                embedding_blob,
            ),
        )
        self._conn.commit()

    def get_skill(self, skill_id: str) -> SkillRecord | None:
        row = self._conn.execute(
            "SELECT * FROM skills WHERE skill_id = ?", (skill_id,)
        ).fetchone()
        return self._row_to_record(row) if row else None

    def find_relevant(
        self,
        topic: str,
        tags: list[str] | None = None,
        skill_type: str | None = None,
        top_k: int = 5,
    ) -> list[SkillRecord]:
        """Find skills relevant to topic/tags using keyword matching.

        Falls back to keyword overlap scoring (no embedding model required).
        """
        conditions = []
        params: list[Any] = []
        if skill_type:
            conditions.append("skill_type = ?")
            params.append(skill_type)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self._conn.execute(
            f"SELECT * FROM skills {where} ORDER BY avg_quality DESC LIMIT 100",
            params,
        ).fetchall()

        if not rows:
            return []

        # Score by keyword overlap
        topic_words = set(topic.lower().split())
        tag_set = set(t.lower() for t in (tags or []))
        scored: list[tuple[float, SkillRecord]] = []

        for row in rows:
            record = self._row_to_record(row)
            score = 0.0
            # Tag overlap
            skill_tags = set(t.lower() for t in record.scenario_tags)
            score += len(tag_set & skill_tags) * 2.0
            # Name/description keyword overlap
            name_words = set(record.name.lower().split())
            desc_words = set(record.description.lower().split())
            score += len(topic_words & name_words) * 1.5
            score += len(topic_words & desc_words) * 0.5
            # Quality bonus
            score += record.avg_quality * 0.3
            if score > 0:
                scored.append((score, record))

        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:top_k]]

    def find_causal_chains(self, topic: str, top_k: int = 3) -> list[SkillRecord]:
        """Find causal chain skills relevant to a topic."""
        return self.find_relevant(topic, skill_type="causal_chain", top_k=top_k)

    def record_usage(self, skill_id: str, success: bool, quality: float) -> None:
        """Record a skill usage outcome."""
        row = self._conn.execute(
            "SELECT total_uses, success_count, avg_quality FROM skills WHERE skill_id = ?",
            (skill_id,),
        ).fetchone()
        if not row:
            return

        total = row["total_uses"] + 1
        succ = row["success_count"] + (1 if success else 0)
        alpha = 1.0 / total
        avg_q = (1 - alpha) * row["avg_quality"] + alpha * quality

        self._conn.execute(
            """UPDATE skills SET total_uses = ?, success_count = ?,
               avg_quality = ?, last_used = ? WHERE skill_id = ?""",
            (total, succ, avg_q, time.time(), skill_id),
        )
        self._conn.commit()

    # ── Prompt candidates ──

    def save_prompt(self, prompt_id: str, role: str, template: str,
                    fitness: float = 0.0, generation: int = 0,
                    parent_id: str = "", source_mode: str = "training") -> None:
        self._conn.execute(
            """INSERT INTO prompt_candidates
               (prompt_id, role, template, fitness, usage_count, generation,
                parent_id, source_mode, created_at)
               VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?)
               ON CONFLICT(prompt_id) DO UPDATE SET
                template=excluded.template, fitness=excluded.fitness,
                generation=excluded.generation""",
            (prompt_id, role, template, fitness, generation,
             parent_id, source_mode, time.time()),
        )
        self._conn.commit()

    def get_best_prompt(self, role: str) -> str | None:
        row = self._conn.execute(
            "SELECT template FROM prompt_candidates WHERE role = ? ORDER BY fitness DESC LIMIT 1",
            (role,),
        ).fetchone()
        return row["template"] if row else None

    def get_prompts_by_role(self, role: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT prompt_id, role, template, fitness, usage_count, generation, parent_id
               FROM prompt_candidates WHERE role = ? ORDER BY fitness DESC LIMIT ?""",
            (role, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def update_prompt_fitness(self, prompt_id: str, fitness: float) -> None:
        self._conn.execute(
            "UPDATE prompt_candidates SET fitness = ?, usage_count = usage_count + 1 WHERE prompt_id = ?",
            (fitness, prompt_id),
        )
        self._conn.commit()

    # ── Causal relations ──

    def add_causal_relation(
        self, cause: str, effect: str, confidence: float = 1.0,
        context: str = "", source: str = "", skill_id: str | None = None,
    ) -> str:
        rel_id = str(uuid.uuid4())[:12]
        self._conn.execute(
            """INSERT INTO causal_relations
               (relation_id, cause, effect, confidence, context, source, skill_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (rel_id, cause, effect, confidence, context, source, skill_id, time.time()),
        )
        self._conn.commit()
        return rel_id

    def find_causal_by_cause(self, cause: str, limit: int = 10) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM causal_relations WHERE cause LIKE ? ORDER BY confidence DESC LIMIT ?",
            (f"%{cause}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def find_causal_by_effect(self, effect: str, limit: int = 10) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM causal_relations WHERE effect LIKE ? ORDER BY confidence DESC LIMIT ?",
            (f"%{effect}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Migration ──

    def migrate_from_json(self, skills_dir: str) -> int:
        """Migrate skills from JSON file directory to SQLite.

        Returns number of skills migrated.
        """
        from debate_rl_v2.skills.skill_manager import DebateSkill

        path = Path(skills_dir)
        if not path.exists():
            return 0

        count = 0
        for f in path.rglob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                skill = DebateSkill.from_dict(data)
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
                self.upsert_skill(record)
                count += 1
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to migrate %s: %s", f, e)

        logger.info("Migrated %d skills from %s", count, skills_dir)
        return count

    # ── Statistics ──

    def get_statistics(self) -> dict[str, Any]:
        skills_count = self._conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
        causal_count = self._conn.execute("SELECT COUNT(*) FROM causal_relations").fetchone()[0]
        prompt_count = self._conn.execute("SELECT COUNT(*) FROM prompt_candidates").fetchone()[0]
        avg_quality = self._conn.execute(
            "SELECT AVG(avg_quality) FROM skills WHERE total_uses > 0"
        ).fetchone()[0]
        return {
            "total_skills": skills_count,
            "total_causal_relations": causal_count,
            "total_prompt_candidates": prompt_count,
            "avg_skill_quality": avg_quality or 0.0,
        }

    # ── Internal ──

    def _row_to_record(self, row: sqlite3.Row) -> SkillRecord:
        embedding = None
        if row["embedding"]:
            embedding = np.frombuffer(row["embedding"], dtype=np.float32)
        causal = None
        if row["causal_chain"]:
            try:
                causal = json.loads(row["causal_chain"])
            except json.JSONDecodeError:
                pass
        return SkillRecord(
            skill_id=row["skill_id"],
            name=row["name"],
            skill_type=row["skill_type"],
            description=row["description"] or "",
            scenario_tags=json.loads(row["scenario_tags"]) if row["scenario_tags"] else [],
            strategy_data=json.loads(row["strategy_data"]) if row["strategy_data"] else {},
            causal_chain=causal,
            prompt_template=row["prompt_template"] or "",
            success_count=row["success_count"],
            total_uses=row["total_uses"],
            avg_quality=row["avg_quality"],
            source_mode=row["source_mode"] or "training",
            created_at=row["created_at"] or 0.0,
            last_used=row["last_used"] or 0.0,
            embedding=embedding,
        )
