"""CausalDataset — JSONL causal dataset loading and batch retrieval.

Supports pre-labeled JSONL files with causal relations, plus domain
and difficulty filtering for curriculum-style training.

JSONL format per line:
{
  "text": "全球变暖导致冰川融化...",
  "relations": [{"cause": "全球变暖", "effect": "冰川融化", "confidence": 0.95}],
  "domain": "环境",
  "difficulty": 2
}
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from debate_rl_v2.logging_config import get_logger

logger = get_logger("causal.dataset")


@dataclass
class CausalRelation:
    """A single cause-effect relationship."""
    cause: str
    effect: str
    confidence: float = 1.0
    context: str = ""
    evidence: list[str] = field(default_factory=list)
    source: str = ""  # "dataset" | "extracted" | "online"

    def to_dict(self) -> dict[str, Any]:
        return {
            "cause": self.cause,
            "effect": self.effect,
            "confidence": self.confidence,
            "context": self.context,
            "evidence": self.evidence,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CausalRelation:
        return cls(
            cause=d["cause"],
            effect=d["effect"],
            confidence=d.get("confidence", 1.0),
            context=d.get("context", ""),
            evidence=d.get("evidence", []),
            source=d.get("source", ""),
        )


@dataclass
class CausalDatasetEntry:
    """A single dataset entry with text and labeled causal relations."""
    text: str
    relations: list[CausalRelation]
    domain: str = ""
    difficulty: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


class CausalDataset:
    """JSONL causal dataset loader with filtering and batching.

    Parameters
    ----------
    seed : int
        Random seed for reproducible batching.
    """

    def __init__(self, seed: int = 42) -> None:
        self._entries: list[CausalDatasetEntry] = []
        self._rng = random.Random(seed)
        self._domain_index: dict[str, list[int]] = {}

    def load(self, path: str) -> None:
        """Load entries from a JSONL file."""
        p = Path(path)
        if not p.exists():
            logger.warning("Dataset file not found: %s", path)
            return

        count = 0
        with open(p, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    relations = [
                        CausalRelation.from_dict(r)
                        for r in data.get("relations", [])
                    ]
                    # Tag source as dataset
                    for rel in relations:
                        if not rel.source:
                            rel.source = "dataset"

                    entry = CausalDatasetEntry(
                        text=data.get("text", ""),
                        relations=relations,
                        domain=data.get("domain", ""),
                        difficulty=data.get("difficulty", 1),
                        metadata={k: v for k, v in data.items()
                                  if k not in ("text", "relations", "domain", "difficulty")},
                    )
                    idx = len(self._entries)
                    self._entries.append(entry)

                    # Build domain index
                    if entry.domain:
                        self._domain_index.setdefault(entry.domain, []).append(idx)
                    count += 1
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning("Skipping line %d in %s: %s", line_num, path, e)

        logger.info("Loaded %d causal entries from %s", count, path)

    def get_batch(
        self,
        size: int,
        difficulty: int | None = None,
        domain: str | None = None,
    ) -> list[CausalDatasetEntry]:
        """Get a random batch of entries, optionally filtered."""
        pool = self._entries
        if domain and domain in self._domain_index:
            pool = [self._entries[i] for i in self._domain_index[domain]]
        if difficulty is not None:
            pool = [e for e in pool if e.difficulty == difficulty]
        if not pool:
            return []
        size = min(size, len(pool))
        return self._rng.sample(pool, size)

    def get_by_domain(self, domain: str) -> list[CausalDatasetEntry]:
        """Get all entries for a specific domain."""
        if domain not in self._domain_index:
            return []
        return [self._entries[i] for i in self._domain_index[domain]]

    @property
    def domains(self) -> list[str]:
        return list(self._domain_index.keys())

    def add_entry(self, entry: CausalDatasetEntry) -> None:
        """Add a dynamically extracted entry."""
        idx = len(self._entries)
        self._entries.append(entry)
        if entry.domain:
            self._domain_index.setdefault(entry.domain, []).append(idx)

    def __len__(self) -> int:
        return len(self._entries)
