"""Memory system — refactored with BaseMemoryStore to eliminate duplication.

Extracts common patterns from ShortTermMemory/LongTermMemory/RAGMemoryStore
into a single base class. Adds hermes-agent inspired frozen snapshot pattern.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from debate_rl_v2.logging_config import get_logger

logger = get_logger("memory")

# ── Shared utilities ──

_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _build_idf(doc_tokens: list[list[str]]) -> dict[str, float]:
    n_docs = len(doc_tokens)
    df: Counter[str] = Counter()
    for tokens in doc_tokens:
        df.update(set(tokens))
    return {t: math.log((1 + n_docs) / (1 + c)) + 1.0 for t, c in df.items()}


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = sum(counts.values())
    return {t: (c / total) * idf.get(t, 0.0) for t, c in counts.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Data model ──

@dataclass
class MemoryEntry:
    """A single memory entry."""
    content: str
    source: str = ""
    importance: float = 0.5
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.access_count += 1
        self.last_accessed = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "source": self.source,
            "importance": self.importance,
            "tags": self.tags,
            "created_at": self.created_at,
            "access_count": self.access_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryEntry:
        return cls(
            content=d["content"],
            source=d.get("source", ""),
            importance=d.get("importance", 0.5),
            tags=d.get("tags", []),
            created_at=d.get("created_at", time.time()),
            access_count=d.get("access_count", 0),
            metadata=d.get("metadata", {}),
        )


# ── Base class (DRY) ──

class BaseMemoryStore:
    """Base class for all memory stores — eliminates code duplication.

    Provides: add, search (keyword), evict, save/load, to_context_string.
    Subclasses override eviction scoring and context formatting.
    """

    def __init__(self, max_entries: int = 500, default_importance: float = 0.5) -> None:
        self.max_entries = max_entries
        self._default_importance = default_importance
        self._entries: list[MemoryEntry] = []

    def add(
        self,
        content: str,
        source: str = "",
        importance: float | None = None,
        tags: list[str] | None = None,
        **metadata: Any,
    ) -> None:
        entry = MemoryEntry(
            content=content,
            source=source,
            importance=importance if importance is not None else self._default_importance,
            tags=tags or [],
            metadata=metadata,
        )
        self._entries.append(entry)
        if len(self._entries) > self.max_entries:
            self._evict()

    def _eviction_score(self, entry: MemoryEntry) -> float:
        """Score for eviction priority (lower = evicted first). Override in subclass."""
        recency = 1.0 / (1 + time.time() - entry.last_accessed) if entry.last_accessed else 0
        return entry.importance * 0.5 + entry.access_count * 0.3 + recency * 0.2

    def _evict(self) -> None:
        self._entries.sort(key=self._eviction_score)
        self._entries = self._entries[-self.max_entries:]

    def search(
        self,
        query: str,
        top_k: int = 5,
        tags: list[str] | None = None,
    ) -> list[MemoryEntry]:
        candidates = self._entries
        if tags:
            tag_set = set(tags)
            candidates = [e for e in candidates if tag_set & set(e.tags)]
        if not candidates:
            return []
        if not query.strip():
            results = sorted(candidates, key=lambda e: -e.importance)[:top_k]
            for r in results:
                r.touch()
            return results

        query_lower = query.lower()
        query_words = set(query_lower.split())
        scored = []
        for entry in candidates:
            content_lower = entry.content.lower()
            word_score = sum(1 for w in query_words if w in content_lower)
            score = word_score + 0.3 * entry.importance + 0.1 * entry.access_count
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda x: -x[0])
        results = [entry for _, entry in scored[:top_k]]
        for r in results:
            r.touch()
        return results

    def get_recent(self, n: int = 10) -> list[MemoryEntry]:
        return self._entries[-n:]

    def _format_header(self) -> str:
        """Override in subclass for section header."""
        return "## 记忆"

    def to_context_string(self, query: str = "", max_entries: int = 5) -> str:
        if query:
            entries = self.search(query, top_k=max_entries)
        else:
            entries = sorted(self._entries, key=lambda e: -e.importance)[:max_entries]
        if not entries:
            return ""
        lines = [self._format_header()]
        for e in entries:
            tag_str = f" [{', '.join(e.tags)}]" if e.tags else ""
            src = f"[{e.source}] " if e.source else ""
            lines.append(f"- {src}{e.content}{tag_str}")
        return "\n".join(lines)

    def save(self, path: str) -> None:
        data = [e.to_dict() for e in self._entries]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        if not Path(path).exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for d in data:
                self._entries.append(MemoryEntry.from_dict(d))
            logger.info("Loaded %d entries from %s", len(data), path)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load memory from %s: %s", path, e)

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


# ── Concrete stores ──

class ShortTermMemory(BaseMemoryStore):
    """Per-episode conversation memory (sliding window)."""

    def __init__(self, max_entries: int = 50) -> None:
        super().__init__(max_entries=max_entries, default_importance=0.5)

    def _eviction_score(self, entry: MemoryEntry) -> float:
        # Short-term: evict by importance only (keep recent high-importance)
        return entry.importance

    def _format_header(self) -> str:
        return "## 近期记忆"


class LongTermMemory(BaseMemoryStore):
    """Persistent cross-episode memory with importance-based retention."""

    def __init__(self, max_entries: int = 500) -> None:
        super().__init__(max_entries=max_entries, default_importance=0.7)

    def _format_header(self) -> str:
        return "## 长期记忆"


class RAGMemoryStore(BaseMemoryStore):
    """Lightweight RAG store with TF-IDF retrieval."""

    def __init__(self, max_entries: int = 2000) -> None:
        super().__init__(max_entries=max_entries, default_importance=0.6)

    def search(
        self,
        query: str,
        top_k: int = 5,
        tags: list[str] | None = None,
    ) -> list[MemoryEntry]:
        """TF-IDF based search (overrides keyword search)."""
        candidates = self._entries
        if tags:
            tag_set = set(tags)
            candidates = [e for e in candidates if tag_set & set(e.tags)]
        if not candidates:
            return []
        if not query.strip():
            results = sorted(candidates, key=lambda e: -e.importance)[:top_k]
            for r in results:
                r.touch()
            return results

        doc_tokens = [_tokenize(e.content) for e in candidates]
        idf = _build_idf(doc_tokens)
        query_vec = _tfidf_vector(_tokenize(query), idf)

        scored = []
        for entry, tokens in zip(candidates, doc_tokens):
            vec = _tfidf_vector(tokens, idf)
            score = _cosine(query_vec, vec) + 0.05 * entry.importance
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda x: -x[0])
        results = [entry for _, entry in scored[:top_k]]
        for r in results:
            r.touch()
        return results

    def _format_header(self) -> str:
        return "## RAG 记忆"


class EntityMemory:
    """Structured knowledge about specific entities."""

    def __init__(self) -> None:
        self._entities: dict[str, dict[str, Any]] = {}

    def update(self, entity: str, facts: dict[str, Any], source: str = "") -> None:
        if entity not in self._entities:
            self._entities[entity] = {"_source": source, "_updated": time.time()}
        self._entities[entity].update(facts)
        self._entities[entity]["_updated"] = time.time()

    def get(self, entity: str) -> dict[str, Any] | None:
        return self._entities.get(entity)

    def query(self, pattern: str) -> dict[str, dict[str, Any]]:
        p = pattern.lower()
        return {k: v for k, v in self._entities.items() if p in k.lower()}

    def to_context_string(self, entities: list[str] | None = None) -> str:
        targets = entities or list(self._entities.keys())
        if not targets:
            return ""
        lines = ["## 实体知识"]
        for name in targets:
            facts = self._entities.get(name)
            if facts:
                items = {k: v for k, v in facts.items() if not k.startswith("_")}
                if items:
                    lines.append(f"### {name}")
                    for k, v in items.items():
                        lines.append(f"  - {k}: {v}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._entities.clear()

    def __len__(self) -> int:
        return len(self._entities)
