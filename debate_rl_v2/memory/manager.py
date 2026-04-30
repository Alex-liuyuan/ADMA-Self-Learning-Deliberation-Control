"""Memory Manager — unified interface with hermes-agent frozen snapshot pattern.

Key improvement over v1: session-start memory is frozen into system prompt,
mid-session writes only update disk. This prevents context window bloat
from accumulating memories during a long debate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from debate_rl_v2.logging_config import get_logger
from debate_rl_v2.memory.base import (
    EntityMemory,
    LongTermMemory,
    MemoryEntry,
    RAGMemoryStore,
    ShortTermMemory,
)

logger = get_logger("memory.manager")


class MemoryManager:
    """Unified memory interface combining all tiers.

    hermes-agent pattern: at session start, freeze long-term memories
    into a snapshot string for system prompt injection. During the session,
    new insights are written to disk but NOT injected into the running
    context (avoids unbounded context growth).

    Parameters
    ----------
    short_term_limit : int
        Max entries in short-term memory.
    long_term_limit : int
        Max entries in long-term memory.
    persist_path : str | None
        Directory for persisting long-term memory.
    rag_enabled : bool
        Enable RAG memory store.
    rag_limit : int
        Max entries in RAG store.
    """

    def __init__(
        self,
        short_term_limit: int = 50,
        long_term_limit: int = 500,
        persist_path: str | None = None,
        rag_enabled: bool = False,
        rag_limit: int = 2000,
    ) -> None:
        self.short_term = ShortTermMemory(max_entries=short_term_limit)
        self.long_term = LongTermMemory(max_entries=long_term_limit)
        self.entity = EntityMemory()
        self._persist_path = persist_path
        self.rag: RAGMemoryStore | None = None
        self._rag_path: str | None = None
        self._frozen_snapshot: str = ""

        # Auto-load
        if persist_path:
            lt_path = str(Path(persist_path) / "long_term.json")
            self.long_term.load(lt_path)

        if rag_enabled:
            self.rag = RAGMemoryStore(max_entries=rag_limit)
            rag_path = str(Path(persist_path) / "rag_store.json") if persist_path else None
            if rag_path:
                self._rag_path = rag_path
                self.rag.load(rag_path)

    # ── Frozen snapshot (hermes-agent pattern) ──

    def freeze_snapshot(self, query: str = "", max_long: int = 10, max_rag: int = 5) -> str:
        """Freeze current long-term memories into a snapshot string.

        Call this once at session/episode start. The snapshot is injected
        into the system prompt and remains static for the session duration.
        """
        parts = []
        lt = self.long_term.to_context_string(query=query, max_entries=max_long)
        if lt:
            parts.append(lt)
        if self.rag:
            rag_ctx = self.rag.to_context_string(query=query, max_entries=max_rag)
            if rag_ctx:
                parts.append(rag_ctx)
        et = self.entity.to_context_string()
        if et:
            parts.append(et)
        self._frozen_snapshot = "\n\n".join(parts)
        logger.info("Frozen memory snapshot: %d chars", len(self._frozen_snapshot))
        return self._frozen_snapshot

    @property
    def frozen_snapshot(self) -> str:
        return self._frozen_snapshot

    # ── Write operations ──

    def add_observation(self, content: str, source: str = "", importance: float = 0.5) -> None:
        """Add to short-term memory (current episode only)."""
        self.short_term.add(content, source=source, importance=importance)

    def add_insight(
        self,
        content: str,
        source: str = "",
        importance: float = 0.7,
        tags: list[str] | None = None,
    ) -> None:
        """Add to long-term memory (persists across episodes)."""
        self.long_term.add(content, source=source, importance=importance, tags=tags)

    def update_entity(self, entity: str, facts: dict[str, Any], source: str = "") -> None:
        self.entity.update(entity, facts, source=source)

    def add_rag(
        self,
        content: str,
        source: str = "",
        importance: float = 0.6,
        tags: list[str] | None = None,
        **metadata: Any,
    ) -> None:
        if self.rag is not None:
            self.rag.add(content, source=source, importance=importance, tags=tags, **metadata)

    # ── Read operations ──

    def build_context(
        self,
        query: str = "",
        entities: list[str] | None = None,
        max_short: int = 10,
        max_long: int = 5,
        include_rag: bool = False,
        max_rag: int = 5,
        use_frozen: bool = True,
    ) -> str:
        """Build combined memory context string for LLM prompts.

        If use_frozen=True and a snapshot exists, uses the frozen snapshot
        for long-term memories (hermes-agent pattern). Short-term is always live.
        """
        parts = []

        # Short-term: always live
        st = self.short_term.to_context_string(max_entries=max_short)
        if st:
            parts.append(st)

        if use_frozen and self._frozen_snapshot:
            parts.append(self._frozen_snapshot)
        else:
            lt = self.long_term.to_context_string(query=query, max_entries=max_long)
            if lt:
                parts.append(lt)
            if include_rag and self.rag:
                rag_ctx = self.rag.to_context_string(query=query, max_entries=max_rag)
                if rag_ctx:
                    parts.append(rag_ctx)
            et = self.entity.to_context_string(entities=entities)
            if et:
                parts.append(et)

        return "\n\n".join(parts)

    def search_rag(self, query: str, top_k: int = 5, tags: list[str] | None = None) -> list[MemoryEntry]:
        if self.rag is None:
            return []
        return self.rag.search(query, top_k=top_k, tags=tags)

    # ── Lifecycle ──

    def distill_episode(self, summary: str, key_facts: dict[str, Any] | None = None) -> None:
        """Distill current episode into long-term memory."""
        self.long_term.add(content=summary, source="episode_distillation",
                           importance=0.8, tags=["episode_summary"])
        if key_facts:
            for entity, facts in key_facts.items():
                self.entity.update(entity, facts, source="distillation")

    def reset_episode(self) -> None:
        """Clear short-term memory for new episode, keep long-term."""
        self.short_term.clear()

    def save(self) -> None:
        if self._persist_path:
            self.long_term.save(str(Path(self._persist_path) / "long_term.json"))
        if self.rag and self._rag_path:
            self.rag.save(self._rag_path)

    def clear_all(self) -> None:
        self.short_term.clear()
        self.long_term.clear()
        self.entity.clear()
        if self.rag:
            self.rag.clear()
        self._frozen_snapshot = ""
