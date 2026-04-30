"""Prompt Caching — hermes-agent inspired prefix caching.

Caches the static prefix of LLM prompts (system prompt + frozen memories)
to reduce token costs in multi-round debates. The prefix is computed once
and reused across all rounds.

Works with providers that support prompt caching (Anthropic, OpenAI).
For providers without native caching, provides client-side deduplication.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from debate_rl_v2.logging_config import get_logger

logger = get_logger("llm.prompt_cache")


@dataclass
class CacheEntry:
    """A cached prompt prefix."""
    prefix_hash: str
    messages: list[dict[str, str]]
    token_count_est: int = 0
    hit_count: int = 0


class PromptCache:
    """Client-side prompt prefix cache.

    For each role, caches the system prompt + frozen memory snapshot
    as a prefix. On subsequent rounds, only the new user message
    needs to be sent (the prefix is reused via cache reference).

    Parameters
    ----------
    enabled : bool
        Whether caching is active.
    max_entries : int
        Maximum cache entries (per role).
    """

    def __init__(self, enabled: bool = True, max_entries: int = 10) -> None:
        self.enabled = enabled
        self.max_entries = max_entries
        self._cache: dict[str, CacheEntry] = {}
        self._stats = {"hits": 0, "misses": 0, "tokens_saved_est": 0}

    def get_or_create_prefix(
        self,
        role: str,
        system_prompt: str,
        frozen_memory: str = "",
    ) -> tuple[str, list[dict[str, str]]]:
        """Get cached prefix or create new one.

        Returns
        -------
        prefix_hash : str
            Hash of the prefix (for cache key).
        prefix_messages : list
            The prefix messages (system + memory).
        """
        if not self.enabled:
            messages = self._build_prefix_messages(system_prompt, frozen_memory)
            return "", messages

        # Compute hash
        content = f"{role}:{system_prompt}:{frozen_memory}"
        prefix_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        if prefix_hash in self._cache:
            entry = self._cache[prefix_hash]
            entry.hit_count += 1
            self._stats["hits"] += 1
            self._stats["tokens_saved_est"] += entry.token_count_est
            return prefix_hash, entry.messages

        # Cache miss — create new entry
        self._stats["misses"] += 1
        messages = self._build_prefix_messages(system_prompt, frozen_memory)
        token_est = sum(len(m.get("content", "")) for m in messages) // 2

        entry = CacheEntry(
            prefix_hash=prefix_hash,
            messages=messages,
            token_count_est=token_est,
        )
        self._cache[prefix_hash] = entry

        # Evict old entries if over limit
        if len(self._cache) > self.max_entries:
            # Remove least-hit entry
            min_key = min(self._cache, key=lambda k: self._cache[k].hit_count)
            del self._cache[min_key]

        return prefix_hash, messages

    def build_messages(
        self,
        role: str,
        system_prompt: str,
        frozen_memory: str,
        conversation: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Build complete message list with cached prefix.

        Parameters
        ----------
        role : str
            Agent role.
        system_prompt : str
            System prompt for this role.
        frozen_memory : str
            Frozen memory snapshot.
        conversation : list
            Dynamic conversation messages.

        Returns
        -------
        messages : list
            Complete message list for LLM call.
        """
        _, prefix = self.get_or_create_prefix(role, system_prompt, frozen_memory)
        return prefix + conversation

    @staticmethod
    def _build_prefix_messages(
        system_prompt: str,
        frozen_memory: str,
    ) -> list[dict[str, str]]:
        """Build the static prefix messages."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            content = system_prompt
            if frozen_memory:
                content += f"\n\n{frozen_memory}"
            messages.append({"role": "system", "content": content})
        elif frozen_memory:
            messages.append({"role": "system", "content": frozen_memory})
        return messages

    @property
    def stats(self) -> dict[str, Any]:
        total = self._stats["hits"] + self._stats["misses"]
        return {
            **self._stats,
            "total_requests": total,
            "hit_rate": self._stats["hits"] / max(total, 1),
            "cache_size": len(self._cache),
        }

    def clear(self) -> None:
        self._cache.clear()
        self._stats = {"hits": 0, "misses": 0, "tokens_saved_est": 0}
