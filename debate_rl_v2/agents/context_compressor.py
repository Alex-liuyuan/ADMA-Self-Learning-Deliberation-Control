"""Context Compressor — hermes-agent inspired three-layer compression.

Manages LLM context window by intelligently compressing conversation
history when it approaches token limits.

Three compression strategies (applied in order):
  1. Prune old tool results (no LLM call needed)
  2. Protect head + tail messages (system prompt + recent N turns)
  3. LLM-summarize middle rounds (expensive, last resort)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from debate_rl_v2.framework.tokenizer import estimate_messages_tokens
from debate_rl_v2.logging_config import get_logger

logger = get_logger("agents.context_compressor")


@dataclass
class CompressionStats:
    """Statistics from a compression pass."""
    original_messages: int = 0
    compressed_messages: int = 0
    original_tokens_est: int = 0
    compressed_tokens_est: int = 0
    strategy_used: str = "none"


class ContextCompressor:
    """Three-layer context compression for debate conversations.

    Parameters
    ----------
    max_tokens : int
        Target maximum context window size in tokens.
    keep_system : bool
        Always preserve the system message.
    keep_recent : int
        Number of recent messages to always preserve.
    tool_result_max_len : int
        Truncate tool results longer than this (chars).
    summarizer : callable | None
        LLM function for summarizing middle rounds.
        Signature: (text: str) -> str
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        keep_system: bool = True,
        keep_recent: int = 6,
        tool_result_max_len: int = 200,
        summarizer: Any | None = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.keep_system = keep_system
        self.keep_recent = keep_recent
        self.tool_result_max_len = tool_result_max_len
        self.summarizer = summarizer

    def compress(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[list[dict[str, str]], CompressionStats]:
        """Compress message list if it exceeds token budget.

        Parameters
        ----------
        messages : list of dict
            OpenAI-format messages [{"role": ..., "content": ...}].

        Returns
        -------
        compressed : list of dict
            Compressed message list.
        stats : CompressionStats
            Compression statistics.
        """
        stats = CompressionStats(
            original_messages=len(messages),
            original_tokens_est=estimate_messages_tokens(messages),
        )

        current_tokens = stats.original_tokens_est
        if current_tokens <= self.max_tokens:
            stats.compressed_messages = len(messages)
            stats.compressed_tokens_est = current_tokens
            stats.strategy_used = "none"
            return messages, stats

        # === Layer 1: Prune tool results ===
        result = self._prune_tool_results(messages)
        current_tokens = estimate_messages_tokens(result)
        if current_tokens <= self.max_tokens:
            stats.compressed_messages = len(result)
            stats.compressed_tokens_est = current_tokens
            stats.strategy_used = "prune_tools"
            logger.info(
                "Compressed via tool pruning: %d→%d msgs, %d→%d tokens",
                stats.original_messages, len(result),
                stats.original_tokens_est, current_tokens,
            )
            return result, stats

        # === Layer 2: Protect head + tail, drop middle ===
        result = self._protect_head_tail(result)
        current_tokens = estimate_messages_tokens(result)
        if current_tokens <= self.max_tokens:
            stats.compressed_messages = len(result)
            stats.compressed_tokens_est = current_tokens
            stats.strategy_used = "head_tail"
            logger.info(
                "Compressed via head/tail protection: %d→%d msgs",
                stats.original_messages, len(result),
            )
            return result, stats

        # === Layer 3: LLM summarization of middle rounds ===
        if self.summarizer is not None:
            result = self._summarize_middle(messages)
            current_tokens = estimate_messages_tokens(result)
            stats.strategy_used = "llm_summary"
        else:
            # Fallback: aggressive truncation
            result = self._aggressive_truncate(messages)
            current_tokens = estimate_messages_tokens(result)
            stats.strategy_used = "aggressive_truncate"

        stats.compressed_messages = len(result)
        stats.compressed_tokens_est = current_tokens
        logger.info(
            "Compressed via %s: %d→%d msgs, %d→%d tokens",
            stats.strategy_used,
            stats.original_messages, len(result),
            stats.original_tokens_est, current_tokens,
        )
        return result, stats

    def _prune_tool_results(
        self,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Layer 1: Truncate verbose tool results in older messages."""
        result = []
        n = len(messages)
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            role = msg.get("role", "")

            # Only prune non-recent messages
            is_recent = (n - i) <= self.keep_recent
            if is_recent or role == "system":
                result.append(msg)
                continue

            # Truncate tool result blocks
            if "tool_calls" in content or "工具" in content:
                if len(content) > self.tool_result_max_len:
                    content = content[:self.tool_result_max_len] + "\n[...工具结果已截断...]"
                    result.append({**msg, "content": content})
                    continue

            # Truncate very long messages
            if len(content) > 500:
                content = content[:400] + "\n[...已截断...]"
                result.append({**msg, "content": content})
            else:
                result.append(msg)

        return result

    def _protect_head_tail(
        self,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Layer 2: Keep system prompt + first exchange + recent N messages."""
        if len(messages) <= self.keep_recent + 2:
            return messages

        head: list[dict[str, str]] = []
        tail: list[dict[str, str]] = []

        # Head: system message + first user/assistant pair
        for msg in messages:
            if msg.get("role") == "system":
                head.append(msg)
            elif len(head) < 3:  # system + first exchange
                head.append(msg)
            else:
                break

        # Tail: recent messages
        tail = messages[-self.keep_recent:]

        # Middle: count dropped rounds
        middle_count = len(messages) - len(head) - len(tail)
        if middle_count > 0:
            separator = {
                "role": "system",
                "content": f"[...省略了 {middle_count} 条中间对话记录...]",
            }
            return head + [separator] + tail

        return head + tail

    def _summarize_middle(
        self,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Layer 3: Use LLM to summarize middle rounds."""
        if len(messages) <= self.keep_recent + 2:
            return messages

        head: list[dict[str, str]] = []
        for msg in messages:
            if msg.get("role") == "system":
                head.append(msg)
            elif len(head) < 3:
                head.append(msg)
            else:
                break

        tail = messages[-self.keep_recent:]
        middle = messages[len(head):-self.keep_recent]

        if not middle:
            return head + tail

        # Build text to summarize
        middle_text = "\n".join(
            f"[{m.get('role', '?')}] {m.get('content', '')[:200]}"
            for m in middle
        )

        try:
            summary = self.summarizer(
                f"请用3-5句话概括以下辩论过程的关键进展:\n\n{middle_text}"
            )
        except Exception as e:
            logger.warning("LLM summarization failed: %s", e)
            summary = f"[省略了 {len(middle)} 条中间对话]"

        summary_msg = {
            "role": "system",
            "content": f"## 前期辩论摘要\n{summary}",
        }
        return head + [summary_msg] + tail

    def _aggressive_truncate(
        self,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Fallback: keep only system + last N messages."""
        system_msgs = [m for m in messages if m.get("role") == "system"]
        recent = messages[-self.keep_recent:]

        # Deduplicate system messages
        seen = set()
        unique_system = []
        for m in system_msgs:
            key = m.get("content", "")[:100]
            if key not in seen:
                seen.add(key)
                unique_system.append(m)

        return unique_system[:1] + recent

    def needs_compression(self, messages: list[dict[str, str]]) -> bool:
        """Check if messages exceed the token budget."""
        return estimate_messages_tokens(messages) > self.max_tokens
