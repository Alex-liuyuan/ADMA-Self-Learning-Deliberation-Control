"""Medical knowledge base tool — structured KB query.

Replaces the fake _fact_check that returned fixed text.
Provides a local knowledge base with medical guidelines,
drug interactions, and treatment protocols.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from debate_rl_v2.tools.registry import ToolRegistry, ToolSchema
from debate_rl_v2.logging_config import get_logger

logger = get_logger("tools.medical_kb")


class MedicalKnowledgeBase:
    """Local medical knowledge base with structured retrieval.

    Supports loading from JSON files and keyword-based search.
    In production, this would connect to a vector DB or API.
    """

    def __init__(self, data_dir: str | None = None) -> None:
        self._entries: list[dict[str, Any]] = []
        self._index: dict[str, list[int]] = {}  # keyword → entry indices
        if data_dir:
            self.load(data_dir)

    def load(self, data_dir: str) -> None:
        """Load knowledge entries from JSON files in directory."""
        path = Path(data_dir)
        if not path.exists():
            logger.warning("Knowledge base directory not found: %s", data_dir)
            return
        for f in path.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    for entry in data:
                        self.add_entry(entry)
                elif isinstance(data, dict):
                    self.add_entry(data)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load KB file %s: %s", f, e)

    def add_entry(self, entry: dict[str, Any]) -> None:
        """Add a knowledge entry and index its keywords."""
        idx = len(self._entries)
        self._entries.append(entry)
        # Index by keywords from title, category, tags
        keywords = set()
        for field in ("title", "category", "disease", "drug", "treatment"):
            val = entry.get(field, "")
            if val:
                keywords.update(val.lower().split())
        for tag in entry.get("tags", []):
            keywords.update(tag.lower().split())
        for kw in keywords:
            self._index.setdefault(kw, []).append(idx)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search knowledge base by keyword overlap."""
        if not self._entries:
            return []
        query_words = set(query.lower().split())
        scores: dict[int, float] = {}
        for word in query_words:
            for idx in self._index.get(word, []):
                scores[idx] = scores.get(idx, 0) + 1.0
        if not scores:
            # Fallback: substring match
            for i, entry in enumerate(self._entries):
                content = json.dumps(entry, ensure_ascii=False).lower()
                overlap = sum(1 for w in query_words if w in content)
                if overlap > 0:
                    scores[i] = overlap * 0.5
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [self._entries[idx] for idx, _ in ranked]

    def __len__(self) -> int:
        return len(self._entries)


# Global KB instance (lazy-loaded)
_kb: MedicalKnowledgeBase | None = None


def _get_kb() -> MedicalKnowledgeBase:
    global _kb
    if _kb is None:
        _kb = MedicalKnowledgeBase()
    return _kb


def init_medical_kb(data_dir: str) -> None:
    """Initialize the medical KB with data from a directory."""
    global _kb
    _kb = MedicalKnowledgeBase(data_dir)
    logger.info("Medical KB initialized with %d entries", len(_kb))


def query_medical_kb(query: str = "", top_k: int = 5) -> str:
    """查询医学知识库，返回相关的指南、药物信息或治疗方案。"""
    if not query:
        return "错误: 请提供查询内容"
    kb = _get_kb()
    if len(kb) == 0:
        return (
            f"⚠️ 医学知识库未加载。关于 '{query}' 的查询需要外部验证。\n"
            f"建议: 1) 提供具体数据来源 2) 引用权威指南 3) 区分事实与观点"
        )
    results = kb.search(query, top_k=top_k)
    if not results:
        return f"未找到与 '{query}' 相关的知识条目。建议查阅最新临床指南。"
    lines = [f"## 知识库查询结果 ({len(results)} 条)"]
    for i, entry in enumerate(results, 1):
        title = entry.get("title", "未命名")
        content = entry.get("content", entry.get("summary", ""))
        source = entry.get("source", "")
        lines.append(f"\n### {i}. {title}")
        if content:
            lines.append(content[:500])
        if source:
            lines.append(f"来源: {source}")
    return "\n".join(lines)


# Auto-register
def _register() -> None:
    registry = ToolRegistry()
    registry.register(
        name="medical_kb",
        description="查询医学知识库，获取指南、药物信息、治疗方案等结构化医学知识",
        handler=query_medical_kb,
        parameters=[
            ToolSchema(name="query", type="string", description="查询内容", required=True),
            ToolSchema(name="top_k", type="number", description="返回结果数量", required=False, default=5),
        ],
        category="medical",
    )


_register()
