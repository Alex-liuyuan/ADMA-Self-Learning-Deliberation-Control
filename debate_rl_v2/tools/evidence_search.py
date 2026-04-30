"""Evidence search tool — literature and evidence retrieval."""

from __future__ import annotations

from debate_rl_v2.tools.registry import ToolRegistry, ToolSchema
from debate_rl_v2.logging_config import get_logger

logger = get_logger("tools.evidence_search")


def search_evidence(query: str = "", source: str = "all", max_results: int = 5) -> str:
    """检索文献证据，支持PubMed、临床试验等来源。

    当前为本地模拟版本。生产环境可对接PubMed API、Semantic Scholar等。
    """
    if not query:
        return "错误: 请提供检索关键词"

    # Placeholder — in production, this would call PubMed/Semantic Scholar API
    return (
        f"📚 证据检索: '{query}' (来源: {source})\n\n"
        f"⚠️ 当前为离线模式，无法访问外部文献数据库。\n"
        f"建议:\n"
        f"  1. 引用已知的权威指南（如NCCN、ESMO、中国临床肿瘤学会指南）\n"
        f"  2. 标注证据等级（I-IV级）和推荐强度（A-D级）\n"
        f"  3. 注明文献发表年份，优先引用近5年研究\n"
        f"  4. 区分随机对照试验(RCT)、荟萃分析和专家共识的证据权重"
    )


def _register() -> None:
    registry = ToolRegistry()
    registry.register(
        name="evidence_search",
        description="检索医学文献证据，支持PubMed、临床试验等来源",
        handler=search_evidence,
        parameters=[
            ToolSchema(name="query", type="string", description="检索关键词", required=True),
            ToolSchema(name="source", type="string", description="来源(pubmed/trials/all)", required=False, default="all"),
            ToolSchema(name="max_results", type="number", description="最大结果数", required=False, default=5),
        ],
        category="research",
    )


_register()
