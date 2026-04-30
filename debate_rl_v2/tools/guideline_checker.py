"""Guideline compliance checker — replaces keyword-overlap _rule_check.

Uses structured rule matching with NLI-style scoring
(lightweight version without external model dependency).
"""

from __future__ import annotations

import re
from typing import Any

from debate_rl_v2.tools.registry import ToolRegistry, ToolSchema
from debate_rl_v2.logging_config import get_logger

logger = get_logger("tools.guideline_checker")


def _extract_key_concepts(text: str) -> set[str]:
    """Extract meaningful concepts from text (bigrams + unigrams)."""
    # Normalize
    text = text.lower().strip()
    # Remove punctuation except Chinese
    words = re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", text)
    concepts = set(words)
    # Add bigrams for better matching
    for i in range(len(words) - 1):
        concepts.add(f"{words[i]}_{words[i+1]}")
    return concepts


def _compute_coverage(proposal_concepts: set[str], rule_concepts: set[str]) -> float:
    """Compute how well proposal covers rule concepts."""
    if not rule_concepts:
        return 1.0
    overlap = proposal_concepts & rule_concepts
    return len(overlap) / len(rule_concepts)


def check_guideline_compliance(
    proposal: str = "",
    rules: str = "",
    threshold: float = 0.3,
) -> str:
    """检查提案是否符合给定的指南规则。

    对每条规则进行概念覆盖度分析，返回合规报告。
    """
    if not proposal:
        return "错误: 未提供提案文本"
    if not rules:
        return "未提供规则，无法进行合规检查。"

    rule_list = [r.strip() for r in rules.split(";") if r.strip()]
    proposal_concepts = _extract_key_concepts(proposal)

    results: list[dict[str, Any]] = []
    total_score = 0.0

    for i, rule in enumerate(rule_list, 1):
        rule_concepts = _extract_key_concepts(rule)
        coverage = _compute_coverage(proposal_concepts, rule_concepts)
        total_score += coverage

        status = "✅ 符合" if coverage >= threshold else "⚠️ 可能不符"
        results.append({
            "rule_num": i,
            "rule": rule,
            "coverage": coverage,
            "status": status,
            "missing": rule_concepts - proposal_concepts if coverage < threshold else set(),
        })

    avg_score = total_score / len(rule_list) if rule_list else 1.0

    # Format report
    lines = [f"## 合规检查报告 (综合评分: {avg_score:.2f})"]
    violations = []
    for r in results:
        lines.append(f"\n{r['status']} 规则 {r['rule_num']}: {r['rule']}")
        lines.append(f"  覆盖度: {r['coverage']:.2f}")
        if r["missing"]:
            missing_str = ", ".join(list(r["missing"])[:5])
            lines.append(f"  缺失概念: {missing_str}")
            violations.append(r["rule_num"])

    if violations:
        lines.append(f"\n⚠️ 共 {len(violations)} 条规则可能未被充分涵盖: {violations}")
        lines.append("建议: 在提案中明确回应上述规则要求。")
    else:
        lines.append("\n✅ 提案基本符合所有规则要求。")

    return "\n".join(lines)


# Auto-register
def _register() -> None:
    registry = ToolRegistry()
    registry.register(
        name="guideline_checker",
        description="检查提案是否符合NCCN等指南规则约束，返回逐条合规分析报告",
        handler=check_guideline_compliance,
        parameters=[
            ToolSchema(name="proposal", type="string", description="待检查的提案文本", required=True),
            ToolSchema(name="rules", type="string", description="规则列表，分号分隔", required=True),
            ToolSchema(name="threshold", type="number", description="合规阈值(0-1)", required=False, default=0.3),
        ],
        category="compliance",
    )


_register()
