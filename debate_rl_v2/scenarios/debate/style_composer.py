"""Debate Style Composer — translates numeric style dimensions to Chinese prompts.

Extracted from the old StrategyBridge (fixes God Object problem).
Implements BaseStyleComposer for the debate scenario.
"""

from __future__ import annotations

from debate_rl_v2.framework.strategy import BaseStyleComposer


class DebateStyleComposer(BaseStyleComposer):
    """Debate-specific style directive composer.

    Maps numeric style dimensions [0,1] to Chinese natural language
    directives injected into LLM system prompts.
    """

    # Threshold-based directive templates per role per dimension
    _TEMPLATES: dict[str, dict[str, dict[str, str]]] = {
        "proposer": {
            "assertiveness": {
                "high": "请用坚定自信的语气阐述你的方案，明确表达立场。",
                "low": "请用谦逊探讨的语气提出方案，留有商榷空间。",
            },
            "detail_level": {
                "high": "请提供详尽的技术细节、数据支撑和具体实施步骤。",
                "low": "请聚焦核心要点，保持简洁明了。",
            },
            "compliance_focus": {
                "high": "请特别注意确保方案完全符合所有既定规则和约束。",
            },
            "incorporation": {
                "high": "请充分吸纳挑战者的反馈，在方案中明确回应每个质疑点。",
                "low": "请坚持你的核心方案思路，用更有力的论据回应质疑。",
            },
        },
        "challenger": {
            "aggressiveness": {
                "high": "请发挥尖锐批评者角色，深入挖掘方案的根本性缺陷和逻辑漏洞。",
                "low": "请以建设性对话的方式温和提出你的疑虑。",
            },
            "specificity": {
                "high": "请精准指出具体的薄弱环节，引用方案中的特定表述进行批驳。",
            },
            "constructiveness": {
                "high": "在指出问题的同时，请提供你认为更优的替代方案或改进建议。",
            },
            "novelty": {
                "high": "请开辟新的质疑角度，提出此前未被讨论过的风险和问题。",
            },
        },
        "arbiter": {
            "strictness": {
                "high": "请以严格标准进行评判，不轻易给出高分。",
                "low": "请以鼓励进步为导向进行评判，认可合理的改进。",
            },
            "detail_feedback": {
                "high": "请给出详尽的评判理由，逐点分析优缺点。",
            },
            "consensus_bias": {
                "high": "如果双方观点已趋于一致且方案质量达标，请积极推动共识。",
                "low": "请保持审慎态度，除非方案确实优秀，否则鼓励继续辩论。",
            },
            "rule_emphasis": {
                "high": "请特别重视规则合规性，将其作为评判的首要标准。",
            },
        },
    }

    HIGH_THRESHOLD = 0.7
    LOW_THRESHOLD = 0.3

    def compose(self, role: str, style: dict[str, float]) -> str:
        """Generate style directive for the given role."""
        templates = self._TEMPLATES.get(role, {})
        if not templates or not style:
            return ""

        parts: list[str] = []
        for dim_name, value in style.items():
            dim_templates = templates.get(dim_name, {})
            if value > self.HIGH_THRESHOLD and "high" in dim_templates:
                parts.append(dim_templates["high"])
            elif value < self.LOW_THRESHOLD and "low" in dim_templates:
                parts.append(dim_templates["low"])

        return "\n".join(parts) if parts else ""
