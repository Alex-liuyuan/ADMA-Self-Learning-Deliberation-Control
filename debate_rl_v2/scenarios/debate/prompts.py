"""Role-specific prompt templates for LLM debate agents.

Each role (Proposer, Challenger, Arbiter, Coordinator) has:
  - A system prompt defining its personality and constraints
  - A user-message formatter that converts debate state → text
  - An output parser that extracts structured actions from LLM JSON

v3 Enhancements:
  - Reflection & self-critique sections for deeper reasoning
  - Evidence citation requirements for stronger argumentation
  - Adaptive difficulty sections based on curriculum level
  - Pattern-aware instructions from cross-session learning
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ======================================================================
# System prompts
# ======================================================================

SYSTEM_PROMPTS: Dict[str, str] = {
    "proposer": """\
你是多智能体辩论系统中的 **提案者 (Proposer)**。

你的职责：
1. 根据辩论主题和上下文信息，提出有建设性的方案
2. 针对挑战者的质疑，改进和完善方案
3. 确保方案符合已有规则约束
4. 在保持方案核心价值的同时，积极吸纳合理建议

**推理要求**（v3增强）：
- 展示你的完整推理链：前提 → 推导 → 结论
- 关键论点必须引用具体数据、文献或案例作为证据
- 提出方案后，主动识别1-2个潜在弱点并给出你的应对思路
- 如果修改了上一轮方案，明确标注修改点和修改原因

输出要求：你必须以 **JSON** 格式回复，包含以下字段：
{
  "reasoning": "你的推理过程（思维链，至少3步）",
  "proposal": "你的具体方案文本",
  "key_points": ["要点1", "要点2", ...],
  "evidence": ["支撑证据1", "支撑证据2", ...],
  "self_critique": "你的方案的主要弱点和应对策略",
  "modifications": ["相比上轮的修改点1", ...],
  "confidence": 0.0到1.0之间的置信度
}
""",

    "challenger": """\
你是多智能体辩论系统中的 **挑战者 (Challenger)**。

你的职责：
1. 批判性地评估提案者的方案，找出潜在问题和风险
2. 提出有建设性的反驳意见和替代方案
3. 保持适度的对抗强度——既不妥协也不过度对立
4. 确保你的质疑有理有据，推动辩论向更优解前进

**推理要求**（v3增强）：
- 你的每个质疑必须明确指出：(a)被质疑的具体论点 (b)质疑的理由 (c)支撑质疑的证据
- 区分「逻辑漏洞」和「事实错误」，分别标注
- 避免重复已被充分回应的旧质疑，优先发现新问题
- 在批评的同时，至少提供一个具体的替代建议

输出要求：你必须以 **JSON** 格式回复，包含以下字段：
{
  "reasoning": "你的推理过程（思维链）",
  "challenge": "你的质疑/反驳文本",
  "key_concerns": [
    {"point": "被质疑的论点", "type": "逻辑|事实|可行性", "evidence": "反面证据"},
    ...
  ],
  "alternative": "你建议的替代方案（必填）",
  "new_angles": ["此前未提及的新质疑角度"],
  "confidence": 0.0到1.0之间的置信度
}
""",

    "arbiter": """\
你是多智能体辩论系统中的 **仲裁者 (Arbiter)**。

你的职责：
1. 公正地评估提案者和挑战者的论点
2. 判断哪一方的论证更有说服力
3. 监督辩论是否符合规则约束
4. 决定是否需要调整合规阈值
5. 推动辩论达成高质量共识

**评判要求**（v3增强）：
- 使用多维度评分框架：逻辑性、可行性、创新性、证据充分性、合规性
- 审查双方引用的证据质量，无证据支撑的论点应降权
- 评估挑战者的质疑是建设性的还是破坏性的
- 判断当前方案与上一轮相比是否有实质性进步
- 明确指出双方各自最强和最弱的论点

输出要求：你必须以 **JSON** 格式回复，包含以下字段：
{
  "reasoning": "你的评判推理",
  "verdict": "你的裁定文本",
  "dimension_scores": {
    "logic": 0.0到1.0, "feasibility": 0.0到1.0,
    "innovation": 0.0到1.0, "evidence": 0.0到1.0,
    "compliance": 0.0到1.0
  },
  "proposal_score": 0.0到1.0（提案者得分）,
  "challenge_score": 0.0到1.0（挑战者得分）,
  "strongest_points": {"proposer": "最强论点", "challenger": "最强论点"},
  "weakest_points": {"proposer": "最弱论点", "challenger": "最弱论点"},
  "improvement_vs_last": true/false,
  "action": "decrease_threshold" | "no_change" | "increase_threshold" | "boost_weights" | "decay_weights",
  "quality_score": 0.0到1.0（本轮辩论质量）,
  "consensus_reached": true/false
}
""",

    "coordinator": """\
你是多智能体辩论系统中的 **协调者 (Coordinator)**。

你的职责：
1. 监控整体辩论动态（对抗强度、合规性、进度）
2. 在宏观层面调整辩论参数
3. 在辩论陷入僵局时进行干预
4. 平衡效率与质量，确保辩论有序推进
5. 【v3】评估边际收益，在质量不再显著提升时建议结束

可用的元动作：
  0: 不做调整
  1: 提高学习率 η（加速适应）
  2: 降低学习率 η（稳定辩论）
  3: 提高对抗系数 α（增加博弈强度）
  4: 降低对抗系数 α（缓和对抗）
  5: 提高低阈值 τ_low（更早介入）
  6: 降低低阈值 τ_low（减少介入）
  7: 提高高阈值 τ_high（放宽切换条件）
  8: 降低高阈值 τ_high（收紧切换条件）
  9: 触发规则挖掘（从历史数据中发现新规则）

输出要求：你必须以 **JSON** 格式回复，包含以下字段：
{
  "reasoning": "你对系统状态的分析",
  "action": 0到9之间的整数,
  "expected_effect": "预期效果描述",
  "quality_trend": "improving|stable|declining",
  "marginal_return": "是否检测到边际收益递减(true/false)",
  "recommend_early_stop": false
}
""",
}


# ======================================================================
# User message formatters
# ======================================================================


def format_proposer_message(
    topic: str,
    context: str,
    round_num: int,
    max_rounds: int,
    prev_proposal: str,
    challenge: str,
    compliance: float,
    disagreement: float,
    rules: List[str],
    history: Optional[List[Dict]] = None,
) -> str:
    """Build the user prompt for the Proposer."""
    rules_text = "\n".join(f"  - {r}" for r in rules) if rules else "  （无活跃规则）"
    history_text = _format_history(history) if history else ""

    return f"""\
## 辩论主题
{topic}

## 背景信息
{context}

## 当前状态
- 轮次：第 {round_num} / {max_rounds} 轮
- 合规得分：{compliance:.3f}
- 分歧度：{disagreement:.3f}
- 活跃规则：
{rules_text}

## 你上一轮的提案
{prev_proposal or "（首轮，尚无提案）"}

## 挑战者的质疑
{challenge or "（首轮，尚无质疑）"}
{history_text}
请根据以上信息提出你的方案。"""


def format_challenger_message(
    topic: str,
    context: str,
    round_num: int,
    max_rounds: int,
    proposal: str,
    prev_challenge: str,
    disagreement: float,
    lambda_adv: float,
    mode: str,
    rules: List[str],
    history: Optional[List[Dict]] = None,
) -> str:
    """Build the user prompt for the Challenger."""
    rules_text = "\n".join(f"  - {r}" for r in rules) if rules else "  （无活跃规则）"
    history_text = _format_history(history) if history else ""

    return f"""\
## 辩论主题
{topic}

## 背景信息
{context}

## 当前状态
- 轮次：第 {round_num} / {max_rounds} 轮
- 分歧度：{disagreement:.3f}
- 对抗强度 λ：{lambda_adv:.3f}
- 当前模式：{mode}
- 活跃规则：
{rules_text}

## 提案者的方案
{proposal}

## 你上一轮的质疑
{prev_challenge or "（首轮，尚无质疑）"}
{history_text}
请对提案进行批判性分析。"""


def format_arbiter_message(
    topic: str,
    context: str,
    round_num: int,
    max_rounds: int,
    proposal: str,
    challenge: str,
    compliance: float,
    disagreement: float,
    rules: List[str],
    history: Optional[List[Dict]] = None,
) -> str:
    """Build the user prompt for the Arbiter."""
    rules_text = "\n".join(f"  - {r}" for r in rules) if rules else "  （无活跃规则）"
    history_text = _format_history(history) if history else ""

    return f"""\
## 辩论主题
{topic}

## 背景信息
{context}

## 当前状态
- 轮次：第 {round_num} / {max_rounds} 轮
- 合规得分：{compliance:.3f}
- 分歧度：{disagreement:.3f}
- 活跃规则：
{rules_text}

## 提案者方案
{proposal}

## 挑战者质疑
{challenge}
{history_text}
请公正地评判本轮辩论。"""


def format_coordinator_message(
    round_num: int,
    max_rounds: int,
    disagreement: float,
    compliance: float,
    lambda_adv: float,
    da_active: bool,
    trend: str,
    recent_quality: float,
) -> str:
    """Build the user prompt for the Coordinator."""
    return f"""\
## 系统状态
- 进度：第 {round_num} / {max_rounds} 轮
- 分歧度：{disagreement:.3f}
- 合规得分：{compliance:.3f}
- 对抗强度 λ：{lambda_adv:.3f}
- 魔鬼代言人：{"激活" if da_active else "未激活"}
- 近期趋势：{trend}
- 辩论质量：{recent_quality:.3f}

请选择最优的元动作来优化辩论进程。"""


# ======================================================================
# Response parsers
# ======================================================================


def parse_proposer_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse proposer JSON response into a standardised dict."""
    return {
        "reasoning": data.get("reasoning", ""),
        "proposal": data.get("proposal", ""),
        "key_points": data.get("key_points", []),
        "evidence": data.get("evidence", []),
        "self_critique": data.get("self_critique", ""),
        "modifications": data.get("modifications", []),
        "confidence": float(data.get("confidence", 0.5)),
    }


def parse_challenger_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse challenger JSON response."""
    # Handle both old format (list of strings) and new format (list of dicts)
    raw_concerns = data.get("key_concerns", [])
    concerns = []
    for c in raw_concerns:
        if isinstance(c, dict):
            concerns.append(c)
        else:
            concerns.append({"point": str(c), "type": "general", "evidence": ""})

    return {
        "reasoning": data.get("reasoning", ""),
        "challenge": data.get("challenge", ""),
        "key_concerns": concerns,
        "alternative": data.get("alternative", ""),
        "new_angles": data.get("new_angles", []),
        "confidence": float(data.get("confidence", 0.5)),
    }


def parse_arbiter_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse arbiter JSON response."""
    action_map = {
        "decrease_threshold": 0,
        "no_change": 1,
        "increase_threshold": 2,
        "boost_weights": 3,
        "decay_weights": 4,
    }
    raw_action = data.get("action", "no_change")
    action_idx = action_map.get(raw_action, 1) if isinstance(raw_action, str) else int(raw_action)

    return {
        "reasoning": data.get("reasoning", ""),
        "verdict": data.get("verdict", ""),
        "dimension_scores": data.get("dimension_scores", {}),
        "proposal_score": float(data.get("proposal_score", 0.5)),
        "challenge_score": float(data.get("challenge_score", 0.5)),
        "strongest_points": data.get("strongest_points", {}),
        "weakest_points": data.get("weakest_points", {}),
        "improvement_vs_last": bool(data.get("improvement_vs_last", False)),
        "action_idx": action_idx,
        "quality_score": float(data.get("quality_score", 0.5)),
        "consensus_reached": bool(data.get("consensus_reached", False)),
    }


def parse_coordinator_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse coordinator JSON response."""
    action = data.get("action", 0)
    if isinstance(action, str):
        action = int(action)
    action = max(0, min(9, action))

    return {
        "reasoning": data.get("reasoning", ""),
        "action_idx": action,
        "expected_effect": data.get("expected_effect", ""),
        "quality_trend": data.get("quality_trend", "stable"),
        "marginal_return": bool(data.get("marginal_return", False)),
        "recommend_early_stop": bool(data.get("recommend_early_stop", False)),
    }


# ======================================================================
# Helpers
# ======================================================================


def _format_history(history: Optional[List[Dict]], max_turns: int = 5) -> str:
    """Format recent debate history into a readable summary."""
    if not history:
        return ""
    recent = history[-max_turns:]
    lines = ["\n## 近期辩论历史"]
    for i, turn in enumerate(recent):
        r = turn.get("round", "?")
        lines.append(f"\n### 第 {r} 轮")
        if "proposal" in turn:
            lines.append(f"提案：{turn['proposal']}")
        if "challenge" in turn:
            lines.append(f"质疑：{turn['challenge']}")
        if "verdict" in turn:
            lines.append(f"裁定：{turn['verdict']}")
    return "\n".join(lines)


# ======================================================================
# v3: Enhanced prompt injection helpers
# ======================================================================


def inject_memory_context(
    base_prompt: str,
    memory_context: str,
    pattern_context: str = "",
) -> str:
    """Inject memory and pattern context into a user prompt.

    Parameters
    ----------
    base_prompt : str
        The original formatted user prompt.
    memory_context : str
        Output from MemoryManager.build_context().
    pattern_context : str
        Output from DebatePatternTracker.build_context().

    Returns
    -------
    enhanced_prompt : str
        Prompt with memory and pattern sections appended.
    """
    parts = [base_prompt]
    if memory_context:
        parts.append(f"\n{memory_context}")
    if pattern_context:
        parts.append(f"\n{pattern_context}")
    return "\n".join(parts)


def inject_style_directive(
    base_prompt: str,
    style_directive: str,
) -> str:
    """Inject RL-generated style directive into a user prompt.

    Parameters
    ----------
    base_prompt : str
        The original formatted user prompt.
    style_directive : str
        Output from StrategyBridge.compose_*_style().

    Returns
    -------
    enhanced_prompt : str
        Prompt with style section prepended.
    """
    if not style_directive:
        return base_prompt
    return f"## 本轮策略指导\n{style_directive}\n\n{base_prompt}"


def get_evolvable_prompt(role: str) -> str:
    """获取可进化的基础提示词（用于初始化种群）。

    Returns the base system prompt for a role, which serves as the
    seed for prompt evolution population initialization.
    """
    if role not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown role '{role}'. Available: {list(SYSTEM_PROMPTS.keys())}")
    return SYSTEM_PROMPTS[role]


def format_difficulty_context(
    level: int,
    level_name: str,
    special_instructions: str = "",
) -> str:
    """Format curriculum difficulty context for prompt injection."""
    difficulty_labels = {
        1: "基础难度 — 请聚焦核心论点，保持清晰简洁",
        2: "进阶难度 — 请注意规则之间的冲突和权衡",
        3: "高级难度 — 请应对动态规则变化，保持策略灵活性",
        4: "专家难度 — 请在时间和资源约束下最大化方案质量",
        5: "对抗难度 — 请发挥最高水平，面对经验丰富的对手",
    }
    context = f"## 当前难度\n{difficulty_labels.get(level, '标准')}"
    if special_instructions:
        context += f"\n{special_instructions}"
    return context

