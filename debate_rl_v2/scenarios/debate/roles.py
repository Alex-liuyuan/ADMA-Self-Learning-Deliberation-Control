"""Debate role definitions — concrete RoleDefinition instances.

Registers the four MDT debate roles with the generic RoleRegistry.
Also provides a factory function for quick setup.
"""

from __future__ import annotations

from debate_rl_v2.framework.roles import RoleDefinition, RoleRegistry


DEBATE_ROLES: dict[str, RoleDefinition] = {
    "proposer": RoleDefinition(
        name="proposer",
        description="Generates and iteratively improves proposals based on feedback",
        system_prompt=(
            "你是多智能体协作系统中的 **提案者 (Proposer)**。\n"
            "职责：提出方案、回应质疑、迭代改进。\n"
            "输出JSON：{\"reasoning\": \"...\", \"proposal\": \"...\", "
            "\"confidence\": 0.0-1.0}"
        ),
        phase="propose",
        action_dim=4,
        style_dimensions=["assertiveness", "detail_level", "compliance_focus", "incorporation"],
        output_schema={
            "reasoning": "string",
            "proposal": "string",
            "confidence": "number",
        },
    ),
    "challenger": RoleDefinition(
        name="challenger",
        description="Critically evaluates proposals, identifies risks and weaknesses",
        system_prompt=(
            "你是多智能体协作系统中的 **挑战者 (Challenger)**。\n"
            "职责：批判性评估方案、指出风险、提供替代建议。\n"
            "输出JSON：{\"reasoning\": \"...\", \"challenge\": \"...\", "
            "\"confidence\": 0.0-1.0}"
        ),
        phase="challenge",
        action_dim=4,
        style_dimensions=["aggressiveness", "specificity", "constructiveness", "novelty"],
        output_schema={
            "reasoning": "string",
            "challenge": "string",
            "confidence": "number",
        },
    ),
    "arbiter": RoleDefinition(
        name="arbiter",
        description="Evaluates proposal quality and challenge validity, judges compliance",
        system_prompt=(
            "你是多智能体协作系统中的 **仲裁者 (Arbiter)**。\n"
            "职责：评判方案质量、挑战有效性、规则合规性。\n"
            "输出JSON：{\"verdict\": \"...\", \"quality_score\": 0.0-1.0, "
            "\"proposal_score\": 0.0-1.0, \"challenge_score\": 0.0-1.0, "
            "\"consensus_reached\": bool}"
        ),
        phase="evaluate",
        action_dim=4,
        is_evaluator=True,
        style_dimensions=["strictness", "detail_feedback", "consensus_bias", "rule_emphasis"],
        output_schema={
            "verdict": "string",
            "quality_score": "number",
            "proposal_score": "number",
            "challenge_score": "number",
            "consensus_reached": "boolean",
            "reasoning": "string",
        },
    ),
    "coordinator": RoleDefinition(
        name="coordinator",
        description="Meta-controller that adjusts collaboration parameters",
        system_prompt=(
            "你是多智能体协作系统中的 **协调者 (Coordinator)**。\n"
            "职责：监控协作进展、调整参数、决定何时终止。\n"
            "输出JSON：{\"action_idx\": 0-9, \"reasoning\": \"...\", "
            "\"expected_effect\": \"...\"}"
        ),
        phase="coordinate",
        action_dim=5,
        is_coordinator=True,
        style_dimensions=[],
        output_schema={
            "action_idx": "integer",
            "reasoning": "string",
            "expected_effect": "string",
        },
    ),
}


def create_debate_role_registry() -> RoleRegistry:
    """Create a RoleRegistry pre-populated with the four debate roles."""
    registry = RoleRegistry()
    for role in DEBATE_ROLES.values():
        registry.register(role)
    return registry
