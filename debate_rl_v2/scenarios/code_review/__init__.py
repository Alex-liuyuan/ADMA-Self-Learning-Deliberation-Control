"""Code Review scenario — demonstrates framework generality.

Three roles:
  - Author: submits code and responds to feedback
  - Reviewer: identifies issues, suggests improvements
  - Maintainer: evaluates overall quality, decides merge/reject

This proves the framework is not debate-specific.
Now includes full RL integration via ScenarioBuilder.
"""

from __future__ import annotations

from typing import Any

from debate_rl_v2.framework.roles import RoleDefinition, RoleRegistry
from debate_rl_v2.framework.types import CollaborationState
from debate_rl_v2.framework.strategy import BaseStyleComposer


CODE_REVIEW_ROLES: dict[str, RoleDefinition] = {
    "author": RoleDefinition(
        name="author",
        description="Submits code changes and addresses reviewer feedback",
        system_prompt=(
            "You are the code **Author**.\n"
            "Submit your code changes, explain design decisions, "
            "and address reviewer feedback.\n"
            'Output JSON: {"code": "...", "explanation": "...", "confidence": 0.0-1.0}'
        ),
        phase="propose",
        action_dim=3,
        style_dimensions=["defensiveness", "detail_level", "refactor_willingness"],
        output_schema={"code": "string", "explanation": "string", "confidence": "number"},
    ),
    "reviewer": RoleDefinition(
        name="reviewer",
        description="Reviews code for bugs, style issues, and design problems",
        system_prompt=(
            "You are the code **Reviewer**.\n"
            "Identify bugs, style issues, performance problems, and design flaws.\n"
            'Output JSON: {"issues": "...", "suggestions": "...", "confidence": 0.0-1.0}'
        ),
        phase="challenge",
        action_dim=3,
        style_dimensions=["thoroughness", "constructiveness", "strictness"],
        output_schema={"issues": "string", "suggestions": "string", "confidence": "number"},
    ),
    "maintainer": RoleDefinition(
        name="maintainer",
        description="Evaluates overall code quality and decides merge/reject",
        system_prompt=(
            "You are the **Maintainer**.\n"
            "Evaluate code quality, review completeness, and decide: merge, request changes, or reject.\n"
            'Output JSON: {"verdict": "...", "quality_score": 0.0-1.0, '
            '"decision": "merge|changes|reject", "reasoning": "..."}'
        ),
        phase="evaluate",
        action_dim=3,
        is_evaluator=True,
        style_dimensions=["strictness", "detail_feedback"],
        output_schema={
            "verdict": "string", "quality_score": "number",
            "decision": "string", "reasoning": "string",
        },
    ),
}


def create_code_review_registry() -> RoleRegistry:
    registry = RoleRegistry()
    for role in CODE_REVIEW_ROLES.values():
        registry.register(role)
    return registry


class CodeReviewStyleComposer(BaseStyleComposer):
    """Style composer for code review scenario."""

    _TEMPLATES = {
        "author": {
            "defensiveness": {
                "high": "Defend your design decisions firmly with technical justification.",
                "low": "Be open to suggestions and willing to refactor.",
            },
            "detail_level": {
                "high": "Provide detailed explanations for every design choice.",
                "low": "Keep explanations concise, focus on the 'why'.",
            },
        },
        "reviewer": {
            "thoroughness": {
                "high": "Review every line carefully. Check edge cases, error handling, and performance.",
                "low": "Focus on the most critical issues only.",
            },
            "constructiveness": {
                "high": "For every issue found, suggest a concrete fix or alternative.",
            },
            "strictness": {
                "high": "Apply strict coding standards. Flag any deviation.",
                "low": "Focus on correctness over style. Be pragmatic.",
            },
        },
        "maintainer": {
            "strictness": {
                "high": "Require all issues to be resolved before merge.",
                "low": "Allow minor issues to be fixed in follow-up PRs.",
            },
        },
    }

    def compose(self, role: str, style: dict[str, float]) -> str:
        templates = self._TEMPLATES.get(role, {})
        parts = []
        for dim, val in style.items():
            t = templates.get(dim, {})
            if val > 0.7 and "high" in t:
                parts.append(t["high"])
            elif val < 0.3 and "low" in t:
                parts.append(t["low"])
        return "\n".join(parts)


class CodeReviewState(CollaborationState):
    """Extended state for code review."""
    code: str = ""
    issues_found: int = 0
    issues_resolved: int = 0
    decision: str = "pending"


def create_code_review_scenario() -> "ScenarioConfig":
    """Convenience factory: create a complete code review ScenarioConfig.

    Returns
    -------
    ScenarioConfig
        Ready-to-use code review scenario configuration.
    """
    from debate_rl_v2.framework.scenario_builder import ScenarioConfig
    from debate_rl_v2.scenarios.code_review.strategy import CodeReviewStrategyBridge
    from debate_rl_v2.scenarios.code_review.observation import CodeReviewObservationEncoder
    from debate_rl_v2.algorithms.role_observations import RoleObservationSpec

    registry = create_code_review_registry()
    bridge = CodeReviewStrategyBridge()
    composer = CodeReviewStyleComposer()
    encoder = CodeReviewObservationEncoder()

    obs_specs = {
        "author_ctrl": RoleObservationSpec(
            name="author_ctrl",
            metrics=["defensiveness", "detail_level", "refactor_willingness"],
            obs_dim=6,
        ),
        "reviewer_ctrl": RoleObservationSpec(
            name="reviewer_ctrl",
            metrics=["thoroughness", "constructiveness", "strictness"],
            obs_dim=6,
        ),
        "maintainer": RoleObservationSpec(
            name="maintainer",
            metrics=["strictness", "detail_feedback"],
            obs_dim=6,
        ),
    }

    return ScenarioConfig(
        name="code_review",
        role_registry=registry,
        strategy_bridge=bridge,
        style_composer=composer,
        observation_encoder=encoder,
        observation_specs=obs_specs,
        online_param_dim=3,
        shared_obs_dim=8,
        role_obs_dim=6,
    )


__all__ = [
    "CODE_REVIEW_ROLES",
    "create_code_review_registry",
    "CodeReviewStyleComposer",
    "CodeReviewState",
    "create_code_review_scenario",
]
