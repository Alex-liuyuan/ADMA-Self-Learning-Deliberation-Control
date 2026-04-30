"""Generic Multi-Agent Collaboration Framework public exports.

Keep this package lightweight: importing ``debate_rl_v2.framework.roles`` should
not eagerly import RL training modules or optional heavy dependencies.
"""

from __future__ import annotations

_EXPORTS = {
    "CollaborationState": "debate_rl_v2.framework.types",
    "RoundRecord": "debate_rl_v2.framework.types",
    "StrategySignals": "debate_rl_v2.framework.types",
    "ComplianceResult": "debate_rl_v2.framework.types",
    "AgentMessage": "debate_rl_v2.framework.types",
    "InteractionPhase": "debate_rl_v2.framework.types",
    "RoleDefinition": "debate_rl_v2.framework.roles",
    "RoleRegistry": "debate_rl_v2.framework.roles",
    "BaseStrategyBridge": "debate_rl_v2.framework.strategy",
    "BaseStyleComposer": "debate_rl_v2.framework.strategy",
    "BaseComplianceVerifier": "debate_rl_v2.framework.compliance",
    "StyleDimension": "debate_rl_v2.framework.compliance",
    "BaseRewardComputer": "debate_rl_v2.framework.reward",
    "RewardWeights": "debate_rl_v2.framework.reward",
    "estimate_tokens": "debate_rl_v2.framework.tokenizer",
    "estimate_messages_tokens": "debate_rl_v2.framework.tokenizer",
    "BaseObservationEncoder": "debate_rl_v2.framework.observation",
    "BaseGameObserver": "debate_rl_v2.framework.observer",
    "BaseMechanismOrchestrator": "debate_rl_v2.framework.mechanism",
    "MechanismSnapshot": "debate_rl_v2.framework.mechanism",
    "BaseKnowledgeAdapter": "debate_rl_v2.framework.knowledge",
    "ScenarioConfig": "debate_rl_v2.framework.scenario_builder",
    "ScenarioBuilder": "debate_rl_v2.framework.scenario_builder",
    "GameToolRegistry": "debate_rl_v2.framework.tool_registry",
    "ToolSpec": "debate_rl_v2.framework.tool_registry",
    "GameToolContext": "debate_rl_v2.framework.tool_context",
    "GameScenario": "debate_rl_v2.framework.game_scenario",
    "GameEngine": "debate_rl_v2.framework.game_engine",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = __import__(module_name, fromlist=[name])
    return getattr(module, name)
