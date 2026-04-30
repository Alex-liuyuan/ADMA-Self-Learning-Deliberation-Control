"""Debate scenario public exports with lazy loading."""

from __future__ import annotations

_EXPORTS = {
    # roles
    "DEBATE_ROLES": "debate_rl_v2.scenarios.debate.roles",
    "create_debate_role_registry": "debate_rl_v2.scenarios.debate.roles",
    "create_debate_registry": "debate_rl_v2.scenarios.debate.roles",
    # strategy
    "DebateStrategyBridge": "debate_rl_v2.scenarios.debate.strategy",
    "DebateStyleComposer": "debate_rl_v2.scenarios.debate.style_composer",
    # types (canonical location)
    "TextDebateState": "debate_rl_v2.scenarios.debate.types",
    "DebateTurn": "debate_rl_v2.scenarios.debate.types",
    "FusionRoundRecord": "debate_rl_v2.scenarios.debate.types",
    # prompts
    "DEBATE_SYSTEM_PROMPTS": "debate_rl_v2.scenarios.debate.prompts",
    "format_proposer_message": "debate_rl_v2.scenarios.debate.prompts",
    "format_challenger_message": "debate_rl_v2.scenarios.debate.prompts",
    "format_arbiter_message": "debate_rl_v2.scenarios.debate.prompts",
    "format_coordinator_message": "debate_rl_v2.scenarios.debate.prompts",
    "parse_proposer_response": "debate_rl_v2.scenarios.debate.prompts",
    "parse_challenger_response": "debate_rl_v2.scenarios.debate.prompts",
    "parse_arbiter_response": "debate_rl_v2.scenarios.debate.prompts",
    "parse_coordinator_response": "debate_rl_v2.scenarios.debate.prompts",
    # v3.0.1: new scenario components
    "DebateGameScenario": "debate_rl_v2.scenarios.debate.scenario",
    "DebateRewardComputer": "debate_rl_v2.scenarios.debate.reward",
    "DebateMechanismOrchestrator": "debate_rl_v2.scenarios.debate.mechanisms",
    "DebateObservationEncoder": "debate_rl_v2.scenarios.debate.observation",
}

__all__ = list(_EXPORTS) + ["create_debate_scenario"]


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = __import__(module_name, fromlist=[name])
    if name == "create_debate_registry":
        return getattr(module, "create_debate_role_registry")
    if name == "DEBATE_SYSTEM_PROMPTS":
        return getattr(module, "SYSTEM_PROMPTS")
    return getattr(module, name)


def create_debate_scenario(
    topic: str = "",
    context: str = "",
    rules: list[str] | None = None,
    max_rounds: int = 10,
    enable_compliance: bool = True,
) -> "ScenarioConfig":
    """创建辩论场景的完整 ScenarioConfig。

    Parameters
    ----------
    topic : str
        辩论主题。
    context : str
        背景信息。
    rules : list[str] | None
        活跃规则列表。
    max_rounds : int
        最大回合数。
    enable_compliance : bool
        是否启用合规验证。

    Returns
    -------
    ScenarioConfig
        可直接传给 ScenarioBuilder 的完整配置。
    """
    from debate_rl_v2.framework.scenario_builder import ScenarioConfig
    from debate_rl_v2.algorithms.role_observations import create_debate_observation_specs
    from debate_rl_v2.scenarios.debate.roles import create_debate_role_registry
    from debate_rl_v2.scenarios.debate.strategy import DebateStrategyBridge
    from debate_rl_v2.scenarios.debate.style_composer import DebateStyleComposer
    from debate_rl_v2.scenarios.debate.reward import DebateRewardComputer
    from debate_rl_v2.scenarios.debate.observation import DebateObservationEncoder
    from debate_rl_v2.scenarios.debate.mechanisms import DebateMechanismOrchestrator
    from debate_rl_v2.scenarios.debate.scenario import DebateGameScenario

    registry = create_debate_role_registry()
    bridge = DebateStrategyBridge(enable_compliance=enable_compliance)
    composer = DebateStyleComposer()
    reward = DebateRewardComputer()
    observation = DebateObservationEncoder()

    compliance_verifier = None
    if enable_compliance:
        try:
            from debate_rl_v2.scenarios.debate.compliance import DebateComplianceVerifier

            compliance_verifier = DebateComplianceVerifier()
        except ImportError:
            pass

    return ScenarioConfig(
        name="debate",
        role_registry=registry,
        strategy_bridge=bridge,
        style_composer=composer,
        reward_computer=reward,
        compliance_verifier=compliance_verifier,
        observation_encoder=observation,
        observation_specs=create_debate_observation_specs(),
        scenario_factory=lambda **kw: DebateGameScenario(
            topic=kw.get("topic", topic),
            context=kw.get("context", context),
            rules=kw.get("rules", rules),
            max_rounds=kw.get("max_rounds", max_rounds),
        ),
        mechanism_factory=lambda **kw: DebateMechanismOrchestrator(
            max_rounds=kw.get("max_rounds", max_rounds),
        ),
        online_param_dim=4,
        shared_obs_dim=observation.shared_obs_dim(),
        role_obs_dim=observation.role_obs_dim(),
    )
