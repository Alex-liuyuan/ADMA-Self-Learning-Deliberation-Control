"""ScenarioBuilder — one-stop factory for creating all scenario components.

Given a ScenarioConfig, creates LLMAgents, MADDPGAgentGroup,
OnlineParameterUpdater, ModeController, etc. — all correctly wired
for the specified scenario.

Usage::

    config = ScenarioConfig(
        name="negotiation",
        role_registry=registry,
        strategy_bridge=NegotiationBridge(),
        style_composer=NegotiationStyleComposer(),
    )
    builder = ScenarioBuilder(config)
    llm_agents = builder.create_llm_agents(client)
    rl_agents = builder.create_maddpg_agents()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from debate_rl_v2.framework.game_scenario import GameScenario
from debate_rl_v2.framework.knowledge import BaseKnowledgeAdapter
from debate_rl_v2.framework.mechanism import BaseMechanismOrchestrator
from debate_rl_v2.framework.observer import BaseGameObserver
from debate_rl_v2.framework.roles import RoleRegistry
from debate_rl_v2.framework.strategy import BaseStrategyBridge, BaseStyleComposer
from debate_rl_v2.framework.reward import BaseRewardComputer
from debate_rl_v2.framework.compliance import BaseComplianceVerifier
from debate_rl_v2.framework.observation import BaseObservationEncoder
from debate_rl_v2.algorithms.role_observations import RoleObservationSpec
from debate_rl_v2.logging_config import get_logger

logger = get_logger("framework.scenario_builder")


@dataclass
class ScenarioConfig:
    """Declarative scenario configuration.

    Defines all components needed for a complete RL-guided LLM scenario.
    """
    name: str
    role_registry: RoleRegistry
    strategy_bridge: BaseStrategyBridge
    style_composer: Optional[BaseStyleComposer] = None
    reward_computer: Optional[BaseRewardComputer] = None
    compliance_verifier: Optional[BaseComplianceVerifier] = None
    observation_encoder: Optional[BaseObservationEncoder] = None
    observation_specs: Optional[Dict[str, RoleObservationSpec]] = None
    scenario_factory: Optional[Callable[..., GameScenario]] = None
    observer_factory: Optional[Callable[..., BaseGameObserver]] = None
    mechanism_factory: Optional[Callable[..., BaseMechanismOrchestrator]] = None
    knowledge_factory: Optional[Callable[..., BaseKnowledgeAdapter]] = None
    runner_factory: Optional[Callable[..., Any]] = None
    fusion_env_factory: Optional[Callable[..., Any]] = None
    online_param_dim: int = 4
    # RL hyperparameters (defaults work for most scenarios)
    shared_obs_dim: int = 14
    role_obs_dim: int = 6
    hidden_dim: int = 128
    critic_hidden_dim: int = 256
    # Deep network architecture: "shallow" | "mlp" | "residual" | "transformer"
    architecture: str = "shallow"
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1


class ScenarioBuilder:
    """One-stop factory for creating all scenario components.

    Parameters
    ----------
    config : ScenarioConfig
        Complete scenario configuration.
    """

    def __init__(self, config: ScenarioConfig) -> None:
        self.config = config
        self._registry = config.role_registry

    def create_llm_agents(
        self,
        client: Any,
        fallback_clients: list | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create LLMAgent instances for all roles.

        Parameters
        ----------
        client : BaseLLMClient
            Primary LLM client.
        fallback_clients : list, optional
            Fallback LLM clients.

        Returns
        -------
        agents : dict
            {role_name: LLMAgent}
        """
        from debate_rl_v2.agents.llm_agent import LLMAgent

        agents = {}
        for role_def in self._registry.get_roles():
            agents[role_def.name] = LLMAgent(
                role=role_def.name,
                client=client,
                role_definition=role_def,
                fallback_clients=fallback_clients,
                **kwargs,
            )
        return agents

    def create_maddpg_agents(
        self,
        device: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Create MADDPGAgentGroup for all roles.

        RL agent names follow the convention: {role_name}_ctrl for base
        roles, and the coordinator role name as-is.

        Returns
        -------
        agent_group : MADDPGAgentGroup
        """
        from debate_rl_v2.agents.maddpg_agent import MADDPGAgentGroup

        cfg = self.config
        total_obs = cfg.shared_obs_dim + cfg.role_obs_dim

        obs_dims: Dict[str, int] = {}
        act_dims: Dict[str, int] = {}

        for role_def in self._registry.get_roles():
            # Coordinator and evaluator keep their names, others get _ctrl suffix
            if role_def.is_coordinator or role_def.is_evaluator:
                rl_name = role_def.name
            else:
                rl_name = f"{role_def.name}_ctrl"
            obs_dims[rl_name] = total_obs
            act_dims[rl_name] = role_def.action_dim

        return MADDPGAgentGroup(
            obs_dims=obs_dims,
            act_dims=act_dims,
            hidden_dim=cfg.hidden_dim,
            critic_hidden_dim=cfg.critic_hidden_dim,
            num_layers=cfg.num_layers,
            architecture=cfg.architecture,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            device=device,
            **kwargs,
        )

    def create_online_updater(self, **kwargs: Any) -> Any:
        """Create OnlineParameterUpdater for all roles.

        Returns
        -------
        updater : OnlineParameterUpdater
        """
        from debate_rl_v2.mode.online_updater import OnlineParameterUpdater

        roles = tuple(self._registry.role_names)
        return OnlineParameterUpdater(
            roles=roles,
            param_dim=self.config.online_param_dim,
            **kwargs,
        )

    def create_mode_controller(self, mode_config: Any = None) -> Any:
        """Create ModeController for dual-mode operation.

        Returns
        -------
        controller : ModeController
        """
        from debate_rl_v2.mode.controller import ModeController

        return ModeController(config=mode_config)

    def create_observation_tracker(self) -> Any:
        """Create a GenericRoleObservationTracker from config specs.

        Returns
        -------
        tracker : GenericRoleObservationTracker
        """
        from debate_rl_v2.algorithms.role_observations import (
            GenericRoleObservationTracker,
            RoleObservationSpec,
        )

        if self.config.observation_specs:
            return GenericRoleObservationTracker(self.config.observation_specs)

        # Auto-generate specs from role registry
        specs: Dict[str, RoleObservationSpec] = {}
        for role_def in self._registry.get_roles():
            rl_name = role_def.name if role_def.is_coordinator else f"{role_def.name}_ctrl"
            # Use style_dimensions as metric names, or default
            metrics = role_def.style_dimensions or ["quality", "confidence"]
            specs[rl_name] = RoleObservationSpec(
                name=rl_name,
                metrics=metrics,
                obs_dim=self.config.role_obs_dim,
            )
        return GenericRoleObservationTracker(specs)

    def create_multi_role_buffer(self) -> Any:
        """Create MultiRoleBuffer for all roles.

        Returns
        -------
        buffer : MultiRoleBuffer
        """
        from debate_rl_v2.algorithms.buffers import MultiRoleBuffer

        roles = tuple(self._registry.role_names)
        return MultiRoleBuffer(roles=roles)

    def create_shapley_credit(self, **kwargs: Any) -> Any:
        """Create ShapleyCredit for all roles.

        Returns
        -------
        credit : ShapleyCredit
        """
        from debate_rl_v2.algorithms.credit_assignment import ShapleyCredit

        roles = tuple(self._registry.role_names)
        return ShapleyCredit(
            num_agents=len(roles),
            roles=roles,
            **kwargs,
        )

    def create_observer(self, **kwargs: Any) -> BaseGameObserver:
        """Create a configured GameEngine observer."""
        if self.config.observer_factory is not None:
            return self.config.observer_factory(**kwargs)
        return BaseGameObserver()

    def create_mechanism_orchestrator(
        self,
        **kwargs: Any,
    ) -> BaseMechanismOrchestrator | None:
        """Create the optional scenario mechanism orchestrator."""
        if self.config.mechanism_factory is None:
            return None
        return self.config.mechanism_factory(**kwargs)

    def create_knowledge_adapter(
        self,
        **kwargs: Any,
    ) -> BaseKnowledgeAdapter | None:
        """Create the optional scenario knowledge adapter."""
        if self.config.knowledge_factory is None:
            return None
        return self.config.knowledge_factory(**kwargs)

    def create_scenario(self, **kwargs: Any) -> GameScenario:
        """Create the scenario instance configured for this builder."""
        if self.config.scenario_factory is None:
            raise ValueError("ScenarioConfig.scenario_factory is not configured")
        return self.config.scenario_factory(**kwargs)

    def create_game_engine(
        self,
        scenario: GameScenario | None = None,
        *,
        tool_registry: Any = None,
        observer: BaseGameObserver | None = None,
        mechanism_orchestrator: BaseMechanismOrchestrator | None = None,
        knowledge_adapter: BaseKnowledgeAdapter | None = None,
        max_rounds: int = 10,
        meta_interval: int = 3,
        **kwargs: Any,
    ) -> Any:
        """Create a GameEngine wired from ScenarioConfig components."""
        from debate_rl_v2.framework.game_engine import GameEngine

        scenario = scenario if scenario is not None else self.create_scenario(max_rounds=max_rounds)

        return GameEngine(
            scenario=scenario,
            role_registry=self._registry,
            strategy_bridge=self.config.strategy_bridge,
            tool_registry=tool_registry,
            observation_encoder=self.config.observation_encoder,
            observer=observer if observer is not None else self.create_observer(),
            mechanism_orchestrator=(
                mechanism_orchestrator
                if mechanism_orchestrator is not None
                else self.create_mechanism_orchestrator(max_rounds=max_rounds)
            ),
            knowledge_adapter=(
                knowledge_adapter
                if knowledge_adapter is not None
                else self.create_knowledge_adapter()
            ),
            reward_computer=self.config.reward_computer,
            max_rounds=max_rounds,
            meta_interval=meta_interval,
            **kwargs,
        )

    def create_dashboard_runner(self, **kwargs: Any) -> Any:
        """Create a scenario-provided dashboard runner.

        Framework stays domain-agnostic; scenario packages provide the factory.
        """
        if self.config.runner_factory is None:
            raise ValueError("ScenarioConfig.runner_factory is not configured")
        return self.config.runner_factory(**kwargs)

    def create_fusion_env(self, **kwargs: Any) -> Any:
        """Create a scenario-provided fusion environment."""
        if self.config.fusion_env_factory is None:
            raise ValueError("ScenarioConfig.fusion_env_factory is not configured")
        return self.config.fusion_env_factory(**kwargs)

    @property
    def role_names(self) -> list[str]:
        return self._registry.role_names

    @property
    def num_roles(self) -> int:
        return len(self._registry)
