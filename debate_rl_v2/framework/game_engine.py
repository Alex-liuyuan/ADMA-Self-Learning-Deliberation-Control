"""通用博弈引擎 — 驱动任意 GameScenario 的完整 episode。

整合：
  - GameScenario: 场景逻辑
  - RoleRegistry: 角色定义
  - GameToolRegistry: 工具注册表
  - BaseStrategyBridge: RL→LLM 策略翻译
  - LLMAgent (with ToolAugmentedAgentLoop): 工具增强智能体

执行流程（每个 episode）：
  1. scenario.create_episode() → 初始状态
  2. 循环（每回合）：
     a. observation_encoder.encode() → RL 观测
     b. rl_agents.act() → 连续动作
     c. strategy_bridge.translate() → 策略信号
     d. 对每个角色：scenario.build_role_prompt() → prompt → llm_agent.act()
     e. scenario.update_state() → 更新状态
     f. scenario.check_terminal() → 是否结束
  3. scenario.compute_rewards(ctx=GameToolContext) → 奖励
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

from debate_rl_v2.framework.game_scenario import GameScenario
from debate_rl_v2.framework.knowledge import BaseKnowledgeAdapter
from debate_rl_v2.framework.mechanism import BaseMechanismOrchestrator, MechanismSnapshot
from debate_rl_v2.framework.observation import BaseObservationEncoder
from debate_rl_v2.framework.observer import BaseGameObserver
from debate_rl_v2.framework.reward import BaseRewardComputer
from debate_rl_v2.framework.roles import RoleRegistry
from debate_rl_v2.framework.strategy import BaseStrategyBridge
from debate_rl_v2.framework.tool_context import GameToolContext
from debate_rl_v2.framework.tool_registry import GameToolRegistry
from debate_rl_v2.framework.types import CollaborationState, StrategySignals
from debate_rl_v2.logging_config import get_logger

if TYPE_CHECKING:
    from debate_rl_v2.agents.llm_agent import LLMAgent
    from debate_rl_v2.algorithms.maddpg_agent import MADDPGAgentGroup

logger = get_logger("framework.game_engine")


class GameEngine:
    """通用博弈引擎 — 驱动任意 GameScenario 的完整 episode。

    Parameters
    ----------
    scenario : GameScenario
        场景实现
    role_registry : RoleRegistry
        角色定义
    strategy_bridge : BaseStrategyBridge | None
        RL→LLM 策略翻译（纯 LLM 模式可为 None）
    tool_registry : GameToolRegistry | None
        工具注册表（None 时自动创建空注册表）
    observation_encoder : BaseObservationEncoder | None
        观测编码器（纯 LLM 模式可为 None）
    max_rounds : int
        最大回合数
    meta_interval : int
        协调者每 N 回合行动一次
    """

    def __init__(
        self,
        scenario: GameScenario,
        role_registry: RoleRegistry,
        strategy_bridge: BaseStrategyBridge | None = None,
        tool_registry: GameToolRegistry | None = None,
        observation_encoder: BaseObservationEncoder | None = None,
        observer: BaseGameObserver | None = None,
        mechanism_orchestrator: BaseMechanismOrchestrator | None = None,
        knowledge_adapter: BaseKnowledgeAdapter | None = None,
        reward_computer: BaseRewardComputer | None = None,
        max_rounds: int = 10,
        meta_interval: int = 3,
    ) -> None:
        self.scenario = scenario
        self.role_registry = role_registry
        self.strategy_bridge = strategy_bridge
        self.tool_registry = tool_registry or GameToolRegistry()
        self.observation_encoder = observation_encoder
        self.observer = observer or BaseGameObserver()
        self.mechanism_orchestrator = mechanism_orchestrator
        self.knowledge_adapter = knowledge_adapter
        self.reward_computer = reward_computer
        self.max_rounds = max_rounds
        self.meta_interval = meta_interval

        # 将 max_rounds 注入 scenario，供 check_terminal 最小轮数判断
        if hasattr(self.scenario, "_max_rounds"):
            self.scenario._max_rounds = max_rounds

        # 注册场景工具
        self.scenario.register_tools(self.tool_registry)

    # ── 公开接口 ──────────────────────────────────────────────────────

    def run_episode(
        self,
        llm_agents: dict[str, LLMAgent | Any],
        rl_agents: MADDPGAgentGroup | None = None,
        explore: bool = True,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """运行一个完整的博弈 episode。

        Parameters
        ----------
        llm_agents : dict
            {role_name: LLMAgent}
        rl_agents : MADDPGAgentGroup | None
            RL 策略控制器（纯 LLM 模式为 None）
        explore : bool
            RL 是否使用探索噪声
        verbose : bool
            是否输出日志

        Returns
        -------
        dict
            Episode 结果，包含 transcript, rewards, transitions 等
        """
        # 初始化 episode
        episode_ctx = self.scenario.create_episode()
        state = CollaborationState()
        state.metadata.update(episode_ctx)

        if verbose:
            logger.info("Episode started: %s", episode_ctx.get("topic", "unknown"))

        self._reset_components(llm_agents, rl_agents)

        history: list[dict[str, Any]] = []
        transitions: list[dict[str, Any]] = []
        self.observer.on_episode_start(episode_ctx, state)

        # 主循环
        for round_num in range(1, self.max_rounds + 1):
            state.round_num = round_num
            if self.observer.should_stop(round_num, state):
                break
            self.observer.on_round_start(round_num, self.max_rounds, state)

            # Step 1: RL 观测 + 动作
            rl_result = self._step_rl_observe(state, round_num, rl_agents, explore)

            # Step 2: 策略翻译 + 应用
            signals = self._step_translate_strategy(rl_result.rl_actions, llm_agents)

            # Step 3: 知识适配器 — 回合前
            knowledge_before = self._step_knowledge_before(state, round_num)

            # Step 4: LLM 执行
            role_outputs = self._step_execute_llm(round_num, state, llm_agents)

            # Step 5: 更新状态
            self._step_update_state(state, round_num, role_outputs, history)

            # Step 6: 合规验证
            compliance = self._step_verify_compliance(signals, role_outputs, state)

            # Step 7: 机制编排
            mechanism_dict = self._step_mechanism_update(
                state, round_num, role_outputs, history,
            )

            # Step 8: 知识适配器 — 回合后
            knowledge_after = self._step_knowledge_after(
                state, round_num, role_outputs, history,
            )

            # Step 9: 记录
            round_record = self._build_round_record(
                round_num, role_outputs, state, mechanism_dict,
                knowledge_after, signals, compliance,
            )
            history.append(round_record)
            self.observer.on_round_end(round_num, state, role_outputs)

            # Step 10: RL transition
            self._record_transition(
                transitions, rl_result, signals, mechanism_dict,
                knowledge_after, role_outputs, compliance,
                state, round_num, rl_agents,
            )

            # Step 10: 终止检查
            if self.scenario.check_terminal(state):
                state.is_terminal = True
                if transitions:
                    transitions[-1]["done"] = True
                if verbose:
                    logger.info(
                        "Terminal at round %d (quality=%.2f)",
                        round_num, state.quality_score,
                    )
                break

        # 后处理
        return self._step_post_episode(
            state, history, transitions, episode_ctx, verbose,
        )

    # ── 内部步骤方法 ──────────────────────────────────────────────────

    def _reset_components(
        self,
        llm_agents: dict[str, Any],
        rl_agents: Any,
    ) -> None:
        """重置所有组件到初始状态。"""
        for agent in llm_agents.values():
            if hasattr(agent, "reset"):
                agent.reset()
        if self.observation_encoder:
            self.observation_encoder.reset()
        if self.strategy_bridge:
            self.strategy_bridge.reset()
        if self.mechanism_orchestrator:
            self.mechanism_orchestrator.reset()
        if self.knowledge_adapter:
            self.knowledge_adapter.reset()

    def _step_rl_observe(
        self,
        state: CollaborationState,
        round_num: int,
        rl_agents: MADDPGAgentGroup | None,
        explore: bool,
    ) -> _RLStepResult:
        """Step 1: RL 观测编码和动作生成。"""
        shared_obs = None
        role_obs: dict[str, np.ndarray] = {}
        rl_actions: dict[str, np.ndarray] = {}

        if self.observation_encoder and rl_agents is not None:
            shared_obs = self.observation_encoder.encode_shared(
                state, round_num, self.max_rounds,
            )
            for role_name in rl_agents.agent_names:
                encoded = self.observation_encoder.encode_role(shared_obs, role_name)
                role_obs[role_name] = encoded
                rl_actions[role_name] = rl_agents[role_name].act(encoded, explore=explore)

        return _RLStepResult(
            shared_obs=shared_obs,
            role_obs=role_obs,
            rl_actions=rl_actions,
        )

    def _step_translate_strategy(
        self,
        rl_actions: dict[str, np.ndarray],
        llm_agents: dict[str, Any],
    ) -> StrategySignals | None:
        """Step 2: 策略翻译并应用到 LLM agents。"""
        if self.strategy_bridge is None:
            return None
        signals = self.strategy_bridge.translate(rl_actions)
        self._apply_signals(signals, llm_agents)
        return signals

    def _step_knowledge_before(
        self,
        state: CollaborationState,
        round_num: int,
    ) -> dict[str, Any]:
        """知识适配器 — 回合前处理。"""
        if not self.knowledge_adapter:
            return {}
        result = self.knowledge_adapter.before_round(
            state=state, round_num=round_num,
        ) or {}
        if result:
            state.metadata.setdefault("knowledge", {}).setdefault(
                "before_round", {},
            )[round_num] = result
        return result

    def _step_knowledge_after(
        self,
        state: CollaborationState,
        round_num: int,
        role_outputs: dict[str, dict[str, Any]],
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """知识适配器 — 回合后处理。"""
        if not self.knowledge_adapter:
            return {}
        result = self.knowledge_adapter.after_round(
            state=state,
            round_num=round_num,
            role_outputs=role_outputs,
            history=history,
        ) or {}
        if result:
            state.metadata.setdefault("knowledge", {}).setdefault(
                "after_round", {},
            )[round_num] = result
            self.observer.on_knowledge_updated(state, result, round_num, role_outputs)
        return result

    def _step_execute_llm(
        self,
        round_num: int,
        state: CollaborationState,
        llm_agents: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Step 3: 按场景定义的步骤执行 LLM agents。"""
        role_outputs: dict[str, dict[str, Any]] = {}
        incrementally_updated_roles: list[str] = []
        round_steps = self.scenario.get_round_steps(
            role_registry=self.role_registry,
            round_num=round_num,
            state=state,
            meta_interval=self.meta_interval,
        )

        for step in round_steps:
            agent = llm_agents.get(step.role_name)
            if agent is None:
                continue
            prompt = self.scenario.build_step_prompt(step, round_num, state)
            result = agent.act(prompt, round_num=round_num)
            role_outputs[step.role_name] = result
            self.observer.on_role_output(
                step.role_name, result, round_num,
                stage=step.stage,
                meta={"label": step.label, **step.meta},
            )
            if step.update_state:
                self.scenario.update_state({step.role_name: result}, state, round_num)
                incrementally_updated_roles.append(step.role_name)
                self.observer.on_state_updated(state, round_num, {step.role_name: result})

        state.metadata["_incrementally_updated_roles"] = incrementally_updated_roles
        return role_outputs

    def _step_update_state(
        self,
        state: CollaborationState,
        round_num: int,
        role_outputs: dict[str, dict[str, Any]],
        history: list[dict[str, Any]],
    ) -> None:
        """Step 4: 更新协作状态。"""
        updated = set(state.metadata.pop("_incrementally_updated_roles", []))
        pending_outputs = {
            role_name: output
            for role_name, output in role_outputs.items()
            if role_name not in updated
        }
        if pending_outputs:
            self.scenario.update_state(pending_outputs, state, round_num)
            self.observer.on_state_updated(state, round_num, pending_outputs)

    def _step_verify_compliance(
        self,
        signals: StrategySignals | None,
        role_outputs: dict[str, dict[str, Any]],
        state: CollaborationState,
    ) -> _ComplianceResult:
        """Step 5: 合规验证。"""
        strategy_compliance: dict[str, Any] = {}
        compliance_rewards: dict[str, float] = {}

        if self.strategy_bridge is not None and signals is not None:
            responses = {
                role_name: self._extract_response_text(output)
                for role_name, output in role_outputs.items()
                if isinstance(output, dict)
            }
            compliance_results = self.strategy_bridge.verify_compliance(signals, responses)
            strategy_compliance = {
                role: {
                    "overall_score": result.overall_score,
                    "dimension_scores": result.dimension_scores,
                    "details": result.details,
                }
                for role, result in compliance_results.items()
            }
            compliance_rewards = self.strategy_bridge.get_compliance_rewards()
            if strategy_compliance:
                state.metadata["strategy_compliance"] = strategy_compliance
            state.metadata["strategy_signals"] = signals.to_dict()

        return _ComplianceResult(
            strategy_compliance=strategy_compliance,
            compliance_rewards=compliance_rewards,
        )

    def _step_mechanism_update(
        self,
        state: CollaborationState,
        round_num: int,
        role_outputs: dict[str, dict[str, Any]],
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Step 6: 机制编排更新。"""
        if not self.mechanism_orchestrator:
            return {}

        mechanism_snapshot = self.mechanism_orchestrator.update(
            state=state,
            round_num=round_num,
            role_outputs=role_outputs,
            history=history,
        )
        mechanism_dict = self._apply_mechanism_snapshot(mechanism_snapshot, state)
        if mechanism_dict:
            state.metadata.setdefault("mechanism_history", []).append({
                "round_num": round_num,
                **mechanism_dict,
            })
            self.observer.on_mechanism_updated(
                state, mechanism_snapshot, round_num, role_outputs,
            )
        return mechanism_dict

    def _step_post_episode(
        self,
        state: CollaborationState,
        history: list[dict[str, Any]],
        transitions: list[dict[str, Any]],
        episode_ctx: dict[str, Any],
        verbose: bool,
    ) -> dict[str, Any]:
        """Episode 后处理：finalize、计算奖励、编译结果。"""
        self.scenario.finalize_episode(state, history)

        ctx = GameToolContext(self.tool_registry)
        try:
            scenario_rewards = self.scenario.compute_rewards(state, history, ctx)
        finally:
            ctx.cleanup()

        framework_rewards = self._compute_framework_rewards(state, history, transitions)
        rewards = scenario_rewards

        # 将奖励附加到 transitions
        for t in transitions:
            t["rewards"] = rewards
            t["framework_rewards"] = framework_rewards

        result = {
            "consensus_reached": state.is_terminal,
            "total_rounds": len(history),
            "final_quality": state.quality_score,
            "final_agreement": state.agreement_level,
            "final_compliance": state.compliance,
            "rewards": rewards,
            "framework_rewards": framework_rewards,
            "reward_breakdown": {
                "scenario_rewards": scenario_rewards,
                "framework_rewards": framework_rewards,
            },
            "transcript": history,
            "transitions": transitions,
            "episode_context": episode_ctx,
            "tool_context_log": ctx.call_log,
            "metadata": state.metadata,
            "strategy_signals": state.metadata.get("strategy_signals", {}),
            "strategy_compliance": state.metadata.get("strategy_compliance", {}),
        }

        self.observer.on_episode_end(result, state)

        if verbose:
            logger.info(
                "Episode done: rounds=%d quality=%.2f rewards=%s",
                len(history), state.quality_score, rewards,
            )

        return result

    def _compute_framework_rewards(
        self,
        state: CollaborationState,
        history: list[dict[str, Any]],
        transitions: list[dict[str, Any]],
    ) -> dict[str, float]:
        if self.reward_computer is None:
            return {}

        if len(history) >= 2:
            prev_record = history[-2].get("state", {})
        else:
            prev_record = {"quality": 0.5, "agreement": 0.5, "compliance": 0.5}
        curr_record = {
            "quality": state.quality_score,
            "agreement": state.agreement_level,
            "compliance": state.compliance,
            "metadata": dict(state.metadata),
        }
        role_names = list(history[-1].get("role_outputs", {}).keys()) if history else None
        role_outputs = history[-1].get("role_outputs", {}) if history else None
        compliance_rewards = transitions[-1].get("compliance_rewards", {}) if transitions else {}
        evaluator_role = "arbiter" if role_names and "arbiter" in role_names else "evaluator"
        coordinator_role = "coordinator" if role_names and "coordinator" in role_names else "coordinator"

        compute_fn = getattr(self.reward_computer, "compute", None)
        if callable(compute_fn):
            rewards = compute_fn(
                prev_state=prev_record,
                curr_state=curr_record,
                role_outputs=role_outputs,
                done=state.is_terminal,
                terminated_successfully=state.is_terminal,
            )
            if compliance_rewards:
                for role, bonus in compliance_rewards.items():
                    if role in rewards:
                        rewards[role] += bonus
            return rewards

        return self.reward_computer.compute_full_rewards(
            prev_state=prev_record,
            curr_state=curr_record,
            done=state.is_terminal,
            terminated_successfully=state.is_terminal,
            compliance_rewards=compliance_rewards,
            role_names=role_names,
            evaluator_role=evaluator_role,
            coordinator_role=coordinator_role,
        )

    # ── 记录辅助 ──────────────────────────────────────────────────────

    def _build_round_record(
        self,
        round_num: int,
        role_outputs: dict[str, dict[str, Any]],
        state: CollaborationState,
        mechanism_dict: dict[str, Any],
        knowledge_after: dict[str, Any],
        signals: StrategySignals | None,
        compliance: _ComplianceResult,
    ) -> dict[str, Any]:
        """构建单回合记录。"""
        record: dict[str, Any] = {
            "round_num": round_num,
            "role_outputs": role_outputs,
            "state": {
                "quality": state.quality_score,
                "agreement": state.agreement_level,
                "compliance": state.compliance,
            },
        }
        if mechanism_dict:
            record["mechanism"] = mechanism_dict
        if knowledge_after:
            record["knowledge"] = knowledge_after
        if signals is not None:
            record["signals"] = signals.to_dict()
        if compliance.strategy_compliance:
            record["strategy_compliance"] = compliance.strategy_compliance
        if compliance.compliance_rewards:
            record["compliance_rewards"] = compliance.compliance_rewards
        return record

    def _record_transition(
        self,
        transitions: list[dict[str, Any]],
        rl_result: _RLStepResult,
        signals: StrategySignals | None,
        mechanism_dict: dict[str, Any],
        knowledge_after: dict[str, Any],
        role_outputs: dict[str, dict[str, Any]],
        compliance: _ComplianceResult,
        state: CollaborationState,
        round_num: int,
        rl_agents: MADDPGAgentGroup | None,
    ) -> None:
        """记录 RL transition（仅在有 RL 观测时）。"""
        if rl_result.shared_obs is None:
            return

        next_obs: dict[str, np.ndarray] = {}
        if self.observation_encoder and rl_agents is not None:
            next_round = min(round_num + 1, self.max_rounds)
            next_shared = self.observation_encoder.encode_shared(
                state, next_round, self.max_rounds,
            )
            for role_name in rl_agents.agent_names:
                next_obs[role_name] = self.observation_encoder.encode_role(
                    next_shared, role_name,
                )

        transitions.append({
            "obs": rl_result.role_obs or rl_result.shared_obs,
            "shared_obs": rl_result.shared_obs,
            "next_obs": next_obs,
            "rl_actions": rl_result.rl_actions,
            "signals": signals.to_dict() if signals else {},
            "mechanism": mechanism_dict,
            "knowledge": knowledge_after,
            "role_outputs": role_outputs,
            "strategy_compliance": compliance.strategy_compliance,
            "compliance_rewards": compliance.compliance_rewards,
            "done": False,
        })

    # ── 信号应用 ──────────────────────────────────────────────────────

    def _apply_signals(self, signals: StrategySignals, llm_agents: dict[str, Any]) -> None:
        """将 RL 策略信号应用到 LLM agents。"""
        for role_name, agent in llm_agents.items():
            # Style directive
            style = signals.get_style(role_name)
            if style and hasattr(agent, "_style_directive"):
                composer = getattr(self.strategy_bridge, "style_composer", None)
                if composer is not None:
                    agent._style_directive = composer.compose(role_name, style)

            # Temperature
            temp = signals.get_temperature(role_name)
            if hasattr(agent, "client") and hasattr(agent.client, "temperature"):
                agent.client.temperature = temp

    # ── 文本提取 ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_response_text(output: dict[str, Any]) -> str:
        """从角色输出中提取可读文本。"""
        for key in (
            "treatment_plan",
            "assessment",
            "ruling",
            "review",
            "summary",
            "key_issues",
            "guidance",
            "verdict",
            "reasoning",
            "suggestions",
            "surgical_assessment",
            "radiation_plan",
            "imaging_assessment",
        ):
            value = output.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return " ".join(
            str(value).strip()
            for key, value in output.items()
            if not key.startswith("_") and isinstance(value, (str, int, float, bool))
        )

    # ── 机制快照 ──────────────────────────────────────────────────────

    @staticmethod
    def _mechanism_to_dict(mechanism_snapshot: Any) -> dict[str, Any]:
        """将机制快照转换为 dict。"""
        if mechanism_snapshot is None:
            return {}
        if isinstance(mechanism_snapshot, MechanismSnapshot):
            return mechanism_snapshot.to_dict()
        if isinstance(mechanism_snapshot, dict):
            return dict(mechanism_snapshot)
        if hasattr(mechanism_snapshot, "to_dict"):
            data = mechanism_snapshot.to_dict()
            if isinstance(data, dict):
                return data
        if hasattr(mechanism_snapshot, "__dict__"):
            data = {
                key: value for key, value in vars(mechanism_snapshot).items()
                if not key.startswith("_")
            }
            if isinstance(data, dict):
                return data
        return {}

    def _apply_mechanism_snapshot(
        self,
        mechanism_snapshot: Any,
        state: CollaborationState,
    ) -> dict[str, Any]:
        """应用机制快照到状态。"""
        data = self._mechanism_to_dict(mechanism_snapshot)
        if data:
            MechanismSnapshot(data).apply_to_state(state)
        return data


# ── 内部数据容器 ──────────────────────────────────────────────────────

class _RLStepResult:
    """RL 观测步骤的结果。"""

    __slots__ = ("shared_obs", "role_obs", "rl_actions")

    def __init__(
        self,
        shared_obs: np.ndarray | None,
        role_obs: dict[str, np.ndarray],
        rl_actions: dict[str, np.ndarray],
    ) -> None:
        self.shared_obs = shared_obs
        self.role_obs = role_obs
        self.rl_actions = rl_actions


class _ComplianceResult:
    """合规验证步骤的结果。"""

    __slots__ = ("strategy_compliance", "compliance_rewards")

    def __init__(
        self,
        strategy_compliance: dict[str, Any],
        compliance_rewards: dict[str, float],
    ) -> None:
        self.strategy_compliance = strategy_compliance
        self.compliance_rewards = compliance_rewards
