"""辩论场景的 GameScenario 实现 — 框架旗舰场景。

从 envs/llm_env.py (TextDebateEnv) 和 envs/debate_logic.py (DebateLogicMixin)
提取核心逻辑，适配到 GameEngine 驱动的 GameScenario 协议。

博弈四角色：
- coordinator: 协调者（控制辩论参数 + 宏观调控）
- proposer:    提案者（提出/改进方案）
- challenger:  挑战者（质疑/反驳方案）
- arbiter:     仲裁者（评判质量 + 共识判定）
"""

from __future__ import annotations

from typing import Any

from debate_rl_v2.agents.tool_agent_loop import AgentTurnResult
from debate_rl_v2.framework.game_scenario import GameScenario, RoundStep
from debate_rl_v2.framework.roles import RoleRegistry
from debate_rl_v2.framework.tool_context import GameToolContext
from debate_rl_v2.framework.tool_registry import GameToolRegistry
from debate_rl_v2.framework.types import CollaborationState
from debate_rl_v2.scenarios.debate.prompts import (
    format_proposer_message,
    format_challenger_message,
    format_arbiter_message,
    format_coordinator_message,
)
from debate_rl_v2.scenarios.debate.reward import DebateRewardComputer
from debate_rl_v2.logging_config import get_logger

logger = get_logger("scenarios.debate.scenario")


class DebateGameScenario(GameScenario):
    """辩论场景的 GameScenario 实现。

    通过 GameEngine.run_episode() 驱动，替代旧版 TextDebateEnv / FusionDebateEnv。

    Parameters
    ----------
    topic : str
        辩论主题。
    context : str
        背景信息。
    rules : list[str] | None
        活跃规则列表。
    max_rounds : int
        最大回合数（由 GameEngine 注入到 _max_rounds）。
    consensus_threshold : float
        共识质量阈值。
    disagreement_consensus : float
        分歧度低于此值视为共识。
    """

    def __init__(
        self,
        topic: str = "",
        context: str = "",
        rules: list[str] | None = None,
        max_rounds: int = 10,
        consensus_threshold: float = 0.8,
        disagreement_consensus: float = 0.15,
    ) -> None:
        self.topic = topic
        self.context = context
        self.rules = rules or []
        self._max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.disagreement_consensus = disagreement_consensus

        # 内部奖励计算器
        self._reward_computer = DebateRewardComputer()

    # ── GameScenario 抽象方法实现 ──

    def setup(self) -> None:
        """初始化场景（辩论场景无需额外加载）。"""
        pass

    def register_tools(self, registry: GameToolRegistry) -> None:
        """注册场景工具（辩论场景暂无专用工具）。"""
        pass

    def create_episode(self) -> dict[str, Any]:
        """创建一个新的辩论 episode。

        Returns
        -------
        dict
            Episode 上下文，存入 CollaborationState.metadata。
        """
        self._reward_computer.reset()
        return {
            "topic": self.topic,
            "context": self.context,
            "rules": self.rules,
            "scenario_type": "debate",
        }

    def get_round_steps(
        self,
        role_registry: RoleRegistry,
        round_num: int,
        state: CollaborationState,
        meta_interval: int,
    ) -> list[RoundStep]:
        """返回辩论回合的执行步骤序列。

        顺序：coordinator(每 meta_interval 轮) → proposer → challenger → arbiter
        """
        steps: list[RoundStep] = []

        # Coordinator 在首轮和每 meta_interval 轮行动
        if round_num % meta_interval == 1 or round_num == 1:
            steps.append(RoundStep(
                role_name="coordinator",
                stage="coordinate",
                label="meta-action",
            ))

        # Proposer → Challenger → Arbiter
        steps.append(RoundStep(
            role_name="proposer",
            stage="propose",
            update_state=True,
            label="proposal",
        ))
        steps.append(RoundStep(
            role_name="challenger",
            stage="challenge",
            update_state=True,
            label="challenge",
        ))
        steps.append(RoundStep(
            role_name="arbiter",
            stage="evaluate",
            update_state=True,
            label="verdict",
        ))

        return steps

    def build_role_prompt(
        self,
        role_name: str,
        round_num: int,
        state: CollaborationState,
    ) -> str:
        """为指定角色构建当前回合的 prompt。"""
        return self._build_prompt_for_role(role_name, round_num, state)

    def build_step_prompt(
        self,
        step: RoundStep,
        round_num: int,
        state: CollaborationState,
    ) -> str:
        """为指定步骤构建 prompt。"""
        return self._build_prompt_for_role(step.role_name, round_num, state)

    def update_state(
        self,
        role_outputs: dict[str, AgentTurnResult | dict[str, Any]],
        state: CollaborationState,
        round_num: int,
    ) -> None:
        """从智能体输出更新协作状态。"""
        meta = state.metadata

        # Proposer 输出
        proposer_out = self._to_dict(role_outputs.get("proposer"))
        if proposer_out:
            meta["proposal"] = proposer_out.get("proposal", "")
            meta["prop_confidence"] = float(proposer_out.get("confidence", 0.5))

        # Challenger 输出
        challenger_out = self._to_dict(role_outputs.get("challenger"))
        if challenger_out:
            meta["challenge"] = challenger_out.get("challenge", "")
            meta["chal_confidence"] = float(challenger_out.get("confidence", 0.5))

        # Arbiter 输出
        arbiter_out = self._to_dict(role_outputs.get("arbiter"))
        if arbiter_out:
            meta["verdict"] = arbiter_out.get("verdict", "")
            state.quality_score = float(arbiter_out.get("quality_score", 0.5))
            meta["prop_score"] = float(arbiter_out.get("proposal_score", 0.5))
            meta["chal_score"] = float(arbiter_out.get("challenge_score", 0.5))
            meta["dimension_scores"] = arbiter_out.get("dimension_scores", {})
            meta["consensus_reached"] = bool(arbiter_out.get("consensus_reached", False))

            # 分歧度由 mechanism orchestrator 统一计算并写回状态。
            # 如果没有启用机制层，才使用仲裁者显式给出的 agreement/disagreement。
            if "agreement_level" in arbiter_out:
                state.agreement_level = max(0.0, min(1.0, float(arbiter_out["agreement_level"])))
            elif "disagreement" in arbiter_out:
                disagreement = max(0.0, min(1.0, float(arbiter_out["disagreement"])))
                state.agreement_level = 1.0 - disagreement

            # 计算合规度
            state.compliance = self._evaluate_compliance(arbiter_out, round_num)

        # Coordinator 输出
        coordinator_out = self._to_dict(role_outputs.get("coordinator"))
        if coordinator_out:
            meta["coordinator_action"] = coordinator_out.get("action_idx", 0)
            meta["coordinator_reasoning"] = coordinator_out.get("reasoning", "")

    def check_terminal(self, state: CollaborationState) -> bool:
        """检查是否达到终止条件。"""
        meta = state.metadata
        disagreement = 1.0 - state.agreement_level

        consensus_ok = (
            meta.get("consensus_reached", False)
            or disagreement < self.disagreement_consensus
        ) and state.quality_score >= self.consensus_threshold

        return consensus_ok

    def compute_rewards(
        self,
        state: CollaborationState,
        history: list[dict[str, Any]],
        ctx: GameToolContext,
    ) -> dict[str, float]:
        """计算各角色奖励。"""
        if len(history) >= 2:
            prev_record = history[-2].get("state", {})
        else:
            prev_record = {"quality": 0.5, "agreement": 0.5, "compliance": 0.5}

        curr_record = {
            "quality": state.quality_score,
            "agreement": state.agreement_level,
            "compliance": state.compliance,
        }

        # 收集最后一轮的角色输出
        role_outputs = history[-1].get("role_outputs", {}) if history else {}

        return self._reward_computer.compute(
            prev_state=prev_record,
            curr_state=curr_record,
            role_outputs=role_outputs,
            done=state.is_terminal,
            terminated_successfully=state.is_terminal,
        )

    def finalize_episode(
        self,
        state: CollaborationState,
        history: list[dict[str, Any]],
    ) -> None:
        """Episode 结束后的清理。"""
        state.metadata["final_proposal"] = state.metadata.get("proposal", "")
        state.metadata["final_verdict"] = state.metadata.get("verdict", "")

    # ── Prompt 构建 ──

    def _build_prompt_for_role(
        self,
        role_name: str,
        round_num: int,
        state: CollaborationState,
    ) -> str:
        """根据角色名分发到对应的 prompt 格式化函数。"""
        meta = state.metadata
        max_rounds = self._max_rounds

        # 构建历史摘要
        history_dicts = meta.get("_recent_history", None)

        if role_name == "proposer":
            return format_proposer_message(
                topic=self.topic,
                context=self.context,
                round_num=round_num,
                max_rounds=max_rounds,
                prev_proposal=meta.get("proposal", ""),
                challenge=meta.get("challenge", ""),
                compliance=state.compliance,
                disagreement=1.0 - state.agreement_level,
                rules=self.rules,
                history=history_dicts,
            )
        elif role_name == "challenger":
            return format_challenger_message(
                topic=self.topic,
                context=self.context,
                round_num=round_num,
                max_rounds=max_rounds,
                proposal=meta.get("proposal", ""),
                prev_challenge=meta.get("challenge", ""),
                disagreement=1.0 - state.agreement_level,
                lambda_adv=state.intensity,
                mode=state.mode,
                rules=self.rules,
                history=history_dicts,
            )
        elif role_name == "arbiter":
            return format_arbiter_message(
                topic=self.topic,
                context=self.context,
                round_num=round_num,
                max_rounds=max_rounds,
                proposal=meta.get("proposal", ""),
                challenge=meta.get("challenge", ""),
                compliance=state.compliance,
                disagreement=1.0 - state.agreement_level,
                rules=self.rules,
                history=history_dicts,
            )
        elif role_name == "coordinator":
            # 计算趋势
            trend = self._compute_trend(state)
            return format_coordinator_message(
                round_num=round_num,
                max_rounds=max_rounds,
                disagreement=1.0 - state.agreement_level,
                compliance=state.compliance,
                lambda_adv=state.intensity,
                da_active=bool(meta.get("da_active", False)),
                trend=trend,
                recent_quality=state.quality_score,
            )
        else:
            # 未知角色，返回通用 prompt
            return (
                f"主题: {self.topic}\n"
                f"第{round_num}/{max_rounds}轮\n"
                f"请以 {role_name} 角色参与讨论。"
            )

    def _evaluate_compliance(
        self,
        arb_result: dict[str, Any],
        round_num: int,
    ) -> float:
        """Compliance score from quality and proposal strength."""
        quality = float(arb_result.get("quality_score", 0.5))
        ps = float(arb_result.get("proposal_score", 0.5))
        t_ratio = round_num / max(self._max_rounds, 1)
        w = 0.4 + 0.3 * t_ratio
        return min(1.0, w * quality + (1.0 - w) * ps)

    def _compute_trend(self, state: CollaborationState) -> str:
        """计算趋势描述。"""
        meta = state.metadata
        prev_quality = meta.get("_prev_quality", 0.5)
        prev_agreement = meta.get("_prev_agreement", 0.5)

        dq = state.quality_score - prev_quality
        da = state.agreement_level - prev_agreement

        parts = []
        parts.append("分歧上升" if da < -0.05 else ("分歧下降" if da > 0.05 else "分歧稳定"))
        parts.append("质量提升" if dq > 0.05 else ("质量下降" if dq < -0.05 else "质量稳定"))

        # 记录当前值供下轮使用
        meta["_prev_quality"] = state.quality_score
        meta["_prev_agreement"] = state.agreement_level

        return "，".join(parts)

    # ── 工具方法 ──

    @staticmethod
    def _to_dict(output: Any) -> dict[str, Any]:
        """将 AgentTurnResult 或 dict 统一转为 dict。"""
        if output is None:
            return {}
        if isinstance(output, dict):
            return output
        if hasattr(output, "parsed"):
            return output.parsed if isinstance(output.parsed, dict) else {}
        if hasattr(output, "__dict__"):
            return vars(output)
        return {}
