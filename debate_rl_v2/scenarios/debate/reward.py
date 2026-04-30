"""辩论场景奖励计算 — 继承通用框架，添加辩论专用组件。

设计原则：
  - 继承 BaseRewardComputer 获得通用的 quality/agreement/step 奖励
  - 叠加辩论专用的 dense quality、curiosity、marginal return 奖励
  - 奖励组合满足势函数条件（potential-based shaping）
"""

from __future__ import annotations

from typing import Any

from debate_rl_v2.framework.reward import BaseRewardComputer, RewardWeights
from debate_rl_v2.logging_config import get_logger

logger = get_logger("scenarios.debate.reward")


class DebateRewardComputer(BaseRewardComputer):
    """辩论场景奖励 — 继承通用框架，添加辩论专用组件。

    在 BaseRewardComputer 的 quality/agreement/step 奖励基础上叠加：
    - Dense quality scoring: 质量分解为多维度加权
    - Curiosity bonus: 鼓励探索新论点角度
    - Marginal return detection: 边际收益递减时给予终止信号

    奖励组合：r_total = r_framework + α * r_scenario
    其中 α 是可配置权重，默认 0.3。
    """

    def __init__(
        self,
        weights: RewardWeights | None = None,
        scenario_weight: float = 0.3,
        curiosity_weight: float = 0.05,
        marginal_penalty: float = 0.02,
        quality_dimensions: dict[str, float] | None = None,
    ) -> None:
        super().__init__(weights)
        self.scenario_weight = scenario_weight
        self.curiosity_weight = curiosity_weight
        self.marginal_penalty = marginal_penalty
        # 质量分解维度权重
        self.quality_dimensions = quality_dimensions or {
            "logic": 0.25,
            "feasibility": 0.25,
            "innovation": 0.15,
            "evidence": 0.20,
            "compliance": 0.15,
        }
        # 用于边际收益检测的历史质量
        self._quality_history: list[float] = []

    def reset(self) -> None:
        """重置 episode 状态。"""
        self._quality_history.clear()

    def compute(
        self,
        prev_state: dict[str, float],
        curr_state: dict[str, float],
        role_outputs: dict[str, dict[str, Any]] | None = None,
        done: bool = False,
        terminated_successfully: bool = False,
    ) -> dict[str, float]:
        """计算辩论场景的完整奖励。

        Parameters
        ----------
        prev_state, curr_state : dict
            包含 quality, agreement, compliance 的状态字典。
        role_outputs : dict, optional
            各角色的输出（用于 dense quality 和 curiosity 计算）。
        done : bool
            是否为终止步。
        terminated_successfully : bool
            是否成功达成共识。

        Returns
        -------
        dict[str, float]
            各角色的奖励值。
        """
        role_names = list(role_outputs.keys()) if role_outputs else [
            "proposer", "challenger", "arbiter", "coordinator",
        ]

        # 1. 基础框架奖励
        rewards = self.compute_full_rewards(
            prev_state=prev_state,
            curr_state=curr_state,
            done=done,
            terminated_successfully=terminated_successfully,
            role_names=role_names,
            evaluator_role="arbiter",
            coordinator_role="coordinator",
        )

        # 2. 辩论专用奖励
        scenario_rewards = self._compute_scenario_rewards(
            prev_state, curr_state, role_outputs, done, role_names,
        )

        # 3. 组合: r_total = r_framework + α * r_scenario
        for role in rewards:
            rewards[role] += self.scenario_weight * scenario_rewards.get(role, 0.0)

        return rewards

    def _compute_scenario_rewards(
        self,
        prev_state: dict[str, float],
        curr_state: dict[str, float],
        role_outputs: dict[str, dict[str, Any]] | None,
        done: bool,
        role_names: list[str],
    ) -> dict[str, float]:
        """计算辩论专用奖励组件。"""
        scenario_r: dict[str, float] = {role: 0.0 for role in role_names}

        quality = curr_state.get("quality", 0.5)
        self._quality_history.append(quality)

        # Dense quality: 从 arbiter 的维度评分中提取
        dense_bonus = self._dense_quality_bonus(role_outputs)

        # Curiosity: 鼓励 challenger 提出新角度
        curiosity_bonus = self._curiosity_bonus(role_outputs)

        # Marginal return: 质量不再提升时给予小惩罚
        marginal_penalty = self._marginal_return_penalty()

        # 分配到各角色
        for role in role_names:
            scenario_r[role] += dense_bonus
            scenario_r[role] -= marginal_penalty

        # Curiosity 只奖励 challenger
        if "challenger" in scenario_r:
            scenario_r["challenger"] += curiosity_bonus

        return scenario_r

    def _dense_quality_bonus(
        self,
        role_outputs: dict[str, dict[str, Any]] | None,
    ) -> float:
        """从 arbiter 的多维度评分中计算 dense quality bonus。"""
        if not role_outputs:
            return 0.0

        arbiter_output = role_outputs.get("arbiter", {})
        dim_scores = arbiter_output.get("dimension_scores", {})
        if not dim_scores:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0
        for dim, weight in self.quality_dimensions.items():
            score = dim_scores.get(dim)
            if score is not None:
                weighted_sum += weight * float(score)
                total_weight += weight

        if total_weight == 0:
            return 0.0

        # 归一化到 [-0.1, 0.1] 范围
        normalized = weighted_sum / total_weight
        return (normalized - 0.5) * 0.2

    def _curiosity_bonus(
        self,
        role_outputs: dict[str, dict[str, Any]] | None,
    ) -> float:
        """鼓励 challenger 提出新质疑角度。"""
        if not role_outputs:
            return 0.0

        challenger_output = role_outputs.get("challenger", {})
        new_angles = challenger_output.get("new_angles", [])
        if not new_angles:
            return 0.0

        # 每个新角度给予小奖励，上限 3 个
        return self.curiosity_weight * min(len(new_angles), 3)

    def _marginal_return_penalty(self) -> float:
        """检测边际收益递减，给予小惩罚鼓励及时终止。"""
        if len(self._quality_history) < 4:
            return 0.0

        # 最近 3 轮的质量变化
        recent = self._quality_history[-3:]
        deltas = [recent[i] - recent[i - 1] for i in range(1, len(recent))]

        # 如果最近 2 轮质量变化都很小（< 0.02），认为边际收益递减
        if all(abs(d) < 0.02 for d in deltas):
            return self.marginal_penalty

        return 0.0
