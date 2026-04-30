"""辩论场景观测编码器 — 将辩论状态编码为 RL 观测向量。

从 core/strategy_bridge.py 的内联逻辑提取，继承 BaseObservationEncoder。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from debate_rl_v2.framework.observation import BaseObservationEncoder
from debate_rl_v2.framework.types import CollaborationState


class DebateObservationEncoder(BaseObservationEncoder):
    """辩论场景观测编码器。

    共享观测维度 (14):
      [0]  round_progress       回合进度 (0-1)
      [1]  quality_score        当前质量分
      [2]  agreement_level      共识水平 (1 - disagreement)
      [3]  compliance           合规度
      [4]  lambda_adv           对抗强度
      [5]  da_active            魔鬼代言人是否激活
      [6]  mode_numeric         模式编码 (standard=0, explore=0.5, exploit=1)
      [7]  prop_confidence      提案者置信度
      [8]  chal_confidence      挑战者置信度
      [9]  quality_trend        质量趋势
      [10] disagreement_trend   分歧趋势
      [11] prop_score           提案者得分
      [12] chal_score           挑战者得分
      [13] time_pressure        时间压力 (round/max_rounds)^2

    角色观测扩展 (6):
      [0] role_phase_encode     角色阶段编码
      [1] role_assertiveness    角色主张度/攻击度
      [2] role_quality_contrib  角色质量贡献
      [3] role_compliance_score 角色合规得分
      [4] role_confidence       角色最近置信度
      [5] role_active           角色是否在本轮活跃
    """

    _SHARED_DIM = 14
    _ROLE_DIM = 6

    _MODE_MAP = {"standard": 0.0, "explore": 0.5, "exploit": 1.0}
    _PHASE_MAP = {
        "proposer": 0.25,
        "proposer_ctrl": 0.25,
        "challenger": 0.5,
        "challenger_ctrl": 0.5,
        "arbiter": 0.75,
        "arbiter_ctrl": 0.75,
        "coordinator": 0.9,
    }

    def __init__(self) -> None:
        self._prev_quality: float = 0.5
        self._prev_agreement: float = 0.5
        self._role_stats: dict[str, dict[str, float]] = {}

    def shared_obs_dim(self) -> int:
        return self._SHARED_DIM

    def role_obs_dim(self) -> int:
        return self._ROLE_DIM

    def encode_shared(
        self,
        state: CollaborationState,
        round_num: int,
        max_rounds: int,
    ) -> np.ndarray:
        obs = np.zeros(self._SHARED_DIM, dtype=np.float32)
        meta = state.metadata

        progress = round_num / max(max_rounds, 1)
        obs[0] = progress
        obs[1] = state.quality_score
        obs[2] = state.agreement_level
        obs[3] = state.compliance
        obs[4] = state.intensity  # lambda_adv
        obs[5] = float(meta.get("da_active", False))
        obs[6] = self._MODE_MAP.get(state.mode, 0.0)
        obs[7] = float(meta.get("prop_confidence", 0.5))
        obs[8] = float(meta.get("chal_confidence", 0.5))

        # 趋势
        quality_trend = state.quality_score - self._prev_quality
        agreement_trend = state.agreement_level - self._prev_agreement
        obs[9] = np.clip(quality_trend, -1.0, 1.0)
        obs[10] = np.clip(agreement_trend, -1.0, 1.0)

        obs[11] = float(meta.get("prop_score", 0.5))
        obs[12] = float(meta.get("chal_score", 0.5))
        obs[13] = progress ** 2  # 时间压力（后期加速）

        # 更新历史
        self._prev_quality = state.quality_score
        self._prev_agreement = state.agreement_level

        return obs

    def encode_role(self, shared_obs: np.ndarray, role: str) -> np.ndarray:
        role_ext = np.zeros(self._ROLE_DIM, dtype=np.float32)

        role_ext[0] = self._PHASE_MAP.get(role, 0.5)

        # 从角色统计中获取（如果有）
        stats = self._role_stats.get(role, {})
        role_ext[1] = stats.get("assertiveness", 0.5)
        role_ext[2] = stats.get("quality_contrib", 0.5)
        role_ext[3] = stats.get("compliance_score", 0.5)
        role_ext[4] = stats.get("confidence", 0.5)
        role_ext[5] = stats.get("active", 1.0)

        return np.concatenate([shared_obs, role_ext])

    def update_role_stats(self, role: str, stats: dict[str, float]) -> None:
        """外部更新角色统计信息（由 GameEngine 或 scenario 调用）。"""
        self._role_stats.setdefault(role, {}).update(stats)

    def reset(self) -> None:
        self._prev_quality = 0.5
        self._prev_agreement = 0.5
        self._role_stats.clear()
