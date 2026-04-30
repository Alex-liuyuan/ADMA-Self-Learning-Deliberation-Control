"""Enhanced Strategy Bridge — RL→LLM fusion with compliance verification.

.. deprecated::
    Use debate_rl_v2.scenarios.debate.strategy.DebateStrategyBridge instead.
    This module is retained for backward compatibility with RL pre-training code.

Migrated from debate_rl/core/strategy_bridge.py with key improvements:
  - Integrated ComplianceVerifier for closed-loop RL→LLM control
  - Role-specific observation encoding (14 shared + 6 role-specific = 20D)
  - Cleaner separation via config dataclass
  - All magic numbers extracted to StrategyBridgeConfig
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from debate_rl_v2.config.rl import StrategyBridgeConfig
from debate_rl_v2.core.compliance_verifier import ComplianceVerifier, ComplianceResult
from debate_rl_v2.algorithms.role_observations import (
    RoleObservationTracker,
    build_role_observation,
    ROLE_OBS_DIM,
)
from debate_rl_v2.logging_config import get_logger

logger = get_logger("core.strategy_bridge")

# Observation dimensions
SHARED_OBS_DIM = 14
FUSION_OBS_DIM = SHARED_OBS_DIM  # backward compat
TOTAL_OBS_DIM = SHARED_OBS_DIM + ROLE_OBS_DIM  # 20

# Action dimensions
PROPOSER_CTRL_ACT_DIM = 4
CHALLENGER_CTRL_ACT_DIM = 4
ARBITER_CTRL_ACT_DIM = 4
COORDINATOR_ACT_DIM = 5


@dataclass
class StrategySignals:
    """Output of the Strategy Bridge — concrete LLM debate parameters."""

    # Per-agent LLM temperatures
    proposer_temperature: float = 0.7
    challenger_temperature: float = 0.7
    arbiter_temperature: float = 0.5

    # Proposer style
    proposer_assertiveness: float = 0.5
    proposer_detail_level: float = 0.5
    proposer_compliance_focus: float = 0.5
    proposer_incorporation: float = 0.5

    # Challenger style
    challenger_aggressiveness: float = 0.5
    challenger_specificity: float = 0.5
    challenger_constructiveness: float = 0.5
    challenger_novelty: float = 0.5

    # Arbiter style
    arbiter_strictness: float = 0.5
    arbiter_detail_feedback: float = 0.5
    arbiter_consensus_bias: float = 0.5
    arbiter_rule_emphasis: float = 0.5

    # Mechanism parameters (Coordinator)
    eta_delta: float = 0.0
    alpha_delta: float = 0.0
    tau_low_delta: float = 0.0
    tau_high_delta: float = 0.0
    exploration_rate: float = 0.3

    # v3: Prompt Evolution Parameters
    proposer_reasoning_depth: float = 0.5
    proposer_self_critique: float = 0.5
    proposer_evidence_demand: float = 0.5
    challenger_attack_angle: float = 0.5
    challenger_depth_vs_breadth: float = 0.5
    challenger_evidence_demand: float = 0.5
    arbiter_framework_weight: float = 0.5
    arbiter_evidence_demand: float = 0.5
    difficulty_level: int = 1

    def to_dict(self) -> Dict[str, float]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class StrategyBridge:
    """Translates MADDPG continuous actions → LLM debate parameters.

    v2 enhancements over v1:
      - Integrated ComplianceVerifier for closed-loop control
      - Role-specific observation encoding (20D per agent)
      - Config-driven (no hardcoded magic numbers)
    """

    def __init__(
        self,
        config: StrategyBridgeConfig | None = None,
        enable_compliance: bool = True,
        enable_role_obs: bool = True,
    ) -> None:
        cfg = config or StrategyBridgeConfig()
        self.temp_range = (cfg.temp_min, cfg.temp_max)
        self.arbiter_temp_range = (cfg.arbiter_temp_min, cfg.arbiter_temp_max)
        self.max_eta_delta = cfg.max_eta_delta
        self.max_alpha_delta = cfg.max_alpha_delta
        self.max_tau_delta = cfg.max_tau_delta

        # v2: Compliance verifier
        self.compliance_verifier = ComplianceVerifier() if enable_compliance else None
        self._last_compliance: Dict[str, ComplianceResult] = {}

        # v2: Role-specific observation tracker
        self.role_tracker = RoleObservationTracker() if enable_role_obs else None
        self.enable_role_obs = enable_role_obs

    # ── Action Translation ──

    def translate(
        self,
        proposer_action: np.ndarray,
        challenger_action: np.ndarray,
        arbiter_action: np.ndarray,
        coordinator_action: np.ndarray,
    ) -> StrategySignals:
        """Convert raw RL actions [-1,1] to debate strategy signals."""
        sig = StrategySignals()

        p = self._to_unit(proposer_action)
        sig.proposer_assertiveness = float(p[0])
        sig.proposer_detail_level = float(p[1])
        sig.proposer_compliance_focus = float(p[2])
        sig.proposer_incorporation = float(p[3])
        sig.proposer_temperature = self._map_temp(
            0.3 * p[0] + 0.7 * (1.0 - p[2]), self.temp_range
        )

        c = self._to_unit(challenger_action)
        sig.challenger_aggressiveness = float(c[0])
        sig.challenger_specificity = float(c[1])
        sig.challenger_constructiveness = float(c[2])
        sig.challenger_novelty = float(c[3])
        sig.challenger_temperature = self._map_temp(
            0.5 * c[0] + 0.3 * c[3] + 0.2 * (1.0 - c[1]), self.temp_range
        )

        a = self._to_unit(arbiter_action)
        sig.arbiter_strictness = float(a[0])
        sig.arbiter_detail_feedback = float(a[1])
        sig.arbiter_consensus_bias = float(a[2])
        sig.arbiter_rule_emphasis = float(a[3])
        sig.arbiter_temperature = self._map_temp(
            0.3 * (1.0 - a[0]) + 0.7 * 0.3, self.arbiter_temp_range
        )

        sig.eta_delta = float(coordinator_action[0]) * self.max_eta_delta
        sig.alpha_delta = float(coordinator_action[1]) * self.max_alpha_delta
        sig.tau_low_delta = float(coordinator_action[2]) * self.max_tau_delta
        sig.tau_high_delta = float(coordinator_action[3]) * self.max_tau_delta
        sig.exploration_rate = float(self._to_unit_scalar(coordinator_action[4]))

        # v3: Prompt evolution signals
        sig.proposer_reasoning_depth = float(p[1])
        sig.proposer_self_critique = float(1.0 - p[0]) * 0.5 + float(p[2]) * 0.5
        sig.proposer_evidence_demand = float(p[1] * 0.6 + p[2] * 0.4)
        sig.challenger_attack_angle = float(c[1])
        sig.challenger_depth_vs_breadth = float(c[3])
        sig.challenger_evidence_demand = float(c[1] * 0.5 + c[2] * 0.5)
        sig.arbiter_framework_weight = float(a[0] * 0.5 + a[1] * 0.5)
        sig.arbiter_evidence_demand = float(a[0] * 0.4 + a[3] * 0.6)

        return sig

    # ── Compliance Verification (v2) ──

    def verify_compliance(
        self,
        signals: StrategySignals,
        responses: Dict[str, str],
    ) -> Dict[str, ComplianceResult]:
        """Verify LLM responses against RL strategy signals.

        Returns per-role compliance results that can be used as reward bonuses.
        """
        if self.compliance_verifier is None:
            return {}

        results: Dict[str, ComplianceResult] = {}

        if "proposer" in responses:
            results["proposer_ctrl"] = self.compliance_verifier.verify_proposer(
                response=responses["proposer"],
                assertiveness=signals.proposer_assertiveness,
                detail_level=signals.proposer_detail_level,
                compliance_focus=signals.proposer_compliance_focus,
                evidence_demand=signals.proposer_evidence_demand,
            )
        if "challenger" in responses:
            results["challenger_ctrl"] = self.compliance_verifier.verify_challenger(
                response=responses["challenger"],
                aggressiveness=signals.challenger_aggressiveness,
                constructiveness=signals.challenger_constructiveness,
                specificity=signals.challenger_specificity,
                evidence_demand=signals.challenger_evidence_demand,
            )
        if "arbiter" in responses:
            results["arbiter_ctrl"] = self.compliance_verifier.verify_arbiter(
                response=responses["arbiter"],
                strictness=signals.arbiter_strictness,
                detail_feedback=signals.arbiter_detail_feedback,
                consensus_bias=signals.arbiter_consensus_bias,
            )

        self._last_compliance = results
        return results

    def get_compliance_rewards(self, weight: float = 0.15) -> Dict[str, float]:
        """Get compliance-based reward bonuses from last verification."""
        if not self._last_compliance or self.compliance_verifier is None:
            return {}
        return self.compliance_verifier.compute_compliance_reward(
            self._last_compliance, weight=weight
        )

    # ── Observation Encoding ──

    @staticmethod
    def encode_observation(
        round_num: int,
        max_rounds: int,
        disagreement: float,
        quality_score: float,
        compliance: float,
        lambda_adv: float,
        da_active: bool,
        mode: str,
        prop_confidence: float,
        chal_confidence: float,
        quality_trend: float = 0.0,
        disagreement_trend: float = 0.0,
        prop_score: float = 0.5,
        chal_score: float = 0.5,
    ) -> np.ndarray:
        """Encode debate state into shared 14D observation vector."""
        mode_standard = float(mode == "standard")
        mode_boost = float(mode == "challenger_boost")
        mode_intervene = float(mode == "arbiter_intervene")

        return np.array([
            disagreement,
            quality_score,
            compliance,
            lambda_adv,
            round_num / max(max_rounds, 1),
            float(da_active),
            mode_standard,
            mode_boost,
            mode_intervene,
            prop_confidence,
            chal_confidence,
            np.clip(quality_trend, -1.0, 1.0),
            np.clip(disagreement_trend, -1.0, 1.0),
            (prop_score + chal_score) / 2.0,
        ], dtype=np.float32)

    def encode_role_observation(
        self,
        shared_obs: np.ndarray,
        role: str,
    ) -> np.ndarray:
        """Encode role-specific 20D observation (shared 14D + role 6D)."""
        if not self.enable_role_obs or self.role_tracker is None:
            return shared_obs
        role_obs = self.role_tracker.encode(role)
        return build_role_observation(shared_obs, role_obs)

    # ── Prompt Style Composers ──

    def compose_proposer_style(self, sig: StrategySignals) -> str:
        parts = []
        if sig.proposer_assertiveness > 0.7:
            parts.append("请用坚定自信的语气阐述你的方案，明确表达立场。")
        elif sig.proposer_assertiveness < 0.3:
            parts.append("请用谦逊探讨的语气提出方案，留有商榷空间。")
        if sig.proposer_detail_level > 0.7:
            parts.append("请提供详尽的技术细节、数据支撑和具体实施步骤。")
        elif sig.proposer_detail_level < 0.3:
            parts.append("请聚焦核心要点，保持简洁明了。")
        if sig.proposer_compliance_focus > 0.7:
            parts.append("请特别注意确保方案完全符合所有既定规则和约束。")
        if sig.proposer_incorporation > 0.7:
            parts.append("请充分吸纳挑战者的反馈，在方案中明确回应每个质疑点。")
        if sig.proposer_reasoning_depth > 0.7:
            parts.append("请进行深度推理，展示完整的思维链，从前提到结论逐步论证。")
        if sig.proposer_self_critique > 0.7:
            parts.append("在提出方案后，请主动审视并列出可能存在的弱点及应对策略。")
        if sig.proposer_evidence_demand > 0.7:
            parts.append("你的每个关键论点都必须引用具体的数据、文献或案例作为支撑证据。")
        return "\n".join(parts) if parts else ""

    def compose_challenger_style(self, sig: StrategySignals) -> str:
        parts = []
        if sig.challenger_aggressiveness > 0.7:
            parts.append("请发挥尖锐批评者角色，深入挖掘方案的根本性缺陷和逻辑漏洞。")
        elif sig.challenger_aggressiveness < 0.3:
            parts.append("请以建设性对话的方式温和提出你的疑虑。")
        if sig.challenger_specificity > 0.7:
            parts.append("请精准指出具体的薄弱环节，引用方案中的特定表述进行批驳。")
        if sig.challenger_constructiveness > 0.7:
            parts.append("在指出问题的同时，请提供你认为更优的替代方案或改进建议。")
        if sig.challenger_novelty > 0.7:
            parts.append("请开辟新的质疑角度，提出此前未被讨论过的风险和问题。")
        if sig.challenger_attack_angle > 0.7:
            parts.append("请重点从事实和数据角度质疑：数据是否准确？前提假设是否成立？")
        elif sig.challenger_attack_angle < 0.3:
            parts.append("请重点从逻辑推理角度质疑：论证是否有逻辑跳跃？因果关系是否成立？")
        if sig.challenger_evidence_demand > 0.7:
            parts.append("你的每个质疑都必须给出反面证据或数据支撑。")
        return "\n".join(parts) if parts else ""

    def compose_arbiter_style(self, sig: StrategySignals) -> str:
        parts = []
        if sig.arbiter_strictness > 0.7:
            parts.append("请以严格标准进行评判，不轻易给出高分。")
        elif sig.arbiter_strictness < 0.3:
            parts.append("请以鼓励进步为导向进行评判，认可合理的改进。")
        if sig.arbiter_detail_feedback > 0.7:
            parts.append("请给出详尽的评判理由，逐点分析优缺点。")
        if sig.arbiter_consensus_bias > 0.7:
            parts.append("如果双方观点已趋于一致且方案质量达标，请积极推动共识。")
        elif sig.arbiter_consensus_bias < 0.3:
            parts.append("请保持审慎态度，除非方案确实优秀，否则鼓励继续辩论。")
        if sig.arbiter_rule_emphasis > 0.7:
            parts.append("请特别重视规则合规性，将其作为评判的首要标准。")
        if sig.arbiter_framework_weight > 0.7:
            parts.append("请使用结构化评分框架，分别从逻辑性、可行性、创新性、证据充分性、合规性五个维度评分。")
        if sig.arbiter_evidence_demand > 0.7:
            parts.append("请特别审查双方的证据质量，无证据支撑的论点应被降权。")
        return "\n".join(parts) if parts else ""

    # ── Reward Computation ──

    @staticmethod
    def compute_reward(
        prev_state: Dict[str, float],
        curr_state: Dict[str, float],
        done: bool,
        consensus_reached: bool,
        compliance_rewards: Dict[str, float] | None = None,
    ) -> Dict[str, float]:
        """Compute per-agent rewards from debate state transitions.

        v2: Adds compliance reward bonus from closed-loop verification.
        """
        dq = curr_state.get("quality", 0.5) - prev_state.get("quality", 0.5)
        dd = prev_state.get("disagreement", 0.5) - curr_state.get("disagreement", 0.5)
        dc = curr_state.get("compliance", 0.5) - prev_state.get("compliance", 0.5)
        quality = curr_state.get("quality", 0.5)

        base = 0.3 * dq + 0.2 * dd - 0.01

        terminal_bonus = 0.0
        if done and consensus_reached:
            terminal_bonus = 2.0 * quality
        elif done and not consensus_reached:
            terminal_bonus = -0.5

        rewards = {
            "proposer_ctrl": base + 0.2 * dq + 0.1 * dc + terminal_bonus,
            "challenger_ctrl": base + 0.15 * abs(dd) + 0.1 * dq + terminal_bonus,
            "arbiter_ctrl": base + 0.1 * dc + 0.1 * dq + terminal_bonus,
            "coordinator": 0.5 * base + 0.3 * dq + 0.1 * dd + terminal_bonus,
        }

        # v2: Add compliance reward bonuses
        if compliance_rewards:
            for role, bonus in compliance_rewards.items():
                if role in rewards:
                    rewards[role] += bonus

        return rewards

    # ── Internal helpers ──

    @staticmethod
    def _to_unit(action: np.ndarray) -> np.ndarray:
        return np.clip((action + 1.0) / 2.0, 0.0, 1.0)

    @staticmethod
    def _to_unit_scalar(val: float) -> float:
        return float(np.clip((val + 1.0) / 2.0, 0.0, 1.0))

    @staticmethod
    def _map_temp(unit_val: float, temp_range: Tuple[float, float]) -> float:
        lo, hi = temp_range
        return lo + float(np.clip(unit_val, 0.0, 1.0)) * (hi - lo)

    def reset(self) -> None:
        """Reset per-episode state."""
        self._last_compliance.clear()
        if self.role_tracker:
            self.role_tracker.reset()

    # ── Online Mode Override ──

    def apply_online_override(
        self,
        signals: StrategySignals,
        online_params: Dict[str, np.ndarray],
    ) -> StrategySignals:
        """Override RL signals with online-accumulated parameters.

        In online mode, the OnlineParameterUpdater provides per-role
        4D parameter vectors that gradually replace RL output as
        confidence grows.

        Parameters
        ----------
        signals : StrategySignals
            Original signals from frozen RL.
        online_params : dict[str, np.ndarray]
            role -> 4D parameter array from OnlineParameterUpdater.

        Returns
        -------
        signals : StrategySignals
            Modified signals with online overrides applied.
        """
        if "proposer" in online_params:
            p = online_params["proposer"]
            signals.proposer_assertiveness = float(p[0])
            signals.proposer_detail_level = float(p[1])
            signals.proposer_compliance_focus = float(p[2])
            signals.proposer_incorporation = float(p[3])
            signals.proposer_temperature = self._map_temp(
                0.3 * p[0] + 0.7 * (1.0 - p[2]), self.temp_range
            )

        if "challenger" in online_params:
            c = online_params["challenger"]
            signals.challenger_aggressiveness = float(c[0])
            signals.challenger_specificity = float(c[1])
            signals.challenger_constructiveness = float(c[2])
            signals.challenger_novelty = float(c[3])
            signals.challenger_temperature = self._map_temp(
                0.5 * c[0] + 0.3 * c[3] + 0.2 * (1.0 - c[1]), self.temp_range
            )

        if "arbiter" in online_params:
            a = online_params["arbiter"]
            signals.arbiter_strictness = float(a[0])
            signals.arbiter_detail_feedback = float(a[1])
            signals.arbiter_consensus_bias = float(a[2])
            signals.arbiter_rule_emphasis = float(a[3])
            signals.arbiter_temperature = self._map_temp(
                0.3 * (1.0 - a[0]) + 0.7 * 0.3, self.arbiter_temp_range
            )

        return signals
