"""辩论场景机制编排 — 封装论文五大核心机制。

参照 mdt_game/mechanisms.py 的模式，将 core/ 下的四个机制组件
（AdversarialIntensityController, SoftSwitchController,
DevilAdvocateVerifier, EvidenceChain）协调为统一接口。

core/ 下的组件保持不变（它们是领域无关的数学组件），
本类负责协调调用顺序和参数传递。
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np

from debate_rl_v2.core.adversarial import AdversarialIntensityController
from debate_rl_v2.core.soft_switch import SoftSwitchController, SwitchState
from debate_rl_v2.core.devil_advocate import DevilAdvocateVerifier, VerificationResult
from debate_rl_v2.core.evidence_chain import EvidenceChain
from debate_rl_v2.framework.mechanism import BaseMechanismOrchestrator, MechanismSnapshot
from debate_rl_v2.framework.types import CollaborationState
from debate_rl_v2.logging_config import get_logger

logger = get_logger("scenarios.debate.mechanisms")


@dataclass
class DebateMechanismState:
    """辩论机制状态快照。"""
    lambda_adv: float = 0.5
    disagreement: float = 0.5
    mode: str = "standard"
    da_active: bool = False
    switch_state: Optional[SwitchState] = None
    da_result: Optional[VerificationResult] = None
    evidence_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        switch_state = data.get("switch_state")
        if isinstance(switch_state, dict):
            data["switch_state"] = dict(switch_state)
        da_result = data.get("da_result")
        if isinstance(da_result, dict):
            data["da_result"] = dict(da_result)
        return data


class DebateMechanismOrchestrator(BaseMechanismOrchestrator):
    """辩论场景机制编排 — 封装论文五大核心机制。

    组件：
    - adversarial: AdversarialIntensityController — 动态对抗强度 λ(t)
    - soft_switch: SoftSwitchController — 概率软切换
    - devil_advocate: DevilAdvocateVerifier — 魔鬼代言人
    - evidence_chain: EvidenceChain — 证据链追踪
    """

    # 用于启发式分歧度计算的模式
    _OPPOSITION_PATTERN = re.compile(
        r"反对|不同意|质疑|风险|不可行|不建议|需要注意|但是|然而|"
        r"不足|问题|担忧|商榷|谨慎|争议|矛盾|不确定"
    )
    _SUPPORT_PATTERN = re.compile(r"可行|合理|支持|同意|一致|推荐|适合")
    _ALTERNATIVE_PATTERN = re.compile(r"替代|改用|可考虑|另一种|备选|优化|调整")
    _TOKEN_PATTERN = re.compile(r"[A-Za-z0-9\+\-]+|[\u4e00-\u9fff]{2,}")
    _TOKEN_STOPWORDS = {
        "当前", "方案", "治疗", "建议", "患者", "需要", "进行", "可以", "因为", "以及",
        "对于", "根据", "同时", "进一步", "评估", "reasoning", "assessment",
    }

    def __init__(
        self,
        max_rounds: int = 10,
        eta: float = 0.2,
        alpha: float = 0.6,
        omega: float = 0.7,
        tau_low: float = 0.3,
        tau_high: float = 0.7,
        steepness: float = 10.0,
        da_disagreement_threshold: float = 0.2,
        da_update_threshold: float = 0.1,
        da_stability_window: int = 2,
        seed: int = 42,
    ) -> None:
        self.max_rounds = max(1, int(max_rounds))
        self.adversarial = AdversarialIntensityController(
            eta=eta, alpha=alpha, omega=omega, max_steps=max_rounds,
        )
        self.soft_switch = SoftSwitchController(
            tau_low=tau_low, tau_high=tau_high, steepness=steepness,
        )
        self.devil_advocate = DevilAdvocateVerifier(
            disagreement_threshold=da_disagreement_threshold,
            update_threshold=da_update_threshold,
            stability_window=da_stability_window,
        )
        self.evidence_chain = EvidenceChain()
        self._rng = np.random.default_rng(seed)

        self._prev_disagreement: float = 0.5
        self._prev_observed_disagreement: float = 0.5
        self._prev_dynamic_disagreement: float = 0.5
        self._last_state = DebateMechanismState()

    def reset(self) -> None:
        """重置所有机制状态。"""
        self.adversarial.reset()
        self.devil_advocate.reset()
        self.evidence_chain.reset()
        self._prev_disagreement = 0.5
        self._prev_observed_disagreement = 0.5
        self._prev_dynamic_disagreement = 0.5
        self._last_state = DebateMechanismState()

    # ── 分歧度计算 ──

    def compute_disagreement(
        self,
        proposer_output: Dict[str, Any],
        challenger_output: Dict[str, Any],
    ) -> float:
        """启发式近似语义分歧度 D(t)。"""
        prop_conf = float(proposer_output.get("confidence", 0.5))
        chal_conf = float(challenger_output.get("confidence", 0.5))
        conf_diff = abs(prop_conf - chal_conf)

        prop_text = str(proposer_output.get("proposal", ""))
        chal_text = str(
            challenger_output.get("challenge", "")
            or challenger_output.get("reasoning", "")
        )

        prop_tokens = {
            t for t in self._TOKEN_PATTERN.findall(prop_text.lower())
            if len(t.strip()) > 1 and t.strip() not in self._TOKEN_STOPWORDS
        }
        chal_tokens = {
            t for t in self._TOKEN_PATTERN.findall(chal_text.lower())
            if len(t.strip()) > 1 and t.strip() not in self._TOKEN_STOPWORDS
        }
        if prop_tokens or chal_tokens:
            overlap = len(prop_tokens & chal_tokens) / max(len(prop_tokens | chal_tokens), 1)
            semantic_gap = 1.0 - overlap
        else:
            semantic_gap = 0.5

        opposition = min(1.0, len(self._OPPOSITION_PATTERN.findall(chal_text)) / 4.0)
        support = min(1.0, len(self._SUPPORT_PATTERN.findall(chal_text)) / 3.0)
        alternative = min(1.0, len(self._ALTERNATIVE_PATTERN.findall(chal_text)) / 3.0)

        disagreement = (
            0.45 * semantic_gap
            + 0.2 * conf_diff
            + 0.25 * opposition
            + 0.1 * alternative
            - 0.15 * support
        )
        return max(0.0, min(1.0, disagreement))

    def shape_disagreement(
        self,
        observed_disagreement: float,
        round_num: int,
        quality: float = 0.0,
    ) -> float:
        """将观测分歧塑造成带阻尼震荡和收敛趋势的 D(t)。"""
        observed = max(0.0, min(1.0, float(observed_disagreement)))
        quality = max(0.0, min(1.0, float(quality)))
        progress = max(0.0, min(1.0, (round_num - 1) / max(self.max_rounds - 1, 1)))

        consensus_attractor = max(0.0, 0.01 + 0.035 * (1.0 - quality))
        decay_target = observed * math.exp(-2.2 * progress) + consensus_attractor
        contraction = 0.42 + 0.48 * progress
        shaped = self._prev_dynamic_disagreement - contraction * (
            self._prev_dynamic_disagreement - decay_target
        )
        if progress >= 0.72:
            shaped = min(shaped, self._prev_dynamic_disagreement - 0.02)

        shaped = max(0.0, min(self._prev_dynamic_disagreement, shaped))
        self._prev_observed_disagreement = observed
        self._prev_dynamic_disagreement = shaped
        return shaped

    def update_intensity(
        self,
        disagreement: float,
        round_num: int,
        quality: float = 0.0,
    ) -> float:
        """更新对抗强度 λ(t)。"""
        base_lam = self.adversarial.update(disagreement, round_num, quality=quality)
        progress = max(0.0, min(1.0, (round_num - 1) / max(self.max_rounds - 1, 1)))
        damping = math.exp(-2.8 * progress)
        oscillation = damping * 0.12 * math.sin(round_num * 1.7 + disagreement * 3.0)
        lam = max(0.0, min(1.0, base_lam + oscillation))
        self.adversarial.lambda_adv = lam
        if self.adversarial.history.lambda_adv:
            self.adversarial.history.lambda_adv[-1] = lam
        self._prev_disagreement = disagreement
        return lam

    # ── 核心协调 ──

    def step(
        self,
        round_num: int,
        proposer_output: Dict[str, Any],
        challenger_output: Dict[str, Any],
        quality: float = 0.0,
    ) -> DebateMechanismState:
        """执行一轮完整的机制更新。"""
        observed_disagreement = self.compute_disagreement(proposer_output, challenger_output)
        disagreement = self.shape_disagreement(
            observed_disagreement=observed_disagreement,
            round_num=round_num,
            quality=quality,
        )
        prev_disagreement = self._prev_disagreement

        lambda_adv = self.update_intensity(disagreement, round_num, quality)
        switch_state = self.soft_switch.decide(lambda_adv, self._rng)

        belief_update = abs(disagreement - prev_disagreement)
        self.devil_advocate.check_stability(disagreement, belief_update)
        da_active = self.devil_advocate.is_active

        da_result = None
        if da_active:
            da_result = self.devil_advocate.process_challenge(disagreement)

        state = DebateMechanismState(
            lambda_adv=lambda_adv,
            disagreement=disagreement,
            mode=switch_state.mode,
            da_active=da_active,
            switch_state=switch_state,
            da_result=da_result,
            evidence_count=len(self.evidence_chain),
        )
        self._last_state = state
        return state

    # ── Framework adapter ──

    def update(
        self,
        *,
        state: CollaborationState,
        round_num: int,
        role_outputs: dict[str, dict[str, Any]],
        history: list[dict[str, Any]],
    ) -> MechanismSnapshot:
        """Framework adapter: 从通用回合输出更新机制状态。"""
        proposer_output = role_outputs.get("proposer", {})
        challenger_output = role_outputs.get("challenger", {})
        mechanism_state = self.step(
            round_num=round_num,
            proposer_output=proposer_output,
            challenger_output=challenger_output,
            quality=state.quality_score,
        )
        state.metadata["_current_mechanism_state"] = mechanism_state
        mechanism_dict = mechanism_state.to_dict()

        if (
            mechanism_state.da_result is not None
            and not mechanism_state.da_result.is_robust
        ):
            state.metadata["da_found_issue"] = True

        state.metadata.setdefault("disagreement_history", []).append(
            mechanism_state.disagreement,
        )

        for role_name, output in role_outputs.items():
            self.evidence_chain.record(
                step=round_num,
                role=role_name,
                action=str(output)[:100],
                triggered_rules=[],
                rule_confidences=[],
                rule_satisfactions=[],
                compliance_score=state.compliance,
                disagreement=mechanism_state.disagreement,
                lambda_adv=mechanism_state.lambda_adv,
                mode=mechanism_state.mode,
                devil_advocate_active=mechanism_state.da_active,
                notes="",
            )

        return MechanismSnapshot(mechanism_dict)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """返回供 Dashboard 展示的数据。"""
        s = self._last_state
        return {
            "lambda_adv": s.lambda_adv,
            "disagreement": s.disagreement,
            "mode": s.mode,
            "da_active": s.da_active,
            "evidence_count": s.evidence_count,
            "intensity_history": {
                "lambda": list(self.adversarial.history.lambda_adv),
                "disagreement": list(self.adversarial.history.disagreement),
            },
        }
