"""Debate Strategy Bridge — concrete implementation on BaseStrategyBridge.

Thin wrapper that maps 4-role continuous actions to debate-specific
StrategySignals. All style composition is delegated to DebateStyleComposer.
"""

from __future__ import annotations


import numpy as np

from debate_rl_v2.framework.strategy import BaseStrategyBridge
from debate_rl_v2.framework.types import StrategySignals
from debate_rl_v2.framework.compliance import BaseComplianceVerifier, StyleDimension
from debate_rl_v2.scenarios.debate.style_composer import DebateStyleComposer
from debate_rl_v2.config.rl import StrategyBridgeConfig
from debate_rl_v2.logging_config import get_logger

logger = get_logger("scenarios.debate.strategy")


def create_debate_compliance_verifier(
    use_embeddings: bool = True,
) -> BaseComplianceVerifier:
    """Create a compliance verifier pre-configured with debate style dimensions."""
    verifier = BaseComplianceVerifier(use_embeddings=use_embeddings)
    verifier.register_dimensions([
        StyleDimension(
            name="assertiveness",
            low_anchors=["或许", "可能", "建议", "考虑", "也许", "perhaps", "maybe", "suggest"],
            high_anchors=["必须", "绝对", "坚定", "明确", "毫无疑问", "must", "definitely", "certain"],
        ),
        StyleDimension(
            name="aggressiveness",
            low_anchors=["温和", "建议", "商榷", "gentle", "mild"],
            high_anchors=["严重", "根本性", "致命", "荒谬", "不可接受", "critical", "fatal", "absurd"],
        ),
        StyleDimension(
            name="detail_level",
            low_anchors=["简要", "概括", "brief"],
            high_anchors=["详细", "具体", "步骤", "数据", "detailed", "specific", "step-by-step"],
        ),
        StyleDimension(
            name="constructiveness",
            low_anchors=["反对", "否定", "reject"],
            high_anchors=["替代", "改进", "建议", "优化", "alternative", "improve", "suggest"],
        ),
        StyleDimension(
            name="strictness",
            low_anchors=["不错", "良好", "认可", "good", "acceptable"],
            high_anchors=["不足", "扣分", "未达", "严格", "insufficient", "strict"],
        ),
        StyleDimension(
            name="specificity",
            low_anchors=["总体", "大致", "overall"],
            high_anchors=["具体", "特定", "针对", "第", "specific", "particular"],
        ),
        StyleDimension(
            name="compliance_focus",
            low_anchors=["创新", "突破", "innovative"],
            high_anchors=["合规", "规则", "指南", "遵循", "compliant", "guideline"],
        ),
        StyleDimension(
            name="consensus_bias",
            low_anchors=["继续", "深入", "不够", "continue", "insufficient"],
            high_anchors=["共识", "一致", "通过", "接受", "consensus", "agree", "accept"],
        ),
        StyleDimension(
            name="incorporation",
            low_anchors=["坚持", "维持", "maintain"],
            high_anchors=["吸纳", "回应", "采纳", "incorporate", "address"],
        ),
        StyleDimension(
            name="novelty",
            low_anchors=["同样", "重复", "same"],
            high_anchors=["新", "另一个", "不同角度", "novel", "new angle"],
        ),
        StyleDimension(
            name="rule_emphasis",
            low_anchors=["灵活", "flexible"],
            high_anchors=["规则", "合规", "标准", "rule", "standard"],
        ),
        StyleDimension(
            name="detail_feedback",
            low_anchors=["简评", "brief"],
            high_anchors=["逐点", "详尽", "分析", "detailed", "point-by-point"],
        ),
    ])
    return verifier


class DebateStrategyBridge(BaseStrategyBridge):
    """Debate-specific strategy bridge.

    Maps 4-role MADDPG actions to debate StrategySignals.
    Style composition delegated to DebateStyleComposer.
    """

    def __init__(
        self,
        config: StrategyBridgeConfig | None = None,
        enable_compliance: bool = True,
    ) -> None:
        cfg = config or StrategyBridgeConfig()
        verifier = create_debate_compliance_verifier() if enable_compliance else None
        super().__init__(
            temp_range=(cfg.temp_min, cfg.temp_max),
            compliance_verifier=verifier,
        )
        self.arbiter_temp_range = (cfg.arbiter_temp_min, cfg.arbiter_temp_max)
        self.max_eta_delta = cfg.max_eta_delta
        self.max_alpha_delta = cfg.max_alpha_delta
        self.max_tau_delta = cfg.max_tau_delta
        self.style_composer = DebateStyleComposer()

    def translate(self, actions: dict[str, np.ndarray]) -> StrategySignals:
        """Translate 4-role MADDPG actions to debate strategy signals."""
        signals = StrategySignals()

        # Proposer
        p = self._to_unit_array(actions.get("proposer_ctrl", np.zeros(4)))
        signals.temperatures["proposer"] = self._map_temp(
            0.3 * float(p[0]) + 0.7 * (1.0 - float(p[2]))
        )
        signals.style_dimensions["proposer"] = {
            "assertiveness": float(p[0]),
            "detail_level": float(p[1]),
            "compliance_focus": float(p[2]),
            "incorporation": float(p[3]),
        }

        # Challenger
        c = self._to_unit_array(actions.get("challenger_ctrl", np.zeros(4)))
        signals.temperatures["challenger"] = self._map_temp(
            0.5 * float(c[0]) + 0.3 * float(c[3]) + 0.2 * (1.0 - float(c[1]))
        )
        signals.style_dimensions["challenger"] = {
            "aggressiveness": float(c[0]),
            "specificity": float(c[1]),
            "constructiveness": float(c[2]),
            "novelty": float(c[3]),
        }

        # Arbiter
        a = self._to_unit_array(actions.get("arbiter_ctrl", np.zeros(4)))
        lo, hi = self.arbiter_temp_range
        signals.temperatures["arbiter"] = lo + float(
            np.clip(0.3 * (1.0 - a[0]) + 0.7 * 0.3, 0, 1)
        ) * (hi - lo)
        signals.style_dimensions["arbiter"] = {
            "strictness": float(a[0]),
            "detail_feedback": float(a[1]),
            "consensus_bias": float(a[2]),
            "rule_emphasis": float(a[3]),
        }

        # Coordinator mechanism deltas
        coord = actions.get("coordinator", np.zeros(5))
        signals.mechanism_deltas = {
            "eta_delta": float(coord[0]) * self.max_eta_delta,
            "alpha_delta": float(coord[1]) * self.max_alpha_delta,
            "tau_low_delta": float(coord[2]) * self.max_tau_delta,
            "tau_high_delta": float(coord[3]) * self.max_tau_delta,
        }
        signals.exploration_rate = self._to_unit(coord[4]) if len(coord) > 4 else 0.3

        return signals

    def compose_style(self, role: str, signals: StrategySignals) -> str:
        """Generate style directive for a role."""
        style = signals.get_style(role)
        return self.style_composer.compose(role, style)
