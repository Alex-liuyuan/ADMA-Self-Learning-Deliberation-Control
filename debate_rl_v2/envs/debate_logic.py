"""Shared Debate Logic Mixin — eliminates duplication between TextDebateEnv and FusionDebateEnv.

Extracts the common methods that were copy-pasted between llm_env.py (~866 lines)
and fusion_env.py (~888 lines). Both environments now inherit from this mixin.
"""

from __future__ import annotations

import math
from typing import Any

from debate_rl_v2.logging_config import get_logger

logger = get_logger("envs.debate_logic")


class DebateLogicMixin:
    """Shared debate logic for TextDebateEnv and FusionDebateEnv.

    Requires the host class to have:
      - self.state (TextDebateState with disagreement, quality_score, etc.)
      - self.history (list[DebateTurn])
      - self.max_rounds (int)
      - self.da_disagreement_threshold (float)
      - self.da_stability_window (int)
      - self._da_active (bool)
      - self._da_stable_count (int)
      - self._da_challenges_issued (int)
      - self._da_max_challenges (int)
    """

    def _compute_disagreement(
        self,
        arb_result: dict[str, Any],
        prop_result: dict[str, Any],
        chal_result: dict[str, Any],
        round_num: int,
    ) -> float:
        """Dynamic disagreement with damped oscillation converging to 0.

        Uses a damped excitation model:
            D(t) = D(t-1) + impulse(t) * envelope(t) - pull(t) * D(t-1)

        Behaviors:
          - Oscillation: tension alternately pushes D up/down
          - Damping: impulse amplitude decays exponentially
          - Convergence: quality-and-time-driven pull draws D toward 0
        """
        ps = arb_result.get("proposal_score", 0.5)
        cs = arb_result.get("challenge_score", 0.5)
        quality = arb_result.get("quality_score", 0.5)
        prop_conf = prop_result.get("confidence", 0.5)
        chal_conf = chal_result.get("confidence", 0.5)

        t = round_num
        T = max(self.max_rounds, 1)
        prev_d = self.state.disagreement

        # Tension: how confrontational is this round
        score_closeness = 1.0 - abs(ps - cs)
        conf_geo_mean = math.sqrt(max(prop_conf, 0.01) * max(chal_conf, 0.01))
        tension = score_closeness * conf_geo_mean

        # Impulse: alternating sign creates wave pattern
        sign = 1.0 if t % 2 == 1 else -1.0
        raw_impulse = sign * tension * 0.5
        amplitude_decay = math.exp(-0.3 * (t - 1))
        impulse = raw_impulse * amplitude_decay * 0.45

        # Convergence pull: draws D toward 0 as consensus forms
        convergence_strength = 0.05 + 0.35 * quality * (t / T) ** 0.6
        pull = convergence_strength * prev_d

        D = prev_d + impulse - pull
        return max(0.0, min(1.0, D))

    def _evaluate_compliance(self, arb_result: dict[str, Any], round_num: int) -> float:
        """Compliance score from quality and proposal strength.

        Early rounds: more weight on proposal score (direct rule adherence).
        Late rounds: more weight on quality (holistic assessment).
        """
        quality = arb_result.get("quality_score", 0.5)
        ps = arb_result.get("proposal_score", 0.5)
        t_ratio = round_num / max(self.max_rounds, 1)
        w = 0.4 + 0.3 * t_ratio  # 0.4 -> 0.7
        return min(1.0, w * quality + (1.0 - w) * ps)

    def _check_devil_advocate(
        self,
        agents: Any,
        verbose: bool,
        round_num: int,
    ) -> bool:
        """Check if devil's advocate should activate and challenge.

        Returns True if DA was triggered.
        """
        d = self.state.disagreement

        if d < self.da_disagreement_threshold:
            self._da_stable_count += 1
        else:
            self._da_stable_count = 0

        if (self._da_stable_count >= self.da_stability_window
                and self._da_challenges_issued < self._da_max_challenges):
            self._da_active = True
            self._da_challenges_issued += 1

            if verbose:
                logger.info("Devil's advocate triggered at round %d", round_num)

            da_prompt = (
                f"⚠️ 系统检测到辩论可能陷入虚假共识。"
                f"请重新审视当前提案，发挥'魔鬼代言人'角色，"
                f"从最刁钻的角度提出质疑。\n\n"
                f"当前提案：{self.state.proposal}"
            )
            da_result = agents["challenger"].act(da_prompt, round_num=round_num)

            if verbose:
                challenge_text = da_result.get("challenge", "")[:150]
                logger.info("DA challenge: %s", challenge_text)

            self._da_stable_count = 0
            self.state.da_active = True
            return True

        self.state.da_active = False
        return False

    def _compute_trend(self) -> str:
        """Summarize recent trend for coordinator."""
        if len(self.history) < 2:
            return "刚开始辩论"
        last2 = self.history[-2:]
        d_trend = last2[-1].state.disagreement - last2[-2].state.disagreement
        q_trend = last2[-1].state.quality_score - last2[-2].state.quality_score
        parts = []
        parts.append("分歧上升" if d_trend > 0.05 else ("分歧下降" if d_trend < -0.05 else "分歧稳定"))
        parts.append("质量提升" if q_trend > 0.05 else ("质量下降" if q_trend < -0.05 else "质量稳定"))
        return "，".join(parts)

    def _compute_quality_trend(self) -> float:
        if len(self.history) < 2:
            return 0.0
        return self.history[-1].state.quality_score - self.history[-2].state.quality_score

    def _compute_disagree_trend(self) -> float:
        if len(self.history) < 2:
            return 0.0
        return self.history[-1].state.disagreement - self.history[-2].state.disagreement

    def _last_prop_conf(self) -> float:
        return self.history[-1].proposal_confidence if self.history else 0.5

    def _last_chal_conf(self) -> float:
        return self.history[-1].challenge_confidence if self.history else 0.5

    def _last_prop_score(self) -> float:
        return self.history[-1].arbiter_scores.get("proposal_score", 0.5) if self.history else 0.5

    def _last_chal_score(self) -> float:
        return self.history[-1].arbiter_scores.get("challenge_score", 0.5) if self.history else 0.5

    def _recent_history_dicts(self, n: int = 3) -> list[dict]:
        """Get recent turns as simple dicts for prompt formatting."""
        return [
            {
                "round": t.round_num,
                "proposal": t.proposal[:100],
                "challenge": t.challenge[:100],
                "verdict": t.verdict[:100],
            }
            for t in self.history[-n:]
        ]

    # ── Prompt builders (shared, DRY) ──

    def _build_proposer_msg(self) -> str:
        """Build proposer prompt. FusionDebateEnv overrides to prepend style."""
        return (
            f"主题: {self.topic}\n背景: {getattr(self, 'context', '')}\n"
            f"第{self.state.round_num}/{self.max_rounds}轮\n"
            f"上轮提案: {self.state.proposal[:200]}\n"
            f"上轮挑战: {self.state.challenge[:200]}\n"
            f"合规度: {self.state.compliance:.2f} 分歧度: {self.state.disagreement:.2f}\n"
            f"规则: {'; '.join(getattr(self, 'rules', [])[:5])}\n"
            f"请提出或改进你的方案。"
        )

    def _build_challenger_msg(self) -> str:
        """Build challenger prompt. FusionDebateEnv overrides to prepend style."""
        return (
            f"主题: {self.topic}\n"
            f"第{self.state.round_num}/{self.max_rounds}轮\n"
            f"当前提案: {self.state.proposal[:300]}\n"
            f"合规度: {self.state.compliance:.2f} 分歧度: {self.state.disagreement:.2f}\n"
            f"规则: {'; '.join(getattr(self, 'rules', [])[:5])}\n"
            f"请对提案进行质疑和挑战。"
        )

    def _build_arbiter_msg(self) -> str:
        """Build arbiter prompt. FusionDebateEnv overrides to prepend style."""
        return (
            f"主题: {self.topic}\n"
            f"第{self.state.round_num}/{self.max_rounds}轮\n"
            f"提案: {self.state.proposal[:200]}\n"
            f"挑战: {self.state.challenge[:200]}\n"
            f"规则: {'; '.join(getattr(self, 'rules', [])[:5])}\n"
            f"请评判提案质量和挑战有效性。"
        )

    def _build_coordinator_msg(self) -> str:
        """Build coordinator prompt."""
        trend = self._compute_trend()
        return (
            f"辩论进展: 第{self.state.round_num}/{self.max_rounds}轮\n"
            f"质量={self.state.quality_score:.2f} 分歧={self.state.disagreement:.2f}\n"
            f"趋势: {trend}\n请选择元动作来调整辩论参数。"
        )
