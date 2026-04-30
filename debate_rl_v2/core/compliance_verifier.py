"""Compliance Verifier — RL→LLM closed-loop validation.

.. deprecated::
    Use debate_rl_v2.scenarios.debate.compliance.DebateComplianceVerifier instead.
    This module is retained for backward compatibility with RL pre-training code.

Fixes the open-loop problem: RL issues strategy signals but never
verifies whether the LLM actually followed them.

This verifier evaluates LLM responses against RL strategy signals
and feeds compliance scores back as additional reward signals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


from debate_rl_v2.logging_config import get_logger

logger = get_logger("core.compliance_verifier")


@dataclass
class ComplianceResult:
    """Result of verifying LLM compliance with RL strategy signals."""
    overall_score: float = 0.5  # 0=completely non-compliant, 1=fully compliant
    dimension_scores: dict[str, float] = field(default_factory=dict)
    details: str = ""


class ComplianceVerifier:
    """Evaluates whether LLM responses comply with RL strategy signals.

    Uses lightweight heuristics (no additional LLM call) to assess:
    - Aggressiveness alignment (challenger)
    - Detail level alignment (proposer)
    - Strictness alignment (arbiter)
    - Evidence usage alignment

    The compliance score is fed back to RL as an additional reward term,
    closing the RL→LLM control loop.
    """

    # Aggressive language indicators (Chinese)
    _AGGRESSIVE_PATTERNS = re.compile(
        r"严重|根本性|致命|完全错误|不可接受|荒谬|站不住脚|漏洞百出|"
        r"强烈反对|必须|绝对不|毫无|彻底|严厉"
    )
    # Mild language indicators
    _MILD_PATTERNS = re.compile(
        r"或许|可能|建议|考虑|也许|一定程度|部分|值得商榷|"
        r"有待|进一步|适当|温和|谨慎"
    )
    # Evidence indicators
    _EVIDENCE_PATTERNS = re.compile(
        r"研究表明|数据显示|根据|文献|指南|试验|统计|证据|"
        r"meta分析|荟萃|RCT|随机|对照|p值|置信区间|"
        r"NCCN|ESMO|WHO|FDA|CFDA"
    )
    # Detail indicators (structured content)
    _DETAIL_PATTERNS = re.compile(
        r"\d+[.、)）]|第[一二三四五六七八九十]|首先|其次|最后|"
        r"具体来说|详细|步骤|方案[一二三]|维度"
    )
    # Self-critique indicators
    _SELF_CRITIQUE_PATTERNS = re.compile(
        r"不足|局限|风险|弱点|缺点|挑战|困难|需要注意|"
        r"潜在问题|可能的|但是|然而|不过"
    )

    def verify_proposer(
        self,
        response: str,
        assertiveness: float,
        detail_level: float,
        compliance_focus: float,
        evidence_demand: float,
    ) -> ComplianceResult:
        """Verify proposer response against RL strategy signals."""
        scores: dict[str, float] = {}

        # Assertiveness check
        agg_count = len(self._AGGRESSIVE_PATTERNS.findall(response))
        mild_count = len(self._MILD_PATTERNS.findall(response))
        total = max(agg_count + mild_count, 1)
        observed_assertiveness = agg_count / total
        scores["assertiveness"] = 1.0 - abs(assertiveness - observed_assertiveness)

        # Detail level check
        detail_count = len(self._DETAIL_PATTERNS.findall(response))
        response_len = len(response)
        observed_detail = min(1.0, detail_count / 5.0) * 0.5 + min(1.0, response_len / 1000) * 0.5
        scores["detail_level"] = 1.0 - abs(detail_level - observed_detail)

        # Evidence usage check
        evidence_count = len(self._EVIDENCE_PATTERNS.findall(response))
        observed_evidence = min(1.0, evidence_count / 3.0)
        scores["evidence"] = 1.0 - abs(evidence_demand - observed_evidence)

        overall = sum(scores.values()) / max(len(scores), 1)
        return ComplianceResult(
            overall_score=overall,
            dimension_scores=scores,
            details=f"proposer compliance: {overall:.2f}",
        )

    def verify_challenger(
        self,
        response: str,
        aggressiveness: float,
        constructiveness: float,
        specificity: float,
        evidence_demand: float,
    ) -> ComplianceResult:
        """Verify challenger response against RL strategy signals."""
        scores: dict[str, float] = {}

        # Aggressiveness check
        agg_count = len(self._AGGRESSIVE_PATTERNS.findall(response))
        mild_count = len(self._MILD_PATTERNS.findall(response))
        total = max(agg_count + mild_count, 1)
        observed_agg = agg_count / total
        scores["aggressiveness"] = 1.0 - abs(aggressiveness - observed_agg)

        # Constructiveness check (presence of alternatives/suggestions)
        constructive_patterns = re.findall(
            r"替代|建议|改进|更好的|可以考虑|另一种|方案|优化", response
        )
        observed_constructive = min(1.0, len(constructive_patterns) / 3.0)
        scores["constructiveness"] = 1.0 - abs(constructiveness - observed_constructive)

        # Specificity check (references to specific parts of proposal)
        specific_patterns = re.findall(r"第\d|具体|特定|关于.*的|针对", response)
        observed_specificity = min(1.0, len(specific_patterns) / 3.0)
        scores["specificity"] = 1.0 - abs(specificity - observed_specificity)

        # Evidence check
        evidence_count = len(self._EVIDENCE_PATTERNS.findall(response))
        observed_evidence = min(1.0, evidence_count / 3.0)
        scores["evidence"] = 1.0 - abs(evidence_demand - observed_evidence)

        overall = sum(scores.values()) / max(len(scores), 1)
        return ComplianceResult(
            overall_score=overall,
            dimension_scores=scores,
            details=f"challenger compliance: {overall:.2f}",
        )

    def verify_arbiter(
        self,
        response: str,
        strictness: float,
        detail_feedback: float,
        consensus_bias: float,
    ) -> ComplianceResult:
        """Verify arbiter response against RL strategy signals."""
        scores: dict[str, float] = {}

        # Strictness check
        strict_patterns = re.findall(r"不足|扣分|低于|未达|不够|欠缺|严格", response)
        lenient_patterns = re.findall(r"不错|良好|优秀|进步|改善|认可", response)
        total = max(len(strict_patterns) + len(lenient_patterns), 1)
        observed_strictness = len(strict_patterns) / total
        scores["strictness"] = 1.0 - abs(strictness - observed_strictness)

        # Detail feedback check
        detail_count = len(self._DETAIL_PATTERNS.findall(response))
        observed_detail = min(1.0, detail_count / 4.0)
        scores["detail_feedback"] = 1.0 - abs(detail_feedback - observed_detail)

        # Consensus bias check
        consensus_patterns = re.findall(r"共识|一致|达成|同意|接受|通过", response)
        continue_patterns = re.findall(r"继续|深入|进一步|不够|需要改进", response)
        total = max(len(consensus_patterns) + len(continue_patterns), 1)
        observed_bias = len(consensus_patterns) / total
        scores["consensus_bias"] = 1.0 - abs(consensus_bias - observed_bias)

        overall = sum(scores.values()) / max(len(scores), 1)
        return ComplianceResult(
            overall_score=overall,
            dimension_scores=scores,
            details=f"arbiter compliance: {overall:.2f}",
        )

    def compute_compliance_reward(
        self,
        results: dict[str, ComplianceResult],
        weight: float = 0.15,
    ) -> dict[str, float]:
        """Convert compliance results to per-agent reward bonuses.

        Parameters
        ----------
        results : dict
            {role: ComplianceResult} from verify_* methods.
        weight : float
            Weight of compliance reward relative to base reward.

        Returns
        -------
        rewards : dict
            {role: float} compliance reward bonuses.
        """
        rewards: dict[str, float] = {}
        for role, result in results.items():
            # Reward = weight * (compliance - 0.5) to center around 0
            rewards[role] = weight * (result.overall_score - 0.5) * 2.0
        return rewards
