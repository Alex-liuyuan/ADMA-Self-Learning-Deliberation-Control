"""Base Compliance Verifier — embedding-based semantic matching.

Replaces the broken regex-based verifier with a multi-strategy approach:
  1. Embedding cosine similarity (if sentence-transformers available)
  2. LLM-as-judge (if LLM client provided, one cheap call)
  3. Enhanced keyword heuristic (fallback, improved over v1 regex)

The verifier is domain-agnostic: it compares "intended style dimensions"
(from RL signals) against "observed style dimensions" (from LLM output)
without hardcoding any debate-specific keywords.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from debate_rl_v2.framework.types import ComplianceResult
from debate_rl_v2.logging_config import get_logger

logger = get_logger("framework.compliance")


@dataclass
class StyleDimension:
    """Definition of a single style dimension for compliance checking."""
    name: str
    description: str = ""
    # Anchor texts for low/high ends of the dimension
    low_anchors: list[str] = field(default_factory=list)
    high_anchors: list[str] = field(default_factory=list)
    weight: float = 1.0


class BaseComplianceVerifier:
    """Domain-agnostic compliance verifier with pluggable backends.

    Usage::

        verifier = BaseComplianceVerifier()
        verifier.register_dimension(StyleDimension(
            name="aggressiveness",
            low_anchors=["温和", "建议", "或许", "gentle", "suggest"],
            high_anchors=["严重", "根本性", "必须", "critical", "must"],
        ))

        result = verifier.verify(
            response="这个方案存在严重缺陷...",
            target_values={"aggressiveness": 0.8},
        )
    """

    def __init__(
        self,
        llm_judge: Callable[[str], dict[str, float]] | None = None,
        use_embeddings: bool = True,
    ) -> None:
        self._dimensions: dict[str, StyleDimension] = {}
        self._llm_judge = llm_judge
        self._embedder: Any = None
        self._use_embeddings = use_embeddings

        if use_embeddings:
            self._try_init_embedder()

    def _try_init_embedder(self) -> None:
        """Try to load a lightweight sentence embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2",
                device="cpu",
            )
            logger.info("Embedding-based compliance verifier initialized")
        except ImportError:
            logger.debug("sentence-transformers not available, using keyword fallback")
            self._embedder = None

    def register_dimension(self, dim: StyleDimension) -> None:
        """Register a style dimension for compliance checking."""
        self._dimensions[dim.name] = dim

    def register_dimensions(self, dims: list[StyleDimension]) -> None:
        for d in dims:
            self.register_dimension(d)

    def verify(
        self,
        response: str,
        target_values: dict[str, float],
    ) -> ComplianceResult:
        """Verify response compliance against target style values.

        Parameters
        ----------
        response : str
            LLM agent's response text.
        target_values : dict
            {dimension_name: target_value} where value is in [0, 1].

        Returns
        -------
        ComplianceResult
            Overall and per-dimension compliance scores.
        """
        if not target_values or not response:
            return ComplianceResult(overall_score=0.5)

        # Strategy 1: Embedding-based (most accurate)
        if self._embedder is not None:
            return self._verify_embedding(response, target_values)

        # Strategy 2: LLM-as-judge
        if self._llm_judge is not None:
            return self._verify_llm_judge(response, target_values)

        # Strategy 3: Enhanced keyword heuristic (fallback)
        return self._verify_keyword(response, target_values)

    def _verify_embedding(
        self, response: str, target_values: dict[str, float]
    ) -> ComplianceResult:
        """Embedding cosine similarity between response and anchor texts."""
        import numpy as np

        scores: dict[str, float] = {}
        response_emb = self._embedder.encode([response], normalize_embeddings=True)[0]

        for dim_name, target in target_values.items():
            dim = self._dimensions.get(dim_name)
            if dim is None or (not dim.low_anchors and not dim.high_anchors):
                scores[dim_name] = 0.5
                continue

            # Compute similarity to low and high anchors
            low_sim = 0.0
            high_sim = 0.0

            if dim.low_anchors:
                low_embs = self._embedder.encode(dim.low_anchors, normalize_embeddings=True)
                low_sim = float(np.max(np.dot(low_embs, response_emb)))

            if dim.high_anchors:
                high_embs = self._embedder.encode(dim.high_anchors, normalize_embeddings=True)
                high_sim = float(np.max(np.dot(high_embs, response_emb)))

            # Observed value: how "high" is the response on this dimension
            total = max(low_sim + high_sim, 1e-8)
            observed = high_sim / total

            # Compliance = 1 - |target - observed|
            scores[dim_name] = max(0.0, 1.0 - abs(target - observed))

        overall = sum(
            scores.get(d, 0.5) * self._dimensions.get(d, StyleDimension(name=d)).weight
            for d in target_values
        ) / max(sum(
            self._dimensions.get(d, StyleDimension(name=d)).weight
            for d in target_values
        ), 1e-8)

        return ComplianceResult(
            overall_score=overall,
            dimension_scores=scores,
            details="embedding-based",
        )

    def _verify_llm_judge(
        self, response: str, target_values: dict[str, float]
    ) -> ComplianceResult:
        """Use a cheap LLM call to judge compliance."""
        try:
            observed = self._llm_judge(response)
            scores: dict[str, float] = {}
            for dim_name, target in target_values.items():
                obs = observed.get(dim_name, 0.5)
                scores[dim_name] = max(0.0, 1.0 - abs(target - obs))
            overall = sum(scores.values()) / max(len(scores), 1)
            return ComplianceResult(
                overall_score=overall,
                dimension_scores=scores,
                details="llm-judge",
            )
        except Exception as e:
            logger.warning("LLM judge failed: %s, falling back to keyword", e)
            return self._verify_keyword(response, target_values)

    def _verify_keyword(
        self, response: str, target_values: dict[str, float]
    ) -> ComplianceResult:
        """Enhanced keyword heuristic — improved over v1 regex counting."""
        scores: dict[str, float] = {}
        response_lower = response.lower()

        for dim_name, target in target_values.items():
            dim = self._dimensions.get(dim_name)
            if dim is None:
                scores[dim_name] = 0.5
                continue

            low_count = sum(1 for a in dim.low_anchors if a.lower() in response_lower)
            high_count = sum(1 for a in dim.high_anchors if a.lower() in response_lower)
            total = max(low_count + high_count, 1)

            # Observed value based on keyword ratio
            observed = high_count / total

            # Also factor in response length as a proxy for detail level
            if dim_name in ("detail_level", "detail_feedback"):
                length_signal = min(1.0, len(response) / 800)
                observed = 0.6 * observed + 0.4 * length_signal

            scores[dim_name] = max(0.0, 1.0 - abs(target - observed))

        overall = sum(scores.values()) / max(len(scores), 1)
        return ComplianceResult(
            overall_score=overall,
            dimension_scores=scores,
            details="keyword-heuristic",
        )

    def compute_reward(
        self,
        results: dict[str, ComplianceResult],
        weight: float = 0.15,
    ) -> dict[str, float]:
        """Convert compliance results to per-role reward bonuses."""
        rewards: dict[str, float] = {}
        for role, result in results.items():
            rewards[role] = weight * (result.overall_score - 0.5) * 2.0
        return rewards
