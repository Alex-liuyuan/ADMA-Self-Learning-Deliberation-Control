"""CausalExtractor — LLM-assisted causal relation extraction from text.

Uses the existing BaseLLMClient + RobustJSONParser to extract
cause-effect relations from debate transcripts and free text.
"""

from __future__ import annotations

from typing import Any

from debate_rl_v2.causal.dataset import CausalRelation
from debate_rl_v2.causal.graph import CausalChain
from debate_rl_v2.logging_config import get_logger

logger = get_logger("causal.extractor")

_EXTRACT_PROMPT = """\
请从以下文本中提取所有因果关系。每个因果关系包含：
- cause: 原因
- effect: 结果
- confidence: 置信度 (0.0-1.0)
- context: 简短背景说明

请以JSON数组格式返回：
[{{"cause": "...", "effect": "...", "confidence": 0.9, "context": "..."}}]

如果没有明确的因果关系，返回空数组 []。

文本：
{text}
"""

_TRAJECTORY_PROMPT = """\
请从以下辩论记录中提取因果推理链。关注辩论中使用的因果论证。

主题：{topic}

辩论记录：
{transcript}

请以JSON数组格式返回因果链：
[{{"chain": [{{"cause": "...", "effect": "...", "confidence": 0.9}}], "topic": "..."}}]
"""


class CausalExtractor:
    """Extract causal relations from text using LLM.

    Parameters
    ----------
    llm_client : BaseLLMClient | None
        LLM client for extraction. If None, extraction is disabled
        and methods return empty results.
    min_confidence : float
        Minimum confidence threshold for extracted relations.
    """

    def __init__(
        self,
        llm_client: Any = None,
        min_confidence: float = 0.5,
    ) -> None:
        self._client = llm_client
        self._min_confidence = min_confidence
        self._parser: Any = None
        if llm_client is not None:
            from debate_rl_v2.llm.json_parser import RobustJSONParser
            self._parser = RobustJSONParser()

    def extract_from_text(self, text: str) -> list[CausalRelation]:
        """Extract causal relations from free text.

        Returns empty list if no LLM client or extraction fails.
        """
        if self._client is None or not text.strip():
            return []

        prompt = _EXTRACT_PROMPT.format(text=text[:2000])
        try:
            response = self._client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
                json_mode=True,
            )
            return self._parse_relations(response.content, source="extracted")
        except Exception as e:
            logger.warning("Causal extraction failed: %s", e)
            return []

    def extract_from_trajectory(
        self,
        transcript: list[dict[str, Any]],
        topic: str,
    ) -> list[CausalChain]:
        """Extract causal chains from a debate trajectory.

        Parameters
        ----------
        transcript : list[dict]
            Debate transcript (list of round dicts with proposal/challenge/verdict).
        topic : str
            Debate topic for context.

        Returns
        -------
        chains : list[CausalChain]
            Extracted causal chains.
        """
        if self._client is None or not transcript:
            return []

        # Condense transcript
        condensed = []
        for t in transcript[-5:]:  # Last 5 rounds max
            parts = []
            if t.get("proposal"):
                parts.append(f"提案: {t['proposal'][:200]}")
            if t.get("challenge"):
                parts.append(f"质疑: {t['challenge'][:200]}")
            if t.get("verdict"):
                parts.append(f"裁定: {t['verdict'][:150]}")
            condensed.append(f"第{t.get('round', '?')}轮: " + " | ".join(parts))

        transcript_text = "\n".join(condensed)
        prompt = _TRAJECTORY_PROMPT.format(topic=topic, transcript=transcript_text)

        try:
            response = self._client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
                json_mode=True,
            )
            return self._parse_chains(response.content, topic)
        except Exception as e:
            logger.warning("Trajectory causal extraction failed: %s", e)
            return []

    def validate_chain(self, chain: CausalChain) -> float:
        """Validate a causal chain and return confidence score.

        Without LLM, returns the chain's own total_confidence.
        With LLM, asks for validation (future enhancement).
        """
        if not chain.relations:
            return 0.0
        return chain.total_confidence

    def _parse_relations(self, content: str, source: str = "") -> list[CausalRelation]:
        """Parse LLM response into CausalRelation list."""
        if self._parser is None:
            return []

        # Try parsing as JSON array
        import json
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try extracting from markdown block
            parsed = self._parser.parse(content)
            if parsed is None:
                return []
            data = parsed if isinstance(parsed, list) else parsed.get("relations", [])

        if not isinstance(data, list):
            data = [data] if isinstance(data, dict) else []

        relations = []
        for item in data:
            if not isinstance(item, dict):
                continue
            cause = item.get("cause", "")
            effect = item.get("effect", "")
            if not cause or not effect:
                continue
            conf = float(item.get("confidence", 0.7))
            if conf < self._min_confidence:
                continue
            relations.append(CausalRelation(
                cause=cause,
                effect=effect,
                confidence=conf,
                context=item.get("context", ""),
                source=source,
            ))
        return relations

    def _parse_chains(self, content: str, topic: str) -> list[CausalChain]:
        """Parse LLM response into CausalChain list."""
        import json
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            if self._parser:
                data = self._parser.parse(content)
            else:
                return []

        if data is None:
            return []
        if not isinstance(data, list):
            data = [data] if isinstance(data, dict) else []

        chains = []
        for item in data:
            if not isinstance(item, dict):
                continue
            chain_rels = item.get("chain", item.get("relations", []))
            if not chain_rels:
                continue
            relations = []
            for r in chain_rels:
                if isinstance(r, dict) and r.get("cause") and r.get("effect"):
                    relations.append(CausalRelation(
                        cause=r["cause"],
                        effect=r["effect"],
                        confidence=float(r.get("confidence", 0.7)),
                        context=r.get("context", ""),
                        source="extracted",
                    ))
            if relations:
                chains.append(CausalChain(
                    relations=relations,
                    topic=item.get("topic", topic),
                ))
        return chains
