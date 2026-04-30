"""Debate Pattern Memory — cross-episode pattern recognition.

Tracks recurring debate dynamics and extracts reusable strategies.
Migrated from debate_rl with cleaner separation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from debate_rl_v2.logging_config import get_logger

logger = get_logger("memory.debate_pattern")


@dataclass
class DebatePattern:
    """A recognized debate pattern that led to success or failure."""
    name: str
    description: str
    success_rate: float = 0.5
    occurrence_count: int = 0
    avg_quality: float = 0.5
    trigger_conditions: dict[str, Any] = field(default_factory=dict)
    example_actions: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def update(self, success: bool, quality: float) -> None:
        self.occurrence_count += 1
        alpha = 1.0 / self.occurrence_count
        self.success_rate = (1 - alpha) * self.success_rate + alpha * float(success)
        self.avg_quality = (1 - alpha) * self.avg_quality + alpha * quality


# Pre-defined pattern detectors
_PATTERN_DETECTORS: dict[str, dict[str, Any]] = {
    "aggressive_productive": {
        "description": "激烈挑战后方案质量大幅提升",
        "condition": lambda s: s.get("challenger_aggressiveness", 0) > 0.7
                     and s.get("quality_delta", 0) > 0.1,
    },
    "consensus_building": {
        "description": "低分歧 + 高合规 → 快速共识",
        "condition": lambda s: s.get("disagreement", 1) < 0.25
                     and s.get("compliance", 0) > 0.75,
    },
    "deadlock": {
        "description": "连续多轮质量无提升，辩论陷入僵局",
        "condition": lambda s: abs(s.get("quality_delta", 0)) < 0.02
                     and s.get("round_progress", 0) > 0.5,
    },
    "evidence_driven": {
        "description": "高证据引用 + 高置信度 → 强说服力",
        "condition": lambda s: s.get("prop_confidence", 0) > 0.85
                     and s.get("compliance", 0) > 0.7,
    },
    "constructive_challenge": {
        "description": "挑战者提供替代方案时质量改善更大",
        "condition": lambda s: s.get("constructiveness", 0) > 0.7
                     and s.get("quality_delta", 0) > 0.05,
    },
    "rule_breakthrough": {
        "description": "规则挖掘后合规性显著提升",
        "condition": lambda s: s.get("mined_rule", False)
                     and s.get("compliance_delta", 0) > 0.1,
    },
}


class DebatePatternTracker:
    """Tracks and learns from debate patterns across episodes."""

    def __init__(self, persist_path: str | None = None) -> None:
        self._patterns: dict[str, DebatePattern] = {}
        self._episode_rounds: list[dict[str, Any]] = []
        self._episode_detected: list[str] = []
        self._persist_path = persist_path

        for name, detector in _PATTERN_DETECTORS.items():
            self._patterns[name] = DebatePattern(name=name, description=detector["description"])

        if persist_path:
            self._load(persist_path)

    def record_round(self, state: dict[str, Any]) -> list[str]:
        """Record a round's state and detect active patterns."""
        self._episode_rounds.append(dict(state))
        detected = []
        for name, detector in _PATTERN_DETECTORS.items():
            try:
                if detector["condition"](state):
                    detected.append(name)
                    if name not in self._episode_detected:
                        self._episode_detected.append(name)
            except (KeyError, TypeError):
                continue
        return detected

    def analyze_episode(
        self,
        consensus_reached: bool,
        final_quality: float,
        total_rounds: int,
        max_rounds: int,
    ) -> dict[str, DebatePattern]:
        success = consensus_reached and final_quality > 0.7
        updated = {}
        for name in self._episode_detected:
            pattern = self._patterns.get(name)
            if pattern:
                pattern.update(success=success, quality=final_quality)
                updated[name] = pattern
        self._episode_rounds.clear()
        self._episode_detected.clear()
        return updated

    def get_successful_patterns(self, min_rate: float = 0.6, min_count: int = 3) -> list[DebatePattern]:
        return [p for p in self._patterns.values()
                if p.success_rate >= min_rate and p.occurrence_count >= min_count]

    def build_context(self, role: str | None = None, max_patterns: int = 5) -> str:
        sorted_patterns = sorted(
            self._patterns.values(),
            key=lambda p: p.success_rate * min(p.occurrence_count, 10),
            reverse=True,
        )[:max_patterns]
        if not sorted_patterns:
            return ""
        lines = ["## 历史辩论模式洞察"]
        for p in sorted_patterns:
            pct = int(p.success_rate * 100)
            icon = "+" if p.success_rate > 0.6 else "~" if p.success_rate > 0.4 else "-"
            lines.append(f"- [{icon}] {p.description} (成功率 {pct}%, 出现 {p.occurrence_count} 次)")
        return "\n".join(lines)

    def save(self) -> None:
        if not self._persist_path:
            return
        data = {}
        for name, p in self._patterns.items():
            data[name] = {
                "name": p.name, "description": p.description,
                "success_rate": p.success_rate, "occurrence_count": p.occurrence_count,
                "avg_quality": p.avg_quality, "tags": p.tags,
            }
        Path(self._persist_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self._persist_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self, path: str) -> None:
        if not Path(path).exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for name, d in data.items():
                if name in self._patterns:
                    self._patterns[name].success_rate = d.get("success_rate", 0.5)
                    self._patterns[name].occurrence_count = d.get("occurrence_count", 0)
                    self._patterns[name].avg_quality = d.get("avg_quality", 0.5)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load patterns: %s", e)
