"""CausalSkill — causal reasoning chain skill type.

Wraps a CausalChain as a DebateSkill-compatible object for storage
in SkillDatabase and retrieval during debates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from debate_rl_v2.causal.dataset import CausalRelation
from debate_rl_v2.causal.graph import CausalChain
from debate_rl_v2.skills.skill_db import SkillRecord


@dataclass
class CausalSkill:
    """A causal reasoning chain packaged as a reusable skill.

    Bridges CausalChain and SkillRecord for unified storage.
    """
    chain: CausalChain
    topic: str = ""
    tags: list[str] = field(default_factory=list)
    source_mode: str = "training"

    def to_skill_record(self) -> SkillRecord:
        """Convert to a SkillRecord for SkillDatabase storage."""
        chain_text = self.chain.to_text()
        return SkillRecord(
            skill_id=f"causal-{self.chain.chain_id}",
            name=f"因果链-{self.topic[:30]}" if self.topic else f"因果链-{self.chain.chain_id[:8]}",
            skill_type="causal_chain",
            description=chain_text[:200] if chain_text else "因果推理链",
            scenario_tags=self.tags or ([self.topic[:30]] if self.topic else []),
            causal_chain=self.chain.to_dict(),
            avg_quality=self.chain.total_confidence,
            success_count=self.chain.success_count,
            total_uses=max(self.chain.usage_count, 1),
            source_mode=self.source_mode,
        )

    @classmethod
    def from_skill_record(cls, record: SkillRecord) -> CausalSkill | None:
        """Reconstruct from a SkillRecord. Returns None if not a causal skill."""
        if record.skill_type != "causal_chain" or record.causal_chain is None:
            return None
        chain = CausalChain.from_dict(record.causal_chain)
        return cls(
            chain=chain,
            topic=record.scenario_tags[0] if record.scenario_tags else "",
            tags=record.scenario_tags,
            source_mode=record.source_mode,
        )

    def to_context_text(self) -> str:
        """Generate context text for LLM prompt injection."""
        text = self.chain.to_text()
        if not text:
            return ""
        prefix = f"[因果推理·{self.topic}] " if self.topic else "[因果推理] "
        return prefix + text
