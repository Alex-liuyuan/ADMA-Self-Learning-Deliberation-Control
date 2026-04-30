"""Knowledge distillation and consolidation — episode-level learning."""

from debate_rl_v2.knowledge.distiller import EpisodeDistiller
from debate_rl_v2.knowledge.consolidator import KnowledgeConsolidator

__all__ = [
    "EpisodeDistiller",
    "KnowledgeConsolidator",
]
