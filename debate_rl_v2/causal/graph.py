"""CausalGraph — adjacency-list causal graph with path queries.

Stores cause→effect relations and supports:
- Chain discovery (BFS/DFS up to max_depth)
- Topic-based semantic query (keyword matching)
- Context generation for LLM prompt injection
"""

from __future__ import annotations

import json
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from debate_rl_v2.causal.dataset import CausalRelation
from debate_rl_v2.logging_config import get_logger

logger = get_logger("causal.graph")


@dataclass
class CausalChain:
    """An ordered sequence of causal relations forming a reasoning chain."""
    chain_id: str = ""
    relations: list[CausalRelation] = field(default_factory=list)
    topic: str = ""
    tags: list[str] = field(default_factory=list)
    total_confidence: float = 1.0
    usage_count: int = 0
    success_count: int = 0

    def __post_init__(self) -> None:
        if not self.chain_id:
            self.chain_id = str(uuid.uuid4())[:12]
        if self.relations and self.total_confidence == 1.0:
            self.total_confidence = 1.0
            for r in self.relations:
                self.total_confidence *= r.confidence

    def to_text(self) -> str:
        """Render chain as readable text for LLM injection."""
        if not self.relations:
            return ""
        parts = []
        for i, r in enumerate(self.relations):
            arrow = " → " if i > 0 else ""
            parts.append(f"{arrow}{r.cause} → {r.effect}")
            if r.context:
                parts.append(f"  (背景: {r.context})")
        conf = f"[置信度: {self.total_confidence:.2f}]"
        return f"{''.join(parts)} {conf}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "relations": [r.to_dict() for r in self.relations],
            "topic": self.topic,
            "tags": self.tags,
            "total_confidence": self.total_confidence,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CausalChain:
        return cls(
            chain_id=d.get("chain_id", ""),
            relations=[CausalRelation.from_dict(r) for r in d.get("relations", [])],
            topic=d.get("topic", ""),
            tags=d.get("tags", []),
            total_confidence=d.get("total_confidence", 1.0),
            usage_count=d.get("usage_count", 0),
            success_count=d.get("success_count", 0),
        )


class CausalGraph:
    """Directed causal graph with adjacency-list storage.

    Nodes are cause/effect strings. Edges carry confidence and context.
    Supports BFS path finding and topic-based chain retrieval.
    """

    def __init__(self) -> None:
        # cause -> [(effect, confidence, context, relation_id)]
        self._forward: dict[str, list[tuple[str, float, str, str]]] = {}
        # effect -> [(cause, confidence, context, relation_id)]
        self._backward: dict[str, list[tuple[str, float, str, str]]] = {}
        # All relations by ID
        self._relations: dict[str, CausalRelation] = {}
        # Pre-built chains
        self._chains: dict[str, CausalChain] = {}

    def add_relation(self, rel: CausalRelation) -> str:
        """Add a single causal relation to the graph. Returns relation ID."""
        rel_id = str(uuid.uuid4())[:12]
        self._relations[rel_id] = rel
        entry = (rel.effect, rel.confidence, rel.context, rel_id)
        self._forward.setdefault(rel.cause, []).append(entry)
        back_entry = (rel.cause, rel.confidence, rel.context, rel_id)
        self._backward.setdefault(rel.effect, []).append(back_entry)
        return rel_id

    def add_chain(self, chain: CausalChain) -> None:
        """Add a pre-built chain and its constituent relations."""
        for rel in chain.relations:
            self.add_relation(rel)
        self._chains[chain.chain_id] = chain

    def query(self, topic: str, max_depth: int = 3, max_chains: int = 5) -> list[CausalChain]:
        """Find causal chains related to a topic.

        Strategy:
        1. Find nodes matching topic keywords
        2. BFS forward from each matching node up to max_depth
        3. Return discovered chains sorted by confidence
        """
        topic_lower = topic.lower()
        topic_words = set(topic_lower.split())

        # Find matching root nodes
        matching_nodes: list[tuple[float, str]] = []
        all_nodes = set(self._forward.keys()) | set(self._backward.keys())
        for node in all_nodes:
            node_lower = node.lower()
            node_words = set(node_lower.split())
            overlap = len(topic_words & node_words)
            if overlap > 0 or topic_lower in node_lower or node_lower in topic_lower:
                score = overlap + (1.0 if topic_lower in node_lower else 0.0)
                matching_nodes.append((score, node))

        matching_nodes.sort(key=lambda x: -x[0])
        roots = [n for _, n in matching_nodes[:10]]

        # BFS from each root
        chains: list[CausalChain] = []
        for root in roots:
            discovered = self._bfs_chains(root, max_depth)
            chains.extend(discovered)

        # Also include pre-built chains matching topic
        for chain in self._chains.values():
            chain_text = " ".join(r.cause + " " + r.effect for r in chain.relations).lower()
            if any(w in chain_text for w in topic_words):
                chains.append(chain)

        # Deduplicate and sort by confidence
        seen: set[str] = set()
        unique: list[CausalChain] = []
        for c in chains:
            if c.chain_id not in seen:
                seen.add(c.chain_id)
                unique.append(c)
        unique.sort(key=lambda c: -c.total_confidence)
        return unique[:max_chains]

    def find_path(self, cause: str, effect: str, max_depth: int = 5) -> CausalChain | None:
        """Find a causal path from cause to effect using BFS."""
        if cause not in self._forward:
            return None

        queue: deque[tuple[str, list[CausalRelation]]] = deque()
        # Seed with direct edges from cause
        for eff, conf, ctx, _ in self._forward.get(cause, []):
            rel = CausalRelation(cause=cause, effect=eff, confidence=conf, context=ctx)
            queue.append((eff, [rel]))

        visited: set[str] = {cause}
        while queue:
            current, path = queue.popleft()
            if current.lower() == effect.lower():
                return CausalChain(relations=path, topic=f"{cause} → {effect}")
            if len(path) >= max_depth:
                continue
            if current in visited:
                continue
            visited.add(current)
            for eff, conf, ctx, _ in self._forward.get(current, []):
                if eff not in visited:
                    rel = CausalRelation(cause=current, effect=eff, confidence=conf, context=ctx)
                    queue.append((eff, path + [rel]))

        return None

    def build_context(self, topic: str, max_chains: int = 3) -> str:
        """Generate causal context text for LLM prompt injection."""
        chains = self.query(topic, max_chains=max_chains)
        if not chains:
            return ""
        lines = ["## 相关因果推理链"]
        for i, chain in enumerate(chains, 1):
            text = chain.to_text()
            if text:
                lines.append(f"{i}. {text}")
        return "\n".join(lines)

    def _bfs_chains(self, root: str, max_depth: int) -> list[CausalChain]:
        """BFS from root, collecting all paths as chains."""
        chains: list[CausalChain] = []
        queue: deque[tuple[str, list[CausalRelation], int]] = deque()

        for eff, conf, ctx, _ in self._forward.get(root, []):
            rel = CausalRelation(cause=root, effect=eff, confidence=conf, context=ctx)
            queue.append((eff, [rel], 1))

        visited: set[str] = {root}
        while queue:
            current, path, depth = queue.popleft()
            # Each path of length >= 1 is a valid chain
            chain = CausalChain(relations=list(path), topic=root)
            chains.append(chain)

            if depth >= max_depth or current in visited:
                continue
            visited.add(current)

            for eff, conf, ctx, _ in self._forward.get(current, []):
                if eff not in visited:
                    rel = CausalRelation(cause=current, effect=eff, confidence=conf, context=ctx)
                    queue.append((eff, path + [rel], depth + 1))

        return chains

    @property
    def node_count(self) -> int:
        return len(set(self._forward.keys()) | set(self._backward.keys()))

    @property
    def edge_count(self) -> int:
        return len(self._relations)

    def save(self, path: str) -> None:
        """Save graph to JSON."""
        data = {
            "relations": [r.to_dict() for r in self._relations.values()],
            "chains": [c.to_dict() for c in self._chains.values()],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("CausalGraph saved: %d nodes, %d edges → %s",
                     self.node_count, self.edge_count, path)

    def load(self, path: str) -> None:
        """Load graph from JSON."""
        p = Path(path)
        if not p.exists():
            logger.warning("CausalGraph file not found: %s", path)
            return
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for rd in data.get("relations", []):
            self.add_relation(CausalRelation.from_dict(rd))
        for cd in data.get("chains", []):
            chain = CausalChain.from_dict(cd)
            self._chains[chain.chain_id] = chain
        logger.info("CausalGraph loaded: %d nodes, %d edges from %s",
                     self.node_count, self.edge_count, path)
