"""Self-Play Evolution — Population-Based Training for Debate Agents.

Implements self-play training where agents improve through mutual
competition across generations. Key components:

  - **OpponentPool**: Stores historical agent versions as opponents.
  - **EloRating**: Tracks agent skill levels using the Elo system.
  - **SelfPlayScheduler**: Manages opponent selection and version saves.
  - **PopulationTrainer**: Orchestrates population-based training loop.

The core insight: by competing against diverse historical versions of
themselves, agents develop robust strategies rather than overfitting
to any single opponent.

Architecture::

    ┌───────────────────────────────────────────────────┐
    │              Self-Play Training Loop               │
    │                                                   │
    │  Generation N:                                    │
    │    Proposer_vN  vs  Challenger_pool               │
    │    Challenger_vN vs Proposer_pool                 │
    │    Arbiter_vN evaluates diverse matchups           │
    │    Coordinator_vN optimizes across matchups        │
    │                                                   │
    │  Every K episodes:                                │
    │    → Save snapshot → Add to pool                  │
    │    → Update Elo ratings                           │
    │    → Select opponents by rating proximity         │
    │                                                   │
    │  Selection strategies:                            │
    │    uniform  : random from pool                    │
    │    latest   : most recent versions                │
    │    elo_match: closest Elo rating (competitive)    │
    │    diverse  : maximize style diversity            │
    └───────────────────────────────────────────────────┘
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore


# ══════════════════════════════════════════════════════════
# Elo Rating System
# ══════════════════════════════════════════════════════════


class EloRating:
    """Elo rating system for tracking agent skill levels.

    Parameters
    ----------
    initial_rating : float
        Starting rating for new agents.
    k_factor : float
        Sensitivity of rating changes after each match.
    """

    def __init__(self, initial_rating: float = 1200.0, k_factor: float = 32.0):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self._ratings: Dict[str, float] = {}
        self._history: Dict[str, List[float]] = {}

    def get_rating(self, agent_id: str) -> float:
        """Get current rating for an agent."""
        if agent_id not in self._ratings:
            self._ratings[agent_id] = self.initial_rating
            self._history[agent_id] = [self.initial_rating]
        return self._ratings[agent_id]

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score (win probability) for agent A."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update(
        self,
        agent_a: str,
        agent_b: str,
        score_a: float,
        score_b: float,
    ) -> Tuple[float, float]:
        """Update ratings after a match.

        Parameters
        ----------
        agent_a, agent_b : str
            Agent identifiers.
        score_a, score_b : float
            Match scores (e.g., quality achieved by each side).

        Returns
        -------
        new_rating_a, new_rating_b : float
            Updated ratings.
        """
        ra = self.get_rating(agent_a)
        rb = self.get_rating(agent_b)

        # Normalize scores to [0, 1] outcome
        total = score_a + score_b
        if total > 0:
            outcome_a = score_a / total
            outcome_b = score_b / total
        else:
            outcome_a = outcome_b = 0.5

        ea = self.expected_score(ra, rb)
        eb = self.expected_score(rb, ra)

        new_ra = ra + self.k_factor * (outcome_a - ea)
        new_rb = rb + self.k_factor * (outcome_b - eb)

        self._ratings[agent_a] = new_ra
        self._ratings[agent_b] = new_rb
        self._history.setdefault(agent_a, []).append(new_ra)
        self._history.setdefault(agent_b, []).append(new_rb)

        return new_ra, new_rb

    def update_group(self, results: Dict[str, float]) -> Dict[str, float]:
        """Update ratings for a group match (all-vs-all).

        Parameters
        ----------
        results : dict
            {agent_id: score} for all participants.

        Returns
        -------
        new_ratings : dict
            Updated ratings for all agents.
        """
        agents = list(results.keys())
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                self.update(
                    agents[i], agents[j],
                    results[agents[i]], results[agents[j]],
                )
        return {a: self.get_rating(a) for a in agents}

    def leaderboard(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get top-rated agents."""
        sorted_agents = sorted(
            self._ratings.items(), key=lambda x: -x[1]
        )
        return sorted_agents[:top_k]

    def get_history(self, agent_id: str) -> List[float]:
        return self._history.get(agent_id, [])

    def save(self, path: str) -> None:
        data = {"ratings": self._ratings, "history": self._history}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        if not Path(path).exists():
            return
        with open(path, "r") as f:
            data = json.load(f)
        self._ratings = data.get("ratings", {})
        self._history = data.get("history", {})


# ══════════════════════════════════════════════════════════
# Opponent Pool
# ══════════════════════════════════════════════════════════


@dataclass
class AgentSnapshot:
    """A frozen snapshot of agent weights at a specific generation."""

    generation: int
    episode: int
    timestamp: float
    rating: float = 1200.0
    checkpoint_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # In-memory state dicts (loaded on demand)
    _state_dicts: Optional[Dict[str, Any]] = field(
        default=None, repr=False, compare=False
    )


class OpponentPool:
    """Pool of historical agent versions for self-play training.

    Stores snapshots of agent weights at various training stages.
    Supports multiple opponent selection strategies.

    Parameters
    ----------
    max_size : int
        Maximum number of snapshots to keep.
    save_dir : str
        Directory for persisting snapshots to disk.
    """

    def __init__(self, max_size: int = 50, save_dir: str = "checkpoints/pool"):
        self.max_size = max_size
        self.save_dir = save_dir
        self._snapshots: List[AgentSnapshot] = []
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    def add(
        self,
        state_dicts: Dict[str, Any],
        generation: int,
        episode: int,
        rating: float = 1200.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentSnapshot:
        """Add a new agent snapshot to the pool.

        Parameters
        ----------
        state_dicts : dict
            {agent_name: state_dict} for all agents.
        generation : int
            Training generation number.
        episode : int
            Training episode number.
        rating : float
            Current Elo rating.
        metadata : dict, optional
            Additional metadata.
        """
        # Save to disk
        ckpt_path = os.path.join(self.save_dir, f"gen_{generation:04d}.pt")
        if torch is not None:
            torch.save(state_dicts, ckpt_path)

        snapshot = AgentSnapshot(
            generation=generation,
            episode=episode,
            timestamp=time.time(),
            rating=rating,
            checkpoint_path=ckpt_path,
            metadata=metadata or {},
        )
        self._snapshots.append(snapshot)

        # Evict if over limit (keep highest-rated)
        if len(self._snapshots) > self.max_size:
            self._snapshots.sort(key=lambda s: s.rating)
            removed = self._snapshots.pop(0)
            # Clean up file
            if os.path.exists(removed.checkpoint_path):
                try:
                    os.remove(removed.checkpoint_path)
                except OSError:
                    pass

        return snapshot

    def sample(
        self,
        strategy: str = "elo_match",
        current_rating: float = 1200.0,
        n: int = 1,
    ) -> List[AgentSnapshot]:
        """Sample opponents from the pool.

        Parameters
        ----------
        strategy : str
            Selection strategy:
            - 'uniform': random selection
            - 'latest': most recent snapshots
            - 'elo_match': closest Elo rating
            - 'diverse': maximize generation spread
        current_rating : float
            Current agent's Elo rating (for elo_match).
        n : int
            Number of opponents to select.

        Returns
        -------
        opponents : list of AgentSnapshot
        """
        if not self._snapshots:
            return []

        n = min(n, len(self._snapshots))

        if strategy == "uniform":
            return random.sample(self._snapshots, n)

        elif strategy == "latest":
            return sorted(
                self._snapshots, key=lambda s: -s.episode
            )[:n]

        elif strategy == "elo_match":
            # Select opponents closest in rating (competitive matches)
            sorted_by_distance = sorted(
                self._snapshots,
                key=lambda s: abs(s.rating - current_rating),
            )
            return sorted_by_distance[:n]

        elif strategy == "diverse":
            # Maximize generation diversity
            if n >= len(self._snapshots):
                return list(self._snapshots)
            step = len(self._snapshots) / n
            return [
                self._snapshots[int(i * step)]
                for i in range(n)
            ]

        else:
            return random.sample(self._snapshots, n)

    def load_snapshot(self, snapshot: AgentSnapshot) -> Optional[Dict[str, Any]]:
        """Load state dicts from a snapshot."""
        if snapshot._state_dicts is not None:
            return snapshot._state_dicts
        if torch is not None and os.path.exists(snapshot.checkpoint_path):
            return torch.load(snapshot.checkpoint_path, map_location="cpu")
        return None

    @property
    def size(self) -> int:
        return len(self._snapshots)

    @property
    def avg_rating(self) -> float:
        if not self._snapshots:
            return 1200.0
        return float(np.mean([s.rating for s in self._snapshots]))

    def summary(self) -> Dict[str, Any]:
        """Get pool summary statistics."""
        if not self._snapshots:
            return {"size": 0}
        ratings = [s.rating for s in self._snapshots]
        return {
            "size": len(self._snapshots),
            "avg_rating": float(np.mean(ratings)),
            "max_rating": float(np.max(ratings)),
            "min_rating": float(np.min(ratings)),
            "generations": [s.generation for s in self._snapshots],
        }


# ══════════════════════════════════════════════════════════
# Self-Play Scheduler
# ══════════════════════════════════════════════════════════


@dataclass
class SelfPlayConfig:
    """Configuration for self-play training."""

    enabled: bool = True
    # Snapshot management
    snapshot_interval: int = 500          # save snapshot every N episodes
    max_pool_size: int = 50               # max snapshots in pool
    pool_dir: str = "checkpoints/pool"    # directory for snapshots
    # Opponent selection
    selection_strategy: str = "elo_match"  # uniform|latest|elo_match|diverse
    opponents_per_round: int = 1          # how many opponents to face
    # Elo system
    initial_elo: float = 1200.0
    elo_k_factor: float = 32.0
    # Self-play mixing
    self_play_ratio: float = 0.5         # fraction of episodes using pool opponents
    # Warmup before self-play starts
    warmup_episodes: int = 500


class SelfPlayScheduler:
    """Manages the self-play training schedule.

    Coordinates when to:
      - Save snapshots to the opponent pool
      - Sample opponents from the pool
      - Update Elo ratings after matches
      - Mix self-play episodes with regular training

    Usage::

        scheduler = SelfPlayScheduler(config)

        for episode in range(total):
            if scheduler.should_use_pool_opponent(episode):
                opponent = scheduler.sample_opponent()
                # ... run episode with pool opponent ...
                scheduler.report_result(episode, scores)
            else:
                # ... run episode with current opponents ...

            if scheduler.should_save_snapshot(episode):
                scheduler.save_snapshot(agent_state_dicts, episode)
    """

    def __init__(self, cfg: SelfPlayConfig):
        self.cfg = cfg
        self.elo = EloRating(
            initial_rating=cfg.initial_elo,
            k_factor=cfg.elo_k_factor,
        )
        self.pool = OpponentPool(
            max_size=cfg.max_pool_size,
            save_dir=cfg.pool_dir,
        )
        self._generation = 0
        self._current_rating = cfg.initial_elo

    def should_use_pool_opponent(self, episode: int) -> bool:
        """Whether this episode should use a pool opponent."""
        if not self.cfg.enabled:
            return False
        if episode < self.cfg.warmup_episodes:
            return False
        if self.pool.size == 0:
            return False
        return random.random() < self.cfg.self_play_ratio

    def should_save_snapshot(self, episode: int) -> bool:
        """Whether to save current weights as a snapshot."""
        if not self.cfg.enabled:
            return False
        if episode < self.cfg.warmup_episodes:
            return False
        return episode % self.cfg.snapshot_interval == 0

    def sample_opponent(self) -> Optional[AgentSnapshot]:
        """Sample an opponent from the pool."""
        opponents = self.pool.sample(
            strategy=self.cfg.selection_strategy,
            current_rating=self._current_rating,
            n=self.cfg.opponents_per_round,
        )
        return opponents[0] if opponents else None

    def load_opponent_weights(
        self, snapshot: AgentSnapshot
    ) -> Optional[Dict[str, Any]]:
        """Load opponent weights from snapshot."""
        return self.pool.load_snapshot(snapshot)

    def save_snapshot(
        self,
        state_dicts: Dict[str, Any],
        episode: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentSnapshot:
        """Save current agent weights to pool."""
        self._generation += 1
        snapshot = self.pool.add(
            state_dicts=state_dicts,
            generation=self._generation,
            episode=episode,
            rating=self._current_rating,
            metadata=metadata,
        )
        return snapshot

    def report_result(
        self,
        episode: int,
        agent_scores: Dict[str, float],
        opponent_scores: Optional[Dict[str, float]] = None,
        opponent_id: str = "pool_opponent",
    ) -> Dict[str, float]:
        """Report match result and update Elo ratings.

        Parameters
        ----------
        agent_scores : dict
            {agent_name: score} for current agents.
        opponent_scores : dict, optional
            Scores for pool opponent (if self-play).
        opponent_id : str
            Identifier for the pool opponent.

        Returns
        -------
        updated_ratings : dict
        """
        # Compute aggregate scores
        current_total = sum(agent_scores.values())
        opponent_total = sum(
            (opponent_scores or agent_scores).values()
        )

        # Update Elo
        current_id = f"current_v{self._generation}"
        new_current, new_opponent = self.elo.update(
            current_id, opponent_id,
            current_total, opponent_total,
        )
        self._current_rating = new_current

        return {
            "current_rating": new_current,
            "opponent_rating": new_opponent,
            "generation": self._generation,
        }

    @property
    def current_rating(self) -> float:
        return self._current_rating

    @property
    def generation(self) -> int:
        return self._generation

    def summary(self) -> Dict[str, Any]:
        """Get self-play training summary."""
        return {
            "enabled": self.cfg.enabled,
            "generation": self._generation,
            "current_rating": self._current_rating,
            "pool": self.pool.summary(),
            "leaderboard": self.elo.leaderboard(top_k=5),
        }

    def save(self, path: str) -> None:
        """Persist scheduler state."""
        elo_path = os.path.join(path, "elo_ratings.json")
        self.elo.save(elo_path)
        meta = {
            "generation": self._generation,
            "current_rating": self._current_rating,
        }
        with open(os.path.join(path, "self_play_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def load(self, path: str) -> None:
        """Load scheduler state."""
        elo_path = os.path.join(path, "elo_ratings.json")
        self.elo.load(elo_path)
        meta_path = os.path.join(path, "self_play_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self._generation = meta.get("generation", 0)
            self._current_rating = meta.get("current_rating", 1200.0)
