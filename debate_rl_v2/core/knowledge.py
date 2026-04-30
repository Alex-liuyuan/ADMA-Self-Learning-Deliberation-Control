"""Neural-Symbolic Knowledge Engine — Section 4.3.

Implements:
  1. Logical Tensor Network (LTN) rules with differentiable satisfaction
  2. Learnable rule confidence w_r ∈ [0,1]
  3. Compliance score: m_k = σ(Σ_r w_r · μ_r(s) − θ)
  4. Gradient-based confidence update: Δw_r ∝ −∂L_task/∂w_r
  5. ILP-inspired online rule mining (Section 4.3.1)

Convergence guarantee (Theorem 3): under Robbins-Monro step-size conditions
and convex task loss, rule confidence w_r converges a.s. to a local optimum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class LogicalTensorRule(nn.Module):
    """A single differentiable logic rule in the LTN framework.

    Each rule computes a satisfaction degree μ_r(s) ∈ [0,1] via a small
    network, and carries a learnable confidence weight w_r.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the input state vector.
    hidden_dim : int
        Hidden layer size for the predicate network.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.predicate = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Raw logit for confidence; actual weight = σ(raw)
        self._confidence_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute satisfaction degree μ_r(s) ∈ [0,1]."""
        return torch.sigmoid(self.predicate(state)).squeeze(-1)

    @property
    def confidence(self) -> torch.Tensor:
        """Learnable confidence w_r ∈ [0,1]."""
        return torch.sigmoid(self._confidence_logit)


class KnowledgeEngine(nn.Module):
    """Neural-symbolic knowledge engine that manages differentiable rules.

    Computes compliance as: c = σ(Σ_r w_r · μ_r(s) − θ)

    Parameters
    ----------
    state_dim : int
        State vector dimensionality.
    num_rules : int
        Initial number of rules.
    threshold : float
        Compliance threshold θ.
    confidence_lr : float
        Learning rate for rule confidence updates.
    max_mined_rules : int
        Maximum number of additionally mined rules.
    """

    def __init__(
        self,
        state_dim: int,
        num_rules: int = 8,
        threshold: float = 0.0,
        confidence_lr: float = 0.01,
        max_mined_rules: int = 8,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.max_mined_rules = max_mined_rules
        self.mined_count = 0

        self.rules = nn.ModuleList(
            [LogicalTensorRule(state_dim) for _ in range(num_rules)]
        )
        self.threshold = nn.Parameter(torch.tensor(threshold))

        # Separate optimizer for rule confidence (can be a different lr)
        self._confidence_lr = confidence_lr

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def compliance_score(self, state: torch.Tensor) -> torch.Tensor:
        """Compute overall compliance score c ∈ [0,1].

        c = σ(Σ_r w_r · μ_r(s) − θ)
        """
        weighted_sum = torch.zeros(state.shape[0], device=state.device)
        for rule in self.rules:
            mu = rule(state)  # (batch,)
            weighted_sum = weighted_sum + rule.confidence * mu
        return torch.sigmoid(weighted_sum - self.threshold)

    def rule_satisfactions(
        self, state: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get per-rule (confidence, satisfaction) pairs."""
        results = []
        for rule in self.rules:
            results.append((rule.confidence.detach(), rule(state).detach()))
        return results

    def get_triggered_rules(
        self, state: torch.Tensor, threshold: float = 0.5
    ) -> List[int]:
        """Return indices of rules with satisfaction > threshold."""
        triggered = []
        for idx, rule in enumerate(self.rules):
            mu = rule(state).mean().item()
            if mu > threshold:
                triggered.append(idx)
        return triggered

    # ------------------------------------------------------------------
    # Confidence learning (Section 4.3)
    # ------------------------------------------------------------------

    def confidence_parameters(self):
        """Return only the confidence logit parameters."""
        return [rule._confidence_logit for rule in self.rules]

    def predicate_parameters(self):
        """Return predicate network parameters (not confidence)."""
        params = []
        for rule in self.rules:
            params.extend(rule.predicate.parameters())
        params.append(self.threshold)
        return params

    # ------------------------------------------------------------------
    # Rule mining (Section 4.3.1)
    # ------------------------------------------------------------------

    def mine_rule(
        self,
        history_states: torch.Tensor,
        history_compliant: torch.Tensor,
    ) -> bool:
        """Mine a new rule from interaction history using simplified ILP.

        Uses a simple linear discriminant to separate compliant vs
        non-compliant states, then adds as a new differentiable rule.

        Returns True if a rule was added.
        """
        if self.mined_count >= self.max_mined_rules:
            return False
        if len(history_states) < 50:
            return False

        # Fit a simple linear separator (simplified ILP)
        pos_mask = history_compliant > 0.5
        neg_mask = ~pos_mask
        if pos_mask.sum() < 5 or neg_mask.sum() < 5:
            return False

        pos_mean = history_states[pos_mask].mean(dim=0)
        neg_mean = history_states[neg_mask].mean(dim=0)
        direction = pos_mean - neg_mean
        direction = direction / (direction.norm() + 1e-8)

        # Create a new rule initialized from the discovered direction
        new_rule = LogicalTensorRule(self.state_dim)
        with torch.no_grad():
            new_rule.predicate[0].weight.data[0] = direction[: new_rule.predicate[0].weight.shape[1]]
            new_rule._confidence_logit.data.fill_(0.0)  # init confidence = 0.5

        self.rules.append(new_rule)
        self.mined_count += 1
        return True


class RuleMiner:
    """Manages the periodic rule mining process (Section 4.3.1).

    Collects interaction history and triggers mining at configurable
    intervals. Supports the coordinator signaling a mine request.
    """

    def __init__(
        self,
        mine_interval: int = 50,
        min_samples: int = 100,
    ) -> None:
        self.mine_interval = mine_interval
        self.min_samples = min_samples
        self._states: List[torch.Tensor] = []
        self._compliant: List[float] = []
        self.episode_count = 0

    def record(self, state: torch.Tensor, compliant: float) -> None:
        self._states.append(state.detach())
        self._compliant.append(compliant)

    def should_mine(self, force: bool = False) -> bool:
        self.episode_count += 1
        if force:
            return len(self._states) >= self.min_samples
        return (
            self.episode_count % self.mine_interval == 0
            and len(self._states) >= self.min_samples
        )

    def get_mining_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        states = torch.stack(self._states)
        compliant = torch.tensor(self._compliant, dtype=torch.float32)
        return states, compliant

    def clear(self) -> None:
        self._states.clear()
        self._compliant.clear()
