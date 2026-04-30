"""Multi-Tier Reward Design for Adversarial-Collaborative Debate RL.

.. deprecated::
    Use debate_rl_v2.scenarios.debate.reward.DebateRewardComputer instead.
    This module is retained for backward compatibility with RL pre-training code.

Implements the hierarchical reward structure:

  Layer 1: **Global Task Reward** — Terminal quality of final proposal.
  Layer 2: **Process Rewards** (per-step, per-role):
    - Divergence reduction reward
    - Compliance reward
    - Constructive adversarial reward
    - Time penalty
  Layer 3: **Meta Rewards** (coordinator, cross-episode):
    - Average quality over debate
    - Convergence speed
    - Rule adaptability

Role-Specific Reward Components:
  - **Proposer**: Scheme generation quality + modification effectiveness
      + arbiter acceptance signal + compliance alignment
  - **Challenger**: Information gain + belief change induction
      + constructive vs. destructive distinction + novelty bonus
  - **Arbiter**: Compliance judgment accuracy + rule confidence calibration
      + consistency bonus + threshold optimization
  - **Coordinator**: Convergence efficiency + quality maintenance
      + rule mining benefit + termination timing

Enhanced v3 Features:
  - **Dense Quality Scoring**: Per-dimension quality decomposition
  - **Marginal Return Detection**: Coordinator early-stop signal
  - **Running Reward Normalization**: Cross-episode Z-score normalization
  - **Curiosity Bonus**: Exploration reward for novel debate states
  - **Opponent Modeling Reward**: Prediction accuracy about other agents
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────

@dataclass
class EnhancedRewardConfig:
    """All tunable weights for the multi-tier reward design."""

    # ---- Layer 1: Global Task ----
    task_weight: float = 2.0
    consensus_bonus: float = 3.0
    no_consensus_penalty: float = -1.0

    # ---- Layer 2: Process Rewards ----
    # Divergence reduction
    divergence_weight: float = 0.25
    # Compliance
    compliance_weight: float = 0.20
    # Constructive adversarial
    constructive_adv_weight: float = 0.20
    # Time penalty (per step)
    step_penalty: float = 0.01

    # ---- Layer 3: Meta Rewards (Coordinator) ----
    meta_avg_quality_weight: float = 0.30
    meta_convergence_speed_weight: float = 0.25
    meta_rule_adaptability_weight: float = 0.20
    meta_efficiency_weight: float = 0.15

    # ---- Proposer Specific ----
    proposer_quality_improve_weight: float = 0.30
    proposer_modification_effectiveness: float = 0.20
    proposer_acceptance_signal_weight: float = 0.15
    proposer_compliance_alignment: float = 0.10

    # ---- Challenger Specific ----
    challenger_info_gain_weight: float = 0.25
    challenger_belief_change_weight: float = 0.20
    challenger_constructiveness_bonus: float = 0.15
    challenger_novelty_weight: float = 0.10
    challenger_destructive_penalty: float = -0.20

    # ---- Arbiter Specific ----
    arbiter_judgment_accuracy_weight: float = 0.25
    arbiter_rule_calibration_weight: float = 0.15
    arbiter_consistency_weight: float = 0.10
    arbiter_threshold_opt_weight: float = 0.10

    # ---- Coordinator Specific ----
    coordinator_convergence_weight: float = 0.25
    coordinator_quality_maintenance: float = 0.20
    coordinator_mining_benefit: float = 0.15
    coordinator_termination_timing: float = 0.10

    # ---- v3 Enhanced Features ----
    # Dense quality scoring
    dense_quality_enabled: bool = True
    dense_quality_dimensions: int = 5  # logic, feasibility, innovation, evidence, compliance
    dense_quality_weights: Tuple[float, ...] = (0.25, 0.20, 0.15, 0.20, 0.20)
    # Marginal return detection
    marginal_return_window: int = 3
    marginal_return_threshold: float = 0.02  # quality improvement < this triggers signal
    marginal_return_bonus: float = 0.3  # bonus for timely termination
    # Reward normalization
    reward_normalization: bool = True
    norm_window: int = 100  # running statistics window
    norm_clip: float = 5.0
    # Curiosity bonus
    curiosity_enabled: bool = True
    curiosity_weight: float = 0.05
    curiosity_decay: float = 0.995  # decay visit counts over episodes
    state_discretization_bins: int = 10
    # Opponent modeling reward
    opponent_modeling_enabled: bool = True
    opponent_modeling_weight: float = 0.08


# ──────────────────────────────────────────────────────────
# State Tracker
# ──────────────────────────────────────────────────────────

@dataclass
class DebateMetrics:
    """Tracks all debate metrics needed for reward computation.

    Accumulated over a full episode to support both per-step and
    terminal reward computation.
    """
    # Per-round snapshots
    quality_history: List[float] = field(default_factory=list)
    disagreement_history: List[float] = field(default_factory=list)
    compliance_history: List[float] = field(default_factory=list)
    lambda_adv_history: List[float] = field(default_factory=list)

    # Proposer
    proposal_scores: List[float] = field(default_factory=list)
    proposal_accepted: List[bool] = field(default_factory=list)

    # Challenger
    belief_changes: List[float] = field(default_factory=list)
    info_gains: List[float] = field(default_factory=list)
    challenge_constructiveness: List[float] = field(default_factory=list)
    challenge_novelty: List[float] = field(default_factory=list)

    # Arbiter
    arbiter_score_deltas: List[float] = field(default_factory=list)
    rule_confidence_changes: List[float] = field(default_factory=list)
    rules_mined: int = 0

    # Coordinator
    mode_switches: int = 0
    mining_triggered: int = 0
    param_adjustments: List[float] = field(default_factory=list)

    def record_round(
        self,
        quality: float,
        disagreement: float,
        compliance: float,
        lambda_adv: float,
        proposal_score: float = 0.5,
        challenge_score: float = 0.5,
        prop_confidence: float = 0.5,
        chal_confidence: float = 0.5,
        arbiter_accepted: bool = False,
        constructiveness: float = 0.5,
        novelty: float = 0.5,
        rule_conf_change: float = 0.0,
        mined_rule: bool = False,
        mode_switched: bool = False,
        param_delta: float = 0.0,
    ) -> None:
        """Record one round of metrics."""
        self.quality_history.append(quality)
        self.disagreement_history.append(disagreement)
        self.compliance_history.append(compliance)
        self.lambda_adv_history.append(lambda_adv)

        self.proposal_scores.append(proposal_score)
        self.proposal_accepted.append(arbiter_accepted)

        # Belief change: how much did disagreement shift?
        if len(self.disagreement_history) >= 2:
            self.belief_changes.append(
                abs(self.disagreement_history[-1] - self.disagreement_history[-2])
            )
        else:
            self.belief_changes.append(0.0)

        # Information gain proxy
        if len(self.quality_history) >= 2:
            self.info_gains.append(
                max(0.0, self.quality_history[-1] - self.quality_history[-2])
            )
        else:
            self.info_gains.append(0.0)

        self.challenge_constructiveness.append(constructiveness)
        self.challenge_novelty.append(novelty)

        # Arbiter
        if len(self.quality_history) >= 2:
            self.arbiter_score_deltas.append(
                self.quality_history[-1] - self.quality_history[-2]
            )
        else:
            self.arbiter_score_deltas.append(0.0)
        self.rule_confidence_changes.append(rule_conf_change)
        if mined_rule:
            self.rules_mined += 1

        if mode_switched:
            self.mode_switches += 1
        if mined_rule:
            self.mining_triggered += 1
        self.param_adjustments.append(param_delta)

    @property
    def avg_quality(self) -> float:
        return float(np.mean(self.quality_history)) if self.quality_history else 0.5

    @property
    def final_quality(self) -> float:
        return self.quality_history[-1] if self.quality_history else 0.5

    @property
    def convergence_speed(self) -> float:
        """How quickly quality converged (higher = faster).

        Measures the area under the quality curve, normalized.
        A debate that reaches high quality early gets a higher score.
        """
        if len(self.quality_history) < 2:
            return 0.0
        total = len(self.quality_history)
        area = sum(q for q in self.quality_history) / total
        return float(area)

    @property
    def rule_adaptability(self) -> float:
        """How well the system adapted rules.

        Based on compliance improvement + successful rule mining.
        """
        if len(self.compliance_history) < 2:
            return 0.0
        comp_improve = max(0.0, self.compliance_history[-1] - self.compliance_history[0])
        mining_score = min(1.0, self.rules_mined * 0.3)
        return float(comp_improve + mining_score)

    def reset(self) -> None:
        self.quality_history.clear()
        self.disagreement_history.clear()
        self.compliance_history.clear()
        self.lambda_adv_history.clear()
        self.proposal_scores.clear()
        self.proposal_accepted.clear()
        self.belief_changes.clear()
        self.info_gains.clear()
        self.challenge_constructiveness.clear()
        self.challenge_novelty.clear()
        self.arbiter_score_deltas.clear()
        self.rule_confidence_changes.clear()
        self.rules_mined = 0
        self.mode_switches = 0
        self.mining_triggered = 0
        self.param_adjustments.clear()


# ──────────────────────────────────────────────────────────
# v3: Dense Quality Scorer
# ──────────────────────────────────────────────────────────

class DenseQualityScorer:
    """Decompose quality into multiple dimensions for finer-grained reward.

    Instead of a single quality scalar, evaluates proposals across
    5 orthogonal dimensions:
      0. Logic coherence   — internal consistency of reasoning
      1. Feasibility       — practical implementability
      2. Innovation        — novelty of the proposed solution
      3. Evidence strength — data/fact backing
      4. Rule compliance   — adherence to constraints

    Each dimension yields a separate reward signal, allowing agents
    to understand *which aspect* of quality they improved or harmed.
    """

    DIMENSION_NAMES = ["logic", "feasibility", "innovation", "evidence", "compliance"]

    def __init__(self, weights: Tuple[float, ...] = (0.25, 0.20, 0.15, 0.20, 0.20)):
        self.weights = np.array(weights, dtype=np.float32)
        self.weights /= self.weights.sum()  # normalize

    def score(self, state: Dict[str, float]) -> np.ndarray:
        """Extract multi-dimensional quality vector from debate state.

        Parameters
        ----------
        state : dict
            Debate state with quality, compliance, disagreement, etc.

        Returns
        -------
        scores : ndarray (5,)
            Per-dimension quality scores in [0, 1].
        """
        q = state.get("quality", 0.5)
        c = state.get("compliance", 0.5)
        d = state.get("disagreement", 0.5)
        prop_conf = state.get("prop_confidence", 0.5)
        chal_conf = state.get("chal_confidence", 0.5)
        novelty = state.get("novelty", 0.5)

        # Heuristic decomposition from available signals
        logic = np.clip(q * 0.7 + (1 - d) * 0.3, 0, 1)
        feasibility = np.clip(c * 0.5 + q * 0.3 + prop_conf * 0.2, 0, 1)
        innovation = np.clip(novelty * 0.6 + (1 - prop_conf) * 0.2 + q * 0.2, 0, 1)
        evidence = np.clip(prop_conf * 0.4 + chal_conf * 0.3 + c * 0.3, 0, 1)
        compliance_score = np.clip(c, 0, 1)

        return np.array([logic, feasibility, innovation, evidence, compliance_score],
                        dtype=np.float32)

    def dense_reward(
        self,
        prev_state: Dict[str, float],
        curr_state: Dict[str, float],
    ) -> Tuple[float, Dict[str, float]]:
        """Compute weighted dense quality reward from state transition.

        Returns
        -------
        total : float
            Weighted sum of per-dimension improvements.
        breakdown : dict
            Per-dimension reward values.
        """
        prev_scores = self.score(prev_state)
        curr_scores = self.score(curr_state)
        deltas = curr_scores - prev_scores

        weighted = deltas * self.weights
        breakdown = {
            name: float(weighted[i])
            for i, name in enumerate(self.DIMENSION_NAMES)
        }
        return float(weighted.sum()), breakdown


# ──────────────────────────────────────────────────────────
# v3: Marginal Return Detector
# ──────────────────────────────────────────────────────────

class MarginalReturnDetector:
    """Detect diminishing returns in debate quality.

    Tracks quality improvement over a sliding window. When the average
    improvement drops below a threshold, signals the coordinator that
    continuing debate has marginal benefit — rewarding timely termination.
    """

    def __init__(self, window: int = 3, threshold: float = 0.02):
        self.window = window
        self.threshold = threshold
        self._history: List[float] = []

    def update(self, quality: float) -> bool:
        """Record quality and return whether marginal return is detected.

        Returns
        -------
        diminishing : bool
            True if recent quality improvement is below threshold.
        """
        self._history.append(quality)
        if len(self._history) < self.window + 1:
            return False
        recent = self._history[-(self.window + 1):]
        improvements = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
        avg_improvement = np.mean(improvements)
        return float(avg_improvement) < self.threshold

    def reset(self) -> None:
        self._history.clear()

    @property
    def avg_recent_improvement(self) -> float:
        if len(self._history) < 2:
            return 0.0
        recent = self._history[-min(self.window + 1, len(self._history)):]
        improvements = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
        return float(np.mean(improvements))


# ──────────────────────────────────────────────────────────
# v3: Running Reward Normalizer
# ──────────────────────────────────────────────────────────

class RewardNormalizer:
    """Running Z-score normalizer for cross-episode reward stability.

    Maintains running mean and std of rewards per agent, and normalizes
    rewards to zero mean, unit variance. This prevents reward magnitude
    drift across training and helps with gradient stability.
    """

    def __init__(self, agents: List[str], window: int = 100, clip: float = 5.0):
        self.clip = clip
        self._means: Dict[str, deque] = {a: deque(maxlen=window) for a in agents}
        self._vars: Dict[str, deque] = {a: deque(maxlen=window) for a in agents}

    def normalize(self, rewards: Dict[str, float]) -> Dict[str, float]:
        """Normalize per-agent rewards using running statistics."""
        normalized = {}
        for agent, r in rewards.items():
            if agent not in self._means:
                self._means[agent] = deque(maxlen=100)
                self._vars[agent] = deque(maxlen=100)

            self._means[agent].append(r)
            mean = np.mean(self._means[agent])
            std = max(np.std(self._means[agent]), 1e-8)
            normed = (r - mean) / std
            normalized[agent] = float(np.clip(normed, -self.clip, self.clip))
        return normalized


# ──────────────────────────────────────────────────────────
# v3: Curiosity Bonus (State Visitation)
# ──────────────────────────────────────────────────────────

class CuriosityBonus:
    """Exploration reward based on state visitation counts.

    Discretizes the continuous debate state space into bins and tracks
    visit counts. Rarely-visited states receive a curiosity bonus,
    encouraging the RL agents to explore diverse debate strategies.
    """

    def __init__(self, bins: int = 10, decay: float = 0.995):
        self.bins = bins
        self.decay = decay
        self._visit_counts: Dict[Tuple[int, ...], int] = {}

    def _discretize(self, state: Dict[str, float]) -> Tuple[int, ...]:
        """Map continuous state to discrete bin tuple."""
        keys = ["quality", "disagreement", "compliance", "prop_confidence"]
        return tuple(
            min(self.bins - 1, int(state.get(k, 0.5) * self.bins))
            for k in keys
        )

    def compute(self, state: Dict[str, float]) -> float:
        """Compute curiosity bonus for a given state.

        Returns a reward inversely proportional to sqrt(visit_count).
        """
        key = self._discretize(state)
        count = self._visit_counts.get(key, 0) + 1
        self._visit_counts[key] = count
        return 1.0 / math.sqrt(count)

    def decay_counts(self) -> None:
        """Apply decay to all visit counts (call between episodes)."""
        for key in self._visit_counts:
            self._visit_counts[key] = max(1, int(self._visit_counts[key] * self.decay))

    def reset(self) -> None:
        self._visit_counts.clear()


# ──────────────────────────────────────────────────────────
# v3: Opponent Modeling Reward
# ──────────────────────────────────────────────────────────

class OpponentModelReward:
    """Reward agents for accurately predicting other agents' behavior.

    Each agent maintains a simple prediction of the other agents'
    next-round confidence/score. Accurate predictions earn a bonus,
    encouraging agents to build internal models of their counterparts.
    """

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha  # EMA smoothing for predictions
        self._predictions: Dict[str, Dict[str, float]] = {}
        self._actuals: Dict[str, Dict[str, float]] = {}

    def predict(self, agent: str, targets: Dict[str, float]) -> None:
        """Record agent's prediction about other agents.

        Parameters
        ----------
        agent : str
            The predicting agent name.
        targets : dict
            {other_agent: predicted_value} predictions.
        """
        self._predictions[agent] = dict(targets)

    def observe(self, agent: str, actuals: Dict[str, float]) -> None:
        """Record actual observed values for other agents."""
        self._actuals[agent] = dict(actuals)

    def compute_reward(self, agent: str) -> float:
        """Compute prediction accuracy reward for an agent.

        Returns
        -------
        reward : float
            Higher when predictions were accurate.
        """
        preds = self._predictions.get(agent, {})
        acts = self._actuals.get(agent, {})
        if not preds or not acts:
            return 0.0

        errors = []
        for target, pred_val in preds.items():
            actual_val = acts.get(target, pred_val)
            errors.append(abs(pred_val - actual_val))

        avg_error = np.mean(errors) if errors else 0.5
        # Reward = 1 - avg_error (higher accuracy = higher reward)
        return float(max(0.0, 1.0 - avg_error * 2.0))

    def auto_predict(self, state: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Generate simple EMA-based predictions for all agent pairs.

        Uses exponential moving average of historical observations
        as a baseline predictor. Returns predictions keyed by agent.
        """
        prop_conf = state.get("prop_confidence", 0.5)
        chal_conf = state.get("chal_confidence", 0.5)
        quality = state.get("quality", 0.5)

        # Simple heuristic predictions
        predictions = {
            "proposer_ctrl": {
                "challenger_confidence": chal_conf,
                "arbiter_quality": quality,
            },
            "challenger_ctrl": {
                "proposer_confidence": prop_conf,
                "arbiter_quality": quality,
            },
            "arbiter_ctrl": {
                "proposer_confidence": prop_conf,
                "challenger_confidence": chal_conf,
            },
            "coordinator": {
                "proposer_confidence": prop_conf,
                "challenger_confidence": chal_conf,
            },
        }

        # Record predictions
        for agent, targets in predictions.items():
            self.predict(agent, targets)

        return predictions

    def auto_observe(self, state: Dict[str, float]) -> None:
        """Record actual observations for all agents."""
        prop_conf = state.get("prop_confidence", 0.5)
        chal_conf = state.get("chal_confidence", 0.5)
        quality = state.get("quality", 0.5)

        for agent in ["proposer_ctrl", "challenger_ctrl", "arbiter_ctrl", "coordinator"]:
            self.observe(agent, {
                "proposer_confidence": prop_conf,
                "challenger_confidence": chal_conf,
                "arbiter_quality": quality,
            })

    def reset(self) -> None:
        self._predictions.clear()
        self._actuals.clear()


# ──────────────────────────────────────────────────────────
# Core Reward Computer
# ──────────────────────────────────────────────────────────

class EnhancedRewardComputer:
    """Multi-tier reward computation engine.

    Computes per-agent rewards based on:
      - Per-step process signals (quality delta, compliance, etc.)
      - Role-specific objectives
      - Terminal task quality
      - Cross-round meta objectives (coordinator)

    Usage::

        computer = EnhancedRewardComputer(cfg)
        # Each round:
        metrics.record_round(...)
        rewards = computer.compute_step_rewards(prev, curr, metrics, done, consensus)
        # Terminal:
        meta = computer.compute_meta_rewards(metrics)
    """

    def __init__(self, cfg: Optional[EnhancedRewardConfig] = None) -> None:
        self.cfg = cfg or EnhancedRewardConfig()

        # v3: Initialize enhanced components
        self.dense_scorer = DenseQualityScorer(
            weights=self.cfg.dense_quality_weights
        ) if self.cfg.dense_quality_enabled else None

        self.marginal_detector = MarginalReturnDetector(
            window=self.cfg.marginal_return_window,
            threshold=self.cfg.marginal_return_threshold,
        )

        self.curiosity = CuriosityBonus(
            bins=self.cfg.state_discretization_bins,
            decay=self.cfg.curiosity_decay,
        ) if self.cfg.curiosity_enabled else None

        self.opponent_model = OpponentModelReward(
        ) if self.cfg.opponent_modeling_enabled else None

        agent_names = ["proposer_ctrl", "challenger_ctrl", "arbiter_ctrl", "coordinator"]
        self.normalizer = RewardNormalizer(
            agents=agent_names,
            window=self.cfg.norm_window,
            clip=self.cfg.norm_clip,
        ) if self.cfg.reward_normalization else None

    # ------------------------------------------------------------------
    # Per-step rewards
    # ------------------------------------------------------------------

    def compute_step_rewards(
        self,
        prev_state: Dict[str, float],
        curr_state: Dict[str, float],
        metrics: DebateMetrics,
        done: bool,
        consensus_reached: bool,
    ) -> Dict[str, float]:
        """Compute multi-tier per-agent step rewards.

        Parameters
        ----------
        prev_state : dict
            Previous round: {quality, disagreement, compliance, ...}
        curr_state : dict
            Current round: {quality, disagreement, compliance,
                            proposal_score, challenge_score,
                            prop_confidence, chal_confidence,
                            constructiveness, novelty, ...}
        metrics : DebateMetrics
            Accumulated debate metrics.
        done : bool
            Whether episode is ending.
        consensus_reached : bool
            Whether quality consensus was achieved.

        Returns
        -------
        rewards : dict
            {role: float} for proposer_ctrl, challenger_ctrl,
            arbiter_ctrl, coordinator.
        """
        c = self.cfg

        # ── Extract deltas ──
        dq = curr_state.get("quality", 0.5) - prev_state.get("quality", 0.5)
        dd = prev_state.get("disagreement", 0.5) - curr_state.get("disagreement", 0.5)  # positive = decreased
        dc = curr_state.get("compliance", 0.5) - prev_state.get("compliance", 0.5)

        quality = curr_state.get("quality", 0.5)
        compliance = curr_state.get("compliance", 0.5)
        disagreement = curr_state.get("disagreement", 0.5)
        prop_score = curr_state.get("proposal_score", 0.5)
        chal_score = curr_state.get("challenge_score", 0.5)
        prop_conf = curr_state.get("prop_confidence", 0.5)
        chal_conf = curr_state.get("chal_confidence", 0.5)
        constructiveness = curr_state.get("constructiveness", 0.5)
        novelty = curr_state.get("novelty", 0.5)

        # ── Layer 2: Shared process rewards ──
        r_divergence = c.divergence_weight * dd
        r_compliance = c.compliance_weight * dc
        r_step_penalty = -c.step_penalty

        shared_process = r_divergence + r_compliance + r_step_penalty

        # ── Layer 1: Terminal task reward ──
        terminal_bonus = 0.0
        if done:
            if consensus_reached:
                terminal_bonus = c.consensus_bonus * quality
            else:
                terminal_bonus = c.no_consensus_penalty

        # ── Proposer reward ──
        r_proposer = self._proposer_reward(
            dq=dq, dc=dc, quality=quality, prop_score=prop_score,
            compliance=compliance, metrics=metrics,
            shared=shared_process, terminal=terminal_bonus,
        )

        # ── Challenger reward ──
        r_challenger = self._challenger_reward(
            dq=dq, dd=dd, disagreement=disagreement,
            chal_score=chal_score, constructiveness=constructiveness,
            novelty=novelty, metrics=metrics,
            shared=shared_process, terminal=terminal_bonus,
        )

        # ── Arbiter reward ──
        r_arbiter = self._arbiter_reward(
            dq=dq, dc=dc, quality=quality, compliance=compliance,
            metrics=metrics,
            shared=shared_process, terminal=terminal_bonus,
        )

        # ── Coordinator reward ──
        r_coordinator = self._coordinator_reward(
            dq=dq, dd=dd, dc=dc, quality=quality,
            metrics=metrics, done=done,
            shared=shared_process, terminal=terminal_bonus,
        )

        return {
            "proposer_ctrl": float(r_proposer),
            "challenger_ctrl": float(r_challenger),
            "arbiter_ctrl": float(r_arbiter),
            "coordinator": float(r_coordinator),
        }

    # ------------------------------------------------------------------
    # v3: Enhanced step rewards with dense scoring & curiosity
    # ------------------------------------------------------------------

    def compute_enhanced_step_rewards(
        self,
        prev_state: Dict[str, float],
        curr_state: Dict[str, float],
        metrics: DebateMetrics,
        done: bool,
        consensus_reached: bool,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Compute multi-tier rewards with v3 enhancements.

        Returns both rewards and detailed diagnostics.

        Returns
        -------
        rewards : dict
            Per-agent rewards (optionally normalized).
        info : dict
            Diagnostic info: dense_breakdown, marginal_return, curiosity, etc.
        """
        # Base rewards
        rewards = self.compute_step_rewards(
            prev_state, curr_state, metrics, done, consensus_reached,
        )

        info: Dict[str, Any] = {}

        # Dense quality scoring
        if self.dense_scorer is not None:
            dense_total, dense_breakdown = self.dense_scorer.dense_reward(
                prev_state, curr_state,
            )
            # Add dense quality bonus to proposer and arbiter
            rewards["proposer_ctrl"] += dense_total * 0.5
            rewards["arbiter_ctrl"] += dense_total * 0.3
            info["dense_breakdown"] = dense_breakdown
            info["dense_total"] = dense_total

        # Marginal return detection for coordinator
        quality = curr_state.get("quality", 0.5)
        diminishing = self.marginal_detector.update(quality)
        info["marginal_return_detected"] = diminishing
        info["avg_recent_improvement"] = self.marginal_detector.avg_recent_improvement
        if diminishing and done and consensus_reached:
            rewards["coordinator"] += self.cfg.marginal_return_bonus

        # Curiosity bonus
        if self.curiosity is not None:
            c_bonus = self.curiosity.compute(curr_state)
            scaled_bonus = self.cfg.curiosity_weight * c_bonus
            for agent in rewards:
                rewards[agent] += scaled_bonus
            info["curiosity_bonus"] = scaled_bonus

        # Opponent modeling reward
        if self.opponent_model is not None:
            self.opponent_model.auto_observe(curr_state)
            for agent in rewards:
                om_reward = self.opponent_model.compute_reward(agent)
                rewards[agent] += self.cfg.opponent_modeling_weight * om_reward
            self.opponent_model.auto_predict(curr_state)
            info["opponent_model_active"] = True

        # Reward normalization
        if self.normalizer is not None:
            rewards = self.normalizer.normalize(rewards)
            info["normalized"] = True

        return rewards, info

    def episode_reset(self) -> None:
        """Reset per-episode state for v3 components."""
        self.marginal_detector.reset()
        if self.curiosity is not None:
            self.curiosity.decay_counts()
        if self.opponent_model is not None:
            self.opponent_model.reset()

    # ------------------------------------------------------------------
    # Role-specific rewards
    # ------------------------------------------------------------------

    def _proposer_reward(
        self, dq: float, dc: float, quality: float, prop_score: float,
        compliance: float, metrics: DebateMetrics,
        shared: float, terminal: float,
    ) -> float:
        """Proposer: generate high-quality proposals that get accepted.

        Objectives:
          1. **Quality improvement** (dq) — proposal gets better over rounds.
          2. **Modification effectiveness** — changes driven by challenge
             feedback lead to quality gains.
          3. **Arbiter acceptance** — high proposal_score indicates
             the arbiter approves the proposal.
          4. **Compliance alignment** — proposal follows rules.
        """
        c = self.cfg

        # 1. Quality improvement
        r_quality = c.proposer_quality_improve_weight * dq

        # 2. Modification effectiveness
        # If quality improved after incorporating feedback, reward.
        # Use correlation between compliance change and quality change.
        if len(metrics.quality_history) >= 2:
            prev_q = metrics.quality_history[-2] if len(metrics.quality_history) >= 2 else 0.5
            curr_q = metrics.quality_history[-1] if metrics.quality_history else 0.5
            modification_eff = max(0.0, curr_q - prev_q)
        else:
            modification_eff = 0.0
        r_modification = c.proposer_modification_effectiveness * modification_eff

        # 3. Arbiter acceptance signal
        # prop_score > 0.6 means arbiter approves; above 0.8 strong approval
        acceptance_signal = max(0.0, prop_score - 0.5) * 2.0  # [0, 1] range
        r_acceptance = c.proposer_acceptance_signal_weight * acceptance_signal

        # 4. Compliance alignment
        r_compliance = c.proposer_compliance_alignment * compliance

        return shared + r_quality + r_modification + r_acceptance + r_compliance + terminal

    def _challenger_reward(
        self, dq: float, dd: float, disagreement: float,
        chal_score: float, constructiveness: float, novelty: float,
        metrics: DebateMetrics, shared: float, terminal: float,
    ) -> float:
        """Challenger: produce informative, constructive challenges.

        Objectives:
          1. **Information gain** — challenge causes quality improvement
             (indicates the challenge was useful, not destructive).
          2. **Belief change induction** — disagreement shifts after challenge,
             showing the challenge was substantive.
          3. **Constructiveness bonus** — higher when challenge provides
             alternatives, not just criticism.
          4. **Novelty** — new challenge angles are rewarded.
          5. **Destructive penalty** — quality drops without reason → penalty.
        """
        c = self.cfg

        # 1. Information gain: quality improved after challenge
        info_gain = max(0.0, dq)  # positive quality change = useful challenge
        r_info = c.challenger_info_gain_weight * info_gain

        # 2. Belief change: disagreement shifted (in either direction)
        belief_change = abs(dd) if len(metrics.belief_changes) < 2 else metrics.belief_changes[-1]
        r_belief = c.challenger_belief_change_weight * belief_change

        # 3. Constructive vs destructive
        # Constructive: challenge_score high AND quality didn't drop significantly
        if constructiveness > 0.6 and dq >= -0.05:
            r_constructive = c.challenger_constructiveness_bonus * constructiveness
        elif dq < -0.1:
            # Quality dropped significantly → destructive challenge
            r_constructive = c.challenger_destructive_penalty * abs(dq)
        else:
            r_constructive = 0.0

        # 4. Novelty bonus
        r_novelty = c.challenger_novelty_weight * max(0.0, novelty - 0.5) * 2.0

        # 5. Smart adversarial: reward calibrated disagreement
        # Not too high (destructive) and not too low (useless)
        target_disagreement = 0.3  # sweet spot
        disagreement_quality = 1.0 - abs(disagreement - target_disagreement) / 0.5
        r_disagree_quality = 0.05 * max(0.0, disagreement_quality)

        return (shared + r_info + r_belief + r_constructive
                + r_novelty + r_disagree_quality + terminal)

    def _arbiter_reward(
        self, dq: float, dc: float, quality: float, compliance: float,
        metrics: DebateMetrics, shared: float, terminal: float,
    ) -> float:
        """Arbiter: accurate judgment + rule learning.

        Two levels:
          **Action level** — Compliance judgment accuracy:
            - Reward for correct compliance assessments.
            - Consistency in scoring (low variance over stable states).

          **Knowledge level** — Gradient-based rule confidence adjustment:
            - Rule confidence changes that correlate with actual compliance.
            - Triggering appropriate rule mining.
            - Threshold optimization.
        """
        c = self.cfg

        # === Action Level ===

        # 1. Judgment accuracy: quality improvement suggests good evaluation
        r_judgment = c.arbiter_judgment_accuracy_weight * dq

        # 2. Consistency: low variance in recent scores = stable judgment
        if len(metrics.quality_history) >= 3:
            recent_q = metrics.quality_history[-3:]
            consistency = 1.0 - float(np.std(recent_q)) * 2.0
            r_consistency = c.arbiter_consistency_weight * max(0.0, consistency)
        else:
            r_consistency = 0.0

        # === Knowledge Level ===

        # 3. Rule calibration: compliance improvement tracks rule adjustments
        if metrics.rule_confidence_changes:
            recent_conf_change = metrics.rule_confidence_changes[-1]
            # Positive compliance delta + positive rule confidence change = good calibration
            calibration_reward = dc * recent_conf_change
            r_calibration = c.arbiter_rule_calibration_weight * np.tanh(calibration_reward * 5.0)
        else:
            r_calibration = 0.0

        # 4. Threshold optimization: compliance near target range [0.7, 0.9]
        target_compliance = 0.8
        compliance_quality = 1.0 - abs(compliance - target_compliance) / 0.3
        r_threshold = c.arbiter_threshold_opt_weight * max(0.0, compliance_quality)

        return shared + r_judgment + r_consistency + r_calibration + r_threshold + terminal

    def _coordinator_reward(
        self, dq: float, dd: float, dc: float, quality: float,
        metrics: DebateMetrics, done: bool,
        shared: float, terminal: float,
    ) -> float:
        """Coordinator: meta-level strategy optimization.

        Objectives:
          1. **Convergence efficiency** — quality converges quickly.
          2. **Quality maintenance** — quality stays high while exploring.
          3. **Rule mining benefit** — mined rules improve compliance.
          4. **Termination timing** — optimal debate length.
        """
        c = self.cfg

        # Use half shared reward (coordinator operates at meta level)
        meta_shared = 0.5 * shared

        # 1. Convergence efficiency: quality improvement rate
        r_convergence = c.coordinator_convergence_weight * dq

        # 2. Quality maintenance: penalize quality drops
        if dq < -0.05:
            r_maintenance = c.coordinator_quality_maintenance * dq  # negative
        else:
            r_maintenance = c.coordinator_quality_maintenance * 0.05 * quality

        # 3. Mining benefit: if rules were recently mined & compliance improved
        mining_benefit = 0.0
        if metrics.mining_triggered > 0 and dc > 0:
            mining_benefit = dc
        r_mining = c.coordinator_mining_benefit * mining_benefit

        # 4. Termination timing
        r_termination = 0.0
        if done:
            total_rounds = len(metrics.quality_history)
            max_possible = max(total_rounds, 1)
            # Reward early consensus with high quality
            if quality >= 0.8 and total_rounds < max_possible * 0.7:
                r_termination = c.coordinator_termination_timing * (1.0 - total_rounds / max_possible)
            # Penalize running to max without quality
            elif quality < 0.6:
                r_termination = -c.coordinator_termination_timing * 0.5

        return meta_shared + r_convergence + r_maintenance + r_mining + r_termination + terminal

    # ------------------------------------------------------------------
    # Meta rewards (cross-episode evaluation)
    # ------------------------------------------------------------------

    def compute_meta_rewards(
        self,
        metrics: DebateMetrics,
        max_rounds: int = 10,
    ) -> Dict[str, float]:
        """Compute meta-level rewards for coordinator (post-episode).

        These rewards evaluate the overall debate quality and can be used
        as additional training signal or for coordinator evaluation.

        Returns
        -------
        meta : dict
            {avg_quality, convergence_speed, rule_adaptability, efficiency}
        """
        c = self.cfg

        # Average quality across all rounds
        avg_q = metrics.avg_quality
        r_avg_quality = c.meta_avg_quality_weight * avg_q

        # Convergence speed (area under quality curve)
        conv_speed = metrics.convergence_speed
        r_conv_speed = c.meta_convergence_speed_weight * conv_speed

        # Rule adaptability
        rule_adapt = metrics.rule_adaptability
        r_rule_adapt = c.meta_rule_adaptability_weight * rule_adapt

        # Efficiency: quality per round (fewer rounds for same quality = better)
        total_rounds = len(metrics.quality_history)
        final_q = metrics.final_quality
        if total_rounds > 0:
            efficiency = final_q / (total_rounds / max_rounds)
            efficiency = min(1.0, efficiency)  # cap at 1
        else:
            efficiency = 0.0
        r_efficiency = c.meta_efficiency_weight * efficiency

        meta_total = r_avg_quality + r_conv_speed + r_rule_adapt + r_efficiency

        return {
            "meta_total": float(meta_total),
            "meta_avg_quality": float(r_avg_quality),
            "meta_convergence_speed": float(r_conv_speed),
            "meta_rule_adaptability": float(r_rule_adapt),
            "meta_efficiency": float(r_efficiency),
            "raw_avg_quality": float(avg_q),
            "raw_convergence_speed": float(conv_speed),
            "raw_rule_adaptability": float(rule_adapt),
            "raw_efficiency": float(efficiency),
        }

    # ------------------------------------------------------------------
    # Numerical env rewards (for pre-training)
    # ------------------------------------------------------------------

    def compute_numerical_rewards(
        self,
        env_rewards: Dict[str, float],
        env_info: Dict[str, Any],
        metrics: DebateMetrics,
        done: bool,
    ) -> Dict[str, float]:
        """Enhanced rewards for the numerical DebateEnv.

        Takes the base env rewards and enriches them with multi-tier signals.

        Parameters
        ----------
        env_rewards : dict
            Raw rewards from DebateEnv.step().
        env_info : dict
            Info dict from DebateEnv.step().
        metrics : DebateMetrics
            Accumulated debate metrics.
        done : bool
            Episode terminated.

        Returns
        -------
        rewards : dict
            Enhanced per-agent rewards for MADDPG training.
        """
        c = self.cfg

        d = env_info.get("disagreement", 0.5)
        comp = env_info.get("compliance", 0.5)
        lam = env_info.get("lambda_adv", 0.5)
        task_r = env_info.get("task_reward", 0.0) if done else 0.0

        # Base rewards from env
        r_prop_base = env_rewards.get("proposer", 0.0)
        r_chal_base = env_rewards.get("challenger", 0.0)
        r_arb_base = env_rewards.get("arbiter", 0.0)
        r_coord_base = env_rewards.get("coordinator", 0.0)

        # Enhanced: quality proxy from compliance + task progress
        quality_proxy = comp * 0.6 + (1.0 - d) * 0.4

        # Enhanced proposer: task alignment bonus
        r_proposer = r_prop_base
        if done and task_r > 0:
            r_proposer += c.proposer_quality_improve_weight * task_r
        r_proposer += c.proposer_compliance_alignment * max(0.0, comp - 0.5)

        # Enhanced challenger: smart adversarial
        r_challenger = r_chal_base
        target_d = 0.3
        smart_adv = 1.0 - abs(d - target_d) / 0.5
        r_challenger += c.challenger_info_gain_weight * max(0.0, smart_adv) * 0.5

        # Enhanced arbiter: compliance quality
        r_arbiter = r_arb_base
        comp_quality = 1.0 - abs(comp - 0.8) / 0.3
        r_arbiter += c.arbiter_judgment_accuracy_weight * max(0.0, comp_quality) * 0.3

        # Enhanced coordinator
        r_coordinator = r_coord_base
        if len(metrics.quality_history) >= 2:
            q_trend = metrics.quality_history[-1] - metrics.quality_history[-2]
            r_coordinator += c.coordinator_convergence_weight * q_trend
        # Mining benefit
        mined = env_info.get("mined_rule", None)
        if mined is not None:
            r_coordinator += c.coordinator_mining_benefit * 0.5

        return {
            "proposer_ctrl": float(r_proposer),
            "challenger_ctrl": float(r_challenger),
            "arbiter_ctrl": float(r_arbiter),
            "coordinator": float(r_coordinator),
        }

    # ------------------------------------------------------------------
    # Information gain helper
    # ------------------------------------------------------------------

    @staticmethod
    def info_gain_reward(disagreement: float) -> float:
        """Reward function for constructive adversarial behavior.

        Encourages moderate disagreement (informative zone):
          - Low disagreement → encourage more challenge (reward ~ d)
          - High disagreement → encourage convergence (reward ~ 1-d)
          - Sweet spot around d=0.3 maximizes reward

        Parameters
        ----------
        disagreement : float
            Current disagreement level [0, 1].

        Returns
        -------
        reward : float
            Information gain reward.
        """
        # Gaussian peak at d=0.3 with sigma=0.25
        target = 0.3
        sigma = 0.25
        return float(np.exp(-0.5 * ((disagreement - target) / sigma) ** 2))

    @staticmethod
    def constructiveness_score(
        quality_delta: float,
        disagreement_delta: float,
    ) -> float:
        """Evaluate whether a challenge was constructive.

        A challenge is constructive if:
          - Quality improved (or didn't drop much) AND
          - Disagreement changed (challenge was substantive)

        Returns
        -------
        score : float
            [0, 1] constructiveness score.
        """
        # Quality didn't drop much → positive component
        quality_component = np.clip(quality_delta + 0.1, 0.0, 0.2) / 0.2

        # Disagreement changed → substance component
        substance_component = min(1.0, abs(disagreement_delta) * 5.0)

        return float(0.6 * quality_component + 0.4 * substance_component)
