"""Base Reward Computer — domain-agnostic, fixes causal confusion.

Key fixes over v1:
  - Challenger reward uses SIGNED disagreement change (not abs)
    to avoid rewarding both escalation and de-escalation
  - Configurable weights (no hardcoded magic numbers)
  - Separate terminal vs step rewards
  - Compliance reward as additive bonus (not mixed into base)
"""

from __future__ import annotations

from dataclasses import dataclass

from debate_rl_v2.logging_config import get_logger

logger = get_logger("framework.reward")


@dataclass
class RewardWeights:
    """Configurable reward weights — eliminates magic numbers."""
    # Step rewards
    quality_improvement: float = 0.3
    agreement_improvement: float = 0.2
    step_penalty: float = 0.01
    # Terminal rewards
    consensus_bonus: float = 2.0
    no_consensus_penalty: float = -0.5
    # Per-role adjustments
    proposer_quality_weight: float = 0.2
    proposer_compliance_weight: float = 0.1
    challenger_agreement_change_weight: float = 0.15  # SIGNED, not abs
    challenger_quality_weight: float = 0.1
    evaluator_compliance_weight: float = 0.1
    evaluator_quality_weight: float = 0.1
    coordinator_quality_weight: float = 0.3
    coordinator_agreement_weight: float = 0.1


class BaseRewardComputer:
    """Domain-agnostic reward computation with causal correctness.

    Key design decisions:
      - Challenger is rewarded for CONSTRUCTIVE disagreement change
        (quality improved AND agreement changed), not raw |Δagreement|
      - All weights are configurable via RewardWeights
      - Compliance rewards are additive bonuses, not mixed into base
    """

    def __init__(self, weights: RewardWeights | None = None) -> None:
        self.w = weights or RewardWeights()

    def compute_step_rewards(
        self,
        prev_state: dict[str, float],
        curr_state: dict[str, float],
        role_names: list[str] | None = None,
        evaluator_role: str = "evaluator",
        coordinator_role: str = "coordinator",
    ) -> dict[str, float]:
        """Compute per-role step rewards from state transition.

        Parameters
        ----------
        prev_state, curr_state : dict
            Must contain: "quality", "agreement", "compliance".
        role_names : list[str], optional
            All role names. If None, uses default 4-role setup.

        Returns
        -------
        rewards : dict[str, float]
        """
        dq = curr_state.get("quality", 0.5) - prev_state.get("quality", 0.5)
        da = curr_state.get("agreement", 0.5) - prev_state.get("agreement", 0.5)
        dc = curr_state.get("compliance", 0.5) - prev_state.get("compliance", 0.5)

        base = self.w.quality_improvement * dq + self.w.agreement_improvement * da - self.w.step_penalty

        rewards: dict[str, float] = {}
        names = role_names or ["proposer", "challenger", evaluator_role, coordinator_role]

        for role in names:
            if role == coordinator_role:
                rewards[role] = (
                    0.5 * base
                    + self.w.coordinator_quality_weight * dq
                    + self.w.coordinator_agreement_weight * da
                )
            elif role == evaluator_role:
                rewards[role] = (
                    base
                    + self.w.evaluator_compliance_weight * dc
                    + self.w.evaluator_quality_weight * dq
                )
            elif "challeng" in role or "critic" in role or "review" in role:
                # FIX: Use SIGNED agreement change × quality improvement
                # Reward constructive challenge: agreement decreased BUT quality improved
                constructive = max(0, dq) * max(0, -da)  # positive only when both conditions met
                destructive_penalty = max(0, -dq) * max(0, -da)  # quality dropped AND agreement dropped
                rewards[role] = (
                    base
                    + self.w.challenger_agreement_change_weight * (constructive - destructive_penalty)
                    + self.w.challenger_quality_weight * dq
                )
            else:
                # Proposer / generic role
                rewards[role] = (
                    base
                    + self.w.proposer_quality_weight * dq
                    + self.w.proposer_compliance_weight * dc
                )

        return rewards

    def compute_terminal_rewards(
        self,
        final_state: dict[str, float],
        terminated_successfully: bool,
        role_names: list[str] | None = None,
    ) -> dict[str, float]:
        """Compute terminal bonus/penalty."""
        quality = final_state.get("quality", 0.5)
        bonus = (
            self.w.consensus_bonus * quality
            if terminated_successfully
            else self.w.no_consensus_penalty
        )
        names = role_names or ["proposer", "challenger", "evaluator", "coordinator"]
        return {role: bonus for role in names}

    def compute_full_rewards(
        self,
        prev_state: dict[str, float],
        curr_state: dict[str, float],
        done: bool,
        terminated_successfully: bool,
        compliance_rewards: dict[str, float] | None = None,
        role_names: list[str] | None = None,
        evaluator_role: str = "evaluator",
        coordinator_role: str = "coordinator",
    ) -> dict[str, float]:
        """Compute step + terminal + compliance rewards."""
        rewards = self.compute_step_rewards(
            prev_state, curr_state, role_names, evaluator_role, coordinator_role
        )

        if done:
            terminal = self.compute_terminal_rewards(
                curr_state, terminated_successfully, role_names
            )
            for role in rewards:
                rewards[role] += terminal.get(role, 0.0)

        if compliance_rewards:
            for role, bonus in compliance_rewards.items():
                if role in rewards:
                    rewards[role] += bonus

        return rewards
