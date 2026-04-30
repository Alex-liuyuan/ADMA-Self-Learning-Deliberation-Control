"""Role-Specific Observation Space — extends shared 14D with per-role features.

Fixes the problem where all 4 RL agents share identical 14D observations,
wasting critic capacity. Each role now gets 14 shared + 6 role-specific = 20D.

Role-specific features:
  - Proposer: proposal quality history, acceptance rate, modification magnitude
  - Challenger: challenge success rate, info gain history, attack angle diversity
  - Arbiter: judgment consistency, rule trigger frequency, calibration error
  - Coordinator: convergence speed, rule mining benefit, termination accuracy
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from debate_rl_v2.logging_config import get_logger

logger = get_logger("algorithms.role_obs")

SHARED_OBS_DIM = 14
ROLE_OBS_DIM = 6
TOTAL_OBS_DIM = SHARED_OBS_DIM + ROLE_OBS_DIM


@dataclass
class RoleObservationTracker:
    """Tracks per-role statistics for observation enrichment.

    Call `update()` each round with role-specific metrics,
    then `encode()` to get the 6D role-specific observation.
    """

    # Rolling windows for statistics
    _window_size: int = 10
    _proposal_qualities: deque = field(default_factory=lambda: deque(maxlen=10))
    _acceptance_flags: deque = field(default_factory=lambda: deque(maxlen=10))
    _modification_magnitudes: deque = field(default_factory=lambda: deque(maxlen=10))
    _challenge_successes: deque = field(default_factory=lambda: deque(maxlen=10))
    _info_gains: deque = field(default_factory=lambda: deque(maxlen=10))
    _attack_angles: deque = field(default_factory=lambda: deque(maxlen=10))
    _judgment_consistencies: deque = field(default_factory=lambda: deque(maxlen=10))
    _rule_triggers: deque = field(default_factory=lambda: deque(maxlen=10))
    _calibration_errors: deque = field(default_factory=lambda: deque(maxlen=10))
    _convergence_speeds: deque = field(default_factory=lambda: deque(maxlen=10))
    _mining_benefits: deque = field(default_factory=lambda: deque(maxlen=10))
    _termination_accuracies: deque = field(default_factory=lambda: deque(maxlen=10))

    def update_proposer(
        self,
        quality: float,
        accepted: bool,
        modification_mag: float,
    ) -> None:
        self._proposal_qualities.append(quality)
        self._acceptance_flags.append(float(accepted))
        self._modification_magnitudes.append(modification_mag)

    def update_challenger(
        self,
        success: bool,
        info_gain: float,
        attack_angle: float,
    ) -> None:
        self._challenge_successes.append(float(success))
        self._info_gains.append(info_gain)
        self._attack_angles.append(attack_angle)

    def update_arbiter(
        self,
        consistency: float,
        rule_trigger_rate: float,
        calibration_error: float,
    ) -> None:
        self._judgment_consistencies.append(consistency)
        self._rule_triggers.append(rule_trigger_rate)
        self._calibration_errors.append(calibration_error)

    def update_coordinator(
        self,
        convergence_speed: float,
        mining_benefit: float,
        termination_accuracy: float,
    ) -> None:
        self._convergence_speeds.append(convergence_speed)
        self._mining_benefits.append(mining_benefit)
        self._termination_accuracies.append(termination_accuracy)

    def encode_proposer(self) -> np.ndarray:
        """6D proposer-specific observation."""
        avg_quality = _safe_mean(self._proposal_qualities)
        quality_trend = _safe_trend(self._proposal_qualities)
        acceptance_rate = _safe_mean(self._acceptance_flags)
        avg_mod_mag = _safe_mean(self._modification_magnitudes)
        mod_trend = _safe_trend(self._modification_magnitudes)
        quality_variance = _safe_std(self._proposal_qualities)
        return np.array([
            avg_quality,       # [0] average proposal quality
            quality_trend,     # [1] quality improvement trend
            acceptance_rate,   # [2] arbiter acceptance rate
            avg_mod_mag,       # [3] average modification magnitude
            mod_trend,         # [4] modification trend
            quality_variance,  # [5] quality variance (stability)
        ], dtype=np.float32)

    def encode_challenger(self) -> np.ndarray:
        """6D challenger-specific observation."""
        success_rate = _safe_mean(self._challenge_successes)
        avg_info_gain = _safe_mean(self._info_gains)
        info_gain_trend = _safe_trend(self._info_gains)
        angle_diversity = _safe_std(self._attack_angles)
        recent_success = _safe_mean(list(self._challenge_successes)[-3:]) if self._challenge_successes else 0.5
        avg_angle = _safe_mean(self._attack_angles)
        return np.array([
            success_rate,      # [0] challenge success rate
            avg_info_gain,     # [1] average information gain
            info_gain_trend,   # [2] info gain trend
            angle_diversity,   # [3] attack angle diversity
            recent_success,    # [4] recent success rate (last 3)
            avg_angle,         # [5] average attack angle
        ], dtype=np.float32)

    def encode_arbiter(self) -> np.ndarray:
        """6D arbiter-specific observation."""
        avg_consistency = _safe_mean(self._judgment_consistencies)
        consistency_trend = _safe_trend(self._judgment_consistencies)
        avg_rule_trigger = _safe_mean(self._rule_triggers)
        avg_calibration = _safe_mean(self._calibration_errors)
        calibration_trend = _safe_trend(self._calibration_errors)
        rule_trigger_var = _safe_std(self._rule_triggers)
        return np.array([
            avg_consistency,    # [0] judgment consistency
            consistency_trend,  # [1] consistency trend
            avg_rule_trigger,   # [2] rule trigger frequency
            avg_calibration,    # [3] calibration error
            calibration_trend,  # [4] calibration improvement trend
            rule_trigger_var,   # [5] rule trigger variance
        ], dtype=np.float32)

    def encode_coordinator(self) -> np.ndarray:
        """6D coordinator-specific observation."""
        avg_convergence = _safe_mean(self._convergence_speeds)
        convergence_trend = _safe_trend(self._convergence_speeds)
        avg_mining = _safe_mean(self._mining_benefits)
        avg_termination = _safe_mean(self._termination_accuracies)
        mining_trend = _safe_trend(self._mining_benefits)
        termination_trend = _safe_trend(self._termination_accuracies)
        return np.array([
            avg_convergence,    # [0] convergence speed
            convergence_trend,  # [1] convergence trend
            avg_mining,         # [2] rule mining benefit
            avg_termination,    # [3] termination timing accuracy
            mining_trend,       # [4] mining benefit trend
            termination_trend,  # [5] termination accuracy trend
        ], dtype=np.float32)

    def encode(self, role: str) -> np.ndarray:
        """Get 6D role-specific observation for the given role."""
        encoders = {
            "proposer_ctrl": self.encode_proposer,
            "challenger_ctrl": self.encode_challenger,
            "arbiter_ctrl": self.encode_arbiter,
            "coordinator": self.encode_coordinator,
        }
        encoder = encoders.get(role)
        if encoder is None:
            return np.zeros(ROLE_OBS_DIM, dtype=np.float32)
        return encoder()

    def reset(self) -> None:
        for attr in vars(self):
            val = getattr(self, attr)
            if isinstance(val, deque):
                val.clear()


def build_role_observation(
    shared_obs: np.ndarray,
    role_obs: np.ndarray,
) -> np.ndarray:
    """Concatenate shared (14D) and role-specific (6D) observations."""
    return np.concatenate([shared_obs, role_obs])


def _safe_mean(d: deque | list) -> float:
    if not d:
        return 0.5
    return float(np.mean(list(d)))


def _safe_std(d: deque | list) -> float:
    if len(d) < 2:
        return 0.0
    return float(np.clip(np.std(list(d)), 0.0, 1.0))


def _safe_trend(d: deque | list) -> float:
    """Compute simple linear trend (last - first) / count."""
    if len(d) < 2:
        return 0.0
    vals = list(d)
    trend = (vals[-1] - vals[0]) / len(vals)
    return float(np.clip(trend, -1.0, 1.0))


# ======================================================================
# Generic Role Observation Tracker — scenario-agnostic
# ======================================================================


@dataclass
class RoleObservationSpec:
    """Specification for a role's observation metrics.

    Parameters
    ----------
    name : str
        Role name (e.g. "buyer", "seller").
    metrics : list[str]
        Metric names tracked for this role (e.g. ["quality", "acceptance_rate"]).
    obs_dim : int
        Output observation dimension. Each metric contributes 2 features
        (mean + trend), so obs_dim should be >= 2 * len(metrics).
        Extra dims are zero-padded.
    """
    name: str
    metrics: list[str] = field(default_factory=list)
    obs_dim: int = 6


class GenericRoleObservationTracker:
    """Scenario-agnostic role observation tracker.

    Unlike the debate-specific RoleObservationTracker, this class works
    with arbitrary roles and metrics defined via RoleObservationSpec.

    Usage::

        specs = {
            "buyer": RoleObservationSpec("buyer", ["offer_quality", "patience"], obs_dim=6),
            "seller": RoleObservationSpec("seller", ["price_firmness", "concession_rate"], obs_dim=6),
        }
        tracker = GenericRoleObservationTracker(specs)
        tracker.update("buyer", offer_quality=0.8, patience=0.6)
        obs = tracker.encode("buyer")  # np.ndarray of shape (6,)
    """

    def __init__(
        self,
        specs: dict[str, RoleObservationSpec],
        window_size: int = 10,
    ) -> None:
        self.specs = specs
        self.window_size = window_size
        # {role: {metric: deque}}
        self._data: dict[str, dict[str, deque]] = {}
        for role, spec in specs.items():
            self._data[role] = {
                m: deque(maxlen=window_size) for m in spec.metrics
            }

    def update(self, role: str, **metrics: float) -> None:
        """Record metric values for a role.

        Unknown metrics are silently ignored.
        """
        role_data = self._data.get(role)
        if role_data is None:
            return
        for metric, value in metrics.items():
            if metric in role_data:
                role_data[metric].append(float(value))

    def encode(self, role: str) -> np.ndarray:
        """Encode role-specific observation as a fixed-size vector.

        For each metric: [mean, trend]. Zero-padded to obs_dim.
        """
        spec = self.specs.get(role)
        if spec is None:
            return np.zeros(ROLE_OBS_DIM, dtype=np.float32)

        features: list[float] = []
        role_data = self._data.get(role, {})
        for metric in spec.metrics:
            d = role_data.get(metric, deque())
            features.append(_safe_mean(d))
            features.append(_safe_trend(d))

        # Pad or truncate to obs_dim
        obs = np.zeros(spec.obs_dim, dtype=np.float32)
        n = min(len(features), spec.obs_dim)
        obs[:n] = features[:n]
        return obs

    def reset(self) -> None:
        """Clear all tracked data."""
        for role_data in self._data.values():
            for d in role_data.values():
                d.clear()


def create_debate_observation_specs() -> dict[str, RoleObservationSpec]:
    """Factory: create observation specs for the standard debate scenario."""
    return {
        "proposer_ctrl": RoleObservationSpec(
            name="proposer_ctrl",
            metrics=["quality", "acceptance_rate", "modification_mag"],
            obs_dim=ROLE_OBS_DIM,
        ),
        "challenger_ctrl": RoleObservationSpec(
            name="challenger_ctrl",
            metrics=["success_rate", "info_gain", "attack_angle"],
            obs_dim=ROLE_OBS_DIM,
        ),
        "arbiter_ctrl": RoleObservationSpec(
            name="arbiter_ctrl",
            metrics=["consistency", "rule_trigger_rate", "calibration_error"],
            obs_dim=ROLE_OBS_DIM,
        ),
        "coordinator": RoleObservationSpec(
            name="coordinator",
            metrics=["convergence_speed", "mining_benefit", "termination_accuracy"],
            obs_dim=ROLE_OBS_DIM,
        ),
    }
