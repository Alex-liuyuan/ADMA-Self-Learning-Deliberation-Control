"""OnlineParameterUpdater — gradient-free parameter accumulation.

In online mode, RL weights are frozen. This updater uses EMA smoothing
and Bayesian conjugate updates to gradually improve strategy parameters
from episode feedback, without any gradient computation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from debate_rl_v2.logging_config import get_logger

logger = get_logger("mode.online_updater")

# 4D strategy parameters per role: [assertiveness/aggressiveness, detail/specificity,
# compliance_focus/constructiveness, incorporation/novelty]
PARAM_DIM = 4
ROLES = ("proposer", "challenger", "arbiter", "coordinator")


@dataclass
class OnlineState:
    """Persistent state for online parameter accumulation."""
    ema_params: dict[str, np.ndarray] = field(default_factory=dict)
    bayesian_means: dict[str, np.ndarray] = field(default_factory=dict)
    bayesian_vars: dict[str, np.ndarray] = field(default_factory=dict)
    total_episodes: int = 0
    _roles: tuple[str, ...] = field(default_factory=lambda: ROLES)
    _param_dim: int = PARAM_DIM

    def __post_init__(self) -> None:
        for role in self._roles:
            if role not in self.ema_params:
                self.ema_params[role] = np.full(self._param_dim, 0.5, dtype=np.float32)
            if role not in self.bayesian_means:
                self.bayesian_means[role] = np.full(self._param_dim, 0.5, dtype=np.float32)
            if role not in self.bayesian_vars:
                self.bayesian_vars[role] = np.full(self._param_dim, 0.25, dtype=np.float32)


class OnlineParameterUpdater:
    """Gradient-free parameter updater for online learning mode.

    Mechanisms:
    - EMA: θ_new = (1-α)·θ_old + α·θ_observed, α=ema_alpha
    - Bayesian: Normal-Normal conjugate update on (param_value, quality)
    - High-confidence params (variance < threshold) auto-solidify

    Parameters
    ----------
    ema_alpha : float
        EMA smoothing factor (default 0.01, slow adaptation).
    obs_noise : float
        Observation noise for Bayesian updates.
    confidence_threshold : float
        Variance threshold below which params are considered converged.
    """

    def __init__(
        self,
        roles: tuple[str, ...] | None = None,
        param_dim: int = PARAM_DIM,
        ema_alpha: float = 0.01,
        obs_noise: float = 0.1,
        confidence_threshold: float = 0.02,
    ) -> None:
        self.ema_alpha = ema_alpha
        self.obs_noise = obs_noise
        self.confidence_threshold = confidence_threshold
        self._roles = roles or ROLES
        self._param_dim = param_dim
        self.state = OnlineState(_roles=self._roles, _param_dim=self._param_dim)

    def update(
        self,
        role: str,
        observed_params: np.ndarray,
        quality: float,
    ) -> None:
        """Update parameters for a role based on observed episode outcome.

        Parameters
        ----------
        role : str
            Agent role name.
        observed_params : np.ndarray
            4D strategy parameters observed this episode.
        quality : float
            Episode quality score (0-1), used to weight the update.
        """
        if role not in self.state.ema_params:
            return

        obs = np.asarray(observed_params, dtype=np.float32)[:self._param_dim]
        quality = float(np.clip(quality, 0.0, 1.0))

        # EMA update (quality-weighted alpha)
        alpha = self.ema_alpha * (0.5 + quality)
        ema = self.state.ema_params[role]
        self.state.ema_params[role] = (1 - alpha) * ema + alpha * obs

        # Bayesian Normal-Normal conjugate update
        prior_mean = self.state.bayesian_means[role]
        prior_var = self.state.bayesian_vars[role]
        obs_var = np.full(self._param_dim, self.obs_noise, dtype=np.float32)

        # Posterior precision = prior precision + obs precision
        post_var = 1.0 / (1.0 / prior_var + 1.0 / obs_var)
        post_mean = post_var * (prior_mean / prior_var + obs / obs_var)

        self.state.bayesian_means[role] = post_mean
        self.state.bayesian_vars[role] = post_var

        self.state.total_episodes += 1

    def get_best_params(self, role: str) -> np.ndarray:
        """Get the best parameter estimate for a role.

        Uses Bayesian posterior mean for converged dimensions,
        EMA for unconverged dimensions.
        """
        if role not in self.state.ema_params:
            return np.full(self._param_dim, 0.5, dtype=np.float32)

        ema = self.state.ema_params[role]
        bayes_mean = self.state.bayesian_means[role]
        bayes_var = self.state.bayesian_vars[role]

        # Use Bayesian mean where converged, EMA otherwise
        converged = bayes_var < self.confidence_threshold
        result = np.where(converged, bayes_mean, ema)
        return np.clip(result, 0.0, 1.0)

    def get_confidence(self, role: str) -> float:
        """Get overall confidence for a role (fraction of converged dims)."""
        if role not in self.state.bayesian_vars:
            return 0.0
        converged = self.state.bayesian_vars[role] < self.confidence_threshold
        return float(np.mean(converged))

    def save(self, path: str) -> None:
        """Save online state to JSON."""
        data = {
            "total_episodes": self.state.total_episodes,
            "ema_params": {r: v.tolist() for r, v in self.state.ema_params.items()},
            "bayesian_means": {r: v.tolist() for r, v in self.state.bayesian_means.items()},
            "bayesian_vars": {r: v.tolist() for r, v in self.state.bayesian_vars.items()},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Online state saved: %s (%d episodes)", path, self.state.total_episodes)

    def load(self, path: str) -> None:
        """Load online state from JSON."""
        p = Path(path)
        if not p.exists():
            logger.warning("Online state file not found: %s", path)
            return
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.state.total_episodes = data.get("total_episodes", 0)
        for key, store in [
            ("ema_params", self.state.ema_params),
            ("bayesian_means", self.state.bayesian_means),
            ("bayesian_vars", self.state.bayesian_vars),
        ]:
            for role, vals in data.get(key, {}).items():
                store[role] = np.array(vals, dtype=np.float32)
        logger.info("Online state loaded: %s (%d episodes)", path, self.state.total_episodes)
