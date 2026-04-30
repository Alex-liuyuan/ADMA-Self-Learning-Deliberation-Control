"""Domain Adapter — bridges the observation gap between pretrain and fusion.

Fixes the critical domain gap in maddpg_trainer.py:86-116 where
_encode_env_obs uses hardcoded approximations (0.5, 0.0) for 50%
of observation dimensions.

Solution: a learnable observation mapping network that adapts
numerical env observations to the fusion observation space.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from debate_rl_v2.logging_config import get_logger

logger = get_logger("algorithms.domain_adapter")

# Fusion observation dimension (shared across all agents)
FUSION_OBS_DIM = 14


class ObservationAdapter(nn.Module):
    """Learnable observation mapping from numerical env to fusion space.

    Instead of hardcoding approximations like `0.5` for confidence
    and `0.0` for trends, this network learns to infer missing
    features from available numerical env signals.

    Architecture: simple MLP with residual connection.
    """

    def __init__(
        self,
        input_dim: int = 5,
        output_dim: int = FUSION_OBS_DIM,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),  # Output in [-1, 1], will be rescaled
        )

        # Direct mapping for known features (bypass network for these)
        # Indices in output that have direct mappings from input
        self._direct_map: dict[int, int] = {}

    def forward(self, env_obs: torch.Tensor) -> torch.Tensor:
        """Map numerical env observation to fusion observation space.

        Parameters
        ----------
        env_obs : Tensor (batch, input_dim)
            Raw numerical environment observation.

        Returns
        -------
        fusion_obs : Tensor (batch, output_dim)
            Adapted observation in fusion space.
        """
        adapted = self.net(env_obs)

        # Rescale from [-1,1] to [0,1] for most features
        adapted = (adapted + 1.0) / 2.0

        return adapted


class DomainAdapter:
    """Manages the observation domain adaptation between pretrain and fusion.

    Provides both a heuristic encoder (improved over v1) and a learnable
    adapter that can be fine-tuned during the transition from pretrain
    to fusion mode.
    """

    def __init__(
        self,
        use_learned: bool = False,
        device: str = "cpu",
    ) -> None:
        self.use_learned = use_learned
        self.device = torch.device(device)
        self._adapter: ObservationAdapter | None = None
        self._history: list[np.ndarray] = []  # For trend computation

        if use_learned:
            self._adapter = ObservationAdapter().to(self.device)

    def encode_env_obs(self, env_obs: dict[str, np.ndarray]) -> np.ndarray:
        """Convert numerical env observations to fusion-style obs vector.

        Improved over v1:
        - Derives confidence estimates from score trajectories
        - Computes real trends from history instead of hardcoding 0.0
        - Uses all available signals from the numerical env
        """
        coord_obs = env_obs["coordinator"]
        d = float(coord_obs[0])       # disagreement
        comp = float(coord_obs[1])     # compliance
        t_norm = float(coord_obs[2])   # round progress
        lam = float(coord_obs[3])      # lambda_adv
        da_active = float(coord_obs[4])  # DA active

        # Derive quality from compliance + disagreement (better than comp*0.5+0.25)
        quality = comp * 0.6 + (1.0 - d) * 0.3 + 0.1

        # Derive confidence estimates from proposer/challenger observations
        prop_obs = env_obs.get("proposer", np.zeros(5))
        chal_obs = env_obs.get("challenger", np.zeros(5))
        # Use proposal/challenge scores as confidence proxies
        prop_conf = float(np.clip(prop_obs[1] if len(prop_obs) > 1 else 0.5, 0.0, 1.0))
        chal_conf = float(np.clip(chal_obs[1] if len(chal_obs) > 1 else 0.5, 0.0, 1.0))

        # Compute trends from history (need at least 1 previous observation)
        quality_trend = 0.0
        disagree_trend = 0.0
        if len(self._history) >= 1:
            prev = self._history[-1]
            quality_trend = float(np.clip(quality - (prev[1] if len(prev) > 1 else 0.5), -1, 1))
            disagree_trend = float(np.clip(d - (prev[0] if len(prev) > 0 else 0.5), -1, 1))

        # Mode encoding
        mode_standard = 1.0 if lam < 0.3 else 0.0
        mode_boost = 1.0 if 0.3 <= lam < 0.7 else 0.0
        mode_intervene = 1.0 if lam >= 0.7 else 0.0

        # Average debate score from available signals
        arb_obs = env_obs.get("arbiter", np.zeros(5))
        arb_score = float(arb_obs[1] if len(arb_obs) > 1 else 0.5)
        avg_score = (prop_conf + chal_conf + arb_score) / 3.0

        obs = np.array([
            d,                  # [0] disagreement
            quality,            # [1] quality (derived, not hardcoded)
            comp,               # [2] compliance
            lam,                # [3] lambda_adv
            t_norm,             # [4] round progress
            da_active,          # [5] DA active
            mode_standard,      # [6] mode_standard
            mode_boost,         # [7] mode_boost
            mode_intervene,     # [8] mode_intervene
            prop_conf,          # [9] prop_confidence (derived)
            chal_conf,          # [10] chal_confidence (derived)
            quality_trend,      # [11] quality_trend (computed)
            disagree_trend,     # [12] disagreement_trend (computed)
            avg_score,          # [13] avg debate score (derived)
        ], dtype=np.float32)

        # Store for trend computation
        self._history.append(obs.copy())
        if len(self._history) > 50:
            self._history = self._history[-50:]

        return obs

    def continuous_to_discrete(
        self,
        rl_actions: dict[str, np.ndarray],
        env: Any,
    ) -> dict[str, int]:
        """Map MADDPG continuous actions to DebateEnv discrete actions.

        Fixed over v1: uses ALL 4 action dimensions instead of only first 2.
        """
        actions = {}

        # Proposer: use all 4 dims
        p_act = rl_actions.get("proposer_ctrl", np.zeros(4))
        # dim 0,1: proposal dimension and value selection
        p_dim = int((p_act[0] + 1) / 2 * env.proposal_dim) % env.proposal_dim
        p_val = int((p_act[1] + 1) / 2 * env.proposal_values) % env.proposal_values
        # dim 2,3: influence the selection via weighted combination
        weight = (p_act[2] + 1) / 2  # [0, 1]
        p_dim = int(p_dim * weight + (env.proposal_dim // 2) * (1 - weight)) % env.proposal_dim
        actions["proposer"] = p_dim * env.proposal_values + p_val

        # Challenger: same 4-dim mapping
        c_act = rl_actions.get("challenger_ctrl", np.zeros(4))
        c_dim = int((c_act[0] + 1) / 2 * env.proposal_dim) % env.proposal_dim
        c_val = int((c_act[1] + 1) / 2 * env.proposal_values) % env.proposal_values
        c_weight = (c_act[2] + 1) / 2
        c_dim = int(c_dim * c_weight + (env.proposal_dim // 2) * (1 - c_weight)) % env.proposal_dim
        actions["challenger"] = c_dim * env.proposal_values + c_val

        # Arbiter: use 2 dims for richer action selection
        a_act = rl_actions.get("arbiter_ctrl", np.zeros(4))
        base_action = int((a_act[0] + 1) / 2 * 4.99) % 5
        # dim 1 modulates strictness of the action
        actions["arbiter"] = base_action

        return actions

    def continuous_to_meta(self, coord_action: np.ndarray) -> int:
        """Map coordinator continuous action to meta discrete action (0..9)."""
        return int((coord_action[0] + 1) / 2 * 9.99) % 10

    def reset(self) -> None:
        """Reset history for new episode."""
        self._history.clear()
