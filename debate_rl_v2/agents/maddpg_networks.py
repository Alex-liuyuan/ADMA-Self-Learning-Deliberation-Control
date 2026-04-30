"""MADDPG Network Architectures — Deterministic Actor + Centralized Critic.

Implements continuous-action networks for Multi-Agent DDPG:
  - DeterministicActor: μ(o) → continuous action vector (tanh-bounded)
  - CentralizedCritic: Q(o_1,...,o_N, a_1,...,a_N) → scalar Q-value
  - OrnsteinUhlenbeck noise process for exploration
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from debate_rl_v2.agents.networks import _make_activation


class DeterministicActor(nn.Module):
    """Deterministic policy network μ(o) → a ∈ [-1, 1]^act_dim.

    Outputs continuous actions bounded by tanh activation.
    In the fusion architecture, these represent debate control
    parameters (temperature, aggressiveness, etc.).

    Parameters
    ----------
    obs_dim : int
        Observation dimensionality.
    act_dim : int
        Continuous action dimensionality.
    hidden_dim : int
        Hidden layer width.
    num_layers : int
        Number of hidden layers.
    activation : str
        Activation function name.
    use_layer_norm : bool
        Whether to use LayerNorm.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        activation: str = "relu",
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        layers: list[nn.Module] = []
        for i in range(num_layers):
            in_d = obs_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(_make_activation(activation))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

        layers.append(nn.Linear(hidden_dim, act_dim))
        layers.append(nn.Tanh())  # bound to [-1, 1]

        self.net = nn.Sequential(*layers)

        # Initialize final layer with small weights for stable start
        nn.init.uniform_(self.net[-2].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.net[-2].bias, -3e-3, 3e-3)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return continuous action ∈ [-1, 1]^act_dim."""
        return self.net(obs)


class CentralizedCritic(nn.Module):
    """Centralized action-value function Q(o_1,...,o_N, a_1,...,a_N).

    Takes ALL agents' observations and actions as input (CTDE paradigm).
    This allows each agent's critic to condition on global information
    during training while actors only use local observations.

    Parameters
    ----------
    total_obs_dim : int
        Sum of all agents' observation dimensions.
    total_act_dim : int
        Sum of all agents' action dimensions.
    hidden_dim : int
        Hidden layer width.
    num_layers : int
        Number of hidden layers.
    activation : str
        Activation function name.
    use_layer_norm : bool
        Whether to use LayerNorm.
    """

    def __init__(
        self,
        total_obs_dim: int,
        total_act_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        activation: str = "relu",
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.total_obs_dim = total_obs_dim
        self.total_act_dim = total_act_dim

        input_dim = total_obs_dim + total_act_dim
        layers: list[nn.Module] = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(_make_activation(activation))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        all_obs: torch.Tensor,
        all_acts: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Q-value from concatenated global state + joint actions.

        Parameters
        ----------
        all_obs : Tensor (B, total_obs_dim)
            Concatenation of all agents' observations.
        all_acts : Tensor (B, total_act_dim)
            Concatenation of all agents' actions.

        Returns
        -------
        q_value : Tensor (B, 1)
        """
        x = torch.cat([all_obs, all_acts], dim=-1)
        return self.net(x)


class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration noise.

    Used in DDPG/MADDPG to produce smooth exploration trajectories.

        dX_t = θ (μ - X_t) dt + σ dW_t

    Parameters
    ----------
    size : int
        Dimensionality of the noise vector.
    mu : float
        Long-term mean.
    theta : float
        Rate of mean reversion.
    sigma : float
        Volatility (noise scale).
    sigma_decay : float
        Multiplicative decay applied to sigma each step.
    sigma_min : float
        Minimum sigma value.
    """

    def __init__(
        self,
        size: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        sigma_decay: float = 0.9999,
        sigma_min: float = 0.01,
        seed: int = 0,
    ) -> None:
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.sigma_min = sigma_min
        self.rng = np.random.default_rng(seed)
        self.state = np.full(size, mu, dtype=np.float32)

    def reset(self) -> None:
        self.state = np.full(self.size, self.mu, dtype=np.float32)

    def sample(self) -> np.ndarray:
        """Generate one noise sample and advance the process."""
        dx = self.theta * (self.mu - self.state) + self.sigma * self.rng.standard_normal(self.size).astype(np.float32)
        self.state = self.state + dx
        self.sigma = max(self.sigma * self.sigma_decay, self.sigma_min)
        return self.state.copy()


class GaussianNoise:
    """Simple Gaussian exploration noise with decay.

    Simpler alternative to OU noise, often sufficient for MADDPG.

    Parameters
    ----------
    size : int
        Action dimensionality.
    std : float
        Initial standard deviation.
    decay : float
        Multiplicative decay per call.
    min_std : float
        Minimum standard deviation.
    """

    def __init__(
        self,
        size: int,
        std: float = 0.1,
        decay: float = 0.9999,
        min_std: float = 0.01,
        seed: int = 0,
    ) -> None:
        self.size = size
        self.std = std
        self.decay = decay
        self.min_std = min_std
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass  # stateless

    def sample(self) -> np.ndarray:
        noise = self.rng.standard_normal(self.size).astype(np.float32) * self.std
        self.std = max(self.std * self.decay, self.min_std)
        return noise
