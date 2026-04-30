"""Neural network architectures for debate agents — Section 5.2.

Implements:
  - Shared feature extractor (bottom layers shared across roles)
  - Role-specific actor networks (with meta-action conditioning)
  - Centralized critic networks
  - Meta-level coordinator networks (for hierarchical PPO)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical


def _make_activation(name: str) -> nn.Module:
    return {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}[name]()


class SharedFeatureExtractor(nn.Module):
    """Shared bottom feature extraction layers.

    Used to reduce parameter count when multiple roles share
    similar input structure.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 64,
        activation: str = "tanh",
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        layers = [nn.Linear(input_dim, output_dim), _make_activation(activation)]
        if use_layer_norm:
            layers.append(nn.LayerNorm(output_dim))
        self.net = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorNetwork(nn.Module):
    """Role-specific actor network π(a|o, g).

    Optionally conditions on a meta-action embedding from the coordinator.

    Parameters
    ----------
    obs_dim : int
        Observation dimensionality.
    act_dim : int
        Number of discrete actions.
    hidden_dim : int
        Hidden layer width.
    num_layers : int
        Number of hidden layers.
    activation : str
        Activation function name.
    meta_embed_dim : int
        Dimensionality of meta-action embedding (0 = no conditioning).
    use_layer_norm : bool
        Whether to use layer normalization.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        activation: str = "tanh",
        meta_embed_dim: int = 0,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.act_dim = act_dim
        self.meta_embed_dim = meta_embed_dim

        input_dim = obs_dim + meta_embed_dim
        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(_make_activation(activation))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Linear(hidden_dim, act_dim))
        self.net = nn.Sequential(*layers)

        # Meta-action embedding layer (if used)
        if meta_embed_dim > 0:
            self.meta_embedder = nn.Embedding(16, meta_embed_dim)
        else:
            self.meta_embedder = None

    def forward(
        self, obs: torch.Tensor, meta_action: Optional[torch.Tensor] = None
    ) -> Categorical:
        x = obs
        if self.meta_embedder is not None:
            if meta_action is not None:
                me = self.meta_embedder(meta_action)
            else:
                # Default zero embedding when no meta action provided
                me = torch.zeros(
                    *obs.shape[:-1], self.meta_embed_dim,
                    device=obs.device, dtype=obs.dtype,
                )
            x = torch.cat([x, me], dim=-1)
        logits = self.net(x)
        return Categorical(logits=logits)


class CriticNetwork(nn.Module):
    """Centralized critic network V(s).

    Takes the full (or partial) state and outputs a scalar value.

    Parameters
    ----------
    obs_dim : int
        Observation dimensionality.
    hidden_dim : int
        Hidden layer width.
    num_layers : int
        Number of hidden layers.
    activation : str
        Activation function name.
    use_layer_norm : bool
        Whether to use layer normalization.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        activation: str = "tanh",
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_d = obs_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(_make_activation(activation))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class MetaActorNetwork(nn.Module):
    """Coordinator meta-level actor π_meta(g|s_meta).

    Outputs probability distribution over meta-actions (protocol
    parameter adjustments, rule mining triggers, termination).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 2,
        activation: str = "tanh",
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_d = obs_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(_make_activation(activation))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Linear(hidden_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.net(obs))


class MetaCriticNetwork(nn.Module):
    """Coordinator meta-level critic V_meta(s_meta)."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        activation: str = "tanh",
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_d = obs_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(_make_activation(activation))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)
