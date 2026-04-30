"""深度 MADDPG 网络架构 — 支持 Transformer、Attention、Residual 等高级组件。

针对复杂任务（如医学辩论）设计的深度网络，支持：
- 多头注意力机制（Multi-Head Attention）
- 残差连接（Residual Connections）
- Transformer Encoder 层
- 更深的网络（4-8层）
- 更大的隐藏维度（256-1024）
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from debate_rl_v2.agents.networks import _make_activation


class TransformerBlock(nn.Module):
    """Transformer Encoder Block with Multi-Head Attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) or (B, D) — auto-expand if needed."""
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, D) → (B, 1, D)
        # Attention with residual
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x.squeeze(1) if x.size(1) == 1 else x


class ResidualBlock(nn.Module):
    """Residual MLP Block with LayerNorm."""

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        activation: str = "relu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            _make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DeepDeterministicActor(nn.Module):
    """深度 Actor 网络，支持 Transformer 和 Residual 架构。

    Parameters
    ----------
    obs_dim : int
        观测维度。
    act_dim : int
        动作维度。
    hidden_dim : int
        隐藏层宽度（推荐 256-512）。
    num_layers : int
        层数（推荐 4-8）。
    architecture : str
        架构类型: "mlp" | "residual" | "transformer"
    num_heads : int
        Transformer 注意力头数（仅 transformer 架构）。
    dropout : float
        Dropout 比例。
    activation : str
        激活函数。
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        architecture: str = "residual",
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.architecture = architecture

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            _make_activation(activation),
            nn.LayerNorm(hidden_dim),
        )

        # Core layers
        if architecture == "transformer":
            self.core = nn.ModuleList([
                TransformerBlock(hidden_dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ])
        elif architecture == "residual":
            self.core = nn.ModuleList([
                ResidualBlock(hidden_dim, activation=activation, dropout=dropout)
                for _ in range(num_layers)
            ])
        else:  # mlp
            layers = []
            for _ in range(num_layers):
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    _make_activation(activation),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                ])
            self.core = nn.Sequential(*layers)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            _make_activation(activation),
            nn.Linear(hidden_dim // 2, act_dim),
            nn.Tanh(),
        )

        # Initialize output layer with small weights
        nn.init.uniform_(self.output_head[-2].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.output_head[-2].bias, -3e-3, 3e-3)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(obs)
        if isinstance(self.core, nn.ModuleList):
            for layer in self.core:
                x = layer(x)
        else:
            x = self.core(x)
        return self.output_head(x)


class DeepCentralizedCritic(nn.Module):
    """深度 Critic 网络，支持 Transformer 和 Residual 架构。

    Parameters
    ----------
    total_obs_dim : int
        所有 agent 观测维度之和。
    total_act_dim : int
        所有 agent 动作维度之和。
    hidden_dim : int
        隐藏层宽度（推荐 512-1024）。
    num_layers : int
        层数（推荐 4-8）。
    architecture : str
        架构类型: "mlp" | "residual" | "transformer"
    num_heads : int
        Transformer 注意力头数。
    dropout : float
        Dropout 比例。
    activation : str
        激活函数。
    """

    def __init__(
        self,
        total_obs_dim: int,
        total_act_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 5,
        architecture: str = "residual",
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.total_obs_dim = total_obs_dim
        self.total_act_dim = total_act_dim
        self.architecture = architecture

        input_dim = total_obs_dim + total_act_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            _make_activation(activation),
            nn.LayerNorm(hidden_dim),
        )

        # Core layers
        if architecture == "transformer":
            self.core = nn.ModuleList([
                TransformerBlock(hidden_dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ])
        elif architecture == "residual":
            self.core = nn.ModuleList([
                ResidualBlock(hidden_dim, activation=activation, dropout=dropout)
                for _ in range(num_layers)
            ])
        else:  # mlp
            layers = []
            for _ in range(num_layers):
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    _make_activation(activation),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                ])
            self.core = nn.Sequential(*layers)

        # Output head (Q-value)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            _make_activation(activation),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        all_obs: torch.Tensor,
        all_acts: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([all_obs, all_acts], dim=-1)
        x = self.input_proj(x)
        if isinstance(self.core, nn.ModuleList):
            for layer in self.core:
                x = layer(x)
        else:
            x = self.core(x)
        return self.output_head(x)


class AttentionActor(nn.Module):
    """基于注意力的 Actor，显式建模角色间交互。

    使用 cross-attention 让当前角色关注其他角色的观测。
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.input_proj = nn.Linear(obs_dim, hidden_dim)
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, act_dim),
            nn.Tanh(),
        )

        nn.init.uniform_(self.output_head[-2].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.output_head[-2].bias, -3e-3, 3e-3)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(obs)
        for layer in self.transformer_layers:
            x = layer(x)
        return self.output_head(x)
