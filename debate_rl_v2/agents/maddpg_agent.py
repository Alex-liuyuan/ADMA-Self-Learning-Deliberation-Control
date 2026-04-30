"""MADDPG Agent — manages actor, target networks, noise, and inference.

Each agent has:
  - actor: μ(o_i) → a_i  (deterministic, local observation only)
  - critic: Q(o_all, a_all)  (centralized, all agents' info)
  - target_actor, target_critic: for stable TD targets (soft update)
  - noise: OU or Gaussian for exploration
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - depends on optional runtime
    torch = None
    nn = None

try:
    from debate_rl_v2.agents.maddpg_networks import (
        DeterministicActor,
        CentralizedCritic,
        OUNoise,
        GaussianNoise,
    )
except ImportError:  # pragma: no cover - torch-backed networks unavailable
    DeterministicActor = None
    CentralizedCritic = None

    class OUNoise:
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
            dx = (
                self.theta * (self.mu - self.state)
                + self.sigma * self.rng.standard_normal(self.size).astype(np.float32)
            )
            self.state = self.state + dx
            self.sigma = max(self.sigma * self.sigma_decay, self.sigma_min)
            return self.state.copy()

    class GaussianNoise:
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
            return None

        def sample(self) -> np.ndarray:
            noise = self.rng.standard_normal(self.size).astype(np.float32) * self.std
            self.std = max(self.std * self.decay, self.min_std)
            return noise


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError(
            "MADDPGAgentGroup requires torch. Install torch or use pure LLM/GameEngine "
            "components that do not require RL networks."
        )


@dataclass
class MADDPGAgent:
    """Single MADDPG agent with actor, critic, and target networks."""

    name: str
    actor: object          # DeterministicActor or DeepDeterministicActor
    critic: object         # CentralizedCritic or DeepCentralizedCritic
    target_actor: object
    target_critic: object
    actor_opt: object
    critic_opt: object
    noise: OUNoise | GaussianNoise
    device: object

    def act(
        self,
        obs: np.ndarray,
        explore: bool = True,
    ) -> np.ndarray:
        """Select action from local observation.

        Parameters
        ----------
        obs : ndarray (obs_dim,)
            Local observation.
        explore : bool
            Add exploration noise.

        Returns
        -------
        action : ndarray (act_dim,)
            Continuous action ∈ [-1, 1].
        """
        _require_torch()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy().squeeze(0)

        if explore:
            noise = self.noise.sample()
            action = action + noise
            action = np.clip(action, -1.0, 1.0)

        return action

    def act_batch(self, obs: torch.Tensor) -> torch.Tensor:
        """Batch forward through actor (for critic updates)."""
        return self.actor(obs)

    def target_act_batch(self, obs: torch.Tensor) -> torch.Tensor:
        """Batch forward through target actor."""
        return self.target_actor(obs)

    def soft_update(self, tau: float = 0.01) -> None:
        """Polyak-average update of target networks.

            θ_target = τ θ_online + (1 − τ) θ_target
        """
        for target_p, online_p in zip(
            self.target_actor.parameters(), self.actor.parameters()
        ):
            target_p.data.copy_(tau * online_p.data + (1.0 - tau) * target_p.data)
        for target_p, online_p in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_p.data.copy_(tau * online_p.data + (1.0 - tau) * target_p.data)

    def reset_noise(self) -> None:
        self.noise.reset()

    def save(self, path: str) -> None:
        _require_torch()
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        _require_torch()
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.target_actor.load_state_dict(ckpt["target_actor"])
        self.target_critic.load_state_dict(ckpt["target_critic"])
        if "actor_opt" in ckpt:
            self.actor_opt.load_state_dict(ckpt["actor_opt"])
        if "critic_opt" in ckpt:
            self.critic_opt.load_state_dict(ckpt["critic_opt"])


class _FallbackActor:
    """Small no-torch actor used for structural tests and mock-only runs."""

    def __init__(self, obs_dim: int, act_dim: int, seed: int = 0) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._rng = np.random.default_rng(seed)
        self._weight = self._rng.standard_normal((obs_dim, act_dim)).astype(np.float32) * 0.05

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if arr.shape[0] != self.obs_dim:
            arr = np.resize(arr, self.obs_dim).astype(np.float32)
        return np.tanh(arr @ self._weight).astype(np.float32)


class _FallbackMADDPGAgent:
    def __init__(
        self,
        name: str,
        obs_dim: int,
        act_dim: int,
        noise: OUNoise | GaussianNoise,
        seed: int = 0,
    ) -> None:
        self.name = name
        self.actor = _FallbackActor(obs_dim, act_dim, seed=seed)
        self.critic = None
        self.target_actor = self.actor
        self.target_critic = None
        self.actor_opt = None
        self.critic_opt = None
        self.noise = noise
        self.device = "numpy"

    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        action = self.actor(obs)
        if explore:
            action = action + self.noise.sample()
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def reset_noise(self) -> None:
        self.noise.reset()

    def soft_update(self, tau: float = 0.01) -> None:
        return None


class MADDPGAgentGroup:
    """Manages all MADDPG agents with shared critic architecture.

    Generalized to support arbitrary role names — roles are inferred
    from obs_dims.keys() instead of a hardcoded ROLES constant.

    Legacy debate roles (proposer_ctrl, challenger_ctrl, arbiter_ctrl,
    coordinator) still work via backward compatibility.

    Parameters
    ----------
    obs_dims : dict
        {role: obs_dim} for each agent.
    act_dims : dict
        {role: act_dim} for each agent.
    hidden_dim : int
        Hidden layer width.
    critic_hidden_dim : int
        Critic hidden width (usually larger).
    num_layers : int
        Number of hidden layers.
    actor_lr : float
        Actor learning rate.
    critic_lr : float
        Critic learning rate.
    noise_type : str
        "ou" or "gaussian".
    noise_std : float
        Initial noise standard deviation.
    noise_decay : float
        Noise decay rate.
    device : torch.device
        Compute device.
    seed : int
        Random seed.
    """

    # Legacy constant kept for backward compatibility (read-only reference)
    ROLES = ("proposer_ctrl", "challenger_ctrl", "arbiter_ctrl", "coordinator")

    # Supported deep architectures
    ARCHITECTURES = ("shallow", "mlp", "residual", "transformer")

    def __init__(
        self,
        obs_dims: Dict[str, int],
        act_dims: Dict[str, int],
        hidden_dim: int = 128,
        critic_hidden_dim: int = 256,
        num_layers: int = 2,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        noise_type: str = "ou",
        noise_std: float = 0.2,
        noise_decay: float = 0.9999,
        device: object = None,
        seed: int = 0,
        architecture: str = "shallow",
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        if device is None:
            device = torch.device("cpu") if torch is not None else "numpy"
        self.device = device
        self.obs_dims = obs_dims
        self.act_dims = act_dims
        self.roles = tuple(obs_dims.keys())  # Dynamic role inference
        self.agents: Dict[str, MADDPGAgent] = {}
        self.architecture = architecture

        # Compute total dims for centralized critic
        total_obs_dim = sum(obs_dims.values())
        total_act_dim = sum(act_dims.values())

        for i, role in enumerate(self.roles):
            if role not in act_dims:
                continue

            # Exploration noise
            if noise_type == "ou":
                noise = OUNoise(
                    size=act_dims[role],
                    sigma=noise_std,
                    sigma_decay=noise_decay,
                    seed=seed + i,
                )
            else:
                noise = GaussianNoise(
                    size=act_dims[role],
                    std=noise_std,
                    decay=noise_decay,
                    seed=seed + i,
                )

            if torch is None:
                self.agents[role] = _FallbackMADDPGAgent(
                    name=role,
                    obs_dim=obs_dims[role],
                    act_dim=act_dims[role],
                    noise=noise,
                    seed=seed + i,
                )
            else:
                actor, critic = self._create_networks(
                    obs_dim=obs_dims[role],
                    act_dim=act_dims[role],
                    total_obs_dim=total_obs_dim,
                    total_act_dim=total_act_dim,
                    hidden_dim=hidden_dim,
                    critic_hidden_dim=critic_hidden_dim,
                    num_layers=num_layers,
                    architecture=architecture,
                    num_heads=num_heads,
                    dropout=dropout,
                    device=device,
                )

                # Target networks (deep copy)
                target_actor = copy.deepcopy(actor)
                target_critic = copy.deepcopy(critic)

                self.agents[role] = MADDPGAgent(
                    name=role,
                    actor=actor,
                    critic=critic,
                    target_actor=target_actor,
                    target_critic=target_critic,
                    actor_opt=torch.optim.Adam(actor.parameters(), lr=actor_lr),
                    critic_opt=torch.optim.Adam(critic.parameters(), lr=critic_lr),
                    noise=noise,
                    device=device,
                )

    def _create_networks(
        self,
        obs_dim: int,
        act_dim: int,
        total_obs_dim: int,
        total_act_dim: int,
        hidden_dim: int,
        critic_hidden_dim: int,
        num_layers: int,
        architecture: str,
        num_heads: int,
        dropout: float,
        device: object,
    ) -> tuple[object, object]:
        """Create actor and critic networks based on architecture choice.

        Parameters
        ----------
        architecture : str
            "shallow" (default 128/256, 2-3层)
            "mlp" (256/512, 4-5层)
            "residual" (256/512, 4-5层, 残差连接)
            "transformer" (256/512, 4-5层, 多头注意力)

        Returns
        -------
        actor, critic : tuple[nn.Module, nn.Module]
        """
        _require_torch()
        if architecture == "shallow":
            # 原版浅层网络
            actor = DeterministicActor(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                activation="relu",
                use_layer_norm=True,
            ).to(device)

            critic = CentralizedCritic(
                total_obs_dim=total_obs_dim,
                total_act_dim=total_act_dim,
                hidden_dim=critic_hidden_dim,
                num_layers=num_layers + 1,
                activation="relu",
                use_layer_norm=True,
            ).to(device)

        else:
            # 深度网络（mlp/residual/transformer）
            from debate_rl_v2.agents.deep_maddpg_networks import (
                DeepDeterministicActor,
                DeepCentralizedCritic,
            )

            actor = DeepDeterministicActor(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                architecture=architecture,
                num_heads=num_heads,
                dropout=dropout,
                activation="relu",
            ).to(device)

            critic = DeepCentralizedCritic(
                total_obs_dim=total_obs_dim,
                total_act_dim=total_act_dim,
                hidden_dim=critic_hidden_dim,
                num_layers=num_layers + 1,
                architecture=architecture,
                num_heads=num_heads,
                dropout=dropout,
                activation="relu",
            ).to(device)

        return actor, critic

    def __getitem__(self, role: str) -> MADDPGAgent:
        return self.agents[role]

    def __iter__(self):
        return iter(self.agents.items())

    def __contains__(self, role: str) -> bool:
        return role in self.agents

    @property
    def agent_names(self) -> List[str]:
        return list(self.agents.keys())

    def soft_update_all(self, tau: float = 0.01) -> None:
        for agent in self.agents.values():
            agent.soft_update(tau)

    def reset_noise_all(self) -> None:
        for agent in self.agents.values():
            agent.reset_noise()

    def save_all(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        for role, agent in self.agents.items():
            agent.save(os.path.join(directory, f"maddpg_{role}.pt"))

    def load_all(self, directory: str) -> None:
        for role, agent in self.agents.items():
            path = os.path.join(directory, f"maddpg_{role}.pt")
            if os.path.exists(path):
                agent.load(path)

    def param_summary(self) -> Dict[str, int]:
        """Count trainable parameters per agent."""
        summary = {}
        for role, agent in self.agents.items():
            actor_p = sum(p.numel() for p in agent.actor.parameters())
            critic_p = sum(p.numel() for p in agent.critic.parameters())
            summary[role] = {"actor": actor_p, "critic": critic_p, "total": actor_p + critic_p}
        total = sum(v["total"] for v in summary.values())
        summary["grand_total"] = total
        return summary
