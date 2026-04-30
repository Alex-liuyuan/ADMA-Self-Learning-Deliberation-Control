"""PPO Agent wrapper — manages actor, critic, and inference for one role."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from debate_rl_v2.agents.networks import (
    ActorNetwork,
    CriticNetwork,
    MetaActorNetwork,
    MetaCriticNetwork,
)
from debate_rl_v2.config import NetworkConfig, PPOConfig, HierarchicalConfig


@dataclass
class PPOAgent:
    """Single-role PPO agent with actor, critic, and optimizers."""
    name: str
    actor: nn.Module
    critic: nn.Module
    pi_opt: torch.optim.Optimizer
    vf_opt: torch.optim.Optimizer
    device: torch.device

    def act(
        self,
        obs: torch.Tensor,
        meta_action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """Sample action, return (action, log_prob, value)."""
        obs = obs.to(self.device)
        with torch.no_grad():
            if isinstance(self.actor, ActorNetwork) and meta_action is not None:
                dist = self.actor(obs.unsqueeze(0), meta_action.unsqueeze(0))
            else:
                dist = self.actor(obs.unsqueeze(0))
            if deterministic:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()
            logp = dist.log_prob(action)
            value = self.critic(obs.unsqueeze(0))
        return int(action.item()), float(logp.item()), float(value.item())

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        meta_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate batch: returns (log_probs, entropy, values)."""
        if isinstance(self.actor, ActorNetwork) and meta_actions is not None:
            dist = self.actor(obs, meta_actions)
        else:
            dist = self.actor(obs)
        logp = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        values = self.critic(obs)
        return logp, entropy, values

    def save(self, path: str) -> None:
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "pi_opt": self.pi_opt.state_dict(),
            "vf_opt": self.vf_opt.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.pi_opt.load_state_dict(ckpt["pi_opt"])
        self.vf_opt.load_state_dict(ckpt["vf_opt"])


class AgentGroup:
    """Manages all four role agents + optional coordinator meta-agent.

    Parameters
    ----------
    obs_dims : dict
        {role_name: obs_dim} for each role.
    act_dims : dict
        {role_name: act_dim} for each role.
    net_cfg : NetworkConfig
        Network architecture config.
    ppo_cfg : PPOConfig
        PPO hyperparameters.
    hier_cfg : HierarchicalConfig
        Hierarchical (coordinator) config.
    device : torch.device
        Compute device.
    """

    def __init__(
        self,
        obs_dims: Dict[str, int],
        act_dims: Dict[str, int],
        net_cfg: NetworkConfig,
        ppo_cfg: PPOConfig,
        hier_cfg: HierarchicalConfig,
        device: torch.device,
    ) -> None:
        self.device = device
        self.agents: Dict[str, PPOAgent] = {}

        nc = net_cfg
        # Base agents (proposer, challenger, arbiter)
        for role in ("proposer", "challenger", "arbiter"):
            actor = ActorNetwork(
                obs_dim=obs_dims[role],
                act_dim=act_dims[role],
                hidden_dim=nc.hidden_dim,
                num_layers=nc.num_layers,
                activation=nc.activation,
                meta_embed_dim=nc.meta_embed_dim,
                use_layer_norm=nc.use_layer_norm,
            ).to(device)
            critic = CriticNetwork(
                obs_dim=obs_dims[role],
                hidden_dim=nc.hidden_dim,
                num_layers=nc.num_layers,
                activation=nc.activation,
                use_layer_norm=nc.use_layer_norm,
            ).to(device)
            self.agents[role] = PPOAgent(
                name=role,
                actor=actor,
                critic=critic,
                pi_opt=torch.optim.Adam(actor.parameters(), lr=ppo_cfg.pi_lr),
                vf_opt=torch.optim.Adam(critic.parameters(), lr=ppo_cfg.vf_lr),
                device=device,
            )

        # Coordinator (meta-level, slower learning rate)
        meta_actor = MetaActorNetwork(
            obs_dim=obs_dims["coordinator"],
            act_dim=act_dims["coordinator"],
            hidden_dim=nc.hidden_dim,
            num_layers=nc.num_layers,
            activation=nc.activation,
            use_layer_norm=nc.use_layer_norm,
        ).to(device)
        meta_critic = MetaCriticNetwork(
            obs_dim=obs_dims["coordinator"],
            hidden_dim=nc.hidden_dim,
            num_layers=nc.num_layers,
            activation=nc.activation,
            use_layer_norm=nc.use_layer_norm,
        ).to(device)
        self.agents["coordinator"] = PPOAgent(
            name="coordinator",
            actor=meta_actor,
            critic=meta_critic,
            pi_opt=torch.optim.Adam(meta_actor.parameters(), lr=hier_cfg.meta_lr),
            vf_opt=torch.optim.Adam(meta_critic.parameters(), lr=hier_cfg.meta_lr),
            device=device,
        )

    def __getitem__(self, role: str) -> PPOAgent:
        return self.agents[role]

    def __iter__(self):
        return iter(self.agents.items())

    def save_all(self, directory: str) -> None:
        import os
        os.makedirs(directory, exist_ok=True)
        for role, agent in self.agents.items():
            agent.save(os.path.join(directory, f"{role}.pt"))

    def load_all(self, directory: str) -> None:
        import os
        for role, agent in self.agents.items():
            path = os.path.join(directory, f"{role}.pt")
            if os.path.exists(path):
                agent.load(path)
