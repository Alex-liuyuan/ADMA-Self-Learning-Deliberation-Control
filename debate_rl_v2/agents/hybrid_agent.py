"""混合动作空间智能体组 — Proposer/Challenger PPO + Arbiter/Coordinator DDPG。

将离散动作（PPO）和连续动作（DDPG）统一管理，支持 hybrid 训练模式。
Proposer/Challenger 保持大规模离散动作空间（proposal_dim × proposal_values），
Arbiter/Coordinator 使用连续动作空间实现精细参数调整。
"""

from __future__ import annotations

import copy
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from debate_rl_v2.agents.ppo_agent import PPOAgent
from debate_rl_v2.agents.networks import ActorNetwork, CriticNetwork
from debate_rl_v2.agents.maddpg_agent import MADDPGAgent
from debate_rl_v2.agents.maddpg_networks import (
    DeterministicActor,
    CentralizedCritic,
    GaussianNoise,
    OUNoise,
)
from debate_rl_v2.config import (
    NetworkConfig,
    PPOConfig,
    HierarchicalConfig,
    ContinuousAgentConfig,
)


class HybridAgentGroup:
    """混合动作空间智能体组。

    Proposer/Challenger: PPO 离散动作（ActorNetwork + CriticNetwork）
    Arbiter/Coordinator: DDPG 连续动作（DeterministicActor + CentralizedCritic）

    Parameters
    ----------
    obs_dims : dict
        {role: obs_dim} 每个角色的观测维度。
    act_dims : dict
        {role: act_dim} 每个角色的动作维度。
    net_cfg : NetworkConfig
        网络架构配置。
    ppo_cfg : PPOConfig
        PPO 超参数（用于 proposer/challenger）。
    hier_cfg : HierarchicalConfig
        分层配置。
    cont_cfg : ContinuousAgentConfig
        连续动作智能体配置（用于 arbiter/coordinator）。
    device : torch.device
        计算设备。
    """

    PPO_ROLES = ("proposer", "challenger")
    DDPG_ROLES = ("arbiter", "coordinator")

    def __init__(
        self,
        obs_dims: Dict[str, int],
        act_dims: Dict[str, int],
        net_cfg: NetworkConfig,
        ppo_cfg: PPOConfig,
        hier_cfg: HierarchicalConfig,
        cont_cfg: Optional[ContinuousAgentConfig] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.device = device
        self.obs_dims = obs_dims
        self.act_dims = act_dims
        cont_cfg = cont_cfg or ContinuousAgentConfig()

        nc = net_cfg
        self.ppo_agents: Dict[str, PPOAgent] = {}
        self.ddpg_agents: Dict[str, MADDPGAgent] = {}

        # --- PPO 智能体：Proposer / Challenger ---
        for role in self.PPO_ROLES:
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
                hidden_dim=nc.critic_hidden_dim,
                num_layers=nc.num_layers,
                activation=nc.activation,
                use_layer_norm=nc.use_layer_norm,
            ).to(device)
            self.ppo_agents[role] = PPOAgent(
                name=role,
                actor=actor,
                critic=critic,
                pi_opt=torch.optim.Adam(actor.parameters(), lr=ppo_cfg.pi_lr),
                vf_opt=torch.optim.Adam(critic.parameters(), lr=ppo_cfg.vf_lr),
                device=device,
            )

        # --- DDPG 智能体：Arbiter / Coordinator ---
        # Arbiter 和 Coordinator 各自独立的 critic（非集中式）
        for role in self.DDPG_ROLES:
            actor = DeterministicActor(
                obs_dim=obs_dims[role],
                act_dim=act_dims[role],
                hidden_dim=cont_cfg.actor_hidden_dim,
                num_layers=nc.num_layers,
                activation="relu",
                use_layer_norm=nc.use_layer_norm,
            ).to(device)
            # Critic 输入：自身观测 + 自身动作
            critic = CentralizedCritic(
                total_obs_dim=obs_dims[role],
                total_act_dim=act_dims[role],
                hidden_dim=cont_cfg.critic_hidden_dim,
                num_layers=nc.num_layers + 1,
                activation="relu",
                use_layer_norm=nc.use_layer_norm,
            ).to(device)
            target_actor = copy.deepcopy(actor)
            target_critic = copy.deepcopy(critic)

            noise = GaussianNoise(
                size=act_dims[role],
                std=cont_cfg.noise_std,
                decay=cont_cfg.noise_decay,
            )

            self.ddpg_agents[role] = MADDPGAgent(
                name=role,
                actor=actor,
                critic=critic,
                target_actor=target_actor,
                target_critic=target_critic,
                actor_opt=torch.optim.Adam(actor.parameters(), lr=cont_cfg.actor_lr),
                critic_opt=torch.optim.Adam(critic.parameters(), lr=cont_cfg.critic_lr),
                noise=noise,
                device=device,
            )

    def __getitem__(self, role: str):
        """按角色名获取智能体（PPOAgent 或 MADDPGAgent）。"""
        if role in self.ppo_agents:
            return self.ppo_agents[role]
        if role in self.ddpg_agents:
            return self.ddpg_agents[role]
        raise KeyError(f"Unknown role: {role}")

    def __contains__(self, role: str) -> bool:
        return role in self.ppo_agents or role in self.ddpg_agents

    def __iter__(self):
        for role, agent in self.ppo_agents.items():
            yield role, agent
        for role, agent in self.ddpg_agents.items():
            yield role, agent

    def is_ppo(self, role: str) -> bool:
        return role in self.ppo_agents

    def is_ddpg(self, role: str) -> bool:
        return role in self.ddpg_agents

    def save_all(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        for role, agent in self.ppo_agents.items():
            agent.save(os.path.join(directory, f"{role}.pt"))
        for role, agent in self.ddpg_agents.items():
            agent.save(os.path.join(directory, f"{role}_ddpg.pt"))

    def load_all(self, directory: str) -> None:
        for role, agent in self.ppo_agents.items():
            path = os.path.join(directory, f"{role}.pt")
            if os.path.exists(path):
                agent.load(path)
        for role, agent in self.ddpg_agents.items():
            path = os.path.join(directory, f"{role}_ddpg.pt")
            if os.path.exists(path):
                agent.load(path)

    def soft_update_ddpg(self, tau: float = 0.005) -> None:
        """对 DDPG 智能体执行 target network 软更新。"""
        for agent in self.ddpg_agents.values():
            agent.soft_update(tau)

    def param_summary(self) -> Dict[str, dict]:
        """统计各智能体参数量。"""
        summary = {}
        total = 0
        for role, agent in self.ppo_agents.items():
            actor_p = sum(p.numel() for p in agent.actor.parameters())
            critic_p = sum(p.numel() for p in agent.critic.parameters())
            summary[role] = {"actor": actor_p, "critic": critic_p, "total": actor_p + critic_p}
            total += actor_p + critic_p
        for role, agent in self.ddpg_agents.items():
            actor_p = sum(p.numel() for p in agent.actor.parameters())
            critic_p = sum(p.numel() for p in agent.critic.parameters())
            target_p = (
                sum(p.numel() for p in agent.target_actor.parameters())
                + sum(p.numel() for p in agent.target_critic.parameters())
            )
            summary[role] = {
                "actor": actor_p,
                "critic": critic_p,
                "target": target_p,
                "total": actor_p + critic_p + target_p,
            }
            total += actor_p + critic_p + target_p
        summary["grand_total"] = total
        return summary
