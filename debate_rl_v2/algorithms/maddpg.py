"""MADDPG (Multi-Agent Deep Deterministic Policy Gradient) Algorithm.

Implements the CTDE (Centralized Training with Decentralized Execution)
framework from Lowe et al. (2017), adapted for the adversarial
collaborative debate system.

Key features:
  - Centralized critic: Q_i(o_1,...,o_N, a_1,...,a_N) for each agent i
  - Decentralized actors: μ_i(o_i) for each agent i
  - Off-policy training with experience replay
  - Soft target network updates (Polyak averaging)
  - Optional prioritized experience replay (PER)
  - Gradient clipping for stability

Training loop per batch:
  1. Sample batch from replay buffer
  2. For each agent i:
     a. Compute target Q using target networks
     b. Update critic by minimizing MSE(Q, y)
     c. Update actor by maximizing Q w.r.t. μ_i(o_i)
  3. Soft-update all target networks
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from debate_rl_v2.agents.maddpg_agent import MADDPGAgent, MADDPGAgentGroup
from debate_rl_v2.algorithms.replay_buffer import MultiAgentReplayBuffer


def maddpg_update(
    agents: MADDPGAgentGroup,
    buffer: MultiAgentReplayBuffer,
    batch_size: int = 256,
    gamma: float = 0.95,
    tau: float = 0.01,
    gradient_clip: float = 0.5,
    prioritized: bool = False,
    per_alpha: float = 0.6,
    per_beta: float = 0.4,
) -> Dict[str, Dict[str, float]]:
    """Perform one MADDPG update step for all agents.

    Parameters
    ----------
    agents : MADDPGAgentGroup
        All MADDPG agents.
    buffer : MultiAgentReplayBuffer
        Shared replay buffer.
    batch_size : int
        Mini-batch size.
    gamma : float
        Discount factor.
    tau : float
        Soft target update rate.
    gradient_clip : float
        Max gradient norm.
    prioritized : bool
        Use prioritized experience replay.
    per_alpha, per_beta : float
        PER hyperparameters.

    Returns
    -------
    stats : dict
        {agent_name: {critic_loss, actor_loss, q_mean, target_q_mean}}
    """
    if len(buffer) < batch_size:
        return {}

    # Sample batch
    batch = buffer.sample(
        batch_size,
        device=list(agents.agents.values())[0].device,
        prioritized=prioritized,
        alpha=per_alpha,
        beta=per_beta,
    )

    agent_names = agents.agent_names
    stats = {}

    # Gather all observations and actions from batch
    all_obs = torch.cat([batch[f"{n}_obs"] for n in agent_names], dim=-1)
    all_acts = torch.cat([batch[f"{n}_act"] for n in agent_names], dim=-1)
    all_next_obs = torch.cat([batch[f"{n}_next_obs"] for n in agent_names], dim=-1)
    done = batch["done"]
    weights = batch["weights"]

    # Compute target actions for all agents (using target actors)
    with torch.no_grad():
        target_actions = []
        for name in agent_names:
            target_act = agents[name].target_act_batch(batch[f"{name}_next_obs"])
            target_actions.append(target_act)
        all_target_acts = torch.cat(target_actions, dim=-1)

    # Update each agent
    for name in agent_names:
        agent = agents[name]
        agent_stats = _update_single_agent(
            agent=agent,
            agents=agents,
            agent_names=agent_names,
            batch=batch,
            all_obs=all_obs,
            all_acts=all_acts,
            all_next_obs=all_next_obs,
            all_target_acts=all_target_acts,
            done=done,
            weights=weights,
            gamma=gamma,
            gradient_clip=gradient_clip,
        )
        stats[name] = agent_stats

    # Soft update ALL target networks
    agents.soft_update_all(tau)

    # Update PER priorities
    if prioritized and "indices" in batch:
        # Use mean TD error across agents
        td_errors = np.zeros(batch_size)
        for name in agent_names:
            td_errors += np.abs(stats[name].get("td_errors", np.zeros(batch_size)))
        td_errors /= len(agent_names)
        buffer.update_priorities(batch["indices"], td_errors)

    return stats


def _update_single_agent(
    agent: MADDPGAgent,
    agents: MADDPGAgentGroup,
    agent_names: List[str],
    batch: Dict[str, torch.Tensor],
    all_obs: torch.Tensor,
    all_acts: torch.Tensor,
    all_next_obs: torch.Tensor,
    all_target_acts: torch.Tensor,
    done: torch.Tensor,
    weights: torch.Tensor,
    gamma: float,
    gradient_clip: float,
) -> Dict[str, float]:
    """Update critic and actor for a single agent."""
    name = agent.name

    # ============================================
    # 1) Critic Update: minimize MSE(Q, y)
    # ============================================

    # Target Q-value: y = r_i + γ * Q_target(o', a'_all) * (1 - done)
    with torch.no_grad():
        target_q = agent.target_critic(all_next_obs, all_target_acts)
        target_q = target_q.squeeze(-1)
        reward = batch[f"{name}_rew"]
        y = reward + gamma * target_q * (1.0 - done)

    # Current Q-value
    current_q = agent.critic(all_obs, all_acts).squeeze(-1)

    # Weighted MSE loss (weights from PER)
    td_error = y - current_q
    critic_loss = (weights * td_error.pow(2)).mean()

    agent.critic_opt.zero_grad()
    critic_loss.backward()
    nn.utils.clip_grad_norm_(agent.critic.parameters(), gradient_clip)
    agent.critic_opt.step()

    # ============================================
    # 2) Actor Update: maximize Q w.r.t. μ_i(o_i)
    # ============================================

    # Replace agent i's action with current policy output
    current_actions = []
    for n in agent_names:
        if n == name:
            # Use current actor for this agent (with gradient)
            act = agent.actor(batch[f"{n}_obs"])
        else:
            # Use stored actions for other agents (detached)
            act = batch[f"{n}_act"]
        current_actions.append(act)
    all_current_acts = torch.cat(current_actions, dim=-1)

    # Actor loss = -E[Q(o, a_all with a_i = μ_i(o_i))]
    actor_loss = -agent.critic(all_obs, all_current_acts).mean()

    agent.actor_opt.zero_grad()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(agent.actor.parameters(), gradient_clip)
    agent.actor_opt.step()

    return {
        "critic_loss": critic_loss.item(),
        "actor_loss": actor_loss.item(),
        "q_mean": current_q.mean().item(),
        "target_q_mean": y.mean().item(),
        "td_errors": td_error.detach().cpu().numpy(),
    }
