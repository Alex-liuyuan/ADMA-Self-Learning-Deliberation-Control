"""Multi-Agent PPO (MAPPO) update — Section 5.

Implements centralized-training-decentralized-execution PPO with
clipped objective, value function loss, and entropy bonus.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from debate_rl_v2.agents.ppo_agent import PPOAgent
from debate_rl_v2.algorithms.buffers import RolloutBuffer
from debate_rl_v2.config import PPOConfig


def mappo_update(
    agent: PPOAgent,
    buffer: RolloutBuffer,
    cfg: PPOConfig,
    last_value: float = 0.0,
    shapley_coef: float = 0.0,
) -> Dict[str, float]:
    """Run PPO update for a single role agent.

    Parameters
    ----------
    agent : PPOAgent
        The agent to update.
    buffer : RolloutBuffer
        Collected experience for this role.
    cfg : PPOConfig
        PPO hyperparameters.
    last_value : float
        Bootstrap value for the last state (from critic).
    shapley_coef : float
        Shapley correction coefficient κ (0 = no correction).

    Returns
    -------
    stats : dict
        Training statistics {pi_loss, vf_loss, entropy, approx_kl}.
    """
    if len(buffer) == 0:
        return {"pi_loss": 0.0, "vf_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}

    device = agent.device
    returns, advantages = buffer.compute_gae(
        gamma=cfg.gamma,
        lam=cfg.gae_lambda,
        last_value=last_value,
        shapley_coef=shapley_coef,
    )

    obs, act, old_logp = buffer.get_tensors(device)
    ret_t = torch.tensor(returns, dtype=torch.float32, device=device)
    adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)

    # Normalize advantages
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    n = len(returns)
    indices = np.arange(n)
    total_pi_loss = 0.0
    total_vf_loss = 0.0
    total_entropy = 0.0
    total_approx_kl = 0.0
    num_updates = 0

    for _ in range(cfg.train_epochs):
        np.random.shuffle(indices)

        for start in range(0, n, cfg.minibatch_size):
            mb_idx = indices[start: start + cfg.minibatch_size]
            mb_obs = obs[mb_idx]
            mb_act = act[mb_idx]
            mb_ret = ret_t[mb_idx]
            mb_adv = adv_t[mb_idx]
            mb_old_logp = old_logp[mb_idx]

            # Forward pass
            logp, entropy, values = agent.evaluate_actions(mb_obs, mb_act)

            # Policy loss (PPO clipped objective)
            ratio = torch.exp(logp - mb_old_logp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * mb_adv
            pi_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            vf_loss = nn.functional.mse_loss(values, mb_ret)

            # Total loss
            loss = pi_loss - cfg.entropy_coef * entropy + 0.5 * vf_loss

            # Update actor
            agent.pi_opt.zero_grad()
            agent.vf_opt.zero_grad()
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(agent.actor.parameters(), cfg.max_grad_norm)
            nn.utils.clip_grad_norm_(agent.critic.parameters(), cfg.max_grad_norm)

            agent.pi_opt.step()
            agent.vf_opt.step()

            # Statistics
            with torch.no_grad():
                approx_kl = ((ratio - 1) - (ratio.log())).mean().item()
            total_pi_loss += pi_loss.item()
            total_vf_loss += vf_loss.item()
            total_entropy += entropy.item()
            total_approx_kl += approx_kl
            num_updates += 1

    num_updates = max(num_updates, 1)
    return {
        "pi_loss": total_pi_loss / num_updates,
        "vf_loss": total_vf_loss / num_updates,
        "entropy": total_entropy / num_updates,
        "approx_kl": total_approx_kl / num_updates,
    }
