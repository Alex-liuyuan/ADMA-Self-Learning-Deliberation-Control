"""Multi-Agent Experience Replay Buffer for MADDPG.

Off-policy replay buffer storing joint transitions (all agents' obs,
actions, rewards, next_obs, dones) for centralized training.

Supports:
  - Uniform random sampling
  - Prioritized Experience Replay (PER) via optional TD-error weights
  - Multi-agent joint storage (all agents share one buffer)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class MultiAgentReplayBuffer:
    """Joint replay buffer for all MADDPG agents.

    Stores transitions as:
        (obs_dict, act_dict, rew_dict, next_obs_dict, done)

    where each dict maps role_name → numpy array.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    agent_names : list of str
        Names of all agents (e.g. ["proposer", "challenger", "arbiter", "coordinator"]).
    obs_dims : dict
        {agent_name: obs_dim} for each agent.
    act_dims : dict
        {agent_name: act_dim} for each agent.
    seed : int
        Random seed for sampling.
    """

    def __init__(
        self,
        capacity: int,
        agent_names: List[str],
        obs_dims: Dict[str, int],
        act_dims: Dict[str, int],
        seed: int = 0,
    ) -> None:
        self.capacity = capacity
        self.agent_names = agent_names
        self.obs_dims = obs_dims
        self.act_dims = act_dims
        self.rng = np.random.default_rng(seed)

        # Pre-allocate numpy arrays for each agent
        self._obs = {
            name: np.zeros((capacity, obs_dims[name]), dtype=np.float32)
            for name in agent_names
        }
        self._act = {
            name: np.zeros((capacity, act_dims[name]), dtype=np.float32)
            for name in agent_names
        }
        self._rew = {
            name: np.zeros(capacity, dtype=np.float32)
            for name in agent_names
        }
        self._next_obs = {
            name: np.zeros((capacity, obs_dims[name]), dtype=np.float32)
            for name in agent_names
        }
        self._done = np.zeros(capacity, dtype=np.float32)

        # Optional: priority weights for PER
        self._priorities = np.ones(capacity, dtype=np.float32)

        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size >= self.capacity

    def add(
        self,
        obs: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        next_obs: Dict[str, np.ndarray],
        done: bool,
    ) -> None:
        """Store a joint transition from all agents."""
        idx = self._ptr
        for name in self.agent_names:
            self._obs[name][idx] = obs[name]
            self._act[name][idx] = actions[name]
            self._rew[name][idx] = rewards[name]
            self._next_obs[name][idx] = next_obs[name]
        self._done[idx] = float(done)
        self._priorities[idx] = max(self._priorities[: max(self._size, 1)].max(), 1.0)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
        prioritized: bool = False,
        alpha: float = 0.6,
        beta: float = 0.4,
    ) -> Dict[str, torch.Tensor]:
        """Sample a mini-batch of joint transitions.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.
        device : torch.device
            Target device for tensors.
        prioritized : bool
            Use prioritized sampling.
        alpha, beta : float
            PER hyperparameters.

        Returns
        -------
        batch : dict
            {
                "{agent}_obs": Tensor (B, obs_dim),
                "{agent}_act": Tensor (B, act_dim),
                "{agent}_rew": Tensor (B,),
                "{agent}_next_obs": Tensor (B, obs_dim),
                "done": Tensor (B,),
                "indices": ndarray (B,),
                "weights": Tensor (B,),   # importance sampling weights
            }
        """
        batch_size = min(batch_size, self._size)

        if prioritized:
            priorities = self._priorities[:self._size] ** alpha
            probs = priorities / priorities.sum()
            indices = self.rng.choice(self._size, size=batch_size, p=probs, replace=False)
            # Importance sampling weights
            weights = (self._size * probs[indices]) ** (-beta)
            weights = weights / weights.max()
            weights = torch.tensor(weights, dtype=torch.float32, device=device)
        else:
            indices = self.rng.choice(self._size, size=batch_size, replace=False)
            weights = torch.ones(batch_size, dtype=torch.float32, device=device)

        batch = {
            "done": torch.tensor(self._done[indices], dtype=torch.float32, device=device),
            "indices": indices,
            "weights": weights,
        }

        for name in self.agent_names:
            batch[f"{name}_obs"] = torch.tensor(
                self._obs[name][indices], dtype=torch.float32, device=device
            )
            batch[f"{name}_act"] = torch.tensor(
                self._act[name][indices], dtype=torch.float32, device=device
            )
            batch[f"{name}_rew"] = torch.tensor(
                self._rew[name][indices], dtype=torch.float32, device=device
            )
            batch[f"{name}_next_obs"] = torch.tensor(
                self._next_obs[name][indices], dtype=torch.float32, device=device
            )

        return batch

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities for PER (call after computing TD errors)."""
        self._priorities[indices] = np.abs(td_errors) + 1e-6

    def stats(self) -> Dict[str, float]:
        """Buffer usage statistics."""
        return {
            "size": self._size,
            "capacity": self.capacity,
            "utilization": self._size / self.capacity,
            "ptr": self._ptr,
        }
