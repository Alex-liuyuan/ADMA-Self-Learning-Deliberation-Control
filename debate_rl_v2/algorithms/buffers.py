"""Enhanced rollout buffers with GAE computation — Section 5.5.

Supports per-role experience storage and Shapley-corrected advantages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

import numpy as np


@dataclass
class RolloutBuffer:
    """Stores rollout data for a single role agent.

    Supports GAE advantage computation and batch tensor generation.
    """
    obs: List[np.ndarray] = field(default_factory=list)
    act: List[int] = field(default_factory=list)
    rew: List[float] = field(default_factory=list)
    val: List[float] = field(default_factory=list)
    logp: List[float] = field(default_factory=list)
    done: List[float] = field(default_factory=list)

    # Optional: Shapley correction values
    shapley_correction: List[float] = field(default_factory=list)

    def add(
        self,
        obs: np.ndarray,
        act: int,
        rew: float,
        val: float,
        logp: float,
        done: float,
    ) -> None:
        self.obs.append(obs)
        self.act.append(int(act))
        self.rew.append(float(rew))
        self.val.append(float(val))
        self.logp.append(float(logp))
        self.done.append(float(done))

    def add_shapley(self, correction: float) -> None:
        self.shapley_correction.append(float(correction))

    def clear(self) -> None:
        self.obs.clear()
        self.act.clear()
        self.rew.clear()
        self.val.clear()
        self.logp.clear()
        self.done.clear()
        self.shapley_correction.clear()

    def __len__(self) -> int:
        return len(self.rew)

    def compute_gae(
        self,
        gamma: float,
        lam: float,
        last_value: float = 0.0,
        shapley_coef: float = 0.0,
    ):
        """Compute generalized advantage estimation (GAE-λ).

        Optionally applies Shapley value correction (Section 5.5):
            Ã^i = A^i + κ · (φ_i − φ̄_i)

        Returns
        -------
        returns : np.ndarray
            Target return values.
        advantages : np.ndarray
            (Optionally Shapley-corrected) advantages.
        """
        n = len(self.rew)
        rews = np.array(self.rew, dtype=np.float32)
        vals = np.array(self.val, dtype=np.float32)
        dones = np.array(self.done, dtype=np.float32)

        adv = np.zeros(n, dtype=np.float32)
        lastgaelam = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_val = last_value
                next_nonterminal = 1.0 - dones[t]
            else:
                next_val = vals[t + 1]
                next_nonterminal = 1.0 - dones[t]

            delta = rews[t] + gamma * next_val * next_nonterminal - vals[t]
            adv[t] = lastgaelam = delta + gamma * lam * next_nonterminal * lastgaelam

        # Apply Shapley correction
        if shapley_coef > 0 and len(self.shapley_correction) == n:
            sc = np.array(self.shapley_correction, dtype=np.float32)
            mean_sc = sc.mean()
            adv = adv + shapley_coef * (sc - mean_sc)

        returns = adv + vals
        return returns, adv

    def get_tensors(self, device: str | Any = "cpu"):
        """Convert stored data to PyTorch tensors.

        Returns
        -------
        obs : Tensor of shape (N, obs_dim)
        act : LongTensor of shape (N,)
        ret : Tensor of shape (N,)
        adv : Tensor of shape (N,)
        old_logp : Tensor of shape (N,)
        """
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on optional runtime
            raise ImportError(
                "RolloutBuffer.get_tensors requires torch. "
                "Install the package with RL dependencies before training."
            ) from exc

        obs = torch.tensor(np.array(self.obs), dtype=torch.float32, device=device)
        act = torch.tensor(self.act, dtype=torch.int64, device=device)
        old_logp = torch.tensor(self.logp, dtype=torch.float32, device=device)
        return obs, act, old_logp


class MultiRoleBuffer:
    """Manages rollout buffers for all roles simultaneously.

    Generalized to support arbitrary role names. Defaults to legacy
    debate roles for backward compatibility.
    """

    # Legacy constant kept for backward compatibility
    ROLES = ("proposer", "challenger", "arbiter", "coordinator")

    def __init__(self, roles: tuple[str, ...] | None = None) -> None:
        self._roles = roles or self.ROLES
        self.buffers = {role: RolloutBuffer() for role in self._roles}

    def __getitem__(self, role: str) -> RolloutBuffer:
        if role not in self.buffers:
            # Lazily create buffer for unknown roles
            self.buffers[role] = RolloutBuffer()
        return self.buffers[role]

    @property
    def roles(self) -> tuple[str, ...]:
        return self._roles

    def clear_all(self) -> None:
        for buf in self.buffers.values():
            buf.clear()

    def total_steps(self) -> int:
        return sum(len(b) for b in self.buffers.values())
