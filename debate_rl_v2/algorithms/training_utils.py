"""Training Utilities — LR scheduling, Q-value monitoring, early stopping.

Addresses training stability issues:
- No LR scheduling → cosine annealing
- No Q-value divergence detection → threshold-based detection
- No early stopping → eval reward patience
- No gradient norm monitoring → TensorBoard logging
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from debate_rl_v2.logging_config import get_logger

logger = get_logger("algorithms.training_utils")


# ── Learning Rate Schedulers ──

class CosineAnnealingScheduler:
    """Cosine annealing LR scheduler with optional warmup.

    lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        lr_min: float = 1e-6,
        warmup_steps: int = 0,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = max(total_steps, 1)
        self.lr_min = lr_min
        self.warmup_steps = warmup_steps
        self._initial_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step = 0

    def step(self) -> float:
        self._step += 1
        lr = self._compute_lr()
        for pg, initial_lr in zip(self.optimizer.param_groups, self._initial_lrs):
            if self._step <= self.warmup_steps:
                # Linear warmup
                pg["lr"] = initial_lr * (self._step / max(self.warmup_steps, 1))
            else:
                pg["lr"] = lr
        return lr

    def _compute_lr(self) -> float:
        if self._step <= self.warmup_steps:
            return self._initial_lrs[0] * (self._step / max(self.warmup_steps, 1))
        progress = (self._step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        progress = min(progress, 1.0)
        return self.lr_min + 0.5 * (self._initial_lrs[0] - self.lr_min) * (1 + math.cos(math.pi * progress))

    @property
    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


class LinearDecayScheduler:
    """Linear LR decay from initial to minimum."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        lr_min: float = 1e-6,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = max(total_steps, 1)
        self.lr_min = lr_min
        self._initial_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step = 0

    def step(self) -> float:
        self._step += 1
        progress = min(self._step / self.total_steps, 1.0)
        for pg, initial_lr in zip(self.optimizer.param_groups, self._initial_lrs):
            pg["lr"] = initial_lr + (self.lr_min - initial_lr) * progress
        return self.optimizer.param_groups[0]["lr"]

    @property
    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# ── Q-Value Divergence Detector ──

class QValueMonitor:
    """Monitors Q-values for divergence and triggers LR reduction.

    If Q-values grow beyond threshold for N consecutive steps,
    reduces learning rate by a factor.
    """

    def __init__(
        self,
        threshold: float = 100.0,
        patience: int = 10,
        lr_reduction_factor: float = 0.5,
        window_size: int = 50,
    ) -> None:
        self.threshold = threshold
        self.patience = patience
        self.lr_reduction_factor = lr_reduction_factor
        self._q_history: deque[float] = deque(maxlen=window_size)
        self._divergence_count = 0
        self._reductions = 0

    def update(self, q_value: float) -> bool:
        """Record a Q-value and check for divergence.

        Returns True if divergence detected (caller should reduce LR).
        """
        self._q_history.append(q_value)

        if len(self._q_history) < 2:
            return False

        # Check absolute magnitude
        if abs(q_value) > self.threshold:
            self._divergence_count += 1
            if self._divergence_count >= self.patience:
                self._divergence_count = 0
                self._reductions += 1
                logger.warning(
                    "Q-value divergence detected (|Q|=%.2f > %.2f). "
                    "Reduction #%d triggered.",
                    abs(q_value), self.threshold, self._reductions,
                )
                return True
        else:
            self._divergence_count = max(0, self._divergence_count - 1)

        # Check growth rate
        if len(self._q_history) >= 10:
            recent = list(self._q_history)[-10:]
            growth = abs(recent[-1]) / max(abs(recent[0]), 1e-8)
            if growth > 5.0:
                logger.warning("Q-value growth rate %.1fx in 10 steps", growth)
                return True

        return False

    @property
    def is_healthy(self) -> bool:
        if not self._q_history:
            return True
        return abs(self._q_history[-1]) < self.threshold

    @property
    def stats(self) -> dict[str, float]:
        if not self._q_history:
            return {"q_mean": 0.0, "q_std": 0.0, "q_max": 0.0}
        vals = list(self._q_history)
        return {
            "q_mean": float(np.mean(vals)),
            "q_std": float(np.std(vals)),
            "q_max": float(np.max(np.abs(vals))),
            "reductions": self._reductions,
        }


# ── Early Stopping ──

class EarlyStopping:
    """Stop training when eval reward stops improving.

    Parameters
    ----------
    patience : int
        Number of eval cycles without improvement before stopping.
    min_delta : float
        Minimum improvement to count as progress.
    """

    def __init__(self, patience: int = 50, min_delta: float = 0.01) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best_value = -float("inf")
        self._counter = 0
        self._should_stop = False

    def update(self, value: float) -> bool:
        """Record an eval metric. Returns True if training should stop."""
        if value > self._best_value + self.min_delta:
            self._best_value = value
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                self._should_stop = True
                logger.info(
                    "Early stopping triggered: no improvement for %d evals (best=%.4f)",
                    self.patience, self._best_value,
                )
                return True
        return False

    @property
    def should_stop(self) -> bool:
        return self._should_stop

    @property
    def best_value(self) -> float:
        return self._best_value


# ── Gradient Norm Monitor ──

def compute_gradient_norm(model: torch.nn.Module) -> float:
    """Compute the total gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


@dataclass
class GradientStats:
    """Accumulated gradient statistics for logging."""
    norms: deque = field(default_factory=lambda: deque(maxlen=100))

    def record(self, norm: float) -> None:
        self.norms.append(norm)

    @property
    def mean_norm(self) -> float:
        return float(np.mean(self.norms)) if self.norms else 0.0

    @property
    def max_norm(self) -> float:
        return float(np.max(self.norms)) if self.norms else 0.0

    def to_dict(self) -> dict[str, float]:
        return {"grad_norm_mean": self.mean_norm, "grad_norm_max": self.max_norm}
