"""Dynamic adversarial intensity controller without hard torch dependency."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


@dataclass
class IntensityHistory:
    disagreement: List[float] = field(default_factory=list)
    delta_disagreement: List[float] = field(default_factory=list)
    time_pressure: List[float] = field(default_factory=list)
    target_intensity: List[float] = field(default_factory=list)
    lambda_adv: List[float] = field(default_factory=list)


class AdversarialIntensityController:
    def __init__(
        self,
        eta: float = 0.2,
        alpha: float = 0.6,
        omega: float = 0.7,
        max_steps: int = 30,
    ) -> None:
        self._eta = float(np.clip(eta, 0.01, 0.99))
        self._alpha = float(np.clip(alpha, 0.01, 0.99))
        self._omega = float(np.clip(omega, 0.01, 0.99))
        self.max_steps = max_steps
        self.lambda_adv: float = 0.5
        self.last_disagreement: float = 1.0
        self.history = IntensityHistory()

    @property
    def eta(self) -> float:
        return self._eta

    @eta.setter
    def eta(self, value: float) -> None:
        self._eta = float(np.clip(value, 0.01, 0.99))

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._alpha = float(np.clip(value, 0.01, 0.99))

    @property
    def omega(self) -> float:
        return self._omega

    @omega.setter
    def omega(self, value: float) -> None:
        self._omega = float(np.clip(value, 0.01, 0.99))

    def reset(self) -> None:
        self.lambda_adv = 0.5
        self.last_disagreement = 1.0
        self.history = IntensityHistory()

    def compute_disagreement(self, p_embed: np.ndarray, c_embed: np.ndarray) -> float:
        return 1.0 - cosine_similarity(p_embed, c_embed)

    def time_pressure(self, t: int) -> float:
        ratio = t / max(self.max_steps, 1)
        return min(1.0, ratio ** 2)

    def update(self, disagreement: float, t: int, *, quality: float = 0.0) -> float:
        delta_d = disagreement - self.last_disagreement
        tp = self.time_pressure(t)
        effective_tp = tp * max(0.0, 1.0 - quality * 0.8)
        u = self.omega * (
            self.alpha * disagreement + (1.0 - self.alpha) * delta_d
        ) + (1.0 - self.omega) * effective_tp
        sigma_u = _sigmoid(u)
        self.lambda_adv = self.lambda_adv + self.eta * (sigma_u - self.lambda_adv)
        self.last_disagreement = disagreement

        self.history.disagreement.append(disagreement)
        self.history.delta_disagreement.append(delta_d)
        self.history.time_pressure.append(tp)
        self.history.target_intensity.append(u)
        self.history.lambda_adv.append(self.lambda_adv)
        return self.lambda_adv

    @property
    def current_intensity(self) -> float:
        return self.lambda_adv


if torch is not None:
    class SemanticEmbedder(torch.nn.Module):
        def __init__(self, input_dim: int, embed_dim: int = 32) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, embed_dim),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            emb = self.net(x)
            return torch.nn.functional.normalize(emb, dim=-1)
else:
    class SemanticEmbedder:
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("SemanticEmbedder requires torch to be installed")
