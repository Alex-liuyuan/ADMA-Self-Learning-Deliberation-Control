"""Probabilistic soft switch without hard torch dependency."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))


@dataclass
class SwitchState:
    p_arbiter_intervene: float = 0.0
    p_challenger_boost: float = 0.0
    mode: str = "standard"


class SoftSwitchController:
    def __init__(
        self,
        tau_low: float = 0.3,
        tau_high: float = 0.7,
        steepness: float = 10.0,
    ) -> None:
        self._tau_low = float(np.clip(tau_low, 0.05, 0.6))
        self._tau_high = float(np.clip(tau_high, 0.4, 0.95))
        self._steepness = max(0.1, float(steepness))

    @property
    def tau_low(self) -> float:
        return self._tau_low

    @tau_low.setter
    def tau_low(self, value: float) -> None:
        self._tau_low = float(np.clip(value, 0.05, 0.6))

    @property
    def tau_high(self) -> float:
        return self._tau_high

    @tau_high.setter
    def tau_high(self, value: float) -> None:
        self._tau_high = float(np.clip(value, 0.4, 0.95))

    @property
    def steepness(self) -> float:
        return self._steepness

    def get_probabilities(self, lambda_adv: float) -> Tuple[float, float]:
        k = self.steepness
        p_arb = _sigmoid(k * (lambda_adv - self.tau_high))
        p_chal = _sigmoid(-k * (lambda_adv - self.tau_low))
        return p_arb, p_chal

    def decide(self, lambda_adv: float, rng: np.random.Generator) -> SwitchState:
        p_arb, p_chal = self.get_probabilities(lambda_adv)
        state = SwitchState(
            p_arbiter_intervene=p_arb,
            p_challenger_boost=p_chal,
        )
        if lambda_adv > self.tau_high and rng.random() < p_arb:
            state.mode = "arbiter_intervene"
        elif lambda_adv < self.tau_low and rng.random() < p_chal:
            state.mode = "challenger_boost"
        return state

    def update_thresholds(self, tau_low: float, tau_high: float) -> None:
        self.tau_low = tau_low
        self.tau_high = tau_high
