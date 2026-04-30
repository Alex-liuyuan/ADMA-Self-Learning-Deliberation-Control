"""Devil's advocate verifier without hard torch dependency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class VerificationResult:
    is_robust: bool
    max_challenge_disagreement: float
    challenges_issued: int
    consensus_confirmed: bool


class DevilAdvocateVerifier:
    def __init__(
        self,
        disagreement_threshold: float = 0.2,
        update_threshold: float = 0.1,
        stability_window: int = 3,
        reactivation_threshold: float = 0.35,
        max_challenges: int = 3,
    ) -> None:
        self._eps_d = float(np.clip(disagreement_threshold, 0.01, 0.99))
        self._eps_p = float(np.clip(update_threshold, 0.01, 0.99))
        self._delta = float(np.clip(reactivation_threshold, 0.01, 0.99))
        self.stability_window = stability_window
        self.max_challenges = max_challenges
        self._stable_count = 0
        self._active = False
        self._challenge_count = 0
        self._challenge_disagreements: List[float] = []

    @property
    def eps_d(self) -> float:
        return self._eps_d

    @property
    def eps_p(self) -> float:
        return self._eps_p

    @property
    def delta(self) -> float:
        return self._delta

    def reset(self) -> None:
        self._stable_count = 0
        self._active = False
        self._challenge_count = 0
        self._challenge_disagreements.clear()

    @property
    def is_active(self) -> bool:
        return self._active

    def check_stability(self, disagreement: float, belief_update_norm: float) -> bool:
        if self._active:
            return True
        if disagreement < self.eps_d and belief_update_norm < self.eps_p:
            self._stable_count += 1
        else:
            self._stable_count = 0

        if self._stable_count >= self.stability_window:
            self._active = True
            self._challenge_count = 0
            self._challenge_disagreements.clear()
            return True
        return False

    def process_challenge(self, challenge_disagreement: float) -> VerificationResult:
        self._challenge_count += 1
        self._challenge_disagreements.append(challenge_disagreement)

        if challenge_disagreement > self.delta:
            self._active = False
            self._stable_count = 0
            return VerificationResult(False, challenge_disagreement, self._challenge_count, False)

        if self._challenge_count >= self.max_challenges:
            self._active = False
            return VerificationResult(
                True,
                max(self._challenge_disagreements),
                self._challenge_count,
                True,
            )

        return VerificationResult(True, challenge_disagreement, self._challenge_count, False)
