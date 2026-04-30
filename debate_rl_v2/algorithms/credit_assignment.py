"""Shapley Value Credit Assignment — Section 5.5.

Implements Monte Carlo approximation of Shapley values for fair credit
assignment in multi-agent debate.

The Shapley value for agent i is:
    φ_i = Σ_{S⊆N\\{i}} |S|!(|N|−|S|−1)! / |N|! · (v(S∪{i}) − v(S))

where v(S) = E_{a^j~π^j, j∉S}[Q_tot(s, a_S, a_{−S})]

The advantage function is corrected as:
    Ã^i = A^i + κ · (φ_i − φ̄_i)

Theorem 8: Shapley values satisfy efficiency, symmetry, dummy player,
and additivity — the unique solution with these properties.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional

import numpy as np


ROLES = ("proposer", "challenger", "arbiter", "coordinator")


class ShapleyCredit:
    """Monte Carlo Shapley value approximation for credit assignment.

    Parameters
    ----------
    num_agents : int
        Number of agents (default 4).
    num_samples : int
        Number of Monte Carlo permutation samples.
    roles : tuple[str, ...] | None
        Role names. Defaults to legacy debate roles.
    """

    def __init__(
        self,
        num_agents: int = 4,
        num_samples: int = 100,
        roles: tuple[str, ...] | None = None,
    ) -> None:
        self.num_agents = num_agents
        self.num_samples = num_samples
        self._roles = roles or ROLES[:num_agents]
        self._history: Dict[str, List[float]] = {r: [] for r in self._roles}

    def compute_shapley_values(
        self,
        values_per_coalition: Dict[frozenset, float],
    ) -> Dict[str, float]:
        """Compute exact Shapley values from coalition values.

        Parameters
        ----------
        values_per_coalition : dict
            Maps frozenset of role names to coalition value v(S).

        Returns
        -------
        shapley_values : dict
            {role_name: φ_i}
        """
        n = self.num_agents
        shapley = {r: 0.0 for r in self._roles}

        for i, role in enumerate(self._roles):
            others = [r for r in self._roles if r != role]
            for size in range(n):
                for subset in itertools.combinations(others, size):
                    s = frozenset(subset)
                    s_with_i = s | {role}
                    marginal = values_per_coalition.get(s_with_i, 0.0) - values_per_coalition.get(s, 0.0)
                    weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
                    shapley[role] += weight * marginal

        return shapley

    def compute_mc_shapley(
        self,
        role_rewards: Dict[str, float],
        total_reward: float,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, float]:
        """Monte Carlo approximation of Shapley values.

        Uses random permutations to approximate the marginal contribution
        of each agent. Efficient for the 4-agent case.

        Parameters
        ----------
        role_rewards : dict
            Per-role rewards for the current episode.
        total_reward : float
            Total team reward.

        Returns
        -------
        shapley_values : dict
            {role_name: φ_i}
        """
        if rng is None:
            rng = np.random.default_rng()

        marginals = {r: [] for r in self._roles}
        roles_list = list(self._roles)

        for _ in range(self.num_samples):
            perm = rng.permutation(roles_list).tolist()
            cumulative = 0.0

            for role in perm:
                # Value with this agent = cumulative + its contribution
                new_cumulative = cumulative + role_rewards.get(role, 0.0)
                marginals[role].append(new_cumulative - cumulative)
                cumulative = new_cumulative

        shapley_vals = {r: float(np.mean(m)) for r, m in marginals.items()}
        return shapley_vals

    def compute_corrections(
        self,
        buffers,
        agents,
        env,
    ) -> Dict[str, List[float]]:
        """Compute Shapley correction values for stored experiences.

        Returns per-role lists of correction values to be added to
        the advantage function.

        This is a simplified version that uses the final episode rewards
        as a proxy for coalition values.
        """
        corrections: Dict[str, List[float]] = {r: [] for r in self._roles}

        # Get total rewards from buffers
        role_totals = {}
        for role in self._roles:
            buf = buffers[role]
            if len(buf) > 0:
                role_totals[role] = sum(buf.rew)
            else:
                role_totals[role] = 0.0

        total = sum(role_totals.values())
        if abs(total) < 1e-8:
            # No meaningful signal
            for role in self._roles:
                corrections[role] = [0.0] * len(buffers[role])
            return corrections

        # Compute MC Shapley values
        shapley_vals = self.compute_mc_shapley(role_totals, total)

        # Update running history
        for role in self._roles:
            self._history[role].append(shapley_vals.get(role, 0.0))

        # Correction = φ_i − φ̄_i
        for role in self._roles:
            hist = self._history[role]
            mean_shapley = np.mean(hist) if hist else 0.0
            correction = shapley_vals.get(role, 0.0) - mean_shapley
            # Apply same correction to all steps in this episode
            corrections[role] = [correction] * len(buffers[role])

        return corrections
