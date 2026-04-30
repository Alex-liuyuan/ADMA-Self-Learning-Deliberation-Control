"""Base Observation Encoder — domain-agnostic observation encoding.

Provides the abstract interface for encoding shared and role-specific
observations that feed into RL agents. Scenario-specific encoders
subclass and implement the encoding logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseObservationEncoder(ABC):
    """Domain-agnostic observation encoder for RL agents.

    Subclasses implement the encoding logic for their scenario.

    Usage::

        class DebateObservationEncoder(BaseObservationEncoder):
            def shared_obs_dim(self) -> int: return 14
            def role_obs_dim(self) -> int: return 6
            def encode_shared(self, state, round_num, max_rounds):
                return np.array([state.disagreement, state.quality_score, ...])
            def encode_role(self, shared_obs, role):
                return np.concatenate([shared_obs, role_specific_features])
    """

    @abstractmethod
    def shared_obs_dim(self) -> int:
        """Dimension of the shared observation vector."""
        ...

    @abstractmethod
    def role_obs_dim(self) -> int:
        """Dimension of the role-specific observation extension."""
        ...

    @property
    def total_obs_dim(self) -> int:
        """Total observation dimension per agent."""
        return self.shared_obs_dim() + self.role_obs_dim()

    @abstractmethod
    def encode_shared(self, state: object, round_num: int, max_rounds: int) -> np.ndarray:
        """Encode the shared observation from current state.

        Parameters
        ----------
        state : object
            Scenario-specific state (e.g. CollaborationState subclass).
        round_num : int
            Current round number.
        max_rounds : int
            Maximum rounds in the episode.

        Returns
        -------
        shared_obs : np.ndarray of shape (shared_obs_dim,)
        """
        ...

    @abstractmethod
    def encode_role(self, shared_obs: np.ndarray, role: str) -> np.ndarray:
        """Encode role-specific observation by extending shared obs.

        Parameters
        ----------
        shared_obs : np.ndarray
            Output of encode_shared().
        role : str
            Role name (e.g. "proposer_ctrl", "buyer_ctrl").

        Returns
        -------
        role_obs : np.ndarray of shape (shared_obs_dim + role_obs_dim,)
        """
        ...

    def reset(self) -> None:
        """Reset any internal state for a new episode. Override if needed."""
        pass
