"""Checkpoint Manager — versioned save/load with architecture hash validation.

Ensures checkpoint compatibility by embedding version and architecture hash.
Prevents silent failures from loading checkpoints with mismatched architectures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from debate_rl_v2.config.master import Config, CHECKPOINT_VERSION
from debate_rl_v2.exceptions import TrainingError
from debate_rl_v2.logging_config import get_logger

logger = get_logger("algorithms.checkpoint")


def save_checkpoint(
    path: str,
    config: Config,
    agent_states: dict[str, Any],
    episode: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a versioned checkpoint with architecture hash.

    Parameters
    ----------
    path : str
        Output file path (.pt).
    config : Config
        Current configuration (for architecture hash).
    agent_states : dict
        Agent state dicts (model weights, optimizer states).
    episode : int
        Current training episode.
    extra : dict, optional
        Additional metadata (eval scores, etc.).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "_version": CHECKPOINT_VERSION,
        "_arch_hash": config.architecture_hash(),
        "_config_snapshot": {
            "network": {
                "hidden_dim": config.network.hidden_dim,
                "num_layers": config.network.num_layers,
                "activation": config.network.activation,
            },
            "maddpg": {
                "actor_hidden_dim": config.maddpg.actor_hidden_dim,
                "critic_hidden_dim": config.maddpg.critic_hidden_dim,
            },
            "env": {
                "context_dim": config.env.context_dim,
                "proposal_dim": config.env.proposal_dim,
            },
        },
        "episode": episode,
        "agent_states": agent_states,
    }
    if extra:
        checkpoint["extra"] = extra

    torch.save(checkpoint, path)
    logger.info(
        "Checkpoint saved: %s (episode=%d, arch=%s)",
        path, episode, config.architecture_hash(),
    )


def load_checkpoint(
    path: str,
    config: Config,
    strict_arch: bool = True,
) -> dict[str, Any]:
    """Load a versioned checkpoint with architecture validation.

    Parameters
    ----------
    path : str
        Checkpoint file path.
    config : Config
        Current configuration (for architecture hash comparison).
    strict_arch : bool
        If True, raise on architecture mismatch. If False, warn only.

    Returns
    -------
    checkpoint : dict
        Loaded checkpoint data.

    Raises
    ------
    TrainingError
        If checkpoint version or architecture is incompatible.
    FileNotFoundError
        If checkpoint file doesn't exist.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # Version check
    ckpt_version = checkpoint.get("_version", "unknown")
    if ckpt_version != CHECKPOINT_VERSION:
        logger.warning(
            "Checkpoint version mismatch: file=%s, current=%s",
            ckpt_version, CHECKPOINT_VERSION,
        )

    # Architecture hash check
    ckpt_hash = checkpoint.get("_arch_hash", "")
    current_hash = config.architecture_hash()
    if ckpt_hash and ckpt_hash != current_hash:
        msg = (
            f"Architecture mismatch: checkpoint={ckpt_hash}, "
            f"current={current_hash}. "
            f"Checkpoint config: {checkpoint.get('_config_snapshot', {})}"
        )
        if strict_arch:
            raise TrainingError(msg)
        logger.warning(msg)

    episode = checkpoint.get("episode", 0)
    logger.info(
        "Checkpoint loaded: %s (episode=%d, version=%s, arch=%s)",
        path, episode, ckpt_version, ckpt_hash,
    )
    return checkpoint
