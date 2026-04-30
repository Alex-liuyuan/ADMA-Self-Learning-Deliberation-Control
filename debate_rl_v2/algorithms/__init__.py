"""RL algorithm public exports with lazy loading."""

from __future__ import annotations

_EXPORTS = {
    "DomainAdapter": "debate_rl_v2.algorithms.domain_adapter",
    "ObservationAdapter": "debate_rl_v2.algorithms.domain_adapter",
    "CosineAnnealingScheduler": "debate_rl_v2.algorithms.training_utils",
    "LinearDecayScheduler": "debate_rl_v2.algorithms.training_utils",
    "QValueMonitor": "debate_rl_v2.algorithms.training_utils",
    "EarlyStopping": "debate_rl_v2.algorithms.training_utils",
    "compute_gradient_norm": "debate_rl_v2.algorithms.training_utils",
    "GradientStats": "debate_rl_v2.algorithms.training_utils",
    "RoleObservationTracker": "debate_rl_v2.algorithms.role_observations",
    "build_role_observation": "debate_rl_v2.algorithms.role_observations",
    "SHARED_OBS_DIM": "debate_rl_v2.algorithms.role_observations",
    "ROLE_OBS_DIM": "debate_rl_v2.algorithms.role_observations",
    "TOTAL_OBS_DIM": "debate_rl_v2.algorithms.role_observations",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = __import__(module_name, fromlist=[name])
    return getattr(module, name)
