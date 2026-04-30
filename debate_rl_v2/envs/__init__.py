"""Debate environment public exports with lazy loading."""

from __future__ import annotations

_EXPORTS = {
    "TextDebateState": "debate_rl_v2.envs.types",
    "DebateTurn": "debate_rl_v2.envs.types",
    "FusionRoundRecord": "debate_rl_v2.envs.types",
    "TextDebateEnv": "debate_rl_v2.envs.llm_env",
    "FusionDebateEnv": "debate_rl_v2.envs.fusion_env",
    "DebateEventEmitter": "debate_rl_v2.envs.event_emitter",
    "DebateEvent": "debate_rl_v2.envs.event_emitter",
    "DebateEventType": "debate_rl_v2.envs.event_emitter",
    "DashboardAdapter": "debate_rl_v2.envs.event_emitter",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = __import__(module_name, fromlist=[name])
    return getattr(module, name)
