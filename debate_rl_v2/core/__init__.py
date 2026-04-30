"""Core mechanism public exports with lazy loading."""

from __future__ import annotations

_EXPORTS = {
    "StrategyBridge": "debate_rl_v2.core.strategy_bridge",
    "StrategySignals": "debate_rl_v2.core.strategy_bridge",
    "FUSION_OBS_DIM": "debate_rl_v2.core.strategy_bridge",
    "TOTAL_OBS_DIM": "debate_rl_v2.core.strategy_bridge",
    "ComplianceVerifier": "debate_rl_v2.core.compliance_verifier",
    "ComplianceResult": "debate_rl_v2.core.compliance_verifier",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = __import__(module_name, fromlist=[name])
    return getattr(module, name)
