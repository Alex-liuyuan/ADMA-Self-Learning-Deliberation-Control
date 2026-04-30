"""debate_rl_v2 — Generic Multi-Agent Collaboration Framework with RL-guided LLM fusion."""

__version__ = "2.2.0"

from debate_rl_v2.logging_config import setup_logging, get_logger

__all__ = [
    "__version__",
    "setup_logging",
    "get_logger",
    "Config",
    "load_config",
]


def __getattr__(name: str):
    if name in {"Config", "load_config"}:
        from debate_rl_v2.config import Config, load_config

        exports = {
            "Config": Config,
            "load_config": load_config,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
