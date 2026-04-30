"""Structured logging configuration for debate_rl_v2.

Replaces all print() calls with proper Python logging.
Supports JSON-structured output for production and human-readable
format for development.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = self.formatException(record.exc_info)
        # Merge extra fields attached via `extra=` kwarg
        for key in ("round_num", "role", "agent", "tool", "episode",
                     "quality", "disagreement", "reward", "duration_ms"):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val
        return json.dumps(entry, ensure_ascii=False)


class HumanFormatter(logging.Formatter):
    """Colored, human-readable formatter for terminal output."""

    COLORS = {
        "DEBUG": "\033[36m",     # cyan
        "INFO": "\033[32m",      # green
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[35m",  # magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        prefix = f"{color}{record.levelname:>8}{self.RESET}"
        name = record.name.split(".")[-1]
        msg = record.getMessage()

        # Append structured extras inline
        extras = []
        for key in ("round_num", "role", "quality", "reward"):
            val = getattr(record, key, None)
            if val is not None:
                extras.append(f"{key}={val}")
        suffix = f"  [{', '.join(extras)}]" if extras else ""

        base = f"{prefix} [{name}] {msg}{suffix}"
        if record.exc_info and record.exc_info[1]:
            base += "\n" + self.formatException(record.exc_info)
        return base


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: str | None = None,
) -> None:
    """Configure the root debate_rl_v2 logger.

    Parameters
    ----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR).
    json_output : bool
        Use JSON formatter (for production / log aggregation).
    log_file : str | None
        Also write logs to this file path.
    """
    root = logging.getLogger("debate_rl_v2")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    formatter: logging.Formatter
    if json_output:
        formatter = JSONFormatter()
    else:
        formatter = HumanFormatter()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    # Optional file handler (always JSON for machine parsing)
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(JSONFormatter())
        root.addHandler(fh)

    # Suppress noisy third-party loggers
    for name in ("httpx", "openai", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the debate_rl_v2 namespace."""
    return logging.getLogger(f"debate_rl_v2.{name}")
