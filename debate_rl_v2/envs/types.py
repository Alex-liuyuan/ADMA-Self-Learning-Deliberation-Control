"""Debate-specific data types — re-export shim for backward compatibility.

权威来源已迁移到 debate_rl_v2.scenarios.debate.types。
此文件保留 re-export 以兼容现有 import。

.. deprecated::
    直接从 debate_rl_v2.scenarios.debate.types 导入。
"""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "debate_rl_v2.envs.types is deprecated. "
    "Import from debate_rl_v2.scenarios.debate.types instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical location
from debate_rl_v2.scenarios.debate.types import (  # noqa: F401, E402
    TextDebateState,
    DebateTurn,
    FusionRoundRecord,
)
