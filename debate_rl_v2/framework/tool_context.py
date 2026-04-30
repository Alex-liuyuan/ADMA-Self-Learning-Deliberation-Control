"""奖励函数/验证函数的工具访问上下文。

用法（在 compute_rewards 中）：
    result = ctx.call_tool("query_nccn_guidelines", {"cancer_type": "NSCLC"})
    guideline_score = evaluate_against_guidelines(result, proposal)
"""

from __future__ import annotations

import uuid
from typing import Any

from debate_rl_v2.framework.tool_registry import GameToolRegistry
from debate_rl_v2.logging_config import get_logger

logger = get_logger("framework.tool_context")


class GameToolContext:
    """给奖励函数/验证函数提供工具访问能力。

    封装 GameToolRegistry 的 dispatch，附加 session 追踪和调用记录。
    """

    def __init__(
        self,
        registry: GameToolRegistry,
        session_id: str = "",
    ) -> None:
        self.registry = registry
        self.session_id = session_id or uuid.uuid4().hex[:8]
        self._call_log: list[dict[str, Any]] = []

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """调用工具并返回结果字符串。"""
        result = self.registry.dispatch(
            name, arguments, session_id=self.session_id,
        )
        self._call_log.append({
            "name": name,
            "arguments": arguments,
            "result_preview": result[:200] if result else "",
        })
        return result

    @property
    def call_log(self) -> list[dict[str, Any]]:
        return list(self._call_log)

    def cleanup(self) -> None:
        """清理 session 资源。"""
        logger.debug(
            "GameToolContext cleanup: session=%s, calls=%d",
            self.session_id, len(self._call_log),
        )
        self._call_log.clear()
