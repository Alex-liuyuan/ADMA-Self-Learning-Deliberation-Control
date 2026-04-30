"""通用工具注册表 — 支持场景级工具集分组和角色级过滤。

与 tools/registry.py 的 ToolRegistry 区别：
  - 非全局单例，每个场景可拥有独立注册表
  - 支持 toolset 分组和角色级工具过滤
  - 输出 OpenAI function calling 格式的 tool schemas
  - 支持同步/异步 handler
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from debate_rl_v2.logging_config import get_logger

logger = get_logger("framework.tool_registry")


@dataclass
class ToolSpec:
    """工具规格声明"""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    handler: Callable[..., str]
    toolset: str = "default"
    check_fn: Callable[[], bool] | None = None
    is_async: bool = False
    allowed_roles: list[str] = field(default_factory=list)  # 空=所有角色可用

    def is_available(self) -> bool:
        if self.check_fn is None:
            return True
        try:
            return self.check_fn()
        except Exception:
            return False

    def to_openai_schema(self) -> dict[str, Any]:
        """转换为 OpenAI function calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class GameToolRegistry:
    """通用工具注册表 — 支持场景级工具集分组。

    与 tools/registry.py 的 ToolRegistry 区别：
    - 非全局单例，每个场景可拥有独立注册表
    - 支持角色级工具过滤（不同角色可用不同工具）
    - 输出 OpenAI function calling 格式的 tool schemas
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        """注册一个工具"""
        self._tools[spec.name] = spec
        logger.debug("Registered game tool: %s [%s]", spec.name, spec.toolset)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def get_schemas(
        self,
        toolset: str | None = None,
        role: str | None = None,
    ) -> list[dict[str, Any]]:
        """获取 OpenAI function calling 格式的 tool schemas。

        Parameters
        ----------
        toolset : str, optional
            按工具集过滤
        role : str, optional
            按角色过滤（只返回该角色可用的工具）
        """
        specs = self._filter(toolset=toolset, role=role)
        return [s.to_openai_schema() for s in specs]

    def dispatch(self, name: str, arguments: dict[str, Any], **ctx: Any) -> str:
        """执行工具调用并返回结果字符串。

        Parameters
        ----------
        name : str
            工具名称
        arguments : dict
            工具参数
        **ctx
            额外上下文（role, round_num 等）

        Returns
        -------
        str
            工具执行结果
        """
        spec = self._tools.get(name)
        if spec is None:
            return json.dumps({"error": f"Tool '{name}' not found"})

        if not spec.is_available():
            return json.dumps({"error": f"Tool '{name}' is currently unavailable"})

        t0 = time.perf_counter()
        try:
            result = spec.handler(**arguments)
            latency = (time.perf_counter() - t0) * 1000
            logger.info(
                "Tool dispatched: %s (%.0fms)", name, latency,
                extra={"tool": name},
            )
            return result if isinstance(result, str) else json.dumps(result)
        except Exception as e:
            logger.error("Tool dispatch failed: %s — %s", name, e)
            return json.dumps({"error": f"Tool '{name}' failed: {e}"})

    def get_tools_for_role(self, role: str) -> list[ToolSpec]:
        """获取指定角色可用的工具列表"""
        return self._filter(role=role)

    def _filter(
        self,
        toolset: str | None = None,
        role: str | None = None,
    ) -> list[ToolSpec]:
        specs = list(self._tools.values())
        if toolset:
            specs = [s for s in specs if s.toolset == toolset]
        if role:
            specs = [
                s for s in specs
                if not s.allowed_roles or role in s.allowed_roles
            ]
        return [s for s in specs if s.is_available()]

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __bool__(self) -> bool:
        return True
