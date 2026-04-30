"""Tool Registry — hermes-agent inspired self-registration pattern.

DEPRECATED: Use framework.tool_registry.GameToolRegistry instead.
This module is kept for backward compatibility but will be removed in v4.0.

Each tool file calls `registry.register()` at import time.
The registry provides schema definitions for LLM prompt injection
and handles tool execution with validation and error reporting.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "debate_rl_v2.tools.registry is deprecated. "
    "Use debate_rl_v2.framework.tool_registry.GameToolRegistry instead.",
    DeprecationWarning,
    stacklevel=2,
)

import inspect
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from debate_rl_v2.exceptions import ToolError, ToolNotFoundError, ToolValidationError
from debate_rl_v2.logging_config import get_logger

logger = get_logger("tools.registry")


@dataclass
class ToolSchema:
    """JSON-schema-style parameter definition for a tool."""
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    default: Any = None


@dataclass
class ToolDefinition:
    """Complete tool definition with metadata and handler."""
    name: str
    description: str
    parameters: list[ToolSchema] = field(default_factory=list)
    handler: Callable[..., Any] = field(default=lambda **kw: "Not implemented")
    category: str = "general"
    availability_check: Callable[[], bool] | None = None

    def is_available(self) -> bool:
        if self.availability_check is None:
            return True
        try:
            return self.availability_check()
        except Exception:
            return False

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function-calling schema format."""
        properties: dict[str, Any] = {}
        required: list[str] = []
        for p in self.parameters:
            properties[p.name] = {
                "type": p.type,
                "description": p.description,
            }
            if p.required:
                required.append(p.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_prompt_text(self) -> str:
        """Format for inclusion in LLM system prompts."""
        params_lines = []
        for p in self.parameters:
            req = " (必填)" if p.required else " (可选)"
            params_lines.append(f"    - {p.name} ({p.type}){req}: {p.description}")
        params_text = "\n".join(params_lines)
        base = f"  - **{self.name}**: {self.description}"
        if params_text:
            base += f"\n{params_text}"
        return base


class ToolRegistry:
    """Central tool registry — singleton pattern.

    Tools self-register at import time. The registry provides:
    - Schema definitions for LLM prompt injection
    - Validated tool execution with structured error reporting
    - Toolset filtering (only expose relevant tools per context)
    """

    _instance: ToolRegistry | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> ToolRegistry:
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._tools = {}
                    cls._instance._initialized = True
        return cls._instance

    def __init__(self) -> None:
        # Avoid re-init on subsequent calls
        if not hasattr(self, "_tools"):
            self._tools: dict[str, ToolDefinition] = {}

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    def register(
        self,
        name: str,
        description: str,
        handler: Callable[..., Any],
        parameters: list[ToolSchema] | None = None,
        category: str = "general",
        availability_check: Callable[[], bool] | None = None,
    ) -> None:
        """Register a tool."""
        tool = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters or [],
            handler=handler,
            category=category,
            availability_check=availability_check,
        )
        self._tools[name] = tool
        logger.debug("Registered tool: %s [%s]", name, category)

    def register_function(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
        category: str = "general",
    ) -> None:
        """Register a plain function as a tool (auto-extract params from signature)."""
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or ""

        # Auto-extract parameters from function signature
        sig = inspect.signature(func)
        params: list[ToolSchema] = []
        for pname, param in sig.parameters.items():
            if pname in ("self", "cls"):
                continue
            ptype = "string"
            annotation = param.annotation
            if annotation != inspect.Parameter.empty:
                if annotation in (int, float):
                    ptype = "number"
                elif annotation is bool:
                    ptype = "boolean"
            required = param.default == inspect.Parameter.empty
            params.append(ToolSchema(
                name=pname,
                type=ptype,
                description="",
                required=required,
                default=None if required else param.default,
            ))

        self.register(
            name=tool_name,
            description=tool_desc.strip(),
            handler=func,
            parameters=params,
            category=category,
        )

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def get_definitions(
        self,
        category: str | None = None,
        available_only: bool = True,
    ) -> list[ToolDefinition]:
        """Get tool definitions, optionally filtered by category."""
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        if available_only:
            tools = [t for t in tools if t.is_available()]
        return tools

    def get_openai_schemas(
        self,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get OpenAI function-calling schemas for available tools."""
        return [t.to_openai_schema() for t in self.get_definitions(category)]

    def handle_call(
        self,
        name: str,
        args: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Execute a tool by name with validation.

        Parameters
        ----------
        name : str
            Tool name.
        args : dict
            Tool arguments.
        context : dict, optional
            Execution context (round_num, role, etc.).

        Returns
        -------
        result : Any
            Tool output.

        Raises
        ------
        ToolNotFoundError
            If tool doesn't exist.
        ToolError
            If execution fails.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(name, list(self._tools.keys()))

        if not tool.is_available():
            raise ToolError(f"Tool '{name}' is currently unavailable")

        # Validate required parameters
        for param in tool.parameters:
            if param.required and param.name not in args:
                raise ToolValidationError(
                    f"Tool '{name}' missing required parameter: '{param.name}'"
                )

        # Fill defaults for optional params
        call_args = {}
        for param in tool.parameters:
            if param.name in args:
                call_args[param.name] = args[param.name]
            elif not param.required and param.default is not None:
                call_args[param.name] = param.default

        try:
            result = tool.handler(**call_args)
            logger.info(
                "Tool executed: %s", name,
                extra={"tool": name},
            )
            return result
        except Exception as e:
            logger.error("Tool execution failed: %s — %s", name, e, extra={"tool": name})
            raise ToolError(f"Tool '{name}' execution failed: {e}") from e

    def to_prompt_text(self) -> str:
        """Generate tool descriptions for LLM system prompts."""
        tools = self.get_definitions(available_only=True)
        if not tools:
            return ""
        lines = [
            "## 可用工具",
            "你可以通过在回复中包含 `tool_calls` 字段来调用工具：",
            '```json',
            '{"tool_calls": [{"name": "工具名", "input": {...}}]}',
            '```',
            "",
            "可用工具列表：",
        ]
        for tool in tools:
            lines.append(tool.to_prompt_text())
        return "\n".join(lines)

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __bool__(self) -> bool:
        return True


def parse_tool_calls(response_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract tool calls from LLM response JSON."""
    calls = response_data.get("tool_calls", [])
    if not isinstance(calls, list):
        return []
    result = []
    for call in calls:
        if isinstance(call, dict) and "name" in call:
            result.append({
                "name": call["name"],
                "input": call.get("input", call.get("arguments", {})),
            })
    return result


def execute_tool_calls(
    tool_calls: list[dict[str, Any]],
    registry: ToolRegistry,
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Execute a batch of tool calls and return results."""
    results = []
    for call in tool_calls:
        name = call["name"]
        args = call.get("input", {})
        try:
            output = registry.handle_call(name, args, context=context)
            results.append({"name": name, "input": args, "output": output, "error": None})
        except (ToolNotFoundError, ToolError) as e:
            results.append({"name": name, "input": args, "output": None, "error": str(e)})
    return results
