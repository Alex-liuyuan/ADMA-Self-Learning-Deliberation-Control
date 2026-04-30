"""Tool system — registry-based, hermes-agent inspired."""

from debate_rl_v2.tools.registry import (
    ToolRegistry,
    ToolDefinition,
    ToolSchema,
    parse_tool_calls,
    execute_tool_calls,
)

__all__ = [
    "ToolRegistry",
    "ToolDefinition",
    "ToolSchema",
    "parse_tool_calls",
    "execute_tool_calls",
]
