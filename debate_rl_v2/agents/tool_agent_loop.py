"""多轮工具调用引擎 — 单个智能体在一个协作回合内的执行。

借鉴 hermes-agent 的 AgentLoop.run()，适配多智能体协作场景：
  - 每个智能体在一个协作回合内可进行多轮 LLM↔Tool 循环
  - 最终输出结构化 JSON（通过 parse_fn 解析）
  - 完整记录工具调用历史供 RL 奖励计算使用
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from debate_rl_v2.framework.tool_registry import GameToolRegistry
from debate_rl_v2.llm.base import BaseLLMClient, LLMResponse
from debate_rl_v2.logging_config import get_logger

logger = get_logger("agents.tool_agent_loop")


@dataclass
class ToolCallRecord:
    """单次工具调用记录"""
    turn: int
    tool_name: str
    arguments: dict[str, Any]
    result: str
    latency_ms: float
    error: str | None = None


@dataclass
class AgentTurnResult:
    """单个智能体在一个协作回合内的完整执行结果"""
    messages: list[dict[str, Any]] = field(default_factory=list)
    final_output: dict[str, Any] = field(default_factory=dict)
    turns_used: int = 0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    finished_naturally: bool = True
    raw_content: str = ""


class ToolAugmentedAgentLoop:
    """多轮工具调用引擎。

    流程：
    1. 构建 messages = [system, *history, user]
    2. 循环（最多 max_tool_turns 次）：
       a. LLM 生成（传 tools= 参数）
       b. 有 tool_calls → 执行 → 结果回注 → 继续
       c. 无 tool_calls → 解析最终输出 → 返回

    Parameters
    ----------
    client : BaseLLMClient
        LLM 客户端
    tool_registry : GameToolRegistry | None
        工具注册表（None 时退化为单次调用）
    role_name : str
        角色名称（用于工具过滤和日志）
    max_tool_turns : int
        最大工具调用轮次
    parse_fn : callable | None
        最终输出解析函数 (str -> dict)
    """

    def __init__(
        self,
        client: BaseLLMClient,
        tool_registry: GameToolRegistry | None = None,
        role_name: str = "",
        max_tool_turns: int = 5,
        parse_fn: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        self.client = client
        self.tool_registry = tool_registry
        self.role_name = role_name
        self.max_tool_turns = max_tool_turns
        self.parse_fn = parse_fn or self._default_parse

    def run(
        self,
        system_prompt: str,
        user_message: str,
        history: list[dict[str, str]] | None = None,
        style_directive: str = "",
        **llm_kwargs: Any,
    ) -> AgentTurnResult:
        """执行一个完整的多轮工具调用循环。

        Parameters
        ----------
        system_prompt : str
            系统提示词
        user_message : str
            当前回合的用户消息
        history : list, optional
            对话历史
        style_directive : str
            RL 策略指导（注入 system prompt）

        Returns
        -------
        AgentTurnResult
            包含最终输出、工具调用记录等
        """
        # 构建 system prompt
        full_system = system_prompt
        if style_directive:
            full_system = f"[策略指导]\n{style_directive}\n\n{full_system}"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": full_system},
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        # 获取工具 schemas
        tool_schemas = None
        if self.tool_registry:
            tool_schemas = self.tool_registry.get_schemas(role=self.role_name)
            if not tool_schemas:
                tool_schemas = None

        result = AgentTurnResult(messages=list(messages))
        tool_records: list[ToolCallRecord] = []

        for turn in range(self.max_tool_turns):
            # LLM 调用
            response = self.client.chat(
                messages, tools=tool_schemas, **llm_kwargs,
            )
            result.turns_used = turn + 1

            # 检查是否有 tool_calls
            if response.tool_calls and self.tool_registry:
                # 将 assistant 消息（含 tool_calls）加入对话
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": response.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments,
                            },
                        }
                        for tc in response.tool_calls
                    ],
                }
                messages.append(assistant_msg)

                # 执行每个工具调用
                for tc in response.tool_calls:
                    t0 = time.perf_counter()
                    try:
                        args = json.loads(tc.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    tool_result = self.tool_registry.dispatch(tc.name, args)
                    latency = (time.perf_counter() - t0) * 1000

                    record = ToolCallRecord(
                        turn=turn,
                        tool_name=tc.name,
                        arguments=args,
                        result=tool_result,
                        latency_ms=latency,
                    )
                    tool_records.append(record)

                    # 将工具结果回注对话
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    })

                logger.debug(
                    "[%s] Turn %d: %d tool calls executed",
                    self.role_name, turn, len(response.tool_calls),
                )
                continue  # 继续下一轮 LLM 调用

            # 无 tool_calls → 最终输出
            result.raw_content = response.content
            result.final_output = self._safe_parse(response.content)
            result.tool_calls = tool_records
            result.finished_naturally = True
            result.messages = messages
            return result

        # 达到最大轮次，使用最后一次 LLM 输出
        logger.warning(
            "[%s] Max tool turns (%d) reached", self.role_name, self.max_tool_turns,
        )
        result.raw_content = response.content if response else ""
        result.final_output = self._safe_parse(result.raw_content)
        result.tool_calls = tool_records
        result.finished_naturally = False
        result.messages = messages
        return result

    def _safe_parse(self, content: str) -> dict[str, Any]:
        """安全解析：先尝试 JSON 解码，再传给 parse_fn。

        parse_fn 可能期望 dict（如 make_schema_parser 生成的解析器），
        也可能期望 str（如自定义解析器）。这里统一处理。
        """
        if not content:
            return {"text": ""}

        # 先尝试 JSON 解码
        parsed_dict: dict[str, Any] | None = None
        try:
            parsed_dict = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            pass

        # 尝试用 parse_fn 解析
        try:
            if parsed_dict is not None:
                return self.parse_fn(parsed_dict)
            return self.parse_fn(content)
        except (AttributeError, TypeError):
            # parse_fn 期望 dict 但收到 str，或反之 → 用已解码的 dict
            if parsed_dict is not None:
                return parsed_dict
            return {"text": content}

    @staticmethod
    def _default_parse(content: str) -> dict[str, Any]:
        """默认解析：尝试 JSON，失败则包装为 text"""
        if not content:
            return {"text": ""}
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"text": content}
