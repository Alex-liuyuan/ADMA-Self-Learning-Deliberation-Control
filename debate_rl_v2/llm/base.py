"""LLM Base Client — abstract interface with retry, fallback, and structured logging.

Migrated from debate_rl/llm/base.py with improvements:
  - Structured error handling (no bare except)
  - Retry with exponential backoff
  - Model routing integration
  - Prompt caching integration
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

from debate_rl_v2.exceptions import LLMError, LLMTimeoutError, LLMRateLimitError
from debate_rl_v2.logging_config import get_logger

logger = get_logger("llm.base")


def _is_numeric_token_value(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _sum_numeric_leaves(value: Any) -> int:
    if _is_numeric_token_value(value):
        return int(value)
    if isinstance(value, dict):
        return sum(_sum_numeric_leaves(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_sum_numeric_leaves(v) for v in value)
    return 0


def _usage_total_tokens(usage: dict[str, Any] | None) -> int:
    """Extract a stable total token count from provider-specific usage payloads."""
    if not usage:
        return 0

    total = usage.get("total_tokens")
    if _is_numeric_token_value(total):
        return int(total)

    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if _is_numeric_token_value(prompt) or _is_numeric_token_value(completion):
        return int(prompt or 0) + int(completion or 0)

    return _sum_numeric_leaves(usage)


@dataclass
class Message:
    """A single chat message."""
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class ToolCallInfo:
    """LLM 返回的工具调用信息（OpenAI function calling 格式）"""
    id: str
    name: str
    arguments: str  # JSON string


@dataclass
class LLMResponse:
    """Structured LLM response."""
    content: str
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    cached: bool = False
    raw: Any = None
    tool_calls: list[ToolCallInfo] = field(default_factory=list)
    finish_reason: str = ""  # "stop" | "tool_calls"


class BaseLLMClient(ABC):
    """Abstract LLM client with retry and structured error handling.

    Subclasses implement `_call()` for the specific provider.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self._call_count = 0
        self._total_tokens = 0
        self._total_latency_ms = 0.0

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Send a chat completion request with retry logic.

        Parameters
        ----------
        messages : list of dict
            OpenAI-format messages.
        temperature : float, optional
            Override default temperature.
        max_tokens : int, optional
            Override default max_tokens.
        json_mode : bool
            Request JSON output format.
        tools : list of dict, optional
            OpenAI function calling tool schemas. When provided, the LLM
            may return tool_calls instead of plain content.

        Returns
        -------
        LLMResponse
            Structured response.

        Raises
        ------
        LLMError
            After all retries exhausted.
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                start = time.monotonic()
                if tools is not None:
                    response = self._call(messages, temp, tokens, json_mode, tools=tools)
                else:
                    response = self._call(messages, temp, tokens, json_mode)
                elapsed = (time.monotonic() - start) * 1000

                response.latency_ms = elapsed
                self._call_count += 1
                self._total_tokens += _usage_total_tokens(response.usage)
                self._total_latency_ms += elapsed

                logger.info(
                    "LLM call: model=%s tokens=%s latency=%.0fms",
                    response.model,
                    response.usage,
                    elapsed,
                )
                return response

            except LLMRateLimitError as e:
                last_error = e
                wait = min(2 ** attempt, 30)
                logger.warning(
                    "Rate limited (attempt %d/%d), waiting %ds",
                    attempt, self.max_retries, wait,
                )
                time.sleep(wait)

            except LLMTimeoutError as e:
                last_error = e
                logger.warning(
                    "Timeout (attempt %d/%d): %s",
                    attempt, self.max_retries, e,
                )

            except LLMError as e:
                last_error = e
                logger.error(
                    "LLM error (attempt %d/%d): %s",
                    attempt, self.max_retries, e,
                )
                if attempt < self.max_retries:
                    time.sleep(1)

        raise LLMError(
            f"All {self.max_retries} retries exhausted. Last error: {last_error}"
        )

    @abstractmethod
    def _call(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Provider-specific implementation. Must raise LLMError subtypes on failure."""
        ...

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "call_count": self._call_count,
            "total_tokens": self._total_tokens,
            "total_latency_ms": self._total_latency_ms,
            "avg_latency_ms": self._total_latency_ms / max(self._call_count, 1),
        }


class OpenAIClient(BaseLLMClient):
    """OpenAI-compatible LLM client (works with OpenAI, DeepSeek, Qwen, etc.)."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str = "",
        base_url: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._api_key = api_key
        self._base_url = base_url
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import openai
                client_kwargs: dict[str, Any] = {}
                if self._api_key:
                    client_kwargs["api_key"] = self._api_key
                if self._base_url:
                    client_kwargs["base_url"] = self._base_url
                self._client = openai.OpenAI(**client_kwargs)
            except ImportError:
                raise LLMError("openai package not installed. Run: pip install openai")
        return self._client

    def _call(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        client = self._ensure_client()
        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if tools:
                kwargs["tools"] = tools

            response = client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            # 解析 tool_calls
            tc_list: list[ToolCallInfo] = []
            if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    tc_list.append(ToolCallInfo(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    ))

            return LLMResponse(
                content=choice.message.content or "",
                model=response.model,
                usage=usage,
                raw=response,
                tool_calls=tc_list,
                finish_reason=choice.finish_reason or "",
            )
        except Exception as e:
            err_str = str(e).lower()
            if "rate" in err_str and "limit" in err_str:
                raise LLMRateLimitError(str(e)) from e
            if "timeout" in err_str or "timed out" in err_str:
                raise LLMTimeoutError(str(e)) from e
            raise LLMError(str(e)) from e
