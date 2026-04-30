"""OpenAI-compatible LLM client.

Works with **any** provider that exposes the ``/v1/chat/completions``
endpoint (OpenAI, DeepSeek, Qwen/DashScope, vLLM, Ollama, etc.).

Concrete convenience subclasses for each provider are registered in
``factory.py``.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

try:
    import openai  # type: ignore
    HAS_OPENAI_SDK = True
except ImportError:
    HAS_OPENAI_SDK = False

try:
    import httpx  # type: ignore
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

import json as _json

from debate_rl_v2.llm.base import BaseLLMClient, LLMResponse, ToolCallInfo


class OpenAICompatibleClient(BaseLLMClient):
    """Generic client for any OpenAI-compatible ``/v1/chat/completions``
    endpoint.

    Uses the official ``openai`` SDK when available; otherwise falls
    back to plain HTTP via ``httpx`` / ``urllib``.

    Parameters
    ----------
    api_key : str | None
        API key.  When *None*, resolved from ``api_key_env``.
    base_url : str
        Base URL of the provider API.
    model : str
        Model name / id.
    api_key_env : str
        Environment variable name for the API key fallback.
    temperature : float
        Default sampling temperature.
    max_tokens : int
        Default max generation tokens.
    timeout : float
        HTTP timeout in seconds.
    max_retries : int
        Number of retries on transient errors.
    extra_headers : dict | None
        Extra HTTP headers added to every request.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: float = 60.0,
        max_retries: int = 3,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.base_url = base_url.rstrip("/")
        self.extra_headers = extra_headers or {}

        # Initialise openai SDK client (preferred) or fallback flag
        self._client: Any = None
        if HAS_OPENAI_SDK:
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=0,  # we handle retries ourselves
            )

    # ------------------------------------------------------------------
    # Core implementation
    # ------------------------------------------------------------------

    def _raw_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        if self._client is not None:
            return self._call_openai_sdk(messages, temp, tokens, json_mode, tools=tools, **kwargs)
        return self._call_http(messages, temp, tokens, json_mode, tools=tools, **kwargs)

    # Bridge: BaseLLMClient.chat() calls _call(), delegate to _raw_chat()
    def _call(self, messages, temperature, max_tokens, json_mode, tools=None):
        return self._raw_chat(messages, temperature, max_tokens, json_mode, tools=tools)

    # ------ openai SDK path ------

    def _call_openai_sdk(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        create_kw: Dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        if json_mode:
            create_kw["response_format"] = {"type": "json_object"}
        if tools:
            create_kw["tools"] = tools

        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(**create_kw)
        latency = (time.perf_counter() - t0) * 1000

        choice = resp.choices[0]
        usage = {}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
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
            model=resp.model or self.model,
            usage=usage,
            finish_reason=choice.finish_reason or "",
            latency_ms=latency,
            tool_calls=tc_list,
        )

    # ------ HTTP fallback path ------

    def _call_http(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            **self.extra_headers,
        }
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if tools:
            body["tools"] = tools
        body.update(kwargs)

        t0 = time.perf_counter()

        if HAS_HTTPX:
            with httpx.Client(timeout=self.timeout) as client:
                http_resp = client.post(url, headers=headers, json=body)
                http_resp.raise_for_status()
                data = http_resp.json()
        else:
            import urllib.request

            req = urllib.request.Request(
                url,
                data=_json.dumps(body).encode(),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = _json.loads(resp.read().decode())

        latency = (time.perf_counter() - t0) * 1000

        choice = data["choices"][0]
        usage = data.get("usage", {})

        # 解析 HTTP 响应中的 tool_calls
        tc_list: list[ToolCallInfo] = []
        msg = choice.get("message", {})
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            tc_list.append(ToolCallInfo(
                id=tc.get("id", ""),
                name=fn.get("name", ""),
                arguments=fn.get("arguments", "{}"),
            ))

        return LLMResponse(
            content=msg.get("content", "") or "",
            model=data.get("model", self.model),
            usage=usage,
            finish_reason=choice.get("finish_reason", ""),
            latency_ms=latency,
            tool_calls=tc_list,
        )


# ======================================================================
# Concrete provider presets
# ======================================================================


class OpenAIClient(OpenAICompatibleClient):
    """OpenAI ChatGPT / GPT-4o client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
            model=model,
            api_key_env="OPENAI_API_KEY",
            **kwargs,
        )


class DeepSeekClient(OpenAICompatibleClient):
    """DeepSeek chat / coder / reasoner client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            model=model,
            api_key_env="DEEPSEEK_API_KEY",
            **kwargs,
        )


class QwenClient(OpenAICompatibleClient):
    """Alibaba Qwen (通义千问) client via DashScope OpenAI-compatible API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-plus",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model=model,
            api_key_env="DASHSCOPE_API_KEY",
            **kwargs,
        )


# ======================================================================
# Mock client for testing (no API key required)
# ======================================================================


class MockLLMClient(BaseLLMClient):
    """Deterministic mock client for unit tests.

    Returns canned JSON responses based on the last user message.
    """

    def __init__(self, model: str = "mock-1.0", **kwargs: Any) -> None:
        # Accept and ignore provider-specific kwargs like api_key
        kwargs.pop("api_key", None)
        kwargs.pop("base_url", None)
        kwargs.pop("api_key_env", None)
        kwargs.pop("extra_headers", None)
        super().__init__(model=model, **kwargs)
        self.call_log: List[Dict] = []

    def _raw_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        self.call_log.append({"messages": messages, "kwargs": kwargs})
        user_msg = messages[-1]["content"] if messages else ""

        # Detect role from system message
        system_msg = ""
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
                break

        # Detect role — check more specific roles first to avoid
        # false matches (e.g. arbiter prompt may mention "proposer")
        def _is_role(role_name: str) -> bool:
            """Check if system message is for a specific role."""
            markers = [
                f"**{role_name.capitalize()}**",          # **Proposer**
                f"({role_name.capitalize()})",             # (Proposer)
                f"You are a {role_name.capitalize()}",     # English
                f"你是.*{role_name.capitalize()}",
                f"**提案者 (Proposer)**" if role_name == "proposer" else "",
                f"**挑战者 (Challenger)**" if role_name == "challenger" else "",
                f"**仲裁者 (Arbiter)**" if role_name == "arbiter" else "",
                f"**协调者 (Coordinator)**" if role_name == "coordinator" else "",
            ]
            return any(m and m in system_msg for m in markers)

        if _is_role("coordinator"):
            resp = _json.dumps({
                "reasoning": "Mock coordinator reasoning",
                "action": 0,
                "expected_effect": "Maintain current dynamics.",
            })
        elif _is_role("arbiter"):
            resp = _json.dumps({
                "reasoning": "Mock arbiter evaluation",
                "verdict": "Proposal has merit but challenger raises valid cost concern.",
                "action": "no_change",
                "quality_score": 0.65,
            })
        elif _is_role("challenger"):
            resp = _json.dumps({
                "reasoning": "Mock challenger reasoning",
                "challenge": "Approach A overlooks cost factor X.",
                "key_concerns": ["cost", "scalability"],
                "confidence": 0.7,
            })
        elif _is_role("proposer") or "Proposer" in system_msg or "proposer" in system_msg:
            resp = _json.dumps({
                "reasoning": "Mock proposer reasoning",
                "proposal": "We should adopt approach A because it minimises risk.",
                "confidence": 0.8,
            })
        else:
            resp = _json.dumps({
                "reasoning": "Mock response",
                "text": "This is a mock response.",
            })

        return LLMResponse(
            content=resp,
            model=self.model,
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            latency_ms=10.0,
        )

    # Bridge: BaseLLMClient.chat() calls _call()
    def _call(self, messages, temperature, max_tokens, json_mode, tools=None):
        return self._raw_chat(messages, temperature, max_tokens, json_mode)
