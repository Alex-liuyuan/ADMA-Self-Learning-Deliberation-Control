"""LLM client factory — create clients by provider name.

Usage::

    from debate_rl_v2.llm import create_llm_client

    client = create_llm_client("openai", model="gpt-4o")
    client = create_llm_client("deepseek", model="deepseek-chat")
    client = create_llm_client("qwen", model="qwen-plus")
    client = create_llm_client("custom", base_url="http://localhost:8000/v1",
                               model="my-model", api_key="sk-xxx")
"""

from __future__ import annotations

from typing import Any, Dict, Type

from debate_rl_v2.llm.base import BaseLLMClient
from debate_rl_v2.llm.openai_compat import (
    DeepSeekClient,
    MockLLMClient,
    OpenAIClient,
    OpenAICompatibleClient,
    QwenClient,
)


PROVIDER_REGISTRY: Dict[str, Type[BaseLLMClient]] = {
    # --- Major providers ---
    "openai": OpenAIClient,
    "chatgpt": OpenAIClient,
    "gpt": OpenAIClient,
    "deepseek": DeepSeekClient,
    "qwen": QwenClient,
    "tongyi": QwenClient,       # 通义
    "dashscope": QwenClient,
    # --- Generic / self-hosted ---
    "custom": OpenAICompatibleClient,
    "vllm": OpenAICompatibleClient,
    "ollama": OpenAICompatibleClient,
    # --- Testing ---
    "mock": MockLLMClient,
}


def create_llm_client(
    provider: str = "openai",
    **kwargs: Any,
) -> BaseLLMClient:
    """Create an LLM client by provider name.

    Parameters
    ----------
    provider : str
        Provider identifier.  Case-insensitive.
        Supported: ``openai``, ``chatgpt``, ``deepseek``, ``qwen``,
        ``tongyi``, ``custom``, ``vllm``, ``ollama``, ``mock``.
    **kwargs
        Forwarded to the provider class constructor (``api_key``,
        ``model``, ``base_url``, ``temperature``, etc.).

    Returns
    -------
    BaseLLMClient
        Ready-to-use client instance.

    Raises
    ------
    ValueError
        If the provider is not registered.
    """
    key = provider.lower().strip()
    if key not in PROVIDER_REGISTRY:
        available = ", ".join(sorted(PROVIDER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Available: {available}"
        )
    cls = PROVIDER_REGISTRY[key]
    return cls(**kwargs)


def register_provider(name: str, cls: Type[BaseLLMClient]) -> None:
    """Register a custom LLM provider at runtime."""
    PROVIDER_REGISTRY[name.lower().strip()] = cls
