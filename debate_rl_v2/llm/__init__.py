"""LLM client layer — enhanced with routing and caching."""

from debate_rl_v2.llm.base import BaseLLMClient, OpenAIClient, LLMResponse, Message, ToolCallInfo
from debate_rl_v2.llm.json_parser import RobustJSONParser
from debate_rl_v2.llm.prompt_cache import PromptCache
from debate_rl_v2.llm.routing import SmartModelRouter, ModelSpec
from debate_rl_v2.llm.factory import create_llm_client

__all__ = [
    "BaseLLMClient",
    "OpenAIClient",
    "LLMResponse",
    "Message",
    "ToolCallInfo",
    "RobustJSONParser",
    "PromptCache",
    "SmartModelRouter",
    "ModelSpec",
    "create_llm_client",
]
