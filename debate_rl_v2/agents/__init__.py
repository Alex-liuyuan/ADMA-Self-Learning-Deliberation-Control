"""Agent framework — hermes-agent inspired rewrite."""

from debate_rl_v2.agents.llm_agent import LLMAgent, LLMAgentGroup
from debate_rl_v2.agents.protocol import AgentMessage, MessageType, ConversationThread
from debate_rl_v2.agents.hooks import HookManager, HookPoint, HookContext
from debate_rl_v2.agents.context_compressor import ContextCompressor
from debate_rl_v2.agents.async_agent import AsyncAgentWrapper, parallel_agent_calls

__all__ = [
    "LLMAgent",
    "LLMAgentGroup",
    "AgentMessage",
    "MessageType",
    "ConversationThread",
    "HookManager",
    "HookPoint",
    "HookContext",
    "ContextCompressor",
    "AsyncAgentWrapper",
    "parallel_agent_calls",
]
