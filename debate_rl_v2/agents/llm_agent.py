"""LLM Agent v2 — integrated with ToolRegistry, ContextCompressor, PromptCache.

Migrated from debate_rl/agents/llm_agent.py with key improvements:
  - ToolRegistry replaces ad-hoc tool system
  - ContextCompressor manages context window
  - PromptCache reduces token costs
  - RobustJSONParser replaces brittle parsing
  - Structured logging replaces print()
  - No bare except blocks

v2.3: Generalized to support arbitrary RoleDefinition (not just debate roles).
  - New optional params: role_definition, parse_fn, expected_fields, msg_type
  - Falls back to legacy debate role maps when role_definition is not provided

v3.0: Tool-augmented agent loop integration.
  - New optional param: game_tool_registry (GameToolRegistry)
  - When provided, act() uses ToolAugmentedAgentLoop for multi-turn tool calls
  - Backward compatible: without game_tool_registry, behavior unchanged
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from debate_rl_v2.agents.protocol import AgentMessage, MessageType
from debate_rl_v2.agents.hooks import HookManager, HookPoint
from debate_rl_v2.agents.tracing import DebateTracer
from debate_rl_v2.agents.context_compressor import ContextCompressor
from debate_rl_v2.scenarios.debate.prompts import (
    SYSTEM_PROMPTS,
    parse_proposer_response,
    parse_challenger_response,
    parse_arbiter_response,
    parse_coordinator_response,
)
from debate_rl_v2.agents.generic_parser import make_schema_parser, schema_to_expected_fields
from debate_rl_v2.agents.tool_agent_loop import ToolAugmentedAgentLoop, AgentTurnResult
from debate_rl_v2.framework.roles import RoleDefinition
from debate_rl_v2.framework.tool_registry import GameToolRegistry
from debate_rl_v2.llm.base import BaseLLMClient, LLMResponse
from debate_rl_v2.llm.json_parser import RobustJSONParser
from debate_rl_v2.llm.prompt_cache import PromptCache
from debate_rl_v2.memory.manager import MemoryManager
from debate_rl_v2.tools.registry import ToolRegistry, parse_tool_calls, execute_tool_calls
from debate_rl_v2.logging_config import get_logger

logger = get_logger("agents.llm_agent")

_ROLE_MSG_TYPE = {
    "proposer": MessageType.PROPOSAL,
    "challenger": MessageType.CHALLENGE,
    "arbiter": MessageType.VERDICT,
    "coordinator": MessageType.META_ACTION,
}

_ROLE_PARSERS = {
    "proposer": parse_proposer_response,
    "challenger": parse_challenger_response,
    "arbiter": parse_arbiter_response,
    "coordinator": parse_coordinator_response,
}

# Expected JSON fields per role (for regex fallback)
_ROLE_FIELDS = {
    "proposer": ["proposal", "confidence", "reasoning"],
    "challenger": ["challenge", "confidence", "reasoning"],
    "arbiter": ["quality_score", "proposal_score", "challenge_score", "verdict", "consensus_reached"],
    "coordinator": ["action_idx", "reasoning", "expected_effect"],
}


class LLMAgent:
    """LLM-based agent with full framework integration.

    Supports two construction modes:
      1. Legacy (debate roles): role in SYSTEM_PROMPTS → uses built-in parsers
      2. Generic (any scenario): provide role_definition from RoleRegistry

    v2.3 generalization:
      - role_definition: RoleDefinition from framework/roles.py
      - parse_fn: custom parser callable
      - expected_fields: field names for regex fallback
      - msg_type: MessageType for act_message()
    """

    def __init__(
        self,
        role: str,
        client: BaseLLMClient,
        role_definition: Optional[RoleDefinition] = None,
        parse_fn: Optional[Callable] = None,
        expected_fields: Optional[List[str]] = None,
        msg_type: Optional[MessageType] = None,
        system_prompt: str = "",
        max_history: int = 20,
        fallback_clients: Optional[List[BaseLLMClient]] = None,
        memory: Optional[MemoryManager] = None,
        tools: Optional[ToolRegistry] = None,
        hooks: Optional[HookManager] = None,
        tracer: Optional[DebateTracer] = None,
        compressor: Optional[ContextCompressor] = None,
        prompt_cache: Optional[PromptCache] = None,
        prompt_evolver: Optional[Any] = None,
        game_tool_registry: Optional[GameToolRegistry] = None,
        max_tool_turns: int = 5,
    ) -> None:
        self.role = role
        self.client = client
        self.fallback_clients = fallback_clients or []
        self.max_history = max_history
        self._history: List[Dict[str, str]] = []
        self._style_directive: str = ""  # Set by StrategyBridge
        self.role_definition = role_definition

        # --- Resolve system_prompt, parse_fn, expected_fields, msg_type ---
        if role_definition is not None:
            # Generic path: derive everything from RoleDefinition
            self.system_prompt = system_prompt or role_definition.system_prompt
            if parse_fn is not None:
                self._parse_fn = parse_fn
            elif role_definition.output_schema:
                self._parse_fn = make_schema_parser(role_definition.output_schema)
            else:
                self._parse_fn = _ROLE_PARSERS.get(role, lambda d: d)
            self._expected_fields = (
                expected_fields
                if expected_fields is not None
                else schema_to_expected_fields(role_definition.output_schema)
                if role_definition.output_schema
                else []
            )
            # Map phase → MessageType
            _PHASE_MSG_TYPE = {
                "propose": MessageType.PROPOSAL,
                "challenge": MessageType.CHALLENGE,
                "evaluate": MessageType.VERDICT,
                "coordinate": MessageType.META_ACTION,
            }
            self._msg_type = msg_type or _PHASE_MSG_TYPE.get(
                role_definition.phase, MessageType.SYSTEM
            )
            self._role_default_output = role_definition.default_output
        elif role in SYSTEM_PROMPTS:
            # Legacy debate path
            self.system_prompt = system_prompt or SYSTEM_PROMPTS[role]
            self._parse_fn = parse_fn or _ROLE_PARSERS.get(role, parse_proposer_response)
            self._expected_fields = expected_fields or _ROLE_FIELDS.get(role, [])
            self._msg_type = msg_type or _ROLE_MSG_TYPE.get(role, MessageType.SYSTEM)
            self._role_default_output = None
        elif system_prompt:
            # Explicit system_prompt without role_definition
            self.system_prompt = system_prompt
            self._parse_fn = parse_fn or (lambda d: d)
            self._expected_fields = expected_fields or []
            self._msg_type = msg_type or MessageType.SYSTEM
            self._role_default_output = None
        else:
            raise ValueError(
                f"Unknown role '{role}'. Provide a role_definition, "
                f"system_prompt, or use a built-in role: {list(SYSTEM_PROMPTS.keys())}"
            )

        # Framework components
        self.memory = memory
        self.tools = tools
        self.hooks = hooks or HookManager()
        self.tracer = tracer
        self.compressor = compressor
        self.prompt_cache = prompt_cache
        self.prompt_evolver = prompt_evolver  # PromptEvolver for dynamic prompt selection

        # v3.0: Game tool registry for multi-turn tool calling
        self.game_tool_registry = game_tool_registry
        self.max_tool_turns = max_tool_turns
        self._tool_loop: Optional[ToolAugmentedAgentLoop] = None
        if game_tool_registry is not None:
            self._tool_loop = ToolAugmentedAgentLoop(
                client=self.client,
                tool_registry=game_tool_registry,
                role_name=self.role,
                max_tool_turns=max_tool_turns,
                parse_fn=self._parse_fn,
            )

        # Internal
        self._parser = RobustJSONParser()
        self._current_prompt_id: str = ""  # Track which evolved prompt is active

        # Stats
        self.total_tokens: int = 0
        self.total_calls: int = 0
        self.total_latency_ms: float = 0.0

    def act(self, user_message: str, round_num: int = 0, **kwargs: Any) -> Dict[str, Any]:
        """Generate a structured response for the current debate state.

        Parameters
        ----------
        user_message : str
            Formatted prompt describing current debate state.
        round_num : int
            Current round number.

        Returns
        -------
        parsed : dict
            Role-specific structured response.
        """
        # Before hooks
        ctx = self.hooks.trigger(
            HookPoint.BEFORE_AGENT_ACT,
            round_num=round_num, role=self.role,
            state={"user_message": user_message},
        )
        if ctx.skip_action and ctx.override_result is not None:
            return ctx.override_result

        # v3.0: Tool-augmented path
        if self._tool_loop is not None:
            return self._act_with_tools(user_message, round_num, **kwargs)

        # Legacy path: single LLM call
        return self._act_simple(user_message, round_num, **kwargs)

    def _act_with_tools(self, user_message: str, round_num: int = 0, **kwargs: Any) -> Dict[str, Any]:
        """Tool-augmented execution path: multi-turn LLM↔Tool loop."""
        turn_result: AgentTurnResult = self._tool_loop.run(
            system_prompt=self.system_prompt,
            user_message=user_message,
            history=list(self._history),
            style_directive=self._style_directive,
        )

        # Update history with final content
        raw = turn_result.raw_content or ""
        self._update_history(user_message, raw)

        parsed = turn_result.final_output
        if not parsed:
            parsed = self._default_response()

        # Attach tool call metadata
        if turn_result.tool_calls:
            parsed["_tool_calls"] = [
                {"name": tc.tool_name, "args": tc.arguments, "result": tc.result[:200]}
                for tc in turn_result.tool_calls
            ]
            parsed["_turns_used"] = turn_result.turns_used

        # Memory update
        if self.memory:
            self.memory.add_observation(
                f"[{self.role}] R{round_num}: {raw[:200]}",
                source=self.role,
                importance=parsed.get("confidence", 0.5),
            )

        # After hooks
        self.hooks.trigger(
            HookPoint.AFTER_AGENT_ACT,
            round_num=round_num, role=self.role,
            result=parsed,
        )

        return parsed

    def _act_simple(self, user_message: str, round_num: int = 0, **kwargs: Any) -> Dict[str, Any]:
        """Legacy single-call execution path (no tool loop)."""

        # Build messages with caching and compression
        messages = self._build_messages(user_message)

        # LLM call with fallback chain
        response = self._call_with_fallback(messages, **kwargs)
        self._update_history(user_message, response.content)
        self._update_stats(response)

        # Parse response
        parsed = self._parse_response(response.content)

        # Tool execution
        if self.tools and "tool_calls" in response.content:
            parsed = self._handle_tool_calls(parsed, response.content, round_num)

        # Memory update
        if self.memory:
            self.memory.add_observation(
                f"[{self.role}] R{round_num}: {response.content[:200]}",
                source=self.role,
                importance=parsed.get("confidence", 0.5),
            )

        # Record prompt evolution fitness
        if self.prompt_evolver and self._current_prompt_id:
            quality = parsed.get("quality_score", parsed.get("confidence", 0.5))
            self.prompt_evolver.record_fitness(self._current_prompt_id, quality)

        # After hooks
        self.hooks.trigger(
            HookPoint.AFTER_AGENT_ACT,
            round_num=round_num, role=self.role,
            result=parsed,
        )

        return parsed

    def _build_messages(self, user_message: str) -> List[Dict[str, str]]:
        """Build message list with caching, memory, compression, and prompt evolution."""
        # Select system prompt: evolved or default
        system = self.system_prompt
        if self.prompt_evolver is not None:
            candidate = self.prompt_evolver.select(self.role)
            if candidate.template:
                system = candidate.template
                self._current_prompt_id = candidate.prompt_id

        if self._style_directive:
            system = f"[策略指导]\n{self._style_directive}\n\n{system}"

        # Add tool descriptions
        if self.tools:
            tool_text = self.tools.to_prompt_text()
            if tool_text:
                system += f"\n\n{tool_text}"

        # Frozen memory snapshot
        frozen = ""
        if self.memory:
            frozen = self.memory.frozen_snapshot

        # Use prompt cache for prefix
        if self.prompt_cache:
            messages = self.prompt_cache.build_messages(
                self.role, system, frozen, self._history
            )
        else:
            messages = [{"role": "system", "content": system}]
            if frozen:
                messages[0]["content"] += f"\n\n{frozen}"
            messages.extend(self._history)

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        # Context compression
        if self.compressor and self.compressor.needs_compression(messages):
            messages, stats = self.compressor.compress(messages)
            logger.debug(
                "Context compressed: %s (%d→%d msgs)",
                stats.strategy_used, stats.original_messages, stats.compressed_messages,
            )

        return messages

    def _call_with_fallback(self, messages: List[Dict[str, str]], **kwargs: Any) -> LLMResponse:
        """Call LLM with automatic fallback chain."""
        clients = [self.client] + self.fallback_clients
        last_error = None

        for i, client in enumerate(clients):
            try:
                response = client.chat(messages, **kwargs)
                if i > 0:
                    logger.info("Fallback to client %d succeeded for %s", i, self.role)
                return response
            except Exception as e:
                last_error = e
                logger.warning(
                    "Client %d failed for %s: %s", i, self.role, e,
                )

        # All clients failed — return safe fallback
        logger.error("All LLM clients failed for %s: %s", self.role, last_error)
        return LLMResponse(
            content=self._safe_fallback_response(),
            model="fallback",
        )

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM response using RobustJSONParser + role-specific parser."""
        # Try robust JSON parsing first
        parsed = self._parser.parse(content, expected_fields=self._expected_fields)
        if parsed is not None:
            return self._normalize_parsed(parsed)

        # Fallback to role-specific parser
        try:
            return self._parse_fn(content)
        except Exception as e:
            logger.warning("Role parser failed for %s: %s", self.role, e)
            return self._default_response()

    def _handle_tool_calls(
        self, parsed: Dict[str, Any], content: str, round_num: int
    ) -> Dict[str, Any]:
        """Execute tool calls found in LLM response."""
        try:
            json_data = self._parser.parse(content) or {}
            calls = parse_tool_calls(json_data)
            if calls:
                results = execute_tool_calls(
                    calls, self.tools,
                    context={"role": self.role, "round_num": round_num},
                )
                # Append tool results to parsed output
                parsed["tool_results"] = results
                logger.info(
                    "Executed %d tool calls for %s", len(results), self.role,
                    extra={"role": self.role, "round_num": round_num},
                )
        except Exception as e:
            logger.warning("Tool execution failed for %s: %s", self.role, e)
        return parsed

    def _normalize_parsed(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all expected fields exist with defaults."""
        defaults = self._default_response()
        for key, default_val in defaults.items():
            if key not in parsed:
                parsed[key] = default_val
        return parsed

    def _default_response(self) -> Dict[str, Any]:
        """Role-specific default response."""
        # Generic path: use RoleDefinition.default_output()
        if self._role_default_output is not None:
            return self._role_default_output()

        # Legacy debate path
        if self.role == "proposer":
            return {"proposal": "", "confidence": 0.5, "reasoning": ""}
        elif self.role == "challenger":
            return {"challenge": "", "confidence": 0.5, "reasoning": ""}
        elif self.role == "arbiter":
            return {
                "verdict": "", "quality_score": 0.5,
                "proposal_score": 0.5, "challenge_score": 0.5,
                "reasoning": "", "consensus_reached": False,
            }
        elif self.role == "coordinator":
            return {"action_idx": 0, "reasoning": "", "expected_effect": ""}
        return {"reasoning": ""}

    def _safe_fallback_response(self) -> str:
        """Generate a safe JSON response when all LLM calls fail."""
        import json
        return json.dumps(self._default_response(), ensure_ascii=False)

    def _update_history(self, user_msg: str, assistant_msg: str) -> None:
        self._history.append({"role": "user", "content": user_msg})
        self._history.append({"role": "assistant", "content": assistant_msg})
        # Trim history
        if len(self._history) > self.max_history * 2:
            self._history = self._history[-(self.max_history * 2):]

    def _update_stats(self, response: LLMResponse) -> None:
        self.total_calls += 1
        self.total_tokens += sum(response.usage.values())
        self.total_latency_ms += response.latency_ms

    def act_message(self, user_message: str, round_num: int = 0, **kwargs: Any) -> AgentMessage:
        """Generate a structured AgentMessage response."""
        parsed = self.act(user_message, round_num=round_num, **kwargs)
        content = parsed.get("reasoning", parsed.get("proposal",
                  parsed.get("challenge", parsed.get("verdict", ""))))
        return AgentMessage(
            msg_type=self._msg_type,
            sender=self.role,
            content=content,
            data=parsed,
            round_num=round_num,
        )

    def reset(self) -> None:
        """Clear conversation history for new episode."""
        self._history.clear()
        self._style_directive = ""
        if self.memory:
            self.memory.reset_episode()

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": self.total_latency_ms / max(self.total_calls, 1),
            "parser_stats": self._parser.stats,
        }


class LLMAgentGroup:
    """Manages a group of LLM agents for a complete debate."""

    def __init__(self, agents: Dict[str, LLMAgent]) -> None:
        self._agents = agents

    def __getitem__(self, role: str) -> LLMAgent:
        return self._agents[role]

    def __contains__(self, role: str) -> bool:
        return role in self._agents

    @property
    def roles(self) -> List[str]:
        return list(self._agents.keys())

    def reset_all(self) -> None:
        for agent in self._agents.values():
            agent.reset()

    @property
    def stats(self) -> Dict[str, Any]:
        return {role: agent.stats for role, agent in self._agents.items()}
