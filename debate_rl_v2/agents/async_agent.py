"""Async Agent Execution — parallel LLM calls for Proposer/Challenger.

Fixes the 4x serial latency problem. Proposer and Challenger can run
in parallel since they both operate on the same previous-round state.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Coroutine

from debate_rl_v2.logging_config import get_logger

logger = get_logger("agents.async_agent")


async def parallel_agent_calls(
    *coros: Coroutine[Any, Any, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run multiple agent calls in parallel.

    Usage::

        prop_result, chal_result = await parallel_agent_calls(
            proposer.async_act(msg1),
            challenger.async_act(msg2),
        )
    """
    start = time.monotonic()
    results = await asyncio.gather(*coros, return_exceptions=True)
    elapsed = (time.monotonic() - start) * 1000

    processed: list[dict[str, Any]] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error("Parallel agent call %d failed: %s", i, result)
            processed.append(_fallback_result(str(result)))
        else:
            processed.append(result)

    logger.info(
        "Parallel agent calls completed: %d calls in %.0fms",
        len(coros), elapsed,
    )
    return processed


async def run_with_timeout(
    coro: Coroutine[Any, Any, dict[str, Any]],
    timeout: float = 60.0,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a coroutine with timeout and fallback."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("Agent call timed out after %.1fs", timeout)
        return fallback or _fallback_result("timeout")
    except Exception as e:
        logger.error("Agent call failed: %s", e)
        return fallback or _fallback_result(str(e))


def _fallback_result(error: str) -> dict[str, Any]:
    """Generate a safe fallback result when an agent call fails."""
    return {
        "proposal": f"[Agent error: {error}]",
        "challenge": f"[Agent error: {error}]",
        "verdict": f"[Agent error: {error}]",
        "confidence": 0.5,
        "quality_score": 0.5,
        "proposal_score": 0.5,
        "challenge_score": 0.5,
        "reasoning": f"Error: {error}",
        "action_idx": 0,
        "expected_effect": "",
        "consensus_reached": False,
        "error": error,
    }


class AsyncAgentWrapper:
    """Wraps a synchronous LLMAgent to provide async interface.

    Uses asyncio.to_thread to run blocking LLM calls in a thread pool.
    """

    def __init__(self, agent: Any) -> None:
        self._agent = agent

    async def async_act(self, message: str, round_num: int = 0) -> dict[str, Any]:
        """Async version of agent.act()."""
        return await asyncio.to_thread(
            self._agent.act, message, round_num=round_num
        )

    @property
    def role(self) -> str:
        return getattr(self._agent, "role", "unknown")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)


class DelegateAgent:
    """Sub-agent delegation — hermes-agent inspired.

    Allows complex sub-tasks to be delegated to specialized sub-agents.
    For example, a proposer might delegate evidence gathering to a
    research sub-agent.
    """

    def __init__(
        self,
        name: str,
        task_description: str,
        agent: Any,
    ) -> None:
        self.name = name
        self.task_description = task_description
        self._agent = agent
        self._wrapper = AsyncAgentWrapper(agent)

    async def execute(self, context: str, round_num: int = 0) -> dict[str, Any]:
        """Execute the delegated task."""
        prompt = (
            f"[子任务委派: {self.task_description}]\n\n"
            f"上下文:\n{context}\n\n"
            f"请完成上述子任务并返回结果。"
        )
        return await self._wrapper.async_act(prompt, round_num=round_num)
