"""通用博弈场景协议 — 新场景只需实现这些方法。

对比 BaseFusionEnvironment 的 3 个抽象方法，GameScenario 提供更完整的
工具增强场景接口，支持：
  - 工具注册 (register_tools)
  - Episode 创建 (create_episode)
  - 角色 prompt 构建 (build_role_prompt)
  - 状态更新 (update_state)
  - 终止检查 (check_terminal)
  - 奖励计算（可使用工具验证）(compute_rewards)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from debate_rl_v2.agents.tool_agent_loop import AgentTurnResult
from debate_rl_v2.framework.roles import RoleRegistry
from debate_rl_v2.framework.tool_context import GameToolContext
from debate_rl_v2.framework.tool_registry import GameToolRegistry
from debate_rl_v2.framework.types import CollaborationState


@dataclass
class RoundStep:
    """A single execution step within one collaboration round."""

    role_name: str
    stage: str = "main"
    update_state: bool = False
    label: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


class GameScenario(ABC):
    """通用博弈场景协议 — 新场景只需实现这些方法。

    接入新场景的步骤：
    1. 继承 GameScenario
    2. 实现 setup / register_tools / create_episode / build_role_prompt /
       update_state / check_terminal / compute_rewards
    3. 将场景实例传给 GameEngine.run_episode()

    对比 hermes-agent 的 HermesAgentBaseEnv:
    - setup()          → setup()
    - get_next_item()  → create_episode()
    - format_prompt()  → build_role_prompt()
    - compute_reward() → compute_rewards()
    - evaluate()       → evaluate()
    """

    @abstractmethod
    def setup(self) -> None:
        """初始化场景（加载数据集、配置工具等）"""

    @abstractmethod
    def register_tools(self, registry: GameToolRegistry) -> None:
        """注册场景特有工具到工具注册表"""

    @abstractmethod
    def create_episode(self) -> dict[str, Any]:
        """创建一个新的博弈 episode。

        Returns
        -------
        dict
            初始状态/上下文，至少包含 "topic" 字段。
            会被存入 CollaborationState.metadata。
        """

    @abstractmethod
    def build_role_prompt(
        self,
        role_name: str,
        round_num: int,
        state: CollaborationState,
    ) -> str:
        """为指定角色构建当前回合的 prompt。

        Parameters
        ----------
        role_name : str
            角色名称
        round_num : int
            当前回合号（1-indexed）
        state : CollaborationState
            当前协作状态
        """

    def get_round_steps(
        self,
        role_registry: RoleRegistry,
        round_num: int,
        state: CollaborationState,
        meta_interval: int,
    ) -> list[RoundStep]:
        """Return the execution plan for one round.

        Default behavior preserves the original GameEngine order:
        coordinator (at meta_interval), then base roles, then evaluator.
        """
        phase_order = ["coordinate", "propose", "challenge", "evaluate", "custom"]
        ordered_roles = sorted(
            role_registry.get_roles(),
            key=lambda role: (
                phase_order.index(role.phase)
                if role.phase in phase_order else len(phase_order)
            ),
        )

        steps: list[RoundStep] = []
        coordinator = role_registry.coordinator
        evaluator = role_registry.evaluator

        if coordinator and (round_num % meta_interval == 1 or round_num == 1):
            steps.append(RoundStep(role_name=coordinator.name, stage="main"))

        for role_def in ordered_roles:
            if role_def.is_coordinator or role_def.is_evaluator:
                continue
            steps.append(RoundStep(role_name=role_def.name, stage="main"))

        if evaluator:
            steps.append(RoundStep(role_name=evaluator.name, stage="main"))

        return steps

    def build_step_prompt(
        self,
        step: RoundStep,
        round_num: int,
        state: CollaborationState,
    ) -> str:
        """Build a prompt for a specific round step.

        Subclasses can override this to support richer multi-stage debates.
        """
        return self.build_role_prompt(step.role_name, round_num, state)

    @abstractmethod
    def update_state(
        self,
        role_outputs: dict[str, AgentTurnResult | dict[str, Any]],
        state: CollaborationState,
        round_num: int,
    ) -> None:
        """从智能体输出更新协作状态。

        Parameters
        ----------
        role_outputs : dict
            {role_name: AgentTurnResult 或 parsed dict}
        state : CollaborationState
            当前状态（就地修改）
        round_num : int
            当前回合号
        """

    @abstractmethod
    def check_terminal(self, state: CollaborationState) -> bool:
        """检查是否达到终止条件。"""

    @abstractmethod
    def compute_rewards(
        self,
        state: CollaborationState,
        history: list[dict[str, Any]],
        ctx: GameToolContext,
    ) -> dict[str, float]:
        """计算各角色奖励（可使用工具验证）。

        Parameters
        ----------
        state : CollaborationState
            最终状态
        history : list
            完整回合历史
        ctx : GameToolContext
            工具访问上下文（可调用工具进行验证）

        Returns
        -------
        dict
            {role_name: reward_value}
        """

    def evaluate(self, episodes: list[dict[str, Any]]) -> dict[str, float]:
        """周期性评估（可选）。默认返回空。"""
        return {}

    def finalize_episode(
        self,
        state: CollaborationState,
        history: list[dict[str, Any]],
    ) -> None:
        """Optional hook for scenario-specific end-of-episode cleanup."""
        return None
