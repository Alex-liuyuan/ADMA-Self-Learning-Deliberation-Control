"""MADDPG Trainer — pre-training and fine-tuning utility for strategy controllers.

.. deprecated::
    Use mdt_game.learning or GameEngine for new training workflows.
    This module is retained for backward compatibility with RL pre-training code.

This trainer is an internal training component, not a top-level runtime mode.

Two training phases:
  1. **Simulation Pre-training** (fast, cheap):
     Train MADDPG agents in the numerical DebateEnv.
     The continuous actions are mapped to discrete env actions.

  2. **Fusion Fine-tuning** (slow, expensive):
     Collect transitions from actual LLM debates and update
     MADDPG policies using the real reward signal.

The pre-training establishes strong strategy priors; fine-tuning
adapts them to the specific LLM model's behavior patterns.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from debate_rl_v2.config import Config
from debate_rl_v2.envs.base_env import DebateEnv, ROLES
from debate_rl_v2.agents.maddpg_agent import MADDPGAgentGroup
from debate_rl_v2.algorithms.replay_buffer import MultiAgentReplayBuffer
from debate_rl_v2.algorithms.maddpg import maddpg_update
from debate_rl_v2.core.strategy_bridge import StrategyBridge, FUSION_OBS_DIM
from debate_rl_v2.core.reward_design import EnhancedRewardComputer, DebateMetrics
from debate_rl_v2.algorithms.domain_adapter import DomainAdapter
from debate_rl_v2.algorithms.training_utils import (
    CosineAnnealingScheduler,
    QValueMonitor,
    EarlyStopping,
)
from debate_rl_v2.utils.logger import Logger
from debate_rl_v2.utils.metrics import MetricsTracker
from debate_rl_v2.logging_config import get_logger

logger = get_logger("algorithms.maddpg_trainer")


class MADDPGTrainer:
    """MADDPG trainer for debate strategy controllers.

    Pre-trains in simulation environment with continuous actions.
    The numerical DebateEnv's discrete actions are mapped from
    MADDPG's continuous outputs.

    Parameters
    ----------
    env : DebateEnv
        Numerical debate environment.
    agents : MADDPGAgentGroup
        MADDPG agents (proposer_ctrl, challenger_ctrl, arbiter_ctrl, coordinator).
    buffer : MultiAgentReplayBuffer
        Shared replay buffer.
    cfg : Config
        Full configuration.
    logger : Logger
        TensorBoard logger.
    bridge : StrategyBridge
        For observation encoding.
    """

    def __init__(
        self,
        env: DebateEnv,
        agents: MADDPGAgentGroup,
        buffer: MultiAgentReplayBuffer,
        cfg: Config,
        logger: Logger,
        bridge: Optional[StrategyBridge] = None,
        dashboard=None,
    ) -> None:
        self.env = env
        self.agents = agents
        self.buffer = buffer
        self.cfg = cfg
        self.logger = logger
        self.bridge = bridge or StrategyBridge()
        self.metrics = MetricsTracker()
        self.dashboard = dashboard  # MADDPGTrainingDashboard (optional)

        # v2: Domain adapter replaces hardcoded _encode_env_obs
        self._domain_adapter = DomainAdapter(use_learned=False)

        # v2: Training stability utilities
        self._q_monitor = QValueMonitor(
            threshold=cfg.training.q_divergence_threshold,
            patience=10,
        )
        self._early_stop = EarlyStopping(
            patience=cfg.training.early_stop_patience,
        )

        self._global_step = 0
        self._best_reward = -float("inf")

    # ------------------------------------------------------------------
    # Observation encoding for numerical env
    # ------------------------------------------------------------------

    def _encode_env_obs(self, env_obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert numerical env observations to fusion-style obs vector.

        v2: Delegates to DomainAdapter which derives confidence/trends
        from history instead of hardcoding 0.5/0.0.
        """
        return self._domain_adapter.encode_env_obs(env_obs)

    def _continuous_to_discrete(
        self,
        rl_actions: Dict[str, np.ndarray],
    ) -> Dict[str, int]:
        """Map MADDPG continuous actions to DebateEnv discrete actions.

        v2: Delegates to DomainAdapter which uses ALL 4 action dimensions
        instead of only the first 2 (fixes 50% unused gradient signal).
        """
        return self._domain_adapter.continuous_to_discrete(rl_actions, self.env)

    def _continuous_to_meta(self, coord_action: np.ndarray) -> int:
        """Map coordinator continuous action to meta discrete action (0..9)."""
        return self._domain_adapter.continuous_to_meta(coord_action)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        total_episodes: int = 5000,
        warmup_episodes: int = 100,
        batch_size: int = 256,
        gamma: float = 0.95,
        tau: float = 0.01,
        update_interval: int = 1,
        eval_interval: int = 100,
        eval_episodes: int = 20,
        save_interval: int = 500,
        log_interval: int = 10,
        gradient_clip: float = 0.5,
    ) -> None:
        """Main MADDPG training loop.

        Phase 1 (warmup): Only collect experience, no updates.
        Phase 2 (training): Collect + update every `update_interval` steps.
        """
        pbar = tqdm(range(1, total_episodes + 1), desc="MADDPG Training")
        agent_names = self.agents.agent_names

        for episode in pbar:
            # Collect one episode
            ep_stats = self._collect_episode(explore=True)

            # Training updates (after warmup)
            update_stats = {}
            if episode > warmup_episodes and episode % update_interval == 0:
                update_stats = maddpg_update(
                    agents=self.agents,
                    buffer=self.buffer,
                    batch_size=batch_size,
                    gamma=gamma,
                    tau=tau,
                    gradient_clip=gradient_clip,
                )

            # Logging
            if episode % log_interval == 0:
                self._log_training(episode, ep_stats, update_stats)

            # Dashboard: record episode data
            if self.dashboard is not None:
                coord_obs = self.env.coordinator_obs if hasattr(self.env, 'coordinator_obs') else None
                lam = float(coord_obs[3]) if coord_obs is not None and len(coord_obs) > 3 else 0.5
                self.dashboard.record_episode(
                    episode=episode,
                    total_reward=ep_stats.get('total_reward', 0),
                    task_reward=ep_stats.get('task_reward', 0),
                    steps=ep_stats.get('steps', 0),
                    disagreement=ep_stats.get('disagreement', 0.5),
                    lambda_adv=lam,
                    meta_quality=ep_stats.get('meta_avg_quality', 0),
                    meta_convergence=ep_stats.get('meta_convergence', 0),
                    meta_efficiency=ep_stats.get('meta_efficiency', 0),
                    buffer_size=len(self.buffer),
                )
                if update_stats:
                    self.dashboard.record_losses(update_stats)
                if self.dashboard.should_update(episode):
                    self.dashboard.update()

            # Evaluation
            if episode % eval_interval == 0:
                eval_stats = self.evaluate(eval_episodes)
                self._log_eval(episode, eval_stats)
                # Dashboard: record eval
                if self.dashboard is not None:
                    self.dashboard.record_eval(
                        episode=episode,
                        task_reward=eval_stats.get('eval/task_reward', 0),
                        task_std=eval_stats.get('eval/task_reward_std', 0),
                        mean_steps=eval_stats.get('eval/mean_steps', 0),
                        disagreement=eval_stats.get('eval/final_disagreement', 0),
                    )
                    self.dashboard.update()

                # v2: Early stopping check
                eval_reward = eval_stats.get('eval/task_reward', 0)
                if self._early_stop.update(eval_reward):
                    logger.info("Early stopping at episode %d (best=%.4f)", episode, self._early_stop.best_value)
                    break

            # v2: Q-value monitoring (from update stats)
            if update_stats:
                for agent_name, stats in update_stats.items():
                    if isinstance(stats, dict) and "critic_loss" in stats:
                        q_val = stats.get("q_mean", stats.get("critic_loss", 0))
                        if self._q_monitor.update(q_val):
                            # Reduce LR on divergence
                            for agent in self.agents.agents.values():
                                if hasattr(agent, 'actor_optimizer'):
                                    for pg in agent.actor_optimizer.param_groups:
                                        pg["lr"] *= 0.5
                                if hasattr(agent, 'critic_optimizer'):
                                    for pg in agent.critic_optimizer.param_groups:
                                        pg["lr"] *= 0.5
                            logger.warning("Q-value divergence: reduced LR by 50%%")
                            break

            # Checkpoint
            if episode % save_interval == 0:
                self._save_checkpoint(episode)

            # Progress bar
            pbar.set_postfix({
                "rew": f"{ep_stats.get('total_reward', 0):.3f}",
                "task": f"{ep_stats.get('task_reward', 0):.3f}",
                "buf": f"{len(self.buffer)}/{self.buffer.capacity}",
            })

        # Final snapshot
        if self.dashboard is not None:
            self.dashboard.update()
            self.dashboard.save_snapshot()
            self.dashboard.generate_static_report()

        self._save_checkpoint("final")
        self.logger.close()

    def _collect_episode(self, explore: bool = True) -> Dict[str, float]:
        """Run one episode, collecting transitions into replay buffer."""
        env = self.env
        env_obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        # v2: Reset domain adapter history for new episode
        self._domain_adapter.reset()

        prev_fusion_obs = self._encode_env_obs(env_obs)

        while not done:
            fusion_obs = self._encode_env_obs(env_obs)

            # RL agents produce continuous actions
            rl_actions = {}
            for role in self.agents.agent_names:
                action = self.agents[role].act(fusion_obs, explore=explore)
                rl_actions[role] = action

            # Map to discrete env actions
            discrete_actions = self._continuous_to_discrete(rl_actions)
            meta_action = None
            if env.t % env.meta_interval == 0:
                coord_act = rl_actions.get("coordinator", np.zeros(5))
                meta_action = self._continuous_to_meta(coord_act)

            # Step environment
            step_out = env.step(actions=discrete_actions, meta_action=meta_action)
            next_env_obs = step_out.obs
            done = step_out.done

            next_fusion_obs = self._encode_env_obs(next_env_obs)

            # Compute per-agent rewards using enhanced reward system
            rewards = env.enhanced_reward.compute_numerical_rewards(
                env_rewards=step_out.rewards,
                env_info=step_out.info,
                metrics=env.debate_metrics,
                done=done,
            )

            # Store transition
            obs_dict = {role: fusion_obs for role in self.agents.agent_names}
            next_obs_dict = {role: next_fusion_obs for role in self.agents.agent_names}
            self.buffer.add(obs_dict, rl_actions, rewards, next_obs_dict, done)

            total_reward += sum(rewards.values())
            steps += 1
            env_obs = next_env_obs
            prev_fusion_obs = next_fusion_obs

        task_reward = step_out.info.get("task_reward", 0.0)

        # Compute meta rewards for logging
        meta_rewards = env.enhanced_reward.compute_meta_rewards(
            metrics=env.debate_metrics,
            max_rounds=env.max_steps,
        )

        return {
            "total_reward": total_reward,
            "task_reward": task_reward,
            "steps": steps,
            "disagreement": step_out.info.get("disagreement", 1.0),
            "meta_avg_quality": meta_rewards.get("raw_avg_quality", 0.0),
            "meta_convergence": meta_rewards.get("raw_convergence_speed", 0.0),
            "meta_efficiency": meta_rewards.get("raw_efficiency", 0.0),
        }

    def evaluate(self, num_episodes: int = 20) -> Dict[str, float]:
        """Run deterministic evaluation (no exploration noise)."""
        task_rewards = []
        total_rewards = []
        steps_list = []
        disagreements = []

        for _ in range(num_episodes):
            stats = self._collect_episode(explore=False)
            total_rewards.append(stats["total_reward"])
            task_rewards.append(stats["task_reward"])
            steps_list.append(stats["steps"])
            disagreements.append(stats["disagreement"])

        return {
            "eval/total_reward": np.mean(total_rewards),
            "eval/task_reward": np.mean(task_rewards),
            "eval/task_reward_std": np.std(task_rewards),
            "eval/mean_steps": np.mean(steps_list),
            "eval/final_disagreement": np.mean(disagreements),
        }

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_training(self, episode, ep_stats, update_stats):
        step = self._global_step
        for key, val in ep_stats.items():
            self.logger.log_scalar(f"maddpg/train/{key}", val, step)

        for agent_name, stats in update_stats.items():
            if isinstance(stats, dict):
                for key, val in stats.items():
                    if key != "td_errors":
                        self.logger.log_scalar(f"maddpg/{agent_name}/{key}", val, step)

        self.logger.log_scalar("maddpg/buffer_size", len(self.buffer), step)
        self._global_step += 1

        logger.info(
            "Ep %d: reward=%.3f task=%.3f steps=%.0f buffer=%d",
            episode, ep_stats['total_reward'], ep_stats['task_reward'],
            ep_stats['steps'], len(self.buffer),
        )

    def _log_eval(self, episode, eval_stats):
        step = self._global_step
        for key, val in eval_stats.items():
            self.logger.log_scalar(key, val, step)

        task_r = eval_stats["eval/task_reward"]
        logger.info(
            "EVAL Ep %d: task_reward=%.3f±%.3f steps=%.1f disagreement=%.3f",
            episode, task_r, eval_stats['eval/task_reward_std'],
            eval_stats['eval/mean_steps'], eval_stats['eval/final_disagreement'],
        )

        if task_r > self._best_reward:
            self._best_reward = task_r
            self._save_checkpoint("best")

    def _save_checkpoint(self, tag) -> None:
        """Save versioned checkpoint with architecture hash validation."""
        from debate_rl_v2.algorithms.checkpoint import save_checkpoint
        ckpt_dir = os.path.join(
            self.cfg.training.checkpoint_dir, "maddpg", str(tag)
        )
        # Legacy per-agent save
        self.agents.save_all(ckpt_dir)
        # v2: Versioned checkpoint with architecture hash
        ckpt_path = os.path.join(ckpt_dir, "trainer_state.pt")
        agent_states = {}
        for name in self.agents.agent_names:
            agent = self.agents[name]
            state = {}
            if hasattr(agent, 'actor') and hasattr(agent.actor, 'state_dict'):
                state["actor"] = agent.actor.state_dict()
            if hasattr(agent, 'critic') and hasattr(agent.critic, 'state_dict'):
                state["critic"] = agent.critic.state_dict()
            agent_states[name] = state
        save_checkpoint(
            ckpt_path, self.cfg, agent_states,
            episode=self._global_step,
            extra={
                "best_reward": self._best_reward,
                "q_monitor_stats": self._q_monitor.stats,
                "early_stop_best": self._early_stop.best_value,
            },
        )
