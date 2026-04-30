"""Hierarchical Trainer — Dual Timescale Framework (Section 5.1).

.. deprecated::
    Use GameEngine for new training workflows.
    This module is retained for backward compatibility with RL pre-training code.

Implements the two-tier training loop:
  - Upper MDP (slow timescale): Coordinator learns meta-policy π_meta(g|s_meta)
  - Lower MDP (fast timescale): Proposer, Challenger, Arbiter learn π_base(a|s,g)

Convergence guarantee (Theorem 6): under dual-timescale stochastic
approximation (Borkar, 1997), the coupled system converges to a locally
optimal hierarchical policy.
"""

from __future__ import annotations
from debate_rl_v2.logging_config import get_logger
logger = get_logger("algorithms.hierarchical")

import os
import time
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from debate_rl_v2.config import Config
from debate_rl_v2.envs.base_env import DebateEnv, ROLES
from debate_rl_v2.agents.ppo_agent import AgentGroup
from debate_rl_v2.algorithms.buffers import MultiRoleBuffer
from debate_rl_v2.algorithms.mappo import mappo_update
from debate_rl_v2.algorithms.credit_assignment import ShapleyCredit
from debate_rl_v2.utils.logger import Logger
from debate_rl_v2.utils.metrics import MetricsTracker
from debate_rl_v2.visualization.live_dashboard import LiveTrainingDashboard


class HierarchicalTrainer:
    """Dual-timescale hierarchical MAPPO trainer.

    Parameters
    ----------
    env : DebateEnv
        The debate environment.
    agents : AgentGroup
        All four role agents.
    cfg : Config
        Full configuration.
    logger : Logger
        TensorBoard logger.
    """

    def __init__(
        self,
        env: DebateEnv,
        agents: AgentGroup,
        cfg: Config,
        logger: Logger,
        dashboard: LiveTrainingDashboard | None = None,
    ) -> None:
        self.env = env
        self.agents = agents
        self.cfg = cfg
        self.logger = logger
        self.device = cfg.resolve_device()

        self.buffers = MultiRoleBuffer()
        self.metrics = MetricsTracker()

        if cfg.credit.use_shapley:
            self.shapley = ShapleyCredit(
                num_agents=4,
                num_samples=cfg.credit.shapley_samples,
            )
        else:
            self.shapley = None

        # --- Live visualization dashboard ---
        self.dashboard = dashboard

        self._global_step = 0
        self._best_reward = -float("inf")

        # --- 端到端训练：核心机制优化器 ---
        # SemanticEmbedder 优化器（较小学习率，避免不稳定）
        self.embedder_opt = torch.optim.Adam(
            self.env.embedder.parameters(),
            lr=cfg.knowledge.confidence_lr * 0.1,
        )
        # KnowledgeEngine predicate 网络 + threshold 优化器
        self.knowledge_opt = torch.optim.Adam(
            self.env.knowledge_engine.predicate_parameters(),
            lr=cfg.knowledge.confidence_lr,
        )
        # 核心机制参数优化器（AdversarialController, SoftSwitch, DevilAdvocate）
        mechanism_params = []
        for module in (self.env.adv_controller, self.env.soft_switch, self.env.devil_advocate):
            if isinstance(module, torch.nn.Module):
                mechanism_params.extend(module.parameters())
        if mechanism_params:
            self.mechanism_opt = torch.optim.Adam(mechanism_params, lr=1e-4)
        else:
            self.mechanism_opt = None

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Main training loop."""
        cfg = self.cfg
        tc = cfg.training
        pbar = tqdm(range(1, tc.total_episodes + 1), desc="Training")

        for episode in pbar:
            # Collect rollout episodes
            episode_stats = self._collect_rollouts(tc.rollout_episodes)

            # --- Lower level update (fast timescale) ---
            base_stats = {}
            for role in ("proposer", "challenger", "arbiter"):
                buf = self.buffers[role]
                if len(buf) > 0:
                    shapley_coef = cfg.credit.correction_coef if self.shapley else 0.0
                    stats = mappo_update(
                        self.agents[role],
                        buf,
                        cfg.ppo,
                        shapley_coef=shapley_coef,
                    )
                    base_stats[role] = stats

            # --- Upper level update (slow timescale) ---
            meta_stats = {}
            if episode % cfg.hierarchical.meta_update_interval == 0:
                buf = self.buffers["coordinator"]
                if len(buf) > 0:
                    from debate_rl_v2.config import PPOConfig
                    meta_ppo = PPOConfig(
                        gamma=cfg.hierarchical.meta_gamma,
                        gae_lambda=cfg.ppo.gae_lambda,
                        clip_ratio=cfg.ppo.clip_ratio,
                        pi_lr=cfg.hierarchical.meta_lr,
                        vf_lr=cfg.hierarchical.meta_lr,
                        entropy_coef=cfg.ppo.entropy_coef * 2,  # more exploration for meta
                        max_grad_norm=cfg.ppo.max_grad_norm,
                        train_epochs=cfg.hierarchical.meta_train_epochs,
                        minibatch_size=min(cfg.ppo.minibatch_size, len(buf)),
                    )
                    meta_stats = mappo_update(
                        self.agents["coordinator"],
                        buf,
                        meta_ppo,
                    )

            # --- Knowledge engine confidence update ---
            self._update_knowledge_confidence(episode_stats)

            # --- 端到端辅助损失更新 ---
            aux_stats = self._update_auxiliary_losses(episode)

            # --- Rule mining ---
            if self.env.rule_miner.should_mine():
                states, compliant = self.env.rule_miner.get_mining_data()
                added = self.env.knowledge_engine.mine_rule(states, compliant)
                if added:
                    self.logger.log_scalar("knowledge/rules_mined", 1.0, self._global_step)
                    if self.dashboard:
                        self.dashboard.record_rule_mined(episode)

            # Clear buffers
            self.buffers.clear_all()

            # --- Live dashboard data recording ---
            if self.dashboard:
                self._record_dashboard_data(episode, episode_stats, base_stats, meta_stats)

            # --- Logging ---
            if episode % tc.log_interval == 0:
                self._log_training(episode, episode_stats, base_stats, meta_stats)

            # --- Evaluation ---
            eval_stats = None
            if episode % tc.eval_interval == 0:
                eval_stats = self.evaluate(tc.eval_episodes)
                self._log_eval(episode, eval_stats)
                if self.dashboard:
                    self.dashboard.record_eval(
                        episode=episode,
                        task_reward=eval_stats.get("eval/task_reward", 0.0),
                        task_reward_std=eval_stats.get("eval/task_reward_std", 0.0),
                        steps=eval_stats.get("eval/mean_steps", 0.0),
                        disagreement=eval_stats.get("eval/final_disagreement", 0.0),
                        da_pass_rate=eval_stats.get("eval/da_pass_rate", 0.0),
                    )

            # --- Checkpointing ---
            if episode % tc.save_interval == 0:
                self._save_checkpoint(episode)

            # --- Live dashboard update ---
            if self.dashboard and self.dashboard.should_update(episode):
                self.dashboard.update()

            # Progress bar
            pbar.set_postfix({
                "rew": f"{episode_stats.get('mean_reward', 0):.3f}",
                "task": f"{episode_stats.get('mean_task_reward', 0):.3f}",
                "steps": f"{episode_stats.get('mean_steps', 0):.1f}",
            })

        # Final snapshot
        if self.dashboard:
            self.dashboard.update()
            self.dashboard.save_snapshot()

        self._save_checkpoint("final")
        self.logger.close()

    # ------------------------------------------------------------------
    # Live dashboard data recording
    # ------------------------------------------------------------------

    def _record_dashboard_data(
        self,
        episode: int,
        episode_stats: Dict[str, float],
        base_stats: Dict[str, Dict],
        meta_stats: Dict,
    ) -> None:
        """Feed current episode data into the live dashboard."""
        db = self.dashboard
        if db is None:
            return

        # Episode-level metrics
        hist = self.env.adv_controller.history
        db.record_episode(
            episode=episode,
            total_reward=episode_stats.get("mean_reward", 0.0),
            task_reward=episode_stats.get("mean_task_reward", 0.0),
            steps=episode_stats.get("mean_steps", 0.0),
            da_pass_rate=episode_stats.get("da_pass_rate", 0.0),
            lambda_adv=hist.lambda_adv[-1] if hist.lambda_adv else 0.5,
            disagreement=hist.disagreement[-1] if hist.disagreement else 0.5,
            time_pressure=hist.time_pressure[-1] if hist.time_pressure else 0.0,
        )

        # Per-role PPO stats
        for role, stats in base_stats.items():
            db.record_ppo_stats(role, stats)
        if meta_stats:
            db.record_ppo_stats("coordinator", meta_stats)

        # Knowledge rule confidences
        import torch as _torch
        rule_confs = []
        for rule in self.env.knowledge_engine.rules:
            with _torch.no_grad():
                rule_confs.append(float(_torch.sigmoid(rule._confidence_logit).item()))
        db.record_rule_confidence(rule_confs)

        # Soft-switch mode counts (from evidence chain)
        evidence = self.env.evidence_chain
        std_count = sum(1 for e in evidence._records if getattr(e, 'mode', 'standard') == 'standard')
        cb_count = sum(1 for e in evidence._records if getattr(e, 'mode', '') == 'challenger_boost')
        ai_count = sum(1 for e in evidence._records if getattr(e, 'mode', '') == 'arbiter_intervene')
        db.record_mode_counts(
            standard=std_count,
            challenger_boost=cb_count,
            arbiter_intervene=ai_count,
        )

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _collect_rollouts(self, num_episodes: int) -> Dict[str, float]:
        """Collect rollout data from multiple episodes."""
        total_rewards = []
        task_rewards = []
        steps_list = []
        da_passes = []

        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False
            ep_reward = 0.0
            current_meta_action = None

            while not done:
                actions = {}
                meta_action = None

                # Coordinator (every meta_interval steps)
                if self.env.t % self.env.meta_interval == 0:
                    o = torch.tensor(obs["coordinator"], dtype=torch.float32)
                    act, logp, val = self.agents["coordinator"].act(o)
                    meta_action = act
                    current_meta_action = act
                    self.buffers["coordinator"].add(
                        obs["coordinator"], act, 0.0, val, logp, 0.0
                    )

                # Base agents
                for role in ("proposer", "challenger", "arbiter"):
                    o = torch.tensor(obs[role], dtype=torch.float32)
                    act, logp, val = self.agents[role].act(o)
                    actions[role] = act
                    self.buffers[role].add(
                        obs[role], act, 0.0, val, logp, 0.0
                    )

                # Environment step
                step_out = self.env.step(actions=actions, meta_action=meta_action)
                obs = step_out.obs
                done = step_out.done

                # Store rewards and done flags
                for role in ("proposer", "challenger", "arbiter"):
                    buf = self.buffers[role]
                    buf.rew[-1] = step_out.rewards[role]
                    buf.done[-1] = float(done)

                if meta_action is not None:
                    # Coordinator reward: combination of base agents performance
                    meta_reward = sum(step_out.rewards[r] for r in ("proposer", "challenger", "arbiter")) / 3.0
                    meta_reward -= self.cfg.reward.meta_convergence_penalty * self.env.t / self.env.max_steps
                    self.buffers["coordinator"].rew[-1] = meta_reward
                    self.buffers["coordinator"].done[-1] = float(done)

                ep_reward += sum(step_out.rewards.values())
                self._global_step += 1

            # Episode stats
            total_rewards.append(ep_reward)
            task_reward = step_out.info.get("task_reward", 0.0)
            task_rewards.append(task_reward)
            steps_list.append(self.env.t)

            da_info = step_out.info.get("devil_advocate", {})
            if da_info.get("confirmed", False):
                da_passes.append(1.0)
            elif da_info.get("triggered", False):
                da_passes.append(0.0)

            # Shapley credit computation
            if self.shapley is not None:
                corrections = self.shapley.compute_corrections(
                    self.buffers, self.agents, self.env
                )
                for role, corr_vals in corrections.items():
                    for cv in corr_vals:
                        self.buffers[role].add_shapley(cv)

        return {
            "mean_reward": np.mean(total_rewards),
            "mean_task_reward": np.mean(task_rewards),
            "mean_steps": np.mean(steps_list),
            "da_pass_rate": np.mean(da_passes) if da_passes else 0.0,
        }

    # ------------------------------------------------------------------
    # Knowledge confidence update
    # ------------------------------------------------------------------

    def _update_knowledge_confidence(self, stats: Dict[str, float]) -> None:
        """Update rule confidence via task reward gradient (Section 4.3)."""
        ke = self.env.knowledge_engine
        task_reward = stats.get("mean_task_reward", 0.0)

        # Use task reward as negative loss signal for confidence params
        for rule in ke.rules:
            # Simple gradient-free update: boost confidence for rules that
            # correlate with high task reward
            with torch.no_grad():
                adjustment = self.cfg.knowledge.confidence_lr * (task_reward - 0.5)
                rule._confidence_logit.data += adjustment

    # ------------------------------------------------------------------
    # Auxiliary losses for end-to-end training
    # ------------------------------------------------------------------

    def _update_auxiliary_losses(self, episode: int) -> Dict[str, float]:
        """更新 SemanticEmbedder 和 KnowledgeEngine predicates。

        通过辅助损失让冻结参数参与端到端训练：
        - Embedder: 时序平滑性 + 目标对齐
        - Knowledge: 合规度预测监督学习
        - Mechanisms: 任务奖励驱动的策略梯度（如果已参数化）
        """
        stats: Dict[str, float] = {}

        # --- SemanticEmbedder 更新 ---
        emb_loss = self.env.compute_embedder_loss()
        if emb_loss is not None:
            self.embedder_opt.zero_grad()
            emb_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.env.embedder.parameters(), 0.5
            )
            self.embedder_opt.step()
            stats["aux/embedder_loss"] = emb_loss.item()
            self.logger.log_scalar("aux/embedder_loss", emb_loss.item(), self._global_step)

        # --- KnowledgeEngine predicate 更新（每 5 个 episode）---
        if episode % 5 == 0:
            know_loss = self.env.compute_knowledge_loss()
            if know_loss is not None:
                self.knowledge_opt.zero_grad()
                know_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.env.knowledge_engine.predicate_parameters(), 0.5
                )
                self.knowledge_opt.step()
                stats["aux/knowledge_loss"] = know_loss.item()
                self.logger.log_scalar("aux/knowledge_loss", know_loss.item(), self._global_step)

        # --- 核心机制参数更新（每 10 个 episode，如果已参数化）---
        if self.mechanism_opt is not None and episode % 10 == 0:
            mech_loss = self._compute_mechanism_loss()
            if mech_loss is not None:
                self.mechanism_opt.zero_grad()
                mech_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self._mechanism_parameters() if p.grad is not None],
                    0.5,
                )
                self.mechanism_opt.step()
                stats["aux/mechanism_loss"] = mech_loss.item()
                self.logger.log_scalar("aux/mechanism_loss", mech_loss.item(), self._global_step)

        return stats

    def _mechanism_parameters(self):
        """收集所有核心机制的可学习参数。"""
        params = []
        for module in (self.env.adv_controller, self.env.soft_switch, self.env.devil_advocate):
            if isinstance(module, torch.nn.Module):
                params.extend(module.parameters())
        return params

    def _compute_mechanism_loss(self) -> Optional[torch.Tensor]:
        """计算核心机制参数的损失。

        目标：让机制参数朝着提升任务奖励的方向更新。
        - AdversarialController: eta/alpha/omega 通过正则化约束到合理范围
        - SoftSwitch: tau_low < tau_high 约束 + 间距正则化
        - DevilAdvocate: eps_d/eps_p/delta 通过任务奖励信号调整
        """
        loss = torch.tensor(0.0)

        # --- AdversarialController 正则化 ---
        # eta 不宜太大（不稳定）也不宜太小（学不动）
        ac = self.env.adv_controller
        eta_t = torch.sigmoid(ac._eta_logit)
        loss = loss + 0.1 * (eta_t - 0.2) ** 2  # 偏好 eta ≈ 0.2

        # alpha 偏好中间值
        alpha_t = torch.sigmoid(ac._alpha_logit)
        loss = loss + 0.05 * (alpha_t - 0.5) ** 2

        # --- SoftSwitch 约束 ---
        ss = self.env.soft_switch
        tau_low_t = torch.sigmoid(ss._tau_low_logit)
        tau_high_t = torch.sigmoid(ss._tau_high_logit)
        # tau_low < tau_high 约束（软约束）
        margin = tau_high_t - tau_low_t
        loss = loss + torch.relu(0.2 - margin)  # 至少 0.2 间距

        # --- DevilAdvocate 约束 ---
        da = self.env.devil_advocate
        eps_d_t = torch.sigmoid(da._eps_d_logit)
        delta_t = torch.sigmoid(da._delta_logit)
        # delta > eps_d 约束（重激活阈值应高于共识阈值）
        loss = loss + torch.relu(eps_d_t - delta_t + 0.1)

        # eps_d 偏好较小值（更严格的共识检测）
        loss = loss + 0.05 * eps_d_t

        return loss

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, num_episodes: int = 20) -> Dict[str, float]:
        """Run deterministic evaluation episodes."""
        task_rewards = []
        steps_list = []
        disagreements = []
        da_passes = []

        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False

            while not done:
                actions = {}
                meta_action = None

                if self.env.t % self.env.meta_interval == 0:
                    o = torch.tensor(obs["coordinator"], dtype=torch.float32)
                    act, _, _ = self.agents["coordinator"].act(o, deterministic=True)
                    meta_action = act

                for role in ("proposer", "challenger", "arbiter"):
                    o = torch.tensor(obs[role], dtype=torch.float32)
                    act, _, _ = self.agents[role].act(o, deterministic=True)
                    actions[role] = act

                step_out = self.env.step(actions=actions, meta_action=meta_action)
                obs = step_out.obs
                done = step_out.done

            task_rewards.append(step_out.info.get("task_reward", 0.0))
            steps_list.append(self.env.t)
            disagreements.append(step_out.info.get("disagreement", 1.0))

            da_info = step_out.info.get("devil_advocate", {})
            if da_info.get("confirmed", False):
                da_passes.append(1.0)
            elif da_info.get("triggered", False):
                da_passes.append(0.0)

        return {
            "eval/task_reward": np.mean(task_rewards),
            "eval/task_reward_std": np.std(task_rewards),
            "eval/mean_steps": np.mean(steps_list),
            "eval/final_disagreement": np.mean(disagreements),
            "eval/da_pass_rate": np.mean(da_passes) if da_passes else 0.0,
        }

    # ------------------------------------------------------------------
    # Logging & Checkpointing
    # ------------------------------------------------------------------

    def _log_training(
        self,
        episode: int,
        ep_stats: Dict,
        base_stats: Dict,
        meta_stats: Dict,
    ) -> None:
        step = self._global_step
        for key, val in ep_stats.items():
            self.logger.log_scalar(f"train/{key}", val, step)

        for role, stats in base_stats.items():
            for key, val in stats.items():
                self.logger.log_scalar(f"train/{role}/{key}", val, step)

        if meta_stats:
            for key, val in meta_stats.items():
                self.logger.log_scalar(f"train/coordinator/{key}", val, step)

        # Adversarial intensity history
        hist = self.env.adv_controller.history
        if hist.lambda_adv:
            self.logger.log_scalar("dynamics/lambda_adv", hist.lambda_adv[-1], step)
        if hist.disagreement:
            self.logger.log_scalar("dynamics/disagreement", hist.disagreement[-1], step)

        logger.info(
            "Ep %d: reward=%.3f task=%.3f steps=%.1f da_pass=%.2f",
            episode, ep_stats['mean_reward'], ep_stats['mean_task_reward'],
            ep_stats['mean_steps'], ep_stats.get('da_pass_rate', 0),
        )

    def _log_eval(self, episode: int, eval_stats: Dict) -> None:
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
        ckpt_dir = os.path.join(self.cfg.training.checkpoint_dir, str(tag))
        os.makedirs(ckpt_dir, exist_ok=True)
        self.agents.save_all(ckpt_dir)

        # Save knowledge engine
        ke_path = os.path.join(ckpt_dir, "knowledge_engine.pt")
        torch.save(self.env.knowledge_engine.state_dict(), ke_path)

        # Save embedder
        emb_path = os.path.join(ckpt_dir, "embedder.pt")
        torch.save(self.env.embedder.state_dict(), emb_path)

        # Save auxiliary optimizers
        aux_opt_path = os.path.join(ckpt_dir, "aux_optimizers.pt")
        aux_state = {
            "embedder_opt": self.embedder_opt.state_dict(),
            "knowledge_opt": self.knowledge_opt.state_dict(),
        }
        if self.mechanism_opt is not None:
            aux_state["mechanism_opt"] = self.mechanism_opt.state_dict()
        torch.save(aux_state, aux_opt_path)

        # Save mechanism modules (if nn.Module)
        for name, module in [
            ("adv_controller", self.env.adv_controller),
            ("soft_switch", self.env.soft_switch),
            ("devil_advocate", self.env.devil_advocate),
        ]:
            if isinstance(module, torch.nn.Module):
                path = os.path.join(ckpt_dir, f"{name}.pt")
                torch.save(module.state_dict(), path)


# ==============================================================================
# Hybrid Trainer — PPO (Proposer/Challenger) + DDPG (Arbiter/Coordinator)
# ==============================================================================

class HybridTrainer:
    """混合训练器：PPO + DDPG 双时间尺度训练。

    Proposer/Challenger: PPO 离散动作（快时间尺度）
    Arbiter: DDPG 连续动作（快时间尺度）
    Coordinator: DDPG 连续动作（慢时间尺度）

    Parameters
    ----------
    env : DebateEnv
        辩论环境（action_mode="hybrid"）。
    agents : HybridAgentGroup
        混合智能体组。
    cfg : Config
        完整配置。
    logger : Logger
        TensorBoard 日志器。
    dashboard : LiveTrainingDashboard | None
        实时可视化面板。
    """

    def __init__(
        self,
        env,
        agents,
        cfg,
        logger,
        dashboard=None,
    ) -> None:
        from debate_rl_v2.agents.hybrid_agent import HybridAgentGroup
        from debate_rl_v2.algorithms.replay_buffer import MultiAgentReplayBuffer

        assert isinstance(agents, HybridAgentGroup), "agents 必须是 HybridAgentGroup"
        assert env.action_mode == "hybrid", "env.action_mode 必须是 'hybrid'"

        self.env = env
        self.agents = agents
        self.cfg = cfg
        self.logger = logger
        self.device = cfg.resolve_device()
        self.dashboard = dashboard

        # PPO buffers (proposer, challenger)
        self.ppo_buffers = MultiRoleBuffer()

        # DDPG replay buffers (arbiter, coordinator)
        cont_cfg = cfg.continuous_agent
        self.arbiter_buffer = MultiAgentReplayBuffer(
            capacity=cont_cfg.buffer_size,
            agent_names=["arbiter"],
            obs_dims={"arbiter": env.obs_dims["arbiter"]},
            act_dims={"arbiter": env.act_dims["arbiter"]},
        )
        self.coordinator_buffer = MultiAgentReplayBuffer(
            capacity=cont_cfg.buffer_size,
            agent_names=["coordinator"],
            obs_dims={"coordinator": env.obs_dims["coordinator"]},
            act_dims={"coordinator": env.act_dims["coordinator"]},
        )

        self.metrics = MetricsTracker()
        self._global_step = 0
        self._best_reward = -float("inf")

        # 端到端训练优化器
        self.embedder_opt = torch.optim.Adam(
            self.env.embedder.parameters(),
            lr=cfg.knowledge.confidence_lr * 0.1,
        )
        self.knowledge_opt = torch.optim.Adam(
            self.env.knowledge_engine.predicate_parameters(),
            lr=cfg.knowledge.confidence_lr,
        )

    def train(self) -> None:
        """主训练循环。"""
        cfg = self.cfg
        tc = cfg.training
        pbar = tqdm(range(1, tc.total_episodes + 1), desc="Hybrid Training")

        for episode in pbar:
            # 收集 rollout
            episode_stats = self._collect_rollouts(tc.rollout_episodes)

            # PPO 更新（proposer, challenger）
            ppo_stats = {}
            for role in ("proposer", "challenger"):
                buf = self.ppo_buffers[role]
                if len(buf) > 0:
                    stats = mappo_update(self.agents[role], buf, cfg.ppo)
                    ppo_stats[role] = stats

            # DDPG 更新（arbiter）
            arbiter_stats = {}
            if len(self.arbiter_buffer) >= cfg.continuous_agent.batch_size:
                arbiter_stats = self._update_ddpg_agent(
                    "arbiter",
                    self.arbiter_buffer,
                    cfg.continuous_agent,
                )

            # DDPG 更新（coordinator，慢时间尺度）
            coord_stats = {}
            if episode % cfg.hierarchical.meta_update_interval == 0:
                if len(self.coordinator_buffer) >= cfg.continuous_agent.batch_size:
                    coord_stats = self._update_ddpg_agent(
                        "coordinator",
                        self.coordinator_buffer,
                        cfg.continuous_agent,
                    )

            # Soft update target networks
            self.agents.soft_update_ddpg(cfg.continuous_agent.tau)

            # 知识引擎置信度更新
            self._update_knowledge_confidence(episode_stats)

            # 辅助损失更新
            aux_stats = self._update_auxiliary_losses(episode)

            # 规则挖掘
            if self.env.rule_miner.should_mine():
                states, compliant = self.env.rule_miner.get_mining_data()
                added = self.env.knowledge_engine.mine_rule(states, compliant)
                if added:
                    self.logger.log_scalar("knowledge/rules_mined", 1.0, self._global_step)

            # 清空 PPO buffers
            self.ppo_buffers.clear_all()

            # 日志记录
            if episode % tc.log_interval == 0:
                self._log_training(episode, episode_stats, ppo_stats, arbiter_stats, coord_stats)

            # 评估
            if episode % tc.eval_interval == 0:
                eval_stats = self.evaluate(tc.eval_episodes)
                self._log_eval(episode, eval_stats)

            # 检查点
            if episode % tc.save_interval == 0:
                self._save_checkpoint(episode)

            # 进度条
            pbar.set_postfix({
                "rew": f"{episode_stats.get('mean_reward', 0):.3f}",
                "task": f"{episode_stats.get('mean_task_reward', 0):.3f}",
            })

        self._save_checkpoint("final")
        self.logger.close()

    def _collect_rollouts(self, num_episodes: int) -> Dict[str, float]:
        """收集 rollout 数据。"""
        total_rewards = []
        task_rewards = []
        steps_list = []

        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                actions = {}
                meta_action = None

                # Coordinator (DDPG, 每 meta_interval 步)
                if self.env.t % self.env.meta_interval == 0:
                    o = torch.tensor(obs["coordinator"], dtype=torch.float32)
                    coord_agent = self.agents["coordinator"]
                    coord_action = coord_agent.act(obs["coordinator"], explore=True)
                    meta_action = coord_action
                    # 存储到 coordinator buffer（稍后）
                    coord_obs_prev = obs["coordinator"].copy()

                # Proposer/Challenger (PPO)
                for role in ("proposer", "challenger"):
                    o = torch.tensor(obs[role], dtype=torch.float32)
                    act, logp, val = self.agents[role].act(o)
                    actions[role] = act
                    self.ppo_buffers[role].add(obs[role], act, 0.0, val, logp, 0.0)

                # Arbiter (DDPG)
                arbiter_agent = self.agents["arbiter"]
                arbiter_action = arbiter_agent.act(obs["arbiter"], explore=True)
                actions["arbiter"] = arbiter_action
                arbiter_obs_prev = obs["arbiter"].copy()

                # 环境 step
                step_out = self.env.step(actions=actions, meta_action=meta_action)
                next_obs = step_out.obs
                done = step_out.done

                # 存储 PPO rewards
                for role in ("proposer", "challenger"):
                    buf = self.ppo_buffers[role]
                    buf.rew[-1] = step_out.rewards[role]
                    buf.done[-1] = float(done)

                # 存储 DDPG transitions
                self.arbiter_buffer.add(
                    obs={"arbiter": arbiter_obs_prev},
                    actions={"arbiter": arbiter_action},
                    rewards={"arbiter": step_out.rewards["arbiter"]},
                    next_obs={"arbiter": next_obs["arbiter"]},
                    done=done,
                )

                if meta_action is not None:
                    coord_reward = sum(step_out.rewards[r] for r in ("proposer", "challenger", "arbiter")) / 3.0
                    self.coordinator_buffer.add(
                        obs={"coordinator": coord_obs_prev},
                        actions={"coordinator": meta_action},
                        rewards={"coordinator": coord_reward},
                        next_obs={"coordinator": next_obs["coordinator"]},
                        done=done,
                    )

                ep_reward += sum(step_out.rewards.values())
                self._global_step += 1
                obs = next_obs

                if done:
                    break

            total_rewards.append(ep_reward)
            task_reward = step_out.info.get("task_reward", 0.0)
            task_rewards.append(task_reward)
            steps_list.append(self.env.t)

        return {
            "mean_reward": np.mean(total_rewards),
            "mean_task_reward": np.mean(task_rewards),
            "mean_steps": np.mean(steps_list),
        }

    def _update_ddpg_agent(
        self,
        role: str,
        buffer,
        cont_cfg,
    ) -> Dict[str, float]:
        """单个 DDPG 智能体更新。"""
        agent = self.agents[role]
        batch = buffer.sample(cont_cfg.batch_size, device=self.device)

        obs = batch[f"{role}_obs"]
        act = batch[f"{role}_act"]
        rew = batch[f"{role}_rew"]
        next_obs = batch[f"{role}_next_obs"]
        done = batch["done"]
        weights = batch["weights"]

        # Critic 更新
        with torch.no_grad():
            next_act = agent.target_actor(next_obs)
            target_q = agent.target_critic(next_obs, next_act).squeeze(-1)
            y = rew + self.cfg.ppo.gamma * target_q * (1.0 - done)

        current_q = agent.critic(obs, act).squeeze(-1)
        critic_loss = (weights * (y - current_q).pow(2)).mean()

        agent.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_opt.step()

        # Actor 更新
        current_act = agent.actor(obs)
        actor_loss = -agent.critic(obs, current_act).mean()

        agent.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
        agent.actor_opt.step()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "q_mean": current_q.mean().item(),
        }

    def _update_knowledge_confidence(self, stats: Dict[str, float]) -> None:
        """更新规则置信度。"""
        ke = self.env.knowledge_engine
        task_reward = stats.get("mean_task_reward", 0.0)
        for rule in ke.rules:
            with torch.no_grad():
                adjustment = self.cfg.knowledge.confidence_lr * (task_reward - 0.5)
                rule._confidence_logit.data += adjustment

    def _update_auxiliary_losses(self, episode: int) -> Dict[str, float]:
        """更新辅助损失。"""
        stats = {}
        emb_loss = self.env.compute_embedder_loss()
        if emb_loss is not None:
            self.embedder_opt.zero_grad()
            emb_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.env.embedder.parameters(), 0.5)
            self.embedder_opt.step()
            stats["aux/embedder_loss"] = emb_loss.item()

        if episode % 5 == 0:
            know_loss = self.env.compute_knowledge_loss()
            if know_loss is not None:
                self.knowledge_opt.zero_grad()
                know_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.env.knowledge_engine.predicate_parameters(), 0.5
                )
                self.knowledge_opt.step()
                stats["aux/knowledge_loss"] = know_loss.item()

        return stats

    def evaluate(self, num_episodes: int = 20) -> Dict[str, float]:
        """确定性评估。"""
        task_rewards = []
        steps_list = []

        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False

            while not done:
                actions = {}
                meta_action = None

                if self.env.t % self.env.meta_interval == 0:
                    coord_agent = self.agents["coordinator"]
                    meta_action = coord_agent.act(obs["coordinator"], explore=False)

                for role in ("proposer", "challenger"):
                    o = torch.tensor(obs[role], dtype=torch.float32)
                    act, _, _ = self.agents[role].act(o, deterministic=True)
                    actions[role] = act

                arbiter_agent = self.agents["arbiter"]
                actions["arbiter"] = arbiter_agent.act(obs["arbiter"], explore=False)

                step_out = self.env.step(actions=actions, meta_action=meta_action)
                obs = step_out.obs
                done = step_out.done

            task_rewards.append(step_out.info.get("task_reward", 0.0))
            steps_list.append(self.env.t)

        return {
            "eval/task_reward": np.mean(task_rewards),
            "eval/task_reward_std": np.std(task_rewards),
            "eval/mean_steps": np.mean(steps_list),
        }

    def _log_training(self, episode, ep_stats, ppo_stats, arbiter_stats, coord_stats):
        step = self._global_step
        for key, val in ep_stats.items():
            self.logger.log_scalar(f"train/{key}", val, step)
        for role, stats in ppo_stats.items():
            for key, val in stats.items():
                self.logger.log_scalar(f"train/{role}/{key}", val, step)
        if arbiter_stats:
            for key, val in arbiter_stats.items():
                self.logger.log_scalar(f"train/arbiter/{key}", val, step)
        if coord_stats:
            for key, val in coord_stats.items():
                self.logger.log_scalar(f"train/coordinator/{key}", val, step)

        logger.info(
            "Ep %d: reward=%.3f task=%.3f arbiter_buf=%d coord_buf=%d",
            episode, ep_stats['mean_reward'], ep_stats['mean_task_reward'],
            len(self.arbiter_buffer), len(self.coordinator_buffer),
        )

    def _log_eval(self, episode, eval_stats):
        step = self._global_step
        for key, val in eval_stats.items():
            self.logger.log_scalar(key, val, step)

        task_r = eval_stats["eval/task_reward"]
        logger.info(
            "EVAL Ep %d: task_reward=%.3f±%.3f",
            episode, task_r, eval_stats['eval/task_reward_std'],
        )

        if task_r > self._best_reward:
            self._best_reward = task_r
            self._save_checkpoint("best")

    def _save_checkpoint(self, tag) -> None:
        ckpt_dir = os.path.join(self.cfg.training.checkpoint_dir, str(tag))
        os.makedirs(ckpt_dir, exist_ok=True)
        self.agents.save_all(ckpt_dir)

        # 保存知识引擎
        ke_path = os.path.join(ckpt_dir, "knowledge_engine.pt")
        torch.save(self.env.knowledge_engine.state_dict(), ke_path)

        # 保存 embedder
        emb_path = os.path.join(ckpt_dir, "embedder.pt")
        torch.save(self.env.embedder.state_dict(), emb_path)

        logger.info("Checkpoint saved: %s", ckpt_dir)
