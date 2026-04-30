"""Base Multi-Agent Debate Environment — Sections 3 & 4.

.. deprecated::
    Use GameEngine + DebateGameScenario instead.
    See debate_rl_v2.scenarios.debate.scenario for the new implementation.

Implements the Constrained Markov Game G = <N, S, {O^i}, {A^i}, P, {R^i}, C, γ>
with four roles: Proposer, Challenger, Arbiter, Coordinator.

Integrates all five core mechanisms:
  1. Dynamic adversarial intensity (Section 4.2)
  2. Neural-symbolic knowledge verification (Section 4.3)
  3. Probabilistic soft-switch (Section 4.4)
  4. Devil's advocate verification (Section 4.5)
  5. Evidence chain tracking (Section 4.3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from debate_rl_v2.core.adversarial import (
    AdversarialIntensityController,
    SemanticEmbedder,
    cosine_similarity,
)
from debate_rl_v2.core.knowledge import KnowledgeEngine, RuleMiner
from debate_rl_v2.core.soft_switch import SoftSwitchController
from debate_rl_v2.core.devil_advocate import DevilAdvocateVerifier
from debate_rl_v2.core.evidence_chain import EvidenceChain
from debate_rl_v2.config import (
    EnvConfig,
    AdversarialConfig,
    KnowledgeConfig,
    SoftSwitchConfig,
    DevilAdvocateConfig,
    RewardConfig,
    ContinuousAgentConfig,
)

# Union type for actions: discrete (int) or continuous (ndarray)
from typing import Union
from debate_rl_v2.core.reward_design import (
    EnhancedRewardComputer,
    EnhancedRewardConfig,
    DebateMetrics,
)

# Role names used as dict keys throughout the codebase
ROLES = ("proposer", "challenger", "arbiter", "coordinator")


@dataclass
class StepOutput:
    """Single-step output from the environment."""
    obs: Dict[str, np.ndarray]
    rewards: Dict[str, float]
    done: bool
    info: Dict[str, Any]


class DebateEnv:
    """Multi-agent debate environment implementing the Constrained Markov Game.

    Parameters
    ----------
    env_cfg : EnvConfig
        Environment dimensions and limits.
    adv_cfg : AdversarialConfig
        Adversarial intensity parameters.
    know_cfg : KnowledgeConfig
        Knowledge engine parameters.
    ss_cfg : SoftSwitchConfig
        Soft-switch parameters.
    da_cfg : DevilAdvocateConfig
        Devil's advocate parameters.
    rew_cfg : RewardConfig
        Reward weights.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        env_cfg: Optional[EnvConfig] = None,
        adv_cfg: Optional[AdversarialConfig] = None,
        know_cfg: Optional[KnowledgeConfig] = None,
        ss_cfg: Optional[SoftSwitchConfig] = None,
        da_cfg: Optional[DevilAdvocateConfig] = None,
        rew_cfg: Optional[RewardConfig] = None,
        cont_cfg: Optional[ContinuousAgentConfig] = None,
        seed: int = 0,
    ) -> None:
        self.env_cfg = env_cfg or EnvConfig()
        self.adv_cfg = adv_cfg or AdversarialConfig()
        self.know_cfg = know_cfg or KnowledgeConfig()
        self.ss_cfg = ss_cfg or SoftSwitchConfig()
        self.da_cfg = da_cfg or DevilAdvocateConfig()
        self.rew_cfg = rew_cfg or RewardConfig()
        self.cont_cfg = cont_cfg or ContinuousAgentConfig()

        ec = self.env_cfg
        self.rng = np.random.default_rng(seed)
        self.action_mode = ec.action_mode  # "discrete" | "hybrid"

        # Dimensions
        self.context_dim = ec.context_dim
        self.proposal_dim = ec.proposal_dim
        self.proposal_values = ec.proposal_values
        self.embed_dim = ec.embed_dim
        self.max_steps = ec.max_steps
        self.meta_interval = ec.meta_interval

        # Action spaces
        self.proposer_act_dim = ec.proposal_dim * ec.proposal_values
        self.challenger_act_dim = ec.proposal_dim * ec.proposal_values
        if self.action_mode == "hybrid":
            # 连续动作空间：arbiter 33维, coordinator 8维
            cc = self.cont_cfg
            self.arbiter_act_dim = cc.arbiter_act_dim
            self.coordinator_act_dim = cc.coordinator_act_dim
        else:
            self.arbiter_act_dim = 5   # {decrease_θ, noop, increase_θ, boost_w, decay_w}
            self.coordinator_act_dim = 10  # {noop, η±, α±, τ_low±, τ_high±, mine_rule}

        # Core mechanisms
        self.adv_controller = AdversarialIntensityController(
            eta=self.adv_cfg.eta,
            alpha=self.adv_cfg.alpha,
            omega=self.adv_cfg.omega,
            max_steps=ec.max_steps,
        )
        self.embedder = SemanticEmbedder(ec.proposal_dim, ec.embed_dim)
        self.knowledge_engine = KnowledgeEngine(
            state_dim=ec.context_dim + ec.proposal_dim,
            num_rules=ec.rule_count,
            threshold=self.know_cfg.initial_threshold,
            confidence_lr=self.know_cfg.confidence_lr,
            max_mined_rules=self.know_cfg.max_mined_rules,
        )
        self.rule_miner = RuleMiner(
            mine_interval=self.know_cfg.mine_interval,
            min_samples=self.know_cfg.ilp_min_samples,
        )
        self.soft_switch = SoftSwitchController(
            tau_low=self.ss_cfg.tau_low,
            tau_high=self.ss_cfg.tau_high,
            steepness=self.ss_cfg.steepness,
        )
        self.devil_advocate = DevilAdvocateVerifier(
            disagreement_threshold=self.da_cfg.disagreement_threshold,
            update_threshold=self.da_cfg.update_threshold,
            stability_window=self.da_cfg.stability_window,
            reactivation_threshold=self.da_cfg.reactivation_threshold,
            max_challenges=self.da_cfg.max_challenges,
        )
        self.evidence_chain = EvidenceChain()

        # Enhanced reward system (v2)
        self._enhanced_reward = EnhancedRewardComputer(EnhancedRewardConfig())
        self._debate_metrics = DebateMetrics()

        self.reset()

    # ------------------------------------------------------------------
    # Observation / action space info (for agent construction)
    # ------------------------------------------------------------------

    @property
    def obs_dims(self) -> Dict[str, int]:
        ec = self.env_cfg
        base = ec.context_dim + ec.proposal_dim + 3  # + [d, comp, t_norm]
        if self.action_mode == "hybrid":
            # Arbiter: base + rule_satisfactions(32) + rule_confidences(32)
            arbiter_dim = ec.context_dim + 2 * ec.proposal_dim + 2 + 2 * ec.rule_count
            # Coordinator: 丰富观测 ~45 维
            coordinator_dim = 45
            return {
                "proposer": base,
                "challenger": base,
                "arbiter": arbiter_dim,
                "coordinator": coordinator_dim,
            }
        return {
            "proposer": base,
            "challenger": base,
            "arbiter": ec.context_dim + 2 * ec.proposal_dim + 2,
            "coordinator": 5,  # [d, comp, t_norm, λ, da_active]
        }

    @property
    def act_dims(self) -> Dict[str, int]:
        return {
            "proposer": self.proposer_act_dim,
            "challenger": self.challenger_act_dim,
            "arbiter": self.arbiter_act_dim,
            "coordinator": self.coordinator_act_dim,
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> Dict[str, np.ndarray]:
        ec = self.env_cfg
        self.t = 0
        self.done = False

        # Task state
        self.context = self.rng.standard_normal(ec.context_dim).astype(np.float32)
        self.target = self.rng.integers(0, ec.proposal_values, size=ec.proposal_dim)
        self.proposal = self.rng.integers(0, ec.proposal_values, size=ec.proposal_dim)
        self.challenge = self.rng.integers(0, ec.proposal_values, size=ec.proposal_dim)
        self.last_proposal = self.proposal.copy()
        self.last_challenge = self.challenge.copy()

        # Reset mechanisms
        self.adv_controller.reset()
        self.devil_advocate.reset()
        self.evidence_chain.reset()
        self._debate_metrics.reset()

        # Reset episode history for auxiliary loss computation
        self._proposal_history = []
        self._challenge_history = []
        self._task_rewards = []

        return self._get_obs()

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _embed(self, vec: np.ndarray) -> torch.Tensor:
        """Compute semantic embedding of a proposal/challenge vector.

        Returns a tensor (not numpy) to preserve gradient flow.
        """
        t = torch.tensor(vec.astype(np.float32) / max(self.proposal_values - 1, 1), requires_grad=False).unsqueeze(0)
        emb = self.embedder(t).squeeze(0)  # Keep as tensor
        return emb

    def _disagreement(self, differentiable: bool = False) -> float:
        """Compute disagreement D(t) = 1 - cos(φ(p), φ(c)).

        Parameters
        ----------
        differentiable : bool
            If True, return a torch.Tensor with gradient; otherwise float.
        """
        p_emb = self._embed(self.proposal)
        c_emb = self._embed(self.challenge)
        cos_sim = torch.nn.functional.cosine_similarity(
            p_emb.unsqueeze(0), c_emb.unsqueeze(0)
        ).squeeze(0)
        d_tensor = 1.0 - cos_sim
        if differentiable:
            return d_tensor
        return d_tensor.detach().item()

    def _compliance(self, differentiable: bool = False) -> Tuple[float, List[int]]:
        """Compute compliance score and triggered rules.

        Parameters
        ----------
        differentiable : bool
            If True, return compliance as torch.Tensor with gradient.
        """
        state = torch.tensor(
            np.concatenate([self.context, self.proposal.astype(np.float32) / max(self.proposal_values - 1, 1)])
        ).unsqueeze(0)
        comp_tensor = self.knowledge_engine.compliance_score(state)
        with torch.no_grad():
            triggered = self.knowledge_engine.get_triggered_rules(state)
        if differentiable:
            return comp_tensor.squeeze(0), triggered
        return comp_tensor.squeeze(0).detach().item(), triggered

    def _get_rule_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取每条规则的满足度和置信度向量（用于 hybrid 模式扩展观测）。"""
        ec = self.env_cfg
        state = torch.tensor(
            np.concatenate([self.context, self.proposal.astype(np.float32) / max(self.proposal_values - 1, 1)])
        ).unsqueeze(0)
        rule_sats_raw = self.knowledge_engine.rule_satisfactions(state)

        rule_sats = np.zeros(ec.rule_count, dtype=np.float32)
        rule_confs = np.zeros(ec.rule_count, dtype=np.float32)
        for i, (w, mu) in enumerate(rule_sats_raw):
            if i >= ec.rule_count:
                break
            rule_confs[i] = float(w)
            rule_sats[i] = float(mu.mean())
        return rule_sats, rule_confs

    def _get_coordinator_rich_obs(
        self, d: float, comp: float, t_norm: float, lam: float, da_active: float,
    ) -> np.ndarray:
        """构建 Coordinator 丰富观测向量 (45 维)。"""
        ac = self.adv_controller
        ss = self.soft_switch
        da = self.devil_advocate

        # 当前机制参数 (6 维)
        eta = float(ac.eta)
        alpha = float(ac.alpha)
        omega = float(ac.omega)
        tau_low = float(ss.tau_low)
        tau_high = float(ss.tau_high)
        steepness = float(ss.steepness) / 20.0  # 归一化

        # DA 参数 (3 维)
        eps_d = float(da.eps_d)
        eps_p = float(da.eps_p)
        delta = float(da.delta)

        # 知识引擎摘要 (2 维)
        rule_sats, rule_confs = self._get_rule_stats()
        mean_rule_conf = float(np.mean(rule_confs)) if len(rule_confs) > 0 else 0.0
        max_rule_sat = float(np.max(rule_sats)) if len(rule_sats) > 0 else 0.0

        # 训练信号 (2 维)
        recent_rewards = self._task_rewards[-5:] if hasattr(self, '_task_rewards') and self._task_rewards else [0.0]
        mean_reward_last5 = float(np.mean(recent_rewards))
        task_reward_ema = mean_reward_last5  # 简化为均值

        # 智能体状态 (2 维) — 用提案/挑战变化率近似熵
        prop_change = float(np.mean(np.abs(self.proposal - self.last_proposal))) / max(self.proposal_values - 1, 1)
        chal_change = float(np.mean(np.abs(self.challenge - self.last_challenge))) / max(self.proposal_values - 1, 1)

        # 进度 (1 维)
        episode_progress = t_norm

        # 当前模式 one-hot (3 维)
        mode_standard = 1.0 if lam < ss.tau_low else 0.0
        mode_boost = 1.0 if ss.tau_low <= lam < ss.tau_high else 0.0
        mode_intervene = 1.0 if lam >= ss.tau_high else 0.0

        # DA 状态 (2 维)
        da_challenge_count = float(da._challenge_count) if hasattr(da, '_challenge_count') else 0.0
        da_stable_count = float(len(da._stability_window)) if hasattr(da, '_stability_window') else 0.0

        # 趋势 (2 维)
        hist = ac.history
        if len(hist.disagreement) >= 2:
            delta_d = hist.disagreement[-1] - hist.disagreement[-2]
        else:
            delta_d = 0.0
        delta_comp = 0.0  # 简化

        obs_vec = np.array([
            d, comp, t_norm, lam, da_active,                    # 原有 5 维
            eta, alpha, omega, tau_low, tau_high, steepness,     # 机制参数 6 维
            eps_d, eps_p, delta,                                  # DA 参数 3 维
            mean_rule_conf, max_rule_sat,                         # 知识引擎 2 维
            mean_reward_last5, task_reward_ema,                   # 训练信号 2 维
            prop_change, chal_change,                             # 智能体状态 2 维
            episode_progress,                                     # 进度 1 维
            mode_standard, mode_boost, mode_intervene,            # 模式 3 维
            da_challenge_count / 10.0, da_stable_count / 10.0,   # DA 状态 2 维
            delta_d, delta_comp,                                  # 趋势 2 维
        ], dtype=np.float32)

        # 填充到 45 维
        target_dim = 45
        if len(obs_vec) < target_dim:
            obs_vec = np.concatenate([obs_vec, np.zeros(target_dim - len(obs_vec), dtype=np.float32)])
        return obs_vec[:target_dim]

    def _get_obs(self) -> Dict[str, np.ndarray]:
        d = self._disagreement()
        comp, _ = self._compliance()
        t_norm = self.t / max(self.max_steps, 1)
        lam = self.adv_controller.current_intensity
        da_active = float(self.devil_advocate.is_active)

        p_norm = self.proposal.astype(np.float32) / max(self.proposal_values - 1, 1)
        c_norm = self.challenge.astype(np.float32) / max(self.proposal_values - 1, 1)

        obs = {
            "proposer": np.concatenate([self.context, p_norm, [d, comp, t_norm]]).astype(np.float32),
            "challenger": np.concatenate([self.context, c_norm, [d, comp, t_norm]]).astype(np.float32),
        }

        if self.action_mode == "hybrid":
            # Arbiter 扩展观测：追加每条规则的满足度和置信度
            rule_sats, rule_confs = self._get_rule_stats()
            obs["arbiter"] = np.concatenate([
                self.context, p_norm, c_norm, [d, t_norm],
                rule_sats, rule_confs,
            ]).astype(np.float32)

            # Coordinator 丰富观测 (~45 维)
            obs["coordinator"] = self._get_coordinator_rich_obs(
                d, comp, t_norm, lam, da_active,
            )
        else:
            obs["arbiter"] = np.concatenate([self.context, p_norm, c_norm, [d, t_norm]]).astype(np.float32)
            obs["coordinator"] = np.array([d, comp, t_norm, lam, da_active], dtype=np.float32)

        return obs

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        actions: Dict[str, Union[int, np.ndarray]],
        meta_action: Optional[Union[int, np.ndarray]] = None,
    ) -> StepOutput:
        if self.done:
            raise RuntimeError("Episode already done; call reset().")

        info: Dict[str, Any] = {}
        rewards = {r: 0.0 for r in ROLES}

        # --- Meta action (coordinator) ---
        if meta_action is not None and self.t % self.meta_interval == 0:
            if self.action_mode == "hybrid" and isinstance(meta_action, np.ndarray):
                self._apply_meta_action_continuous(meta_action, info)
            else:
                self._apply_meta_action(meta_action, info)

        # --- Soft switch decision ---
        lam = self.adv_controller.current_intensity
        switch_state = self.soft_switch.decide(lam, self.rng)
        info["switch_mode"] = switch_state.mode

        # --- Proposer action ---
        self.last_proposal = self.proposal.copy()
        self._apply_mask_action(self.proposal, actions["proposer"])
        # Record for auxiliary loss
        self._proposal_history.append(self.proposal.copy())

        # --- Challenger action ---
        self.last_challenge = self.challenge.copy()
        self._apply_mask_action(self.challenge, actions["challenger"])
        self._challenge_history.append(self.challenge.copy())

        # Challenger boost in low-adversarial regime
        if switch_state.mode == "challenger_boost":
            random_act = self.rng.integers(0, self.proposer_act_dim)
            self._apply_mask_action(self.challenge, random_act)

        # --- Devil's advocate ---
        d = self._disagreement()
        belief_update = float(
            np.linalg.norm(self.proposal - self.last_proposal)
            + np.linalg.norm(self.challenge - self.last_challenge)
        )
        da_triggered = self.devil_advocate.check_stability(d, belief_update)
        if self.devil_advocate.is_active:
            # Issue devil's advocate challenge
            da_challenge = self.rng.integers(0, self.proposer_act_dim)
            saved_challenge = self.challenge.copy()
            self._apply_mask_action(self.challenge, da_challenge)
            da_d = self._disagreement()
            da_result = self.devil_advocate.process_challenge(da_d)
            info["devil_advocate"] = {
                "triggered": True,
                "is_robust": da_result.is_robust,
                "confirmed": da_result.consensus_confirmed,
            }
            if not da_result.is_robust:
                # Reactivate debate — keep the adversarial challenge
                pass
            else:
                # Restore original challenge
                self.challenge = saved_challenge

        # --- Arbiter action & compliance ---
        comp, triggered = self._compliance()
        arbiter_action = actions["arbiter"]
        if self.action_mode == "hybrid" and isinstance(arbiter_action, np.ndarray):
            self._apply_arbiter_action_continuous(arbiter_action, comp)
        else:
            self._apply_arbiter_action(arbiter_action, comp)
        info["triggered_rules"] = triggered
        info["compliance"] = comp

        # --- Record evidence ---
        rule_sats = self.knowledge_engine.rule_satisfactions(
            torch.tensor(
                np.concatenate([self.context, self.proposal.astype(np.float32) / max(self.proposal_values - 1, 1)])
            ).unsqueeze(0)
        )
        self.evidence_chain.record(
            step=self.t,
            role="all",
            action=actions,
            triggered_rules=triggered,
            rule_confidences=[float(w) for w, _ in rule_sats[:len(triggered)]],
            rule_satisfactions=[float(mu.mean()) for _, mu in rule_sats[:len(triggered)]],
            compliance_score=comp,
            disagreement=d,
            lambda_adv=lam,
            mode=switch_state.mode,
            devil_advocate_active=self.devil_advocate.is_active,
        )

        # --- Record for rule mining ---
        state_tensor = torch.tensor(
            np.concatenate([self.context, self.proposal.astype(np.float32) / max(self.proposal_values - 1, 1)])
        )
        self.rule_miner.record(state_tensor, comp)

        # --- Update adversarial intensity ---
        d = self._disagreement()
        self.adv_controller.update(d, self.t)
        info["lambda_adv"] = self.adv_controller.current_intensity
        info["disagreement"] = d

        # --- Rewards (Section 5.4 — Enhanced Multi-Tier) ---
        rc = self.rew_cfg
        delta_d = d - (self.adv_controller.history.disagreement[-2]
                       if len(self.adv_controller.history.disagreement) > 1 else 1.0)

        r_div = -delta_d  # reward for reducing disagreement
        r_comp = comp - 0.5
        r_info = self._info_gain_reward(d)

        base_reward = rc.divergence_weight * r_div + rc.compliance_weight * r_comp - rc.step_penalty
        rewards["proposer"] = base_reward + self._proposal_quality_reward()
        rewards["challenger"] = base_reward + rc.info_gain_weight * r_info
        rewards["arbiter"] = base_reward + rc.compliance_weight * r_comp
        rewards["coordinator"] = 0.0  # meta reward computed at meta update

        # Record metrics for enhanced reward system
        quality_proxy = comp * 0.6 + (1.0 - d) * 0.4
        self._debate_metrics.record_round(
            quality=quality_proxy,
            disagreement=d,
            compliance=comp,
            lambda_adv=self.adv_controller.current_intensity,
            proposal_score=self._proposal_quality_reward(),
            constructiveness=0.5,
            novelty=0.5,
        )

        # --- Termination ---
        self.t += 1
        done = self.t >= self.max_steps

        # Devil advocate confirmed consensus → early termination
        da_info = info.get("devil_advocate", {})
        if da_info.get("confirmed", False):
            done = True
            info["early_stop_reason"] = "consensus_confirmed"

        self.done = done

        # --- Final task reward ---
        if done:
            task_reward = self._task_reward()
            for role in ROLES:
                rewards[role] += rc.task_weight * task_reward
            info["task_reward"] = task_reward
            info["evidence_chain"] = self.evidence_chain.get_justification()

        return StepOutput(
            obs=self._get_obs(),
            rewards=rewards,
            done=done,
            info=info,
        )

    # ------------------------------------------------------------------
    # Action helpers
    # ------------------------------------------------------------------

    def _apply_mask_action(self, vec: np.ndarray, action: int) -> None:
        """Decode (dimension, value) from flat action index."""
        k = action // self.proposal_values
        v = action % self.proposal_values
        k = min(k, self.proposal_dim - 1)
        vec[k] = v

    def _apply_arbiter_action(self, action: int, comp: float) -> None:
        """0=decrease_θ, 1=noop, 2=increase_θ, 3=boost_weights, 4=decay_weights."""
        ke = self.knowledge_engine
        if action == 0:
            with torch.no_grad():
                ke.threshold.data -= 0.05
        elif action == 2:
            with torch.no_grad():
                ke.threshold.data += 0.05
        elif action == 3:
            for rule in ke.rules:
                with torch.no_grad():
                    rule._confidence_logit.data += 0.02 * (comp - 0.5)
        elif action == 4:
            for rule in ke.rules:
                with torch.no_grad():
                    rule._confidence_logit.data -= 0.02 * (comp - 0.5)

    def _apply_meta_action(self, action: int, info: Dict[str, Any]) -> None:
        """Coordinator meta actions.

        0=noop, 1/2=η±, 3/4=α±, 5/6=τ_low±, 7/8=τ_high±, 9=mine_rule
        """
        ac = self.adv_controller
        if action == 1:
            ac.eta = min(0.5, ac.eta + 0.02)
        elif action == 2:
            ac.eta = max(0.05, ac.eta - 0.02)
        elif action == 3:
            ac.alpha = min(0.9, ac.alpha + 0.05)
        elif action == 4:
            ac.alpha = max(0.1, ac.alpha - 0.05)
        elif action == 5:
            self.soft_switch.update_thresholds(
                self.soft_switch.tau_low + 0.05, self.soft_switch.tau_high
            )
        elif action == 6:
            self.soft_switch.update_thresholds(
                self.soft_switch.tau_low - 0.05, self.soft_switch.tau_high
            )
        elif action == 7:
            self.soft_switch.update_thresholds(
                self.soft_switch.tau_low, self.soft_switch.tau_high + 0.05
            )
        elif action == 8:
            self.soft_switch.update_thresholds(
                self.soft_switch.tau_low, self.soft_switch.tau_high - 0.05
            )
        elif action == 9:
            if self.rule_miner.should_mine(force=True):
                states, compliant = self.rule_miner.get_mining_data()
                added = self.knowledge_engine.mine_rule(states, compliant)
                info["mined_rule"] = added

    def _apply_arbiter_action_continuous(self, action: np.ndarray, comp: float) -> None:
        """连续动作：action ∈ [-1, 1]^33。

        a[0]: 阈值 θ 调整量，映射到 [-0.1, 0.1]
        a[1:33]: 每条规则置信度调整量，映射到 [-0.05, 0.05]
        """
        ke = self.knowledge_engine
        with torch.no_grad():
            ke.threshold.data += float(action[0]) * 0.1
        for i, rule in enumerate(ke.rules):
            if i + 1 < len(action):
                with torch.no_grad():
                    rule._confidence_logit.data += float(action[i + 1]) * 0.05

    def _apply_meta_action_continuous(self, action: np.ndarray, info: Dict[str, Any]) -> None:
        """Coordinator 连续动作：action ∈ [-1, 1]^8。

        a[0]: Δη      映射到 [-0.05, 0.05]
        a[1]: Δα      映射到 [-0.1, 0.1]
        a[2]: Δω      映射到 [-0.1, 0.1]
        a[3]: Δτ_low  映射到 [-0.05, 0.05]
        a[4]: Δτ_high 映射到 [-0.05, 0.05]
        a[5]: Δsteepness 映射到 [-1.0, 1.0]
        a[6]: Δε_D    映射到 [-0.02, 0.02]
        a[7]: Δδ      映射到 [-0.02, 0.02]
        """
        ac = self.adv_controller
        ss = self.soft_switch
        da = self.devil_advocate

        ac.eta = float(np.clip(ac.eta + action[0] * 0.05, 0.05, 0.5))
        ac.alpha = float(np.clip(ac.alpha + action[1] * 0.1, 0.1, 0.9))
        ac.omega = float(np.clip(ac.omega + action[2] * 0.1, 0.1, 0.9))

        new_tau_low = float(np.clip(ss.tau_low + action[3] * 0.05, 0.05, 0.6))
        new_tau_high = float(np.clip(ss.tau_high + action[4] * 0.05, 0.4, 0.95))
        ss.update_thresholds(new_tau_low, new_tau_high)

        if hasattr(ss, '_steepness_logit'):
            with torch.no_grad():
                ss._steepness_logit.data += float(action[5]) * 0.1

        # DA 参数调整
        if hasattr(da, '_eps_d_logit'):
            with torch.no_grad():
                da._eps_d_logit.data += float(action[6]) * 0.05
        if hasattr(da, '_delta_logit'):
            with torch.no_grad():
                da._delta_logit.data += float(action[7]) * 0.05

        # 规则挖掘：当 action 均值偏正时触发
        if float(np.mean(action)) > 0.5:
            if self.rule_miner.should_mine(force=True):
                states, compliant = self.rule_miner.get_mining_data()
                added = self.knowledge_engine.mine_rule(states, compliant)
                info["mined_rule"] = added

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _task_reward(self) -> float:
        """Terminal reward: fraction of proposal matching target."""
        return 1.0 - float(np.mean(self.proposal != self.target))

    def _proposal_quality_reward(self) -> float:
        """Intermediate reward for proposal quality."""
        match = 1.0 - float(np.mean(self.proposal != self.target))
        return 0.1 * match

    def _info_gain_reward(self, d: float) -> float:
        """Reward challenger for constructive adversarial behavior."""
        if d < 0.3:
            return d  # encourage increasing useful disagreement
        return 1.0 - d  # encourage convergence when disagreement is high

    @property
    def debate_metrics(self) -> DebateMetrics:
        """Expose debate metrics for enhanced reward computation."""
        return self._debate_metrics

    @property
    def enhanced_reward(self) -> EnhancedRewardComputer:
        """Expose enhanced reward computer."""
        return self._enhanced_reward

    # ------------------------------------------------------------------
    # Auxiliary losses for end-to-end training
    # ------------------------------------------------------------------

    def compute_embedder_loss(self) -> Optional[torch.Tensor]:
        """辅助损失：让嵌入空间有语义意义。

        约束：
        1. 相邻步的提案嵌入应相似（时序平滑性）
        2. 提案越接近目标，嵌入越接近目标嵌入（目标对齐）

        Returns None if insufficient history.
        """
        if len(self._proposal_history) < 3:
            return None

        proposals = self._proposal_history
        target = self.target

        # 计算嵌入（带梯度）
        embeddings = [self._embed(p) for p in proposals]
        target_emb = self._embed(target)

        loss = torch.tensor(0.0)

        # 时序平滑性：相邻步提案嵌入应相似
        for i in range(len(embeddings) - 1):
            cos_sim = torch.nn.functional.cosine_similarity(
                embeddings[i].unsqueeze(0), embeddings[i + 1].unsqueeze(0)
            )
            loss = loss + (1.0 - cos_sim.squeeze(0))

        # 目标对齐：与目标匹配度高的提案 → 嵌入更接近目标
        for i, p in enumerate(proposals):
            match_rate = 1.0 - float(np.mean(p != target))
            cos_to_target = torch.nn.functional.cosine_similarity(
                embeddings[i].unsqueeze(0), target_emb.unsqueeze(0)
            ).squeeze(0)
            # 匹配度高 → 鼓励嵌入接近目标
            loss = loss - match_rate * cos_to_target

        return loss / max(len(proposals), 1)

    def compute_knowledge_loss(self) -> Optional[torch.Tensor]:
        """辅助损失：让规则满足度与任务成功正相关。

        使用 rule_miner 收集的历史数据进行监督学习。

        Returns None if insufficient data.
        """
        if len(self.rule_miner._states) < 20:
            return None

        states = torch.stack(self.rule_miner._states[-100:])  # 最近 100 条
        compliant = torch.tensor(
            self.rule_miner._compliant[-100:], dtype=torch.float32
        )

        # 预测合规度
        pred_comp = self.knowledge_engine.compliance_score(states)
        # 监督损失：预测合规度应接近实际合规度
        loss = torch.nn.functional.mse_loss(pred_comp, compliant)
        return loss
