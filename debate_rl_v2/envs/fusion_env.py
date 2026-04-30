"""Refactored Fusion Debate Environment — RL-Guided LLM Debate.

.. deprecated::
    Use GameEngine + DebateGameScenario instead.
    See debate_rl_v2.scenarios.debate.scenario for the new implementation.

Slimmed from ~888 lines by using DebateLogicMixin, EventEmitter,
and the enhanced StrategyBridge with compliance verification.

Key v2 improvements:
  - Closed-loop RL→LLM control via ComplianceVerifier
  - Role-specific observations (20D per agent)
  - Event-driven dashboard updates (no try/except pass)
  - Context compression for long debates
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from debate_rl_v2.core.strategy_bridge import (
    StrategyBridge,
    StrategySignals,
)
from debate_rl_v2.scenarios.debate.types import TextDebateState, DebateTurn, FusionRoundRecord
from debate_rl_v2.envs.debate_logic import DebateLogicMixin
from debate_rl_v2.envs.event_emitter import (
    DebateEventEmitter,
    DebateEvent,
    DebateEventType,
)
from debate_rl_v2.logging_config import get_logger

logger = get_logger("envs.fusion_env")


class FusionDebateEnv(DebateLogicMixin):
    """RL-Guided LLM Debate Environment — refactored v2.

    Combines MADDPG controllers with LLM agents. RL-learned strategies
    dynamically guide LLM behavior each round via the StrategyBridge.

    v2 improvements:
      - DebateLogicMixin eliminates code duplication with TextDebateEnv
      - EventEmitter replaces scattered try/except dashboard updates
      - ComplianceVerifier closes the RL→LLM control loop
      - Role-specific observations for richer critic learning
    """

    def __init__(
        self,
        topic: str,
        context: str = "",
        rules: Optional[List[str]] = None,
        max_rounds: int = 10,
        meta_interval: int = 3,
        consensus_threshold: float = 0.8,
        disagreement_consensus: float = 0.15,
        da_stability_window: int = 3,
        da_disagreement_threshold: float = 0.15,
        da_max_challenges: int = 3,
        bridge: Optional[StrategyBridge] = None,
        emitter: Optional[DebateEventEmitter] = None,
    ) -> None:
        self.topic = topic
        self.context = context
        self.rules = rules or []
        self.max_rounds = max_rounds
        self.meta_interval = meta_interval
        self.consensus_threshold = consensus_threshold
        self.disagreement_consensus = disagreement_consensus

        # Devil's advocate (used by DebateLogicMixin)
        self.da_disagreement_threshold = da_disagreement_threshold
        self.da_stability_window = da_stability_window
        self._da_active = False
        self._da_stable_count = 0
        self._da_challenges_issued = 0
        self._da_max_challenges = da_max_challenges

        # Core components
        self.bridge = bridge or StrategyBridge()
        self.emitter = emitter or DebateEventEmitter()

        # State
        self.state = TextDebateState()
        self.history: List[DebateTurn] = []
        self.fusion_history: List[FusionRoundRecord] = []
        self._current_signals: Optional[StrategySignals] = None

        self.rng = np.random.default_rng(42)

    def run(
        self,
        rl_agents: Any,
        llm_agents: Dict[str, Any],
        verbose: bool = True,
        explore: bool = True,
        adv_controller: Any = None,
        soft_switch: Any = None,
        reward_computer: Any = None,
        debate_metrics: Any = None,
        mode_controller: Any = None,
        online_updater: Any = None,
        episode_distiller: Any = None,
    ) -> Dict[str, Any]:
        """Run a complete fusion debate.

        Parameters
        ----------
        rl_agents : MADDPGAgentGroup
            RL strategy controllers.
        llm_agents : dict
            {"proposer": LLMAgent, "challenger": LLMAgent, ...}
        verbose : bool
            Log progress.
        explore : bool
            RL exploration noise.
        adv_controller : AdversarialIntensityController, optional
        soft_switch : SoftSwitchController, optional
        reward_computer : EnhancedRewardComputer, optional
        debate_metrics : DebateMetrics, optional
        mode_controller : ModeController, optional
            Dual-mode controller. When online, freezes RL and uses OnlineUpdater.
        online_updater : OnlineParameterUpdater, optional
            Gradient-free parameter accumulator for online mode.
        episode_distiller : EpisodeDistiller, optional
            Episode-level knowledge distillation.

        Returns
        -------
        result : dict
            Debate result with transcript, RL transitions, metrics.
        """
        # Override explore flag based on mode
        if mode_controller is not None and mode_controller.is_online:
            explore = False
        self.state = TextDebateState()
        self.history.clear()
        self.fusion_history.clear()
        self._da_active = False
        self._da_stable_count = 0
        self._da_challenges_issued = 0
        self.bridge.reset()

        self.emitter.emit(DebateEvent(
            event_type=DebateEventType.DEBATE_START,
            data={"topic": self.topic, "max_rounds": self.max_rounds, "mode": "fusion"},
        ))

        prev_state_dict: Dict[str, float] = {
            "quality": 0.5, "disagreement": 0.5, "compliance": 0.5,
        }

        for round_num in range(1, self.max_rounds + 1):
            self.state.round_num = round_num

            self.emitter.emit(DebateEvent(
                event_type=DebateEventType.ROUND_START,
                round_num=round_num,
                data={"max_rounds": self.max_rounds},
            ))

            # ── Step 1: Encode observation ──
            quality_trend = self._compute_quality_trend()
            disagree_trend = self._compute_disagree_trend()
            shared_obs = self.bridge.encode_observation(
                round_num=round_num,
                max_rounds=self.max_rounds,
                disagreement=self.state.disagreement,
                quality_score=self.state.quality_score,
                compliance=self.state.compliance,
                lambda_adv=self.state.lambda_adv,
                da_active=self._da_active,
                mode=self.state.mode,
                prop_confidence=self._last_prop_conf(),
                chal_confidence=self._last_chal_conf(),
                quality_trend=quality_trend,
                disagreement_trend=disagree_trend,
                prop_score=self._last_prop_score(),
                chal_score=self._last_chal_score(),
            )

            # ── Step 2: RL agents produce actions ──
            rl_actions: Dict[str, np.ndarray] = {}
            for role in rl_agents.agent_names:
                # v2: Role-specific observation (20D)
                role_obs = self.bridge.encode_role_observation(shared_obs, role)
                action = rl_agents[role].act(role_obs, explore=explore)
                rl_actions[role] = action

            # ── Step 3: Strategy Bridge → signals ──
            signals = self.bridge.translate(
                proposer_action=rl_actions.get("proposer_ctrl", np.zeros(4)),
                challenger_action=rl_actions.get("challenger_ctrl", np.zeros(4)),
                arbiter_action=rl_actions.get("arbiter_ctrl", np.zeros(4)),
                coordinator_action=rl_actions.get("coordinator", np.zeros(5)),
            )

            # ── Step 3b: Online mode override ──
            if mode_controller is not None and mode_controller.is_online and online_updater is not None:
                online_params = {
                    role: online_updater.get_best_params(role)
                    for role in ("proposer", "challenger", "arbiter", "coordinator")
                }
                signals = self.bridge.apply_online_override(signals, online_params)

            self._current_signals = signals

            self.emitter.emit(DebateEvent(
                event_type=DebateEventType.RL_SIGNALS,
                round_num=round_num,
                data={
                    "role": "system",
                    "content": (
                        f"RL策略: prop_temp={signals.proposer_temperature:.2f} "
                        f"chal_temp={signals.challenger_temperature:.2f} "
                        f"assertive={signals.proposer_assertiveness:.2f} "
                        f"aggressive={signals.challenger_aggressiveness:.2f}"
                    ),
                },
            ))

            # ── Step 4: Apply strategy signals to LLM agents ──
            self._apply_signals(signals, llm_agents)

            # ── Step 5: Execute LLM debate round ──

            # Coordinator (LLM)
            coord_action = 0
            if round_num % self.meta_interval == 1 or round_num == 1:
                coord_result = llm_agents["coordinator"].act(
                    self._build_coordinator_msg(), round_num=round_num
                )
                coord_action = coord_result.get("action_idx", 0)
                _ = coord_result.get("reasoning", "")  # available for logging
                if adv_controller:
                    self._apply_coordinator_action_from_signals(signals, adv_controller, soft_switch)

            # Proposer
            prop_result = llm_agents["proposer"].act(
                self._build_proposer_msg(), round_num=round_num
            )
            self.state.proposal = prop_result.get("proposal", "")

            self.emitter.emit(DebateEvent(
                event_type=DebateEventType.PROPOSAL,
                round_num=round_num,
                data={"role": "proposer", "content": self.state.proposal[:200]},
            ))

            # Challenger
            chal_result = llm_agents["challenger"].act(
                self._build_challenger_msg(), round_num=round_num
            )
            self.state.challenge = chal_result.get("challenge", "")

            self.emitter.emit(DebateEvent(
                event_type=DebateEventType.CHALLENGE,
                round_num=round_num,
                data={"role": "challenger", "content": self.state.challenge[:200]},
            ))

            # Arbiter
            arb_result = llm_agents["arbiter"].act(
                self._build_arbiter_msg(), round_num=round_num
            )
            self.state.verdict = arb_result.get("verdict", "")
            self.state.quality_score = arb_result.get("quality_score", 0.5)

            self.emitter.emit(DebateEvent(
                event_type=DebateEventType.VERDICT,
                round_num=round_num,
                data={"role": "arbiter", "content": self.state.verdict[:200],
                      "quality": self.state.quality_score},
            ))

            # ── Step 6: Update mechanism state ──
            self.state.disagreement = self._compute_disagreement(
                arb_result, prop_result, chal_result, round_num
            )
            self.state.compliance = self._evaluate_compliance(arb_result, round_num)

            if adv_controller:
                adv_controller.update(self.state.disagreement, self.state.compliance)
                self.state.lambda_adv = adv_controller.current_intensity

            # ── Step 6b: Compliance verification (v2 closed-loop) ──
            compliance_results = self.bridge.verify_compliance(
                signals,
                responses={
                    "proposer": self.state.proposal,
                    "challenger": self.state.challenge,
                    "arbiter": self.state.verdict,
                },
            )
            compliance_rewards = self.bridge.get_compliance_rewards()

            # ── Step 7: Compute rewards ──
            curr_state_dict = {
                "quality": self.state.quality_score,
                "disagreement": self.state.disagreement,
                "compliance": self.state.compliance,
            }

            if reward_computer and debate_metrics:
                rewards = reward_computer.compute_step_rewards(
                    prev_state=prev_state_dict,
                    curr_state=curr_state_dict,
                    metrics=debate_metrics,
                    done=False,
                    consensus_reached=False,
                )
                # Add compliance bonuses
                for role, bonus in compliance_rewards.items():
                    if role in rewards:
                        rewards[role] += bonus
            else:
                rewards = self.bridge.compute_reward(
                    prev_state=prev_state_dict,
                    curr_state=curr_state_dict,
                    done=False,
                    consensus_reached=False,
                    compliance_rewards=compliance_rewards,
                )

            # ── Step 8: Update role observation tracker (v2) ──
            if self.bridge.role_tracker is not None:
                quality_delta = self._compute_quality_trend()
                accept_threshold = getattr(self, '_accept_threshold', 0.6)
                challenge_threshold = getattr(self, '_challenge_threshold', 0.5)
                quality_target = getattr(self, '_quality_target', 0.7)
                self.bridge.role_tracker.update_proposer(
                    quality=self.state.quality_score,
                    accepted=arb_result.get("proposal_score", 0.5) > accept_threshold,
                    modification_mag=abs(quality_delta),
                )
                self.bridge.role_tracker.update_challenger(
                    success=arb_result.get("challenge_score", 0.5) > challenge_threshold,
                    info_gain=abs(self._compute_disagree_trend()),
                    attack_angle=signals.get_style("challenger").get("novelty", 0.5) if hasattr(signals, 'get_style') else 0.5,
                )
                self.bridge.role_tracker.update_arbiter(
                    consistency=1.0 - abs(quality_delta),
                    rule_trigger_rate=self.state.compliance,
                    calibration_error=abs(self.state.quality_score - quality_target),
                )
                self.bridge.role_tracker.update_coordinator(
                    convergence_speed=max(0, -self._compute_disagree_trend()),
                    mining_benefit=self.state.compliance,
                    termination_accuracy=self.state.quality_score,
                )

            # ── Step 9: Record ──
            turn = DebateTurn(
                round_num=round_num,
                proposal=self.state.proposal,
                proposal_confidence=prop_result.get("confidence", 0.5),
                challenge=self.state.challenge,
                challenge_confidence=chal_result.get("confidence", 0.5),
                verdict=self.state.verdict,
                arbiter_scores={
                    "proposal_score": arb_result.get("proposal_score", 0.5),
                    "challenge_score": arb_result.get("challenge_score", 0.5),
                    "quality_score": arb_result.get("quality_score", 0.5),
                },
                coordinator_action=coord_action,
                state=TextDebateState(**vars(self.state)),
            )
            self.history.append(turn)

            fusion_rec = FusionRoundRecord(
                round_num=round_num,
                observation=shared_obs,
                rl_actions=rl_actions,
                strategy_signals=signals.to_dict(),
                rewards=rewards,
                proposal=self.state.proposal[:200],
                challenge=self.state.challenge[:200],
                verdict=self.state.verdict[:200],
                quality_score=self.state.quality_score,
                disagreement=self.state.disagreement,
                compliance=self.state.compliance,
                lambda_adv=self.state.lambda_adv,
                mode=self.state.mode,
                compliance_scores={
                    r: c.overall_score for r, c in compliance_results.items()
                },
            )
            self.fusion_history.append(fusion_rec)

            prev_state_dict = curr_state_dict

            self.emitter.emit(DebateEvent(
                event_type=DebateEventType.ROUND_END,
                round_num=round_num,
                data={
                    "round_num": round_num,
                    "disagreement": self.state.disagreement,
                    "quality_score": self.state.quality_score,
                    "compliance": self.state.compliance,
                    "lambda_adv": self.state.lambda_adv,
                    "mode": self.state.mode,
                    "da_active": self._da_active,
                },
            ))

            # Devil's advocate
            self._check_devil_advocate(llm_agents, verbose, round_num)

            # Consensus check
            consensus_ok = (
                (arb_result.get("consensus_reached", False)
                 or self.state.disagreement < self.disagreement_consensus)
                and self.state.quality_score >= self.consensus_threshold
            )

            if consensus_ok:
                self.state.consensus_reached = True
                # Recompute terminal rewards
                terminal_rewards = self.bridge.compute_reward(
                    prev_state=prev_state_dict,
                    curr_state=curr_state_dict,
                    done=True,
                    consensus_reached=True,
                    compliance_rewards=compliance_rewards,
                )
                self.fusion_history[-1].rewards = terminal_rewards

                self.emitter.emit(DebateEvent(
                    event_type=DebateEventType.CONSENSUS_REACHED,
                    round_num=round_num,
                    data={"quality": self.state.quality_score},
                ))
                if verbose:
                    logger.info(
                        "Fusion consensus at round %d (quality=%.2f)",
                        round_num, self.state.quality_score,
                    )
                break

        self.emitter.emit(DebateEvent(
            event_type=DebateEventType.DEBATE_END,
            data={"consensus": self.state.consensus_reached,
                  "total_rounds": len(self.history), "mode": "fusion"},
        ))

        result = self._compile_result()

        # ── Post-episode: Online parameter update ──
        if mode_controller is not None and mode_controller.is_online and online_updater is not None:
            if self._current_signals is not None:
                for role in ("proposer", "challenger", "arbiter", "coordinator"):
                    params = self._extract_role_params(self._current_signals, role)
                    if params is not None:
                        online_updater.update(role, params, self.state.quality_score)

        # ── Post-episode: Knowledge distillation ──
        if episode_distiller is not None:
            should_distill = True
            if mode_controller is not None:
                should_distill = mode_controller.should_distill(self.state.quality_score)
            if should_distill:
                episode_distiller.distill(
                    result, result.get("transcript", []),
                    topic=self.topic,
                )

        # ── Post-episode: Record mode episode ──
        if mode_controller is not None:
            mode_controller.record_episode()

        return result

    # ── Signal application ──

    @staticmethod
    def _extract_role_params(signals: StrategySignals, role: str) -> Optional[np.ndarray]:
        """Extract 4D parameter vector from signals for a given role."""
        if role == "proposer":
            return np.array([
                signals.proposer_assertiveness,
                signals.proposer_detail_level,
                signals.proposer_compliance_focus,
                signals.proposer_incorporation,
            ], dtype=np.float32)
        elif role == "challenger":
            return np.array([
                signals.challenger_aggressiveness,
                signals.challenger_specificity,
                signals.challenger_constructiveness,
                signals.challenger_novelty,
            ], dtype=np.float32)
        elif role == "arbiter":
            return np.array([
                signals.arbiter_strictness,
                signals.arbiter_detail_feedback,
                signals.arbiter_consensus_bias,
                signals.arbiter_rule_emphasis,
            ], dtype=np.float32)
        elif role == "coordinator":
            return np.array([
                signals.eta_delta,
                signals.alpha_delta,
                signals.tau_low_delta,
                signals.exploration_rate,
            ], dtype=np.float32)
        return None

    def _apply_signals(self, signals: StrategySignals, llm_agents: Dict[str, Any]) -> None:
        """Apply RL strategy signals to LLM agents."""
        # Temperature
        for role, temp in [
            ("proposer", signals.proposer_temperature),
            ("challenger", signals.challenger_temperature),
            ("arbiter", signals.arbiter_temperature),
        ]:
            agent = llm_agents.get(role)
            if agent and hasattr(agent, "client") and hasattr(agent.client, "temperature"):
                agent.client.temperature = temp

        # Style directives (injected as system prompt prefix)
        style_map = {
            "proposer": self.bridge.compose_proposer_style(signals),
            "challenger": self.bridge.compose_challenger_style(signals),
            "arbiter": self.bridge.compose_arbiter_style(signals),
        }
        for role, style in style_map.items():
            agent = llm_agents.get(role)
            if agent and style and hasattr(agent, "_style_directive"):
                agent._style_directive = style

    def _apply_coordinator_action_from_signals(
        self, signals: StrategySignals, adv_controller: Any, soft_switch: Any
    ) -> None:
        """Apply coordinator RL signals to mechanism parameters."""
        if adv_controller:
            adv_controller.eta = max(0.01, min(0.5, adv_controller.eta + signals.eta_delta))
            adv_controller.alpha = max(0.1, min(0.95, adv_controller.alpha + signals.alpha_delta))
        if soft_switch:
            new_low = max(0.1, min(0.6, soft_switch.tau_low + signals.tau_low_delta))
            new_high = max(0.4, min(0.95, soft_switch.tau_high + signals.tau_high_delta))
            if new_low < new_high:
                soft_switch.update_thresholds(new_low, new_high)

    # ── Prompt builders: override Mixin to prepend RL style directives ──

    def _build_proposer_msg(self) -> str:
        base = super()._build_proposer_msg()
        style = self.bridge.compose_proposer_style(self._current_signals) if self._current_signals else ""
        return f"[策略指导]\n{style}\n\n{base}" if style else base

    def _build_challenger_msg(self) -> str:
        base = super()._build_challenger_msg()
        style = self.bridge.compose_challenger_style(self._current_signals) if self._current_signals else ""
        return f"[策略指导]\n{style}\n\n{base}" if style else base

    def _build_arbiter_msg(self) -> str:
        base = super()._build_arbiter_msg()
        style = self.bridge.compose_arbiter_style(self._current_signals) if self._current_signals else ""
        return f"[策略指导]\n{style}\n\n{base}" if style else base

    def _build_coordinator_msg(self) -> str:
        trend = self._compute_trend()
        return (
            f"辩论进展: 第{self.state.round_num}/{self.max_rounds}轮\n"
            f"质量={self.state.quality_score:.2f} 分歧={self.state.disagreement:.2f}\n"
            f"趋势: {trend}\n请选择元动作来调整辩论参数。"
        )

    # ── Transition extraction ──

    def get_transitions(self) -> List[Dict[str, Any]]:
        """Extract (obs, act, rew, next_obs, done) tuples for replay buffer."""
        transitions = []
        for i, rec in enumerate(self.fusion_history):
            is_last = (i == len(self.fusion_history) - 1)
            next_obs = (
                self.fusion_history[i + 1].observation
                if not is_last
                else rec.observation
            )
            transitions.append({
                "obs": {role: rec.observation for role in rec.rl_actions},
                "actions": rec.rl_actions,
                "rewards": rec.rewards,
                "next_obs": {role: next_obs for role in rec.rl_actions},
                "done": is_last and self.state.consensus_reached,
            })
        return transitions

    def _compile_result(self) -> Dict[str, Any]:
        return {
            "consensus_reached": self.state.consensus_reached,
            "total_rounds": len(self.history),
            "final_quality": self.state.quality_score,
            "final_disagreement": self.state.disagreement,
            "final_compliance": self.state.compliance,
            "final_proposal": self.state.proposal,
            "transcript": [
                {
                    "round": t.round_num,
                    "proposal": t.proposal,
                    "challenge": t.challenge,
                    "verdict": t.verdict,
                    "quality": t.state.quality_score,
                    "disagreement": t.state.disagreement,
                }
                for t in self.history
            ],
            "fusion_data": [
                {
                    "round": r.round_num,
                    "rewards": r.rewards,
                    "signals": r.strategy_signals,
                    "compliance_scores": r.compliance_scores,
                    "quality_score": r.quality_score,
                    "disagreement": r.disagreement,
                    "compliance": r.compliance,
                    "lambda_adv": r.lambda_adv,
                    "mode": r.mode,
                }
                for r in self.fusion_history
            ],
            "transitions": self.get_transitions(),
        }
