"""Refactored Text Debate Environment — uses DebateLogicMixin + EventEmitter.

.. deprecated::
    Use GameEngine + DebateGameScenario instead.
    See debate_rl_v2.scenarios.debate.scenario for the new implementation.

Slimmed from ~866 lines to ~300 lines by extracting shared logic to
DebateLogicMixin and dashboard updates to DebateEventEmitter.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from debate_rl_v2.scenarios.debate.types import TextDebateState, DebateTurn
from debate_rl_v2.envs.debate_logic import DebateLogicMixin
from debate_rl_v2.envs.event_emitter import (
    DebateEventEmitter,
    DebateEvent,
    DebateEventType,
)
from debate_rl_v2.logging_config import get_logger

logger = get_logger("envs.llm_env")


class TextDebateEnv(DebateLogicMixin):
    """Natural-language debate environment — refactored v2.

    Key improvements over v1:
      - Uses DebateLogicMixin for shared logic (DRY)
      - Uses DebateEventEmitter for dashboard/tracing (no try/except pass)
      - Structured logging replaces print()
      - Context compression support
    """

    def __init__(
        self,
        topic: str,
        context: str = "",
        rules: Optional[List[str]] = None,
        max_rounds: int = 10,
        meta_interval: int = 3,
        adv_eta: float = 0.2,
        consensus_threshold: float = 0.8,
        disagreement_consensus: float = 0.15,
        da_stability_window: int = 3,
        da_disagreement_threshold: float = 0.15,
        da_max_challenges: int = 3,
        emitter: Optional[DebateEventEmitter] = None,
    ) -> None:
        self.topic = topic
        self.context = context
        self.rules = rules or []
        self.max_rounds = max_rounds
        self.meta_interval = meta_interval
        self.consensus_threshold = consensus_threshold
        self.disagreement_consensus = disagreement_consensus

        # Devil's advocate state (used by DebateLogicMixin)
        self.da_disagreement_threshold = da_disagreement_threshold
        self.da_stability_window = da_stability_window
        self._da_active = False
        self._da_stable_count = 0
        self._da_challenges_issued = 0
        self._da_max_challenges = da_max_challenges

        # Event emitter (replaces dashboard try/except)
        self.emitter = emitter or DebateEventEmitter()

        # State
        self.state = TextDebateState()
        self.history: List[DebateTurn] = []

        # RNG for soft switch
        self.rng = np.random.default_rng(42)

    def run(
        self,
        agents: Dict[str, Any],
        verbose: bool = True,
        adv_controller: Any = None,
        soft_switch: Any = None,
        skill_db: Any = None,
        causal_graph: Any = None,
        episode_distiller: Any = None,
        mode_controller: Any = None,
    ) -> Dict[str, Any]:
        """Run a complete text debate.

        Parameters
        ----------
        agents : dict
            {"proposer": LLMAgent, "challenger": LLMAgent,
             "arbiter": LLMAgent, "coordinator": LLMAgent}
        verbose : bool
            Log debate progress.
        adv_controller : AdversarialIntensityController, optional
        soft_switch : SoftSwitchController, optional
        skill_db : SkillDatabase, optional
            SQLite skill database for skill/causal context injection.
        causal_graph : CausalGraph, optional
            Causal graph for context injection.
        episode_distiller : EpisodeDistiller, optional
            Episode-level knowledge distillation.
        mode_controller : ModeController, optional
            Dual-mode controller.

        Returns
        -------
        result : dict
            Debate result with transcript, metrics, etc.
        """
        # Build extra context from skills and causal graph
        extra_context = self._build_extra_context(skill_db, causal_graph)

        self.state = TextDebateState()
        self.history.clear()
        self._da_active = False
        self._da_stable_count = 0
        self._da_challenges_issued = 0

        self.emitter.emit(DebateEvent(
            event_type=DebateEventType.DEBATE_START,
            data={"topic": self.topic, "max_rounds": self.max_rounds},
        ))

        if verbose:
            logger.info("Debate started: %s (max %d rounds)", self.topic, self.max_rounds)

        for round_num in range(1, self.max_rounds + 1):
            self.state.round_num = round_num

            self.emitter.emit(DebateEvent(
                event_type=DebateEventType.ROUND_START,
                round_num=round_num,
                data={"max_rounds": self.max_rounds},
            ))

            # Soft switch mode
            if soft_switch:
                lam = adv_controller.current_intensity if adv_controller else 0.5
                switch = soft_switch.decide(lam, self.rng)
                self.state.mode = switch.mode
                self.state.lambda_adv = lam

            # Coordinator (every meta_interval rounds)
            coord_action = 0
            coord_reasoning = ""
            if round_num % self.meta_interval == 1 or round_num == 1:
                coord_result = agents["coordinator"].act(
                    self._build_coordinator_msg(), round_num=round_num
                )
                coord_action = coord_result.get("action_idx", 0)
                coord_reasoning = coord_result.get("reasoning", "")
                if adv_controller:
                    self._apply_coordinator_action(coord_action, adv_controller, soft_switch)

                self.emitter.emit(DebateEvent(
                    event_type=DebateEventType.COORDINATOR_ACTION,
                    round_num=round_num,
                    data={"role": "coordinator", "content": coord_reasoning[:200],
                          "meta": f"action={coord_action}"},
                ))

            # Proposer
            proposer_msg = self._build_proposer_msg()
            if extra_context:
                proposer_msg = f"{extra_context}\n\n{proposer_msg}"
            prop_result = agents["proposer"].act(
                proposer_msg, round_num=round_num
            )
            self.state.proposal = prop_result.get("proposal", "")

            self.emitter.emit(DebateEvent(
                event_type=DebateEventType.PROPOSAL,
                round_num=round_num,
                data={"role": "proposer", "content": self.state.proposal[:200],
                      "confidence": prop_result.get("confidence", 0.5)},
            ))

            # Challenger
            chal_result = agents["challenger"].act(
                self._build_challenger_msg(), round_num=round_num
            )
            self.state.challenge = chal_result.get("challenge", "")

            self.emitter.emit(DebateEvent(
                event_type=DebateEventType.CHALLENGE,
                round_num=round_num,
                data={"role": "challenger", "content": self.state.challenge[:200],
                      "confidence": chal_result.get("confidence", 0.5)},
            ))

            # Arbiter
            arb_result = agents["arbiter"].act(
                self._build_arbiter_msg(), round_num=round_num
            )
            self.state.verdict = arb_result.get("verdict", "")
            self.state.quality_score = arb_result.get("quality_score", 0.5)

            # Update mechanism state (via Mixin)
            self.state.disagreement = self._compute_disagreement(
                arb_result, prop_result, chal_result, round_num
            )
            self.state.compliance = self._evaluate_compliance(arb_result, round_num)

            # Update adversarial intensity
            if adv_controller:
                adv_controller.update(self.state.disagreement, self.state.compliance)
                self.state.lambda_adv = adv_controller.current_intensity

            self.emitter.emit(DebateEvent(
                event_type=DebateEventType.VERDICT,
                round_num=round_num,
                data={"role": "arbiter", "content": self.state.verdict[:200],
                      "quality": self.state.quality_score},
            ))

            # Record turn
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
                reasoning={
                    "proposer": prop_result.get("reasoning", ""),
                    "challenger": chal_result.get("reasoning", ""),
                    "arbiter": arb_result.get("reasoning", ""),
                    "coordinator": coord_reasoning,
                },
            )
            self.history.append(turn)

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
                    "prop_confidence": prop_result.get("confidence", 0.5),
                    "chal_confidence": chal_result.get("confidence", 0.5),
                    "da_active": self._da_active,
                },
            ))

            # Devil's advocate check (via Mixin)
            self._check_devil_advocate(agents, verbose, round_num)

            # Consensus check
            consensus_ok = (
                (arb_result.get("consensus_reached", False)
                 or self.state.disagreement < self.disagreement_consensus)
                and self.state.quality_score >= self.consensus_threshold
            )

            if consensus_ok:
                self.state.consensus_reached = True
                self.emitter.emit(DebateEvent(
                    event_type=DebateEventType.CONSENSUS_REACHED,
                    round_num=round_num,
                    data={"quality": self.state.quality_score},
                ))
                if verbose:
                    logger.info(
                        "Consensus reached at round %d (quality=%.2f)",
                        round_num, self.state.quality_score,
                    )
                break

        self.emitter.emit(DebateEvent(
            event_type=DebateEventType.DEBATE_END,
            data={"consensus": self.state.consensus_reached,
                  "total_rounds": len(self.history)},
        ))

        result = self._compile_result()

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

        if mode_controller is not None:
            mode_controller.record_episode()

        return result

    # Prompt builders inherited from DebateLogicMixin (DRY)

    def _apply_coordinator_action(
        self, action: int, adv_controller: Any, soft_switch: Any
    ) -> None:
        ac = adv_controller
        if action == 1:
            ac.eta = min(0.5, ac.eta + 0.02)
        elif action == 2:
            ac.eta = max(0.05, ac.eta - 0.02)
        elif action == 3:
            ac.alpha = min(0.9, ac.alpha + 0.05)
        elif action == 4:
            ac.alpha = max(0.1, ac.alpha - 0.05)
        elif action == 5 and soft_switch:
            soft_switch.update_thresholds(soft_switch.tau_low + 0.05, soft_switch.tau_high)
        elif action == 6 and soft_switch:
            soft_switch.update_thresholds(soft_switch.tau_low - 0.05, soft_switch.tau_high)
        elif action == 7 and soft_switch:
            soft_switch.update_thresholds(soft_switch.tau_low, soft_switch.tau_high + 0.05)
        elif action == 8 and soft_switch:
            soft_switch.update_thresholds(soft_switch.tau_low, soft_switch.tau_high - 0.05)

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
        }

    def _build_extra_context(self, skill_db: Any, causal_graph: Any) -> str:
        """Build extra context from skill DB and causal graph for prompt injection."""
        parts: List[str] = []

        # Skill context
        if skill_db is not None:
            records = skill_db.find_relevant(self.topic, top_k=3)
            if records:
                lines = ["## 相关辩论技能"]
                for r in records:
                    lines.append(f"- {r.name}: {r.description[:100]} (质量={r.avg_quality:.2f})")
                parts.append("\n".join(lines))

        # Causal context
        if causal_graph is not None:
            causal_text = causal_graph.build_context(self.topic, max_chains=3)
            if causal_text:
                parts.append(causal_text)

        return "\n\n".join(parts)
