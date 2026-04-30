"""MDT (Multi-Disciplinary Team) Tumor Debate Environment — Section 6.1.

.. deprecated::
    Use GameEngine + MDTScenario instead.
    See mdt_game.scenario for the new implementation.

A specialized debate environment simulating multi-disciplinary tumor board
consultations. Agents negotiate treatment plans for synthetic cancer cases.

Treatment dimensions:
  0: Surgery        (0=none, 1=minimal, 2=standard, 3=radical)
  1: Chemotherapy   (0=none, 1=adjuvant, 2=neoadjuvant, 3=high-dose)
  2: Radiotherapy   (0=none, 1=local, 2=extended, 3=full)
  3: Immunotherapy  (0=none, 1=checkpoint, 2=combination, 3=experimental)
  4: Hormone        (0=none, 1=tamoxifen, 2=aromatase, 3=combined)
  5: Targeted       (0=none, 1=single-agent, 2=dual, 3=triple)
  6: Palliative     (0=none, 1=pain, 2=full-supportive, 3=hospice)
  7: Surveillance   (0=none, 1=quarterly, 2=monthly, 3=weekly)
  8: Clinical trial (0=no, 1=phase3, 2=phase2, 3=phase1)
  9: Rehab          (0=none, 1=physical, 2=psycho, 3=comprehensive)

Patient features (context):
  age, tumor_size, grade, stage, ki67, er, pr, her2, brca, comorbidity,
  performance_status, prior_treatments, ... (24 dims)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from debate_rl_v2.config import (
    AdversarialConfig,
    DevilAdvocateConfig,
    EnvConfig,
    KnowledgeConfig,
    RewardConfig,
    SoftSwitchConfig,
)
from debate_rl_v2.envs.base_env import DebateEnv


class MDTDebateEnv(DebateEnv):
    """MDT tumor board consultation environment.

    Extends DebateEnv with:
      - Domain-specific patient case generation
      - Medical protocol rules (hard + soft constraints)
      - Realistic treatment outcome simulation
    """

    # Indices into context vector
    IDX_AGE = 0
    IDX_TUMOR_SIZE = 1
    IDX_GRADE = 2
    IDX_STAGE = 3
    IDX_KI67 = 4
    IDX_ER = 5
    IDX_PR = 6
    IDX_HER2 = 7
    IDX_BRCA = 8
    IDX_COMORBIDITY = 9
    IDX_PERFORMANCE = 10

    def __init__(
        self,
        env_cfg: Optional[EnvConfig] = None,
        adv_cfg: Optional[AdversarialConfig] = None,
        know_cfg: Optional[KnowledgeConfig] = None,
        ss_cfg: Optional[SoftSwitchConfig] = None,
        da_cfg: Optional[DevilAdvocateConfig] = None,
        rew_cfg: Optional[RewardConfig] = None,
        seed: int = 0,
    ) -> None:
        # Default MDT dimensions
        if env_cfg is None:
            env_cfg = EnvConfig(
                name="mdt",
                context_dim=24,
                proposal_dim=10,
                proposal_values=4,
                embed_dim=32,
                rule_count=12,
                max_steps=40,
                meta_interval=5,
            )
        super().__init__(env_cfg, adv_cfg, know_cfg, ss_cfg, da_cfg, rew_cfg, seed)

    def reset(self) -> Dict[str, np.ndarray]:
        """Generate a new synthetic patient case and reset debate."""
        obs = super().reset()
        self._generate_patient()
        self._generate_target_treatment()
        return self._get_obs()

    def _generate_patient(self) -> None:
        """Generate a synthetic cancer patient profile."""
        rng = self.rng

        patient = np.zeros(self.context_dim, dtype=np.float32)
        patient[self.IDX_AGE] = rng.normal(0.55, 0.15)  # normalized age
        patient[self.IDX_TUMOR_SIZE] = rng.exponential(0.3)
        patient[self.IDX_GRADE] = rng.choice([0.33, 0.66, 1.0])  # grade 1-3
        patient[self.IDX_STAGE] = rng.choice([0.25, 0.5, 0.75, 1.0])  # I-IV
        patient[self.IDX_KI67] = rng.beta(2, 5)
        patient[self.IDX_ER] = float(rng.random() > 0.3)  # ER+
        patient[self.IDX_PR] = float(rng.random() > 0.35)  # PR+
        patient[self.IDX_HER2] = float(rng.random() > 0.8)  # HER2+
        patient[self.IDX_BRCA] = float(rng.random() > 0.9)  # BRCA mutation
        patient[self.IDX_COMORBIDITY] = rng.beta(2, 8)
        patient[self.IDX_PERFORMANCE] = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0])

        # Remaining dims: random noise features
        for i in range(11, self.context_dim):
            patient[i] = rng.standard_normal() * 0.3

        self.context = np.clip(patient, -2.0, 2.0)

    def _generate_target_treatment(self) -> None:
        """Generate a guideline-adherent target treatment based on patient."""
        rng = self.rng
        target = np.zeros(self.proposal_dim, dtype=int)
        stage = self.context[self.IDX_STAGE]
        er_pos = self.context[self.IDX_ER] > 0.5
        her2_pos = self.context[self.IDX_HER2] > 0.5
        perf = self.context[self.IDX_PERFORMANCE]

        # Simplified guideline logic
        if stage <= 0.5:
            # Early stage: surgery + adjuvant
            target[0] = 2  # standard surgery
            target[1] = 1 if stage > 0.25 else 0  # adjuvant chemo
            target[2] = 1  # local radio
        elif stage <= 0.75:
            # Locally advanced: neoadjuvant + surgery + radio
            target[0] = 2
            target[1] = 2  # neoadjuvant chemo
            target[2] = 2  # extended radio
        else:
            # Metastatic: systemic + palliative
            target[0] = rng.choice([0, 1])
            target[1] = 3  # high-dose chemo
            target[2] = 1
            target[6] = 2  # palliative

        if er_pos:
            target[4] = rng.choice([1, 2])  # hormone therapy
        if her2_pos:
            target[5] = rng.choice([1, 2])  # targeted therapy
        target[3] = 1 if stage > 0.5 else 0  # immunotherapy for advanced

        target[7] = 1 if stage <= 0.5 else 2  # surveillance frequency
        target[8] = rng.choice([0, 1]) if stage > 0.75 else 0  # clinical trial
        target[9] = 1 if perf > 0.5 else 0  # rehab

        self.target = target.astype(int)

    def _task_reward(self) -> float:
        """Domain-specific reward: weighted match with clinical importance."""
        # Surgery, chemo, radio are most critical
        weights = np.array([3.0, 2.5, 2.0, 1.5, 1.5, 1.5, 2.0, 1.0, 0.5, 0.5])
        weights = weights / weights.sum()
        matches = (self.proposal == self.target).astype(float)
        return float(np.dot(weights, matches))
