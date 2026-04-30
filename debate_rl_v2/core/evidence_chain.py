"""Evidence Chain Tracking — Section 4.3.

Records decision justification at every debate step for explainability
and downstream credit assignment.  Each record contains triggered rules,
confidence scores, actions taken, and compliance outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import json


@dataclass
class EvidenceRecord:
    """A single evidence record in the chain."""
    step: int
    role: str                                  # which agent acted
    action: Any                                # action taken
    triggered_rules: List[int]                 # indices of rules with μ > 0.5
    rule_confidences: List[float]              # current w_r for triggered rules
    rule_satisfactions: List[float]            # μ_r(s) for triggered rules
    compliance_score: float                    # overall compliance c
    disagreement: float                        # D_i(t)
    lambda_adv: float                          # current adversarial intensity
    mode: str = "standard"                     # soft-switch mode
    devil_advocate_active: bool = False
    notes: str = ""


class EvidenceChain:
    """Maintains a chain of evidence records for one debate episode.

    Provides methods to:
      - Record evidence at each step
      - Query the chain for specific patterns
      - Export the chain for visualization / audit
    """

    def __init__(self) -> None:
        self._records: List[EvidenceRecord] = []

    def reset(self) -> None:
        self._records.clear()

    def record(
        self,
        step: int,
        role: str,
        action: Any,
        triggered_rules: List[int],
        rule_confidences: List[float],
        rule_satisfactions: List[float],
        compliance_score: float,
        disagreement: float,
        lambda_adv: float,
        mode: str = "standard",
        devil_advocate_active: bool = False,
        notes: str = "",
    ) -> None:
        self._records.append(
            EvidenceRecord(
                step=step,
                role=role,
                action=action,
                triggered_rules=triggered_rules,
                rule_confidences=rule_confidences,
                rule_satisfactions=rule_satisfactions,
                compliance_score=compliance_score,
                disagreement=disagreement,
                lambda_adv=lambda_adv,
                mode=mode,
                devil_advocate_active=devil_advocate_active,
                notes=notes,
            )
        )

    def get_justification(self) -> str:
        """Generate a human-readable justification summary."""
        if not self._records:
            return "No evidence recorded."

        lines = ["=== Decision Evidence Chain ===\n"]
        for rec in self._records:
            lines.append(
                f"Step {rec.step} [{rec.role}] action={rec.action}  "
                f"compliance={rec.compliance_score:.3f}  "
                f"disagreement={rec.disagreement:.3f}  "
                f"λ={rec.lambda_adv:.3f}  mode={rec.mode}"
            )
            if rec.triggered_rules:
                rule_info = ", ".join(
                    f"R{idx}(w={w:.2f},μ={mu:.2f})"
                    for idx, w, mu in zip(
                        rec.triggered_rules,
                        rec.rule_confidences,
                        rec.rule_satisfactions,
                    )
                )
                lines.append(f"  Rules: {rule_info}")
            if rec.devil_advocate_active:
                lines.append("  ⚡ Devil's Advocate ACTIVE")
            if rec.notes:
                lines.append(f"  Note: {rec.notes}")
        return "\n".join(lines)

    def export_json(self) -> str:
        """Export evidence chain as JSON for programmatic analysis."""
        records = []
        for rec in self._records:
            records.append({
                "step": rec.step,
                "role": rec.role,
                "action": str(rec.action),
                "triggered_rules": rec.triggered_rules,
                "rule_confidences": rec.rule_confidences,
                "rule_satisfactions": rec.rule_satisfactions,
                "compliance_score": rec.compliance_score,
                "disagreement": rec.disagreement,
                "lambda_adv": rec.lambda_adv,
                "mode": rec.mode,
                "devil_advocate_active": rec.devil_advocate_active,
                "notes": rec.notes,
            })
        return json.dumps(records, indent=2, ensure_ascii=False)

    @property
    def length(self) -> int:
        return len(self._records)

    def __len__(self) -> int:
        return len(self._records)
