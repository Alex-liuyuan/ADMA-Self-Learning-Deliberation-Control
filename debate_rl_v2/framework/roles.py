"""Role abstraction — domain-agnostic role definitions.

Any collaboration scenario defines its roles by subclassing BaseRole.
The RoleRegistry manages role discovery and validation.

Examples:
  - Debate: Proposer, Challenger, Arbiter, Coordinator
  - Code Review: Author, Reviewer, Maintainer
  - Negotiation: Buyer, Seller, Mediator
  - Brainstorm: Ideator, Critic, Synthesizer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RoleDefinition:
    """Declarative role specification — no code, just metadata."""
    name: str
    description: str
    system_prompt: str
    phase: str = "propose"  # which InteractionPhase this role acts in
    action_dim: int = 4     # RL action dimensionality for this role
    is_evaluator: bool = False
    is_coordinator: bool = False
    output_schema: dict[str, Any] = field(default_factory=dict)
    style_dimensions: list[str] = field(default_factory=list)

    def default_output(self) -> dict[str, Any]:
        """Generate a safe fallback output matching the schema."""
        defaults: dict[str, Any] = {}
        for key, spec in self.output_schema.items():
            dtype = spec if isinstance(spec, str) else spec.get("type", "string")
            if dtype == "string":
                defaults[key] = ""
            elif dtype == "number":
                defaults[key] = 0.5
            elif dtype == "boolean":
                defaults[key] = False
            elif dtype == "integer":
                defaults[key] = 0
            else:
                defaults[key] = ""
        return defaults


class RoleRegistry:
    """Registry of roles for a collaboration scenario.

    Usage::

        registry = RoleRegistry()
        registry.register(RoleDefinition(
            name="proposer",
            description="Generates proposals",
            system_prompt="You are a proposer...",
            phase="propose",
            style_dimensions=["assertiveness", "detail_level"],
            output_schema={"proposal": "string", "confidence": "number"},
        ))

        for role in registry.get_roles():
            ...  # role.name, role.phase
    """

    def __init__(self) -> None:
        self._roles: dict[str, RoleDefinition] = {}

    def register(self, role: RoleDefinition) -> None:
        self._roles[role.name] = role

    def get(self, name: str) -> RoleDefinition | None:
        return self._roles.get(name)

    def get_roles(self, phase: str | None = None) -> list[RoleDefinition]:
        roles = list(self._roles.values())
        if phase:
            roles = [r for r in roles if r.phase == phase]
        return roles

    @property
    def role_names(self) -> list[str]:
        return list(self._roles.keys())

    @property
    def evaluator(self) -> RoleDefinition | None:
        for r in self._roles.values():
            if r.is_evaluator:
                return r
        return None

    @property
    def coordinator(self) -> RoleDefinition | None:
        for r in self._roles.values():
            if r.is_coordinator:
                return r
        return None

    def __len__(self) -> int:
        return len(self._roles)

    def __contains__(self, name: str) -> bool:
        return name in self._roles
