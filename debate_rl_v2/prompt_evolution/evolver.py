"""PromptEvolver — evolutionary algorithm for prompt optimization.

Maintains a population of prompt candidates per role, using tournament
selection, LLM-driven mutation, and crossover to evolve better prompts.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Any

from debate_rl_v2.logging_config import get_logger

logger = get_logger("prompt_evolution.evolver")


@dataclass
class PromptCandidate:
    """A single prompt candidate in the evolutionary population."""
    prompt_id: str = ""
    role: str = ""
    template: str = ""
    fitness: float = 0.0
    usage_count: int = 0
    generation: int = 0
    parent_id: str = ""

    def __post_init__(self) -> None:
        if not self.prompt_id:
            self.prompt_id = str(uuid.uuid4())[:12]


_MUTATION_PROMPT = """\
你是一个提示词优化专家。请对以下系统提示词进行局部改写，使其在{aspect}方面更强。

要求：
1. 保持原始提示词的核心结构和JSON输出格式要求不变
2. 只修改措辞、增删指令、调整重点
3. 修改幅度适中，不要完全重写

原始提示词：
{template}

请直接返回修改后的完整提示词，不要包含其他说明。
"""

_CROSSOVER_PROMPT = """\
你是一个提示词优化专家。请将以下两个提示词的优点结合，生成一个新的提示词。

提示词A：
{parent_a}

提示词B：
{parent_b}

要求：
1. 保持JSON输出格式要求不变
2. 从A和B各取最有效的指令组合
3. 确保新提示词连贯一致

请直接返回合并后的完整提示词。
"""

_MUTATION_ASPECTS = [
    "逻辑推理深度",
    "证据引用要求",
    "自我批判能力",
    "建设性反馈",
    "结构化输出",
    "创新性思维",
    "规则合规性",
    "简洁表达",
]


class PromptEvolver:
    """Evolutionary prompt optimizer.

    Maintains per-role populations and evolves them through:
    - Tournament selection (select best from random subset)
    - LLM-driven mutation (rewrite with specific focus)
    - LLM-driven crossover (combine two parents)

    Parameters
    ----------
    config : PromptEvolutionConfig
        Evolution hyperparameters.
    skill_db : SkillDatabase | None
        For persisting prompt candidates.
    llm_client : BaseLLMClient | None
        For mutation/crossover operations. If None, evolution is disabled.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        config: Any = None,
        skill_db: Any = None,
        llm_client: Any = None,
        seed: int = 42,
    ) -> None:
        from debate_rl_v2.config.rl import PromptEvolutionConfig
        self._config = config or PromptEvolutionConfig()
        self._skill_db = skill_db
        self._client = llm_client
        self._rng = random.Random(seed)

        # role -> list[PromptCandidate]
        self._populations: dict[str, list[PromptCandidate]] = {}
        # prompt_id -> list[float] (fitness samples)
        self._fitness_history: dict[str, list[float]] = {}

    def initialize_population(self, role: str, base_prompt: str) -> None:
        """Initialize population for a role from a base prompt.

        Creates population_size candidates by mutating the base prompt.
        If no LLM client, creates copies with slight label variations.
        """
        pop_size = self._config.population_size
        population = [
            PromptCandidate(role=role, template=base_prompt, generation=0)
        ]

        if self._client is not None:
            # Generate mutations via LLM
            for i in range(pop_size - 1):
                aspect = self._rng.choice(_MUTATION_ASPECTS)
                mutated = self._mutate_with_llm(base_prompt, aspect)
                if mutated:
                    population.append(PromptCandidate(
                        role=role, template=mutated, generation=0,
                        parent_id=population[0].prompt_id,
                    ))
        else:
            # Without LLM, just duplicate the base (will differentiate via fitness)
            for _ in range(pop_size - 1):
                population.append(PromptCandidate(
                    role=role, template=base_prompt, generation=0,
                    parent_id=population[0].prompt_id,
                ))

        self._populations[role] = population[:pop_size]

        # Persist to DB
        if self._skill_db is not None:
            for c in self._populations[role]:
                self._skill_db.save_prompt(
                    c.prompt_id, c.role, c.template,
                    c.fitness, c.generation, c.parent_id,
                )

        logger.info("Initialized population for %s: %d candidates", role, len(self._populations[role]))

    def select(self, role: str) -> PromptCandidate:
        """Select the best candidate for a role (tournament selection).

        Returns the candidate with highest fitness from a random tournament.
        """
        pop = self._populations.get(role, [])
        if not pop:
            return PromptCandidate(role=role, template="")

        # Tournament selection
        k = min(self._config.tournament_size, len(pop))
        tournament = self._rng.sample(pop, k)
        return max(tournament, key=lambda c: c.fitness)

    def get_best(self, role: str) -> PromptCandidate:
        """Get the absolute best candidate for a role."""
        pop = self._populations.get(role, [])
        if not pop:
            return PromptCandidate(role=role, template="")
        return max(pop, key=lambda c: c.fitness)

    def record_fitness(self, prompt_id: str, quality: float) -> None:
        """Record a fitness observation for a prompt candidate."""
        self._fitness_history.setdefault(prompt_id, []).append(quality)

        # Update candidate fitness (running average)
        for pop in self._populations.values():
            for c in pop:
                if c.prompt_id == prompt_id:
                    samples = self._fitness_history[prompt_id]
                    c.fitness = sum(samples) / len(samples)
                    c.usage_count = len(samples)
                    # Persist
                    if self._skill_db is not None:
                        self._skill_db.update_prompt_fitness(prompt_id, c.fitness)
                    return

    def evolve(self) -> None:
        """Execute one generation of evolution across all roles.

        Steps:
        1. Keep elite candidates unchanged
        2. Tournament-select parents
        3. Apply mutation or crossover
        4. Replace worst candidates with offspring
        """
        for role, pop in self._populations.items():
            if len(pop) < 2:
                continue

            # Sort by fitness
            pop.sort(key=lambda c: -c.fitness)
            elite_count = min(self._config.elite_count, len(pop))
            new_pop = pop[:elite_count]  # Keep elites

            # Generate offspring to fill remaining slots
            target_size = self._config.population_size
            generation = max(c.generation for c in pop) + 1

            while len(new_pop) < target_size:
                if self._rng.random() < self._config.crossover_rate and len(pop) >= 2:
                    # Crossover
                    parent_a = self._tournament_select(pop)
                    parent_b = self._tournament_select(pop)
                    child_template = self._crossover_with_llm(
                        parent_a.template, parent_b.template
                    )
                    if child_template:
                        child = PromptCandidate(
                            role=role, template=child_template,
                            generation=generation,
                            parent_id=parent_a.prompt_id,
                        )
                        new_pop.append(child)
                        continue

                if self._rng.random() < self._config.mutation_rate:
                    # Mutation
                    parent = self._tournament_select(pop)
                    aspect = self._rng.choice(_MUTATION_ASPECTS)
                    mutated = self._mutate_with_llm(parent.template, aspect)
                    if mutated:
                        child = PromptCandidate(
                            role=role, template=mutated,
                            generation=generation,
                            parent_id=parent.prompt_id,
                        )
                        new_pop.append(child)
                        continue

                # Fallback: clone a tournament winner
                parent = self._tournament_select(pop)
                child = PromptCandidate(
                    role=role, template=parent.template,
                    generation=generation,
                    parent_id=parent.prompt_id,
                )
                new_pop.append(child)

            self._populations[role] = new_pop[:target_size]

            # Persist new candidates
            if self._skill_db is not None:
                for c in self._populations[role]:
                    self._skill_db.save_prompt(
                        c.prompt_id, c.role, c.template,
                        c.fitness, c.generation, c.parent_id,
                    )

            logger.info(
                "Evolved %s: gen=%d, best_fitness=%.3f",
                role, generation, new_pop[0].fitness if new_pop else 0.0,
            )

    def get_population(self, role: str) -> list[PromptCandidate]:
        return list(self._populations.get(role, []))

    def load_from_db(self, role: str) -> None:
        """Load population from SkillDatabase."""
        if self._skill_db is None:
            return
        rows = self._skill_db.get_prompts_by_role(role, limit=self._config.population_size)
        if not rows:
            return
        pop = []
        for r in rows:
            pop.append(PromptCandidate(
                prompt_id=r["prompt_id"],
                role=r["role"],
                template=r["template"],
                fitness=r["fitness"],
                usage_count=r["usage_count"],
                generation=r["generation"],
                parent_id=r.get("parent_id", ""),
            ))
        self._populations[role] = pop
        logger.info("Loaded %d candidates for %s from DB", len(pop), role)

    # ── Internal ──

    def _tournament_select(self, pop: list[PromptCandidate]) -> PromptCandidate:
        k = min(self._config.tournament_size, len(pop))
        tournament = self._rng.sample(pop, k)
        return max(tournament, key=lambda c: c.fitness)

    def _mutate_with_llm(self, template: str, aspect: str) -> str | None:
        """Use LLM to mutate a prompt template."""
        if self._client is None:
            return None
        prompt = _MUTATION_PROMPT.format(aspect=aspect, template=template[:3000])
        try:
            response = self._client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=2048,
            )
            result = response.content.strip()
            # Basic validation: must be non-trivial
            if len(result) > 50:
                return result
        except Exception as e:
            logger.warning("Mutation LLM call failed: %s", e)
        return None

    def _crossover_with_llm(self, parent_a: str, parent_b: str) -> str | None:
        """Use LLM to crossover two prompt templates."""
        if self._client is None:
            return None
        prompt = _CROSSOVER_PROMPT.format(
            parent_a=parent_a[:2000], parent_b=parent_b[:2000]
        )
        try:
            response = self._client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2048,
            )
            result = response.content.strip()
            if len(result) > 50:
                return result
        except Exception as e:
            logger.warning("Crossover LLM call failed: %s", e)
        return None
