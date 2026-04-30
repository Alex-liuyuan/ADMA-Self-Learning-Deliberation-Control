"""LLM and text debate configuration with validation."""

from __future__ import annotations

from dataclasses import dataclass

from debate_rl_v2.exceptions import ConfigError


@dataclass
class LLMConfig:
    """LLM agent configuration."""
    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: float = 60.0
    max_retries: int = 3
    max_history: int = 20
    # Per-role provider overrides
    proposer_provider: str = ""
    proposer_model: str = ""
    challenger_provider: str = ""
    challenger_model: str = ""
    arbiter_provider: str = ""
    arbiter_model: str = ""
    coordinator_provider: str = ""
    coordinator_model: str = ""
    # v2: Model routing
    cheap_model: str = ""
    cheap_provider: str = ""
    routing_strategy: str = "fixed"  # "fixed" | "smart"
    # v2: Prompt caching
    enable_prompt_cache: bool = True
    cache_prefix_tokens: int = 0  # 0 = auto

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ConfigError("llm.temperature", "must be non-negative")
        if self.max_tokens <= 0:
            raise ConfigError("llm.max_tokens", "must be positive")
        if self.timeout <= 0:
            raise ConfigError("llm.timeout", "must be positive")
        if self.max_retries < 0:
            raise ConfigError("llm.max_retries", "must be non-negative")
        if self.routing_strategy not in ("fixed", "smart"):
            raise ConfigError("llm.routing_strategy", f"unsupported: '{self.routing_strategy}'")


@dataclass
class TextDebateConfig:
    """Text-based debate environment configuration."""
    topic: str = "Should we adopt policy A or policy B?"
    context: str = ""
    rules: str = ""
    max_rounds: int = 10
    meta_interval: int = 3
    consensus_threshold: float = 0.8
    export_transcript: str = ""
    verbose: bool = True
    # v2 framework features
    enable_tools: bool = True
    enable_memory: bool = True
    enable_tracing: bool = False
    trace_export_path: str = ""
    enable_hooks: bool = True
    quality_gate: float = 0.0
    token_budget: int = 0
    enable_human_review: bool = False
    # v2: Context compression
    enable_context_compression: bool = True
    compression_threshold_tokens: int = 8000
    compression_keep_recent: int = 3

    def __post_init__(self) -> None:
        if self.max_rounds <= 0:
            raise ConfigError("text_debate.max_rounds", "must be positive")
        if not 0 < self.consensus_threshold <= 1:
            raise ConfigError("text_debate.consensus_threshold", "must be in (0, 1]")
        if self.meta_interval <= 0:
            raise ConfigError("text_debate.meta_interval", "must be positive")
