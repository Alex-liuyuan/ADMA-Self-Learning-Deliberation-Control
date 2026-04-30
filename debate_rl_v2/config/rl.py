"""RL algorithm configurations with validation."""

from __future__ import annotations

from dataclasses import dataclass

from debate_rl_v2.exceptions import ConfigError


@dataclass
class NetworkConfig:
    hidden_dim: int = 640
    num_layers: int = 3
    activation: str = "tanh"
    shared_feature_dim: int = 256
    meta_embed_dim: int = 64
    use_layer_norm: bool = True
    critic_hidden_dim: int = 1024

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ConfigError("network.hidden_dim", "must be positive")
        if self.num_layers < 1:
            raise ConfigError("network.num_layers", "must be >= 1")
        if self.activation not in ("tanh", "relu", "gelu", "silu"):
            raise ConfigError("network.activation", f"unsupported: '{self.activation}'")


@dataclass
class ContinuousAgentConfig:
    """Arbiter/Coordinator continuous action space config."""
    arbiter_act_dim: int = 33
    coordinator_act_dim: int = 8
    actor_hidden_dim: int = 640
    critic_hidden_dim: int = 1024
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    tau: float = 0.005
    noise_type: str = "gaussian"
    noise_std: float = 0.1
    noise_decay: float = 0.9999
    buffer_size: int = 100_000
    batch_size: int = 256
    warmup_steps: int = 1000

    def __post_init__(self) -> None:
        if self.actor_lr <= 0 or self.critic_lr <= 0:
            raise ConfigError("continuous_agent.lr", "learning rates must be positive")
        if not 0 < self.tau < 1:
            raise ConfigError("continuous_agent.tau", "must be in (0, 1)")


@dataclass
class PPOConfig:
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    pi_lr: float = 3e-4
    vf_lr: float = 1e-3
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    train_epochs: int = 4
    minibatch_size: int = 256

    def __post_init__(self) -> None:
        if not 0 < self.gamma <= 1:
            raise ConfigError("ppo.gamma", "must be in (0, 1]")
        if not 0 < self.clip_ratio < 1:
            raise ConfigError("ppo.clip_ratio", "must be in (0, 1)")


@dataclass
class HierarchicalConfig:
    meta_lr: float = 1e-4
    meta_gamma: float = 0.99
    meta_train_epochs: int = 2
    meta_update_interval: int = 5


@dataclass
class CreditConfig:
    use_shapley: bool = True
    shapley_samples: int = 100
    correction_coef: float = 0.1


@dataclass
class RewardConfig:
    task_weight: float = 1.0
    divergence_weight: float = 0.2
    compliance_weight: float = 0.2
    info_gain_weight: float = 0.2
    step_penalty: float = 0.01
    meta_convergence_penalty: float = 0.1
    meta_violation_penalty: float = 0.2

    # Enhanced multi-tier reward (v2)
    use_enhanced: bool = True
    consensus_bonus: float = 3.0
    no_consensus_penalty: float = -1.0
    constructive_adv_weight: float = 0.20

    # Layer 3: Meta (coordinator)
    meta_avg_quality_weight: float = 0.30
    meta_convergence_speed_weight: float = 0.25
    meta_rule_adaptability_weight: float = 0.20
    meta_efficiency_weight: float = 0.15

    # Proposer
    proposer_quality_improve_weight: float = 0.30
    proposer_modification_effectiveness: float = 0.20
    proposer_acceptance_signal_weight: float = 0.15
    proposer_compliance_alignment: float = 0.10

    # Challenger
    challenger_info_gain_weight: float = 0.25
    challenger_belief_change_weight: float = 0.20
    challenger_constructiveness_bonus: float = 0.15
    challenger_novelty_weight: float = 0.10
    challenger_destructive_penalty: float = -0.20

    # Arbiter
    arbiter_judgment_accuracy_weight: float = 0.25
    arbiter_rule_calibration_weight: float = 0.15
    arbiter_consistency_weight: float = 0.10
    arbiter_threshold_opt_weight: float = 0.10

    # Coordinator
    coordinator_convergence_weight: float = 0.25
    coordinator_quality_maintenance: float = 0.20
    coordinator_mining_benefit: float = 0.15
    coordinator_termination_timing: float = 0.10


@dataclass
class MADDPGConfig:
    """MADDPG algorithm configuration."""
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 256
    num_layers: int = 2
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma: float = 0.95
    tau: float = 0.01
    buffer_size: int = 100_000
    batch_size: int = 256
    noise_type: str = "ou"
    noise_std: float = 0.2
    noise_decay: float = 0.9999
    noise_min: float = 0.01
    total_episodes: int = 5000
    warmup_episodes: int = 100
    update_interval: int = 1
    gradient_clip: float = 0.5
    eval_interval: int = 100
    eval_episodes: int = 20
    save_interval: int = 500
    log_interval: int = 10

    def __post_init__(self) -> None:
        if self.actor_lr <= 0 or self.critic_lr <= 0:
            raise ConfigError("maddpg.lr", "learning rates must be positive")
        if not 0 < self.gamma <= 1:
            raise ConfigError("maddpg.gamma", "must be in (0, 1]")
        if self.warmup_episodes < 0:
            raise ConfigError("maddpg.warmup_episodes", "must be non-negative")


@dataclass
class StrategyBridgeConfig:
    """RL → LLM Strategy Bridge configuration."""
    temp_min: float = 0.3
    temp_max: float = 1.2
    arbiter_temp_min: float = 0.2
    arbiter_temp_max: float = 0.8
    max_eta_delta: float = 0.05
    max_alpha_delta: float = 0.1
    max_tau_delta: float = 0.1

    def __post_init__(self) -> None:
        if self.temp_min >= self.temp_max:
            raise ConfigError("strategy_bridge.temp", "temp_min must be < temp_max")
        if self.arbiter_temp_min >= self.arbiter_temp_max:
            raise ConfigError("strategy_bridge.arbiter_temp", "min must be < max")


@dataclass
class SelfPlayTrainingConfig:
    enabled: bool = True
    snapshot_interval: int = 500
    max_pool_size: int = 50
    selection_strategy: str = "elo_match"
    self_play_ratio: float = 0.5
    warmup_episodes: int = 500
    save_dir: str = "self_play_pool"


@dataclass
class CurriculumTrainingConfig:
    enabled: bool = True
    initial_level: int = 1
    max_level: int = 5
    promotion_threshold: float = 0.80
    promotion_window: int = 20
    min_episodes_per_level: int = 100
    demotion_threshold: float = 0.50
    exploration_rate: float = 0.1


@dataclass
class EnhancedRewardTrainingConfig:
    dense_quality_enabled: bool = True
    dense_quality_dimensions: int = 5
    marginal_return_window: int = 3
    marginal_return_threshold: float = 0.02
    reward_normalization: bool = True
    norm_window: int = 100
    norm_clip: float = 5.0
    curiosity_enabled: bool = True
    curiosity_weight: float = 0.05
    curiosity_decay: float = 0.995
    opponent_modeling_enabled: bool = True
    opponent_modeling_weight: float = 0.08


@dataclass
class ModeConfig:
    """Dual-mode configuration — training vs online."""
    mode: str = "training"  # "training" | "online"
    exploration_noise: float = 0.1
    distill_quality_threshold: float = 0.6
    enable_prompt_evolution: bool = True
    prompt_evolve_interval: int = 5
    enable_skill_extraction: bool = True
    enable_causal_extraction: bool = True

    def __post_init__(self) -> None:
        if self.mode not in ("training", "online"):
            raise ConfigError("mode.mode", f"must be 'training' or 'online', got '{self.mode}'")


@dataclass
class OnlineUpdateConfig:
    """Online learning parameter update configuration."""
    ema_alpha: float = 0.01
    obs_noise: float = 0.1
    confidence_threshold: float = 0.02
    state_path: str = "online_state.json"

    def __post_init__(self) -> None:
        if not 0 < self.ema_alpha < 1:
            raise ConfigError("online_update.ema_alpha", "must be in (0, 1)")


@dataclass
class CausalConfig:
    """Causal reasoning configuration."""
    dataset_path: str = ""
    max_graph_depth: int = 3
    max_chains_per_query: int = 3
    extraction_confidence_threshold: float = 0.5
    graph_save_path: str = "causal_graph.json"


@dataclass
class PromptEvolutionConfig:
    """Prompt evolution algorithm configuration."""
    population_size: int = 8
    tournament_size: int = 3
    mutation_rate: float = 0.3
    crossover_rate: float = 0.2
    elite_count: int = 2
    min_fitness_samples: int = 3


@dataclass
class TrainingConfig:
    total_episodes: int = 1000
    rollout_episodes: int = 4
    log_interval: int = 10
    save_interval: int = 100
    eval_interval: int = 50
    eval_episodes: int = 20
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"
    # Live visualization
    live_viz: bool = False
    live_viz_interval: int = 5
    live_viz_smoothing: int = 10
    live_viz_save_dir: str = ""
    live_viz_dark_mode: bool = False
    # v2: LR scheduling & early stopping
    lr_schedule: str = "cosine"  # "cosine" | "constant" | "linear"
    early_stop_patience: int = 50
    q_divergence_threshold: float = 100.0

    def __post_init__(self) -> None:
        if self.total_episodes <= 0:
            raise ConfigError("training.total_episodes", "must be positive")
        if self.lr_schedule not in ("cosine", "constant", "linear"):
            raise ConfigError("training.lr_schedule", f"unsupported: '{self.lr_schedule}'")
