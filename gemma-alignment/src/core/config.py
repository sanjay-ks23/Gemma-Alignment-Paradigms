"""
Configuration management for Gemma alignment experiments.

This module provides dataclasses for experiment configuration and utilities
for loading configurations from YAML files with validation and defaults.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import yaml


@dataclass
class ModelConfig:
    """Configuration for model loading and PEFT adapters."""
    
    base_checkpoint: str = "google/gemma-3-270m-it"
    size: Literal["270m", "1b"] = "270m"
    peft_type: Literal["lora", "qlora", "none"] = "lora"
    peft_rank: int = 8
    peft_alpha: int = 16
    peft_dropout: float = 0.1
    peft_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    load_in_4bit: bool = False
    load_in_8bit: bool = False


@dataclass
class TrainingConfig:
    """Configuration for training loop parameters."""
    
    mode: Literal["sft", "rl", "staged"] = "sft"
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    max_seq_length: int = 512
    fp16: bool = False
    bf16: bool = True


@dataclass
class RLConfig:
    """Configuration for reinforcement learning algorithms."""
    
    algorithm: Literal["ppo", "grpo", "dpo"] = "ppo"
    rollout_size: int = 64
    num_ppo_epochs: int = 4
    ppo_clip: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    kl_coef: float = 0.1
    gamma: float = 1.0
    lam: float = 0.95
    target_kl: Optional[float] = None
    dpo_beta: float = 0.1


@dataclass
class HeuristicRewardConfig:
    """Configuration for heuristic reward signals."""
    
    toxicity_threshold: float = 0.3
    toxicity_weight: float = 0.5
    length_penalty: float = 0.0
    repetition_penalty: float = 0.1


@dataclass
class TrainableRewardConfig:
    """Configuration for trainable reward model."""
    
    architecture: Literal["small-bert", "mlp", "transformer"] = "small-bert"
    hidden_size: int = 768
    num_layers: int = 3
    learning_rate: float = 2e-5
    checkpoint_path: Optional[str] = None


@dataclass
class RewardConfig:
    """Configuration for reward computation strategy."""
    
    type: Literal["heuristic", "trainable", "hybrid"] = "hybrid"
    heuristic: HeuristicRewardConfig = field(default_factory=HeuristicRewardConfig)
    trainable: TrainableRewardConfig = field(default_factory=TrainableRewardConfig)
    alpha: float = 0.5  # Weight for heuristic component in hybrid
    beta: float = 0.5   # Weight for trainable component in hybrid
    normalize_rewards: bool = True
    clip_rewards: float = 10.0


@dataclass
class LoggingConfig:
    """Configuration for logging and experiment tracking."""
    
    wandb_enabled: bool = False
    wandb_project: str = "gemma-alignment"
    wandb_entity: Optional[str] = None
    log_interval: int = 50
    eval_interval: int = 500
    save_interval: int = 1000
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"


@dataclass
class ExperimentConfig:
    """
    Main configuration dataclass for Gemma alignment experiments.
    
    This class aggregates all sub-configurations and provides the complete
    experiment specification loaded from YAML config files.
    
    Attributes:
        task: The alignment task to run (safety, clinical, conala).
        dataset_path: Path to the dataset directory.
        model: Model loading and PEFT configuration.
        training: Training loop parameters.
        rl: Reinforcement learning algorithm settings.
        reward: Reward model and computation settings.
        logging: Logging and checkpointing settings.
        seed: Random seed for reproducibility.
        device: Device to run on (cuda, cpu, auto).
        debug: Whether to run in debug mode (single batch).
    """
    
    task: Literal["safety", "clinical", "conala"] = "safety"
    dataset_path: str = "data/datasets"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42
    device: str = "auto"
    debug: bool = False


def _deep_update(base: dict, updates: dict) -> dict:
    """Recursively update a nested dictionary."""
    result = base.copy()
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _dict_to_dataclass(data: dict, cls: type) -> Any:
    """Convert a dictionary to a dataclass, handling nested dataclasses."""
    if not hasattr(cls, "__dataclass_fields__"):
        return data
    
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    
    for field_name, field_type in field_types.items():
        if field_name in data:
            value = data[field_name]
            # Handle nested dataclasses
            if hasattr(field_type, "__dataclass_fields__"):
                kwargs[field_name] = _dict_to_dataclass(value, field_type)
            else:
                kwargs[field_name] = value
    
    return cls(**kwargs)


def load_config(path: str, defaults_path: Optional[str] = None) -> ExperimentConfig:
    """
    Load experiment configuration from a YAML file.
    
    This function loads a YAML configuration file and converts it to an
    ExperimentConfig dataclass. If a defaults file is provided, the main
    config will override the defaults.
    
    Args:
        path: Path to the YAML configuration file.
        defaults_path: Optional path to a defaults YAML file.
    
    Returns:
        ExperimentConfig: Fully populated configuration dataclass.
    
    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML is malformed.
        ValueError: If required fields are missing or invalid.
    
    Example:
        >>> cfg = load_config("configs/sft_lora_270m.yml")
        >>> print(cfg.model.base_checkpoint)
        google/gemma-3-270m-it
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    # Load defaults if provided
    base_config: dict = {}
    if defaults_path:
        defaults_file = Path(defaults_path)
        if defaults_file.exists():
            with open(defaults_file) as f:
                base_config = yaml.safe_load(f) or {}
    
    # Load main config
    with open(config_path) as f:
        main_config = yaml.safe_load(f) or {}
    
    # Merge configs (main overrides defaults)
    merged = _deep_update(base_config, main_config)
    
    # Build nested dataclass structure
    config_dict = {}
    
    # Simple fields
    for field_name in ["task", "dataset_path", "seed", "device", "debug"]:
        if field_name in merged:
            config_dict[field_name] = merged[field_name]
    
    # Nested config objects
    if "model" in merged:
        config_dict["model"] = _dict_to_dataclass(merged["model"], ModelConfig)
    if "training" in merged:
        config_dict["training"] = _dict_to_dataclass(merged["training"], TrainingConfig)
    if "rl" in merged:
        config_dict["rl"] = _dict_to_dataclass(merged["rl"], RLConfig)
    if "reward" in merged:
        reward_dict = merged["reward"].copy()
        if "heuristic" in reward_dict:
            reward_dict["heuristic"] = _dict_to_dataclass(
                reward_dict["heuristic"], HeuristicRewardConfig
            )
        if "trainable" in reward_dict:
            reward_dict["trainable"] = _dict_to_dataclass(
                reward_dict["trainable"], TrainableRewardConfig
            )
        config_dict["reward"] = RewardConfig(**reward_dict)
    if "logging" in merged:
        config_dict["logging"] = _dict_to_dataclass(merged["logging"], LoggingConfig)
    
    return ExperimentConfig(**config_dict)


def save_config(config: ExperimentConfig, path: str) -> None:
    """
    Save an ExperimentConfig to a YAML file.
    
    Args:
        config: The configuration to save.
        path: Path to write the YAML file.
    """
    from dataclasses import asdict
    
    config_dict = asdict(config)
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path_obj, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
