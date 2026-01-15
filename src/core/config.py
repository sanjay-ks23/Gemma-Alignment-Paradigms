"""Configuration management for Gemma alignment experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import yaml


@dataclass
class ModelConfig:
    """Model loading and PEFT adapter configuration."""
    base_checkpoint: str = "google/gemma-3-270m-it"
    peft_type: Literal["lora", "none"] = "lora"
    peft_rank: int = 8
    peft_alpha: int = 16
    peft_dropout: float = 0.1
    peft_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )


@dataclass
class TrainingConfig:
    """Training loop parameters."""
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
    """Reinforcement learning algorithm parameters."""
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
    """Heuristic reward signal parameters."""
    toxicity_threshold: float = 0.3
    toxicity_weight: float = 0.5
    length_penalty: float = 0.0
    repetition_penalty: float = 0.1


@dataclass
class TrainableRewardConfig:
    """Trainable reward model parameters."""
    hidden_size: int = 768
    num_layers: int = 3
    learning_rate: float = 2e-5
    checkpoint_path: Optional[str] = None


@dataclass
class RewardConfig:
    """Reward computation strategy configuration."""
    type: Literal["heuristic", "trainable", "hybrid"] = "hybrid"
    heuristic: HeuristicRewardConfig = field(default_factory=HeuristicRewardConfig)
    trainable: TrainableRewardConfig = field(default_factory=TrainableRewardConfig)
    alpha: float = 0.5
    beta: float = 0.5
    normalize_rewards: bool = True
    clip_rewards: float = 10.0


@dataclass
class LoggingConfig:
    """Logging and experiment tracking configuration."""
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
    """Main experiment configuration."""
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
    result = base.copy()
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _dict_to_dataclass(data: dict, cls: type) -> Any:
    if not hasattr(cls, "__dataclass_fields__"):
        return data
    
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    
    for field_name, field_type in field_types.items():
        if field_name in data:
            value = data[field_name]
            if hasattr(field_type, "__dataclass_fields__"):
                kwargs[field_name] = _dict_to_dataclass(value, field_type)
            else:
                kwargs[field_name] = value
    
    return cls(**kwargs)


def load_config(path: str, defaults_path: Optional[str] = None) -> ExperimentConfig:
    """Load experiment configuration from YAML file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    base_config: dict = {}
    if defaults_path:
        defaults_file = Path(defaults_path)
        if defaults_file.exists():
            with open(defaults_file) as f:
                base_config = yaml.safe_load(f) or {}
    
    with open(config_path) as f:
        main_config = yaml.safe_load(f) or {}
    
    merged = _deep_update(base_config, main_config)
    config_dict = {}
    
    for field_name in ["task", "dataset_path", "seed", "device", "debug"]:
        if field_name in merged:
            config_dict[field_name] = merged[field_name]
    
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
    """Save configuration to YAML file."""
    from dataclasses import asdict
    
    config_dict = asdict(config)
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path_obj, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
