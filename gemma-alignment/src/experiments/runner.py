"""
Experiment runner CLI for Gemma alignment experiments.

This module provides the main entry point for running experiments
from configuration files.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.config import ExperimentConfig, load_config, save_config
from src.core.registry import get_component
from src.core.utils import get_device, get_logger, set_seed, ensure_dir
from src.data.safety_dataset import SafetyDataset
from src.data.clinical_dataset import ClinicalDataset
from src.data.conala_dataset import CoNaLaDataset
from src.models.gemma_wrapper import GemmaWrapper
from src.models.peft_adapters import create_adapter
from src.models.reward_model import RewardModel
from src.tokenization.tokenizer_wrapper import TokenizerWrapper
from src.trainers.sft_trainer import SFTTrainer
from src.trainers.rl_trainer import RLTrainer
from src.trainers.staged_trainer import StagedTrainer
from src.evaluation.evaluator import Evaluator


def run_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    """
    Run a complete experiment based on configuration.
    
    Args:
        cfg: Experiment configuration.
    
    Returns:
        Dictionary containing experiment results.
    """
    logger = get_logger("gemma-alignment")
    
    # Set seed
    set_seed(cfg.seed)
    logger.info(f"Seed set to {cfg.seed}")
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.task}_{cfg.training.mode}_{timestamp}"
    output_dir = ensure_dir(os.path.join(cfg.logging.output_dir, run_name))
    checkpoint_dir = ensure_dir(os.path.join(cfg.logging.checkpoint_dir, run_name))
    
    # Update config with run-specific paths
    cfg.logging.output_dir = str(output_dir)
    cfg.logging.checkpoint_dir = str(checkpoint_dir)
    
    # Save config
    save_config(cfg, os.path.join(output_dir, "config.yml"))
    
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {cfg.model.base_checkpoint}")
    tokenizer = TokenizerWrapper.from_pretrained(cfg.model.base_checkpoint)
    
    # Load dataset
    logger.info(f"Loading {cfg.task} dataset")
    dataset_cls = _get_dataset_class(cfg.task)
    train_dataset = dataset_cls(
        split="train",
        cfg=cfg,
        tokenizer=tokenizer,
        include_rejected=(cfg.training.mode == "rl" and cfg.rl.algorithm == "dpo"),
    )
    val_dataset = dataset_cls(
        split="val",
        cfg=cfg,
        tokenizer=tokenizer,
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Run based on training mode
    if cfg.training.mode == "sft":
        results = _run_sft(cfg, tokenizer, train_dataset, val_dataset)
    elif cfg.training.mode == "rl":
        results = _run_rl(cfg, tokenizer, train_dataset, val_dataset)
    elif cfg.training.mode == "staged":
        results = _run_staged(cfg, tokenizer, train_dataset, val_dataset)
    else:
        raise ValueError(f"Unknown training mode: {cfg.training.mode}")
    
    # Save results
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(_serialize_results(results), f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    
    return results


def _run_sft(
    cfg: ExperimentConfig,
    tokenizer: TokenizerWrapper,
    train_dataset,
    val_dataset,
) -> Dict[str, Any]:
    """Run SFT training."""
    logger = get_logger("gemma-alignment")
    
    # Load model
    logger.info(f"Loading model: {cfg.model.base_checkpoint}")
    model = GemmaWrapper(
        checkpoint=cfg.model.base_checkpoint,
        device=cfg.device,
        load_in_4bit=cfg.model.peft_type == "qlora",
        load_in_8bit=cfg.model.load_in_8bit,
    )
    
    # Create PEFT adapter
    peft_adapter = None
    if cfg.model.peft_type != "none":
        peft_adapter = create_adapter(
            adapter_type=cfg.model.peft_type,
            rank=cfg.model.peft_rank,
            alpha=cfg.model.peft_alpha,
            dropout=cfg.model.peft_dropout,
            target_modules=cfg.model.peft_target_modules,
        )
    
    # Create data loaders
    train_loader = train_dataset.get_dataloader(
        batch_size=cfg.training.batch_size,
        shuffle=True,
    )
    val_loader = val_dataset.get_dataloader(
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        cfg=cfg,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        peft_adapter=peft_adapter,
    )
    
    # Train
    metrics = trainer.train()
    
    # Export LoRA if applicable
    if peft_adapter is not None:
        trainer.export_lora_checkpoint(
            os.path.join(cfg.logging.checkpoint_dir, "final_lora")
        )
    
    return {"training": metrics, "mode": "sft"}


def _run_rl(
    cfg: ExperimentConfig,
    tokenizer: TokenizerWrapper,
    train_dataset,
    val_dataset,
) -> Dict[str, Any]:
    """Run RL training."""
    logger = get_logger("gemma-alignment")
    
    # Load model
    logger.info(f"Loading model: {cfg.model.base_checkpoint}")
    model = GemmaWrapper(
        checkpoint=cfg.model.base_checkpoint,
        device=cfg.device,
    )
    
    # Create reward model
    logger.info("Creating reward model")
    reward_model = RewardModel(
        hidden_size=cfg.reward.trainable.hidden_size,
        num_layers=cfg.reward.trainable.num_layers,
        reward_config=cfg.reward,
    )
    
    # Train reward model on preference pairs if available
    if hasattr(train_dataset, "get_preference_pairs"):
        pairs = train_dataset.get_preference_pairs()
        if pairs:
            logger.info(f"Training reward model on {len(pairs)} preference pairs")
            prompts, chosen, rejected = zip(*pairs[:1000])
            reward_model.train_on_pairs(
                list(chosen), list(rejected), tokenizer, epochs=2
            )
    
    # Create reference model
    ref_model = GemmaWrapper(
        checkpoint=cfg.model.base_checkpoint,
        device=cfg.device,
    )
    ref_model.freeze()
    
    # Create data loader
    train_loader = train_dataset.get_dataloader(
        batch_size=cfg.training.batch_size,
        shuffle=True,
    )
    
    # Create trainer
    trainer = RLTrainer(
        policy_model=model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        cfg=cfg,
        train_dataloader=train_loader,
        ref_model=ref_model,
    )
    
    # Train
    metrics = trainer.train()
    
    return {"training": metrics, "mode": "rl"}


def _run_staged(
    cfg: ExperimentConfig,
    tokenizer: TokenizerWrapper,
    train_dataset,
    val_dataset,
) -> Dict[str, Any]:
    """Run staged SFT -> RL training."""
    # Create staged trainer
    trainer = StagedTrainer(
        cfg=cfg,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    
    # Run
    results = trainer.run()
    
    return {"training": results, "mode": "staged"}


def _get_dataset_class(task: str):
    """Get dataset class for task."""
    if task == "safety":
        return SafetyDataset
    elif task == "clinical":
        return ClinicalDataset
    elif task == "conala":
        return CoNaLaDataset
    else:
        raise ValueError(f"Unknown task: {task}")


def _serialize_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize results for JSON saving."""
    serialized = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serialized[key] = _serialize_results(value)
        elif isinstance(value, list):
            serialized[key] = [
                v if isinstance(v, (int, float, str, bool, type(None)))
                else str(v) for v in value
            ]
        elif isinstance(value, (int, float, str, bool, type(None))):
            serialized[key] = value
        else:
            serialized[key] = str(value)
    return serialized


def main():
    """Main entry point for experiment runner."""
    parser = argparse.ArgumentParser(
        description="Gemma Alignment Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run SFT with LoRA on 270M model
    python -m src.experiments.runner --config configs/sft_lora_270m.yml
    
    # Run in debug mode (single batch)
    python -m src.experiments.runner --config configs/sft_lora_270m.yml --debug
    
    # Run PPO on 270M model
    python -m src.experiments.runner --config configs/ppo_rl_270m.yml
        """,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (single batch)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda, cpu, auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    from src.core.utils import setup_logging
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Gemma Alignment Experiment Runner")
    logger.info("=" * 60)
    
    # Load config
    logger.info(f"Loading config: {args.config}")
    
    # Find defaults file if exists
    config_dir = Path(args.config).parent
    defaults_path = config_dir / "defaults.yml"
    
    cfg = load_config(
        args.config,
        defaults_path=str(defaults_path) if defaults_path.exists() else None,
    )
    
    # Apply CLI overrides
    if args.debug:
        cfg.debug = True
        logger.info("Debug mode enabled")
    
    if args.device:
        cfg.device = args.device
    
    if args.seed:
        cfg.seed = args.seed
    
    if args.output_dir:
        cfg.logging.output_dir = args.output_dir
    
    # Log config
    logger.info(f"Task: {cfg.task}")
    logger.info(f"Model: {cfg.model.base_checkpoint}")
    logger.info(f"Training mode: {cfg.training.mode}")
    logger.info(f"PEFT: {cfg.model.peft_type}")
    logger.info(f"Device: {cfg.device}")
    
    # Run experiment
    try:
        results = run_experiment(cfg)
        logger.info("Experiment completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        if cfg.debug:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
