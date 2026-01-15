"""Experiment runner CLI."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from src.core.config import ExperimentConfig, load_config, save_config
from src.core.utils import ensure_dir, get_logger, set_seed, setup_logging
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


DATASET_CLASSES = {
    "safety": SafetyDataset,
    "clinical": ClinicalDataset,
    "conala": CoNaLaDataset,
}


def run_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Run experiment based on configuration."""
    logger = get_logger("gemma-alignment")
    set_seed(cfg.seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.task}_{cfg.training.mode}_{timestamp}"
    output_dir = ensure_dir(os.path.join(cfg.logging.output_dir, run_name))
    checkpoint_dir = ensure_dir(os.path.join(cfg.logging.checkpoint_dir, run_name))
    
    cfg.logging.output_dir = str(output_dir)
    cfg.logging.checkpoint_dir = str(checkpoint_dir)
    save_config(cfg, os.path.join(output_dir, "config.yml"))
    
    tokenizer = TokenizerWrapper.from_pretrained(cfg.model.base_checkpoint)
    
    dataset_cls = DATASET_CLASSES[cfg.task]
    train_dataset = dataset_cls(
        split="train",
        cfg=cfg,
        tokenizer=tokenizer,
        include_rejected=(cfg.training.mode == "rl" and cfg.rl.algorithm == "dpo"),
    )
    val_dataset = dataset_cls(split="val", cfg=cfg, tokenizer=tokenizer)
    
    logger.info(f"Training: {len(train_dataset)} samples, Validation: {len(val_dataset)} samples")
    
    if cfg.training.mode == "sft":
        results = _run_sft(cfg, tokenizer, train_dataset, val_dataset)
    elif cfg.training.mode == "rl":
        results = _run_rl(cfg, tokenizer, train_dataset, val_dataset)
    elif cfg.training.mode == "staged":
        results = _run_staged(cfg, tokenizer, train_dataset, val_dataset)
    else:
        raise ValueError(f"Unknown training mode: {cfg.training.mode}")
    
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(_serialize(results), f, indent=2)
    
    return results


def _run_sft(cfg, tokenizer, train_dataset, val_dataset):
    model = GemmaWrapper(
        checkpoint=cfg.model.base_checkpoint,
        device=cfg.device,
        load_in_4bit=cfg.model.peft_type == "qlora",
        load_in_8bit=cfg.model.load_in_8bit,
    )
    
    peft_adapter = None
    if cfg.model.peft_type != "none":
        peft_adapter = create_adapter(
            adapter_type=cfg.model.peft_type,
            rank=cfg.model.peft_rank,
            alpha=cfg.model.peft_alpha,
            dropout=cfg.model.peft_dropout,
            target_modules=cfg.model.peft_target_modules,
        )
    
    trainer = SFTTrainer(
        model=model,
        cfg=cfg,
        train_dataloader=train_dataset.get_dataloader(cfg.training.batch_size, shuffle=True),
        val_dataloader=val_dataset.get_dataloader(cfg.training.batch_size, shuffle=False),
        peft_adapter=peft_adapter,
    )
    
    metrics = trainer.train()
    
    if peft_adapter is not None:
        trainer.export_lora_checkpoint(os.path.join(cfg.logging.checkpoint_dir, "final_lora"))
    
    return {"training": metrics, "mode": "sft"}


def _run_rl(cfg, tokenizer, train_dataset, val_dataset):
    model = GemmaWrapper(checkpoint=cfg.model.base_checkpoint, device=cfg.device)
    
    reward_model = RewardModel(
        hidden_size=cfg.reward.trainable.hidden_size,
        num_layers=cfg.reward.trainable.num_layers,
        reward_config=cfg.reward,
    )
    
    ref_model = GemmaWrapper(checkpoint=cfg.model.base_checkpoint, device=cfg.device)
    ref_model.freeze()
    
    trainer = RLTrainer(
        policy_model=model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        cfg=cfg,
        train_dataloader=train_dataset.get_dataloader(cfg.training.batch_size, shuffle=True),
        ref_model=ref_model,
    )
    
    return {"training": trainer.train(), "mode": "rl"}


def _run_staged(cfg, tokenizer, train_dataset, val_dataset):
    trainer = StagedTrainer(
        cfg=cfg,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    return {"training": trainer.run(), "mode": "staged"}


def _serialize(obj):
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize(v) for v in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Gemma Alignment Experiment Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    config_dir = Path(args.config).parent
    defaults_path = config_dir / "defaults.yml"
    
    cfg = load_config(
        args.config,
        defaults_path=str(defaults_path) if defaults_path.exists() else None,
    )
    
    if args.debug:
        cfg.debug = True
    if args.device:
        cfg.device = args.device
    if args.seed:
        cfg.seed = args.seed
    
    logger.info(f"Task: {cfg.task}, Mode: {cfg.training.mode}, Model: {cfg.model.base_checkpoint}")
    
    try:
        run_experiment(cfg)
        return 0
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        if cfg.debug:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
