"""
Staged trainer orchestrating SFT followed by RL refinement.

This module provides the StagedTrainer class that runs a two-phase
training pipeline: first SFT with PEFT, then RL-based refinement.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from src.core.config import ExperimentConfig
from src.core.utils import get_logger, set_seed
from src.data.base_dataset import BaseDataset
from src.models.base_model import ModelWrapper
from src.models.gemma_wrapper import GemmaWrapper
from src.models.peft_adapters import create_adapter
from src.models.reward_model import RewardModel
from src.tokenization.tokenizer_wrapper import TokenizerWrapper
from src.trainers.sft_trainer import SFTTrainer
from src.trainers.rl_trainer import RLTrainer


class StagedTrainer:
    """
    Orchestrates a staged training pipeline: SFT -> RL.
    
    This trainer runs supervised fine-tuning first to establish a good
    initial policy, then refines it with reinforcement learning.
    
    The pipeline:
    1. Load base model and apply PEFT adapter.
    2. Run SFT training on the dataset.
    3. Export SFT checkpoint.
    4. Initialize RL trainer with SFT model.
    5. Run RL refinement.
    6. Save final model.
    
    Attributes:
        cfg: Experiment configuration.
        tokenizer: Tokenizer wrapper.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
    
    Example:
        >>> trainer = StagedTrainer(cfg)
        >>> results = trainer.run()
        >>> print(results["sft_metrics"], results["rl_metrics"])
    """
    
    def __init__(
        self,
        cfg: ExperimentConfig,
        tokenizer: Optional[TokenizerWrapper] = None,
        train_dataset: Optional[BaseDataset] = None,
        val_dataset: Optional[BaseDataset] = None,
        reward_model: Optional[RewardModel] = None,
    ):
        """
        Initialize the staged trainer.
        
        Args:
            cfg: Experiment configuration.
            tokenizer: Optional tokenizer (loaded from config if None).
            train_dataset: Optional training dataset.
            val_dataset: Optional validation dataset.
            reward_model: Optional reward model for RL phase.
        """
        self.cfg = cfg
        self.logger = get_logger("gemma-alignment")
        
        # Set seed for reproducibility
        set_seed(cfg.seed)
        
        # Load tokenizer if not provided
        if tokenizer is None:
            tokenizer = TokenizerWrapper.from_pretrained(cfg.model.base_checkpoint)
        self.tokenizer = tokenizer
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.reward_model = reward_model
        
        # Training state
        self.sft_checkpoint_path: Optional[str] = None
        self.rl_checkpoint_path: Optional[str] = None
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete staged training pipeline.
        
        Returns:
            Dictionary containing:
            - sft_metrics: Metrics from SFT phase.
            - rl_metrics: Metrics from RL phase.
            - sft_checkpoint: Path to SFT checkpoint.
            - final_checkpoint: Path to final model.
        """
        self.logger.info("=" * 50)
        self.logger.info("Starting Staged Training Pipeline")
        self.logger.info("=" * 50)
        
        # Phase 1: SFT
        self.logger.info("\n[Phase 1] Supervised Fine-Tuning")
        sft_metrics = self._run_sft_phase()
        
        # Phase 2: RL
        self.logger.info("\n[Phase 2] Reinforcement Learning Refinement")
        rl_metrics = self._run_rl_phase()
        
        self.logger.info("\n" + "=" * 50)
        self.logger.info("Staged Training Complete")
        self.logger.info("=" * 50)
        
        return {
            "sft_metrics": sft_metrics,
            "rl_metrics": rl_metrics,
            "sft_checkpoint": self.sft_checkpoint_path,
            "final_checkpoint": self.rl_checkpoint_path,
        }
    
    def _run_sft_phase(self) -> Dict[str, Any]:
        """
        Run the SFT training phase.
        
        Returns:
            SFT training metrics.
        """
        # Load model
        model = GemmaWrapper(
            checkpoint=self.cfg.model.base_checkpoint,
            device=self.cfg.device,
            load_in_4bit=self.cfg.model.load_in_4bit,
            load_in_8bit=self.cfg.model.load_in_8bit,
        )
        
        # Create PEFT adapter
        peft_adapter = create_adapter(
            adapter_type=self.cfg.model.peft_type,
            rank=self.cfg.model.peft_rank,
            alpha=self.cfg.model.peft_alpha,
            dropout=self.cfg.model.peft_dropout,
            target_modules=self.cfg.model.peft_target_modules,
        )
        
        # Create data loaders
        train_loader = self.train_dataset.get_dataloader(
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
        )
        val_loader = None
        if self.val_dataset is not None:
            val_loader = self.val_dataset.get_dataloader(
                batch_size=self.cfg.training.batch_size,
                shuffle=False,
            )
        
        # Create SFT trainer
        sft_trainer = SFTTrainer(
            model=model,
            cfg=self.cfg,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            peft_adapter=peft_adapter,
        )
        
        # Train
        metrics = sft_trainer.train()
        
        # Save checkpoint
        self.sft_checkpoint_path = os.path.join(
            self.cfg.logging.checkpoint_dir, "sft_final"
        )
        sft_trainer.save_checkpoint(self.sft_checkpoint_path)
        
        # Export LoRA weights separately
        if peft_adapter is not None:
            lora_path = os.path.join(self.cfg.logging.checkpoint_dir, "sft_lora")
            sft_trainer.export_lora_checkpoint(lora_path)
        
        return metrics
    
    def _run_rl_phase(self) -> Dict[str, Any]:
        """
        Run the RL refinement phase.
        
        Returns:
            RL training metrics.
        """
        # Load SFT checkpoint
        if self.sft_checkpoint_path is None:
            raise RuntimeError("SFT phase must run before RL phase")
        
        # Reload model from SFT checkpoint
        model = GemmaWrapper(
            checkpoint=self.cfg.model.base_checkpoint,
            device=self.cfg.device,
        )
        
        # Load SFT weights
        checkpoint_path = Path(self.sft_checkpoint_path)
        if (checkpoint_path / "model").exists():
            model.load(str(checkpoint_path / "model"))
        
        # Create or load reward model
        if self.reward_model is None:
            self.reward_model = RewardModel(
                hidden_size=self.cfg.reward.trainable.hidden_size,
                num_layers=self.cfg.reward.trainable.num_layers,
                reward_config=self.cfg.reward,
            )
            
            # Optionally train reward model on preference pairs
            if hasattr(self.train_dataset, "get_preference_pairs"):
                pairs = self.train_dataset.get_preference_pairs()
                if pairs:
                    prompts, chosen, rejected = zip(*pairs)
                    self.reward_model.train_on_pairs(
                        list(chosen), list(rejected), self.tokenizer, epochs=2
                    )
        
        # Create reference model (frozen copy for KL penalty)
        ref_model = GemmaWrapper(
            checkpoint=self.cfg.model.base_checkpoint,
            device=self.cfg.device,
        )
        ref_model.freeze()
        
        # Create data loader
        train_loader = self.train_dataset.get_dataloader(
            batch_size=self.cfg.training.batch_size // 2,  # Smaller batch for RL
            shuffle=True,
        )
        
        # Create RL trainer
        rl_trainer = RLTrainer(
            policy_model=model,
            reward_model=self.reward_model,
            tokenizer=self.tokenizer,
            cfg=self.cfg,
            train_dataloader=train_loader,
            ref_model=ref_model,
        )
        
        # Modify config for RL phase (fewer epochs, different LR)
        original_epochs = self.cfg.training.epochs
        self.cfg.training.epochs = max(1, original_epochs // 2)
        
        # Train
        metrics = rl_trainer.train()
        
        # Restore config
        self.cfg.training.epochs = original_epochs
        
        # Save final checkpoint
        self.rl_checkpoint_path = os.path.join(
            self.cfg.logging.checkpoint_dir, "staged_final"
        )
        rl_trainer.save_checkpoint(self.rl_checkpoint_path)
        
        return metrics
    
    def run_sft_only(self) -> Dict[str, Any]:
        """
        Run only the SFT phase (useful for debugging).
        
        Returns:
            SFT training metrics.
        """
        self.logger.info("Running SFT-only mode")
        return self._run_sft_phase()
    
    def run_rl_only(self, sft_checkpoint: str) -> Dict[str, Any]:
        """
        Run only the RL phase from a pre-trained SFT checkpoint.
        
        Args:
            sft_checkpoint: Path to SFT checkpoint.
        
        Returns:
            RL training metrics.
        """
        self.logger.info("Running RL-only mode from existing SFT checkpoint")
        self.sft_checkpoint_path = sft_checkpoint
        return self._run_rl_phase()
