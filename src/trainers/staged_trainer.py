"""Staged trainer orchestrating SFT followed by RL refinement."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.config import ExperimentConfig
from src.core.utils import get_logger, set_seed
from src.data.base_dataset import BaseDataset
from src.models.gemma_wrapper import GemmaWrapper
from src.models.peft_adapters import create_adapter
from src.models.reward_model import RewardModel
from src.tokenization.tokenizer_wrapper import TokenizerWrapper
from src.trainers.sft_trainer import SFTTrainer
from src.trainers.rl_trainer import RLTrainer


class StagedTrainer:
    """Orchestrates SFT -> RL training pipeline."""
    
    def __init__(
        self,
        cfg: ExperimentConfig,
        tokenizer: Optional[TokenizerWrapper] = None,
        train_dataset: Optional[BaseDataset] = None,
        val_dataset: Optional[BaseDataset] = None,
        reward_model: Optional[RewardModel] = None,
    ):
        self.cfg = cfg
        self.logger = get_logger("gemma-alignment")
        
        set_seed(cfg.seed)
        
        self.tokenizer = tokenizer or TokenizerWrapper.from_pretrained(cfg.model.base_checkpoint)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.reward_model = reward_model
        
        self.sft_checkpoint_path: Optional[str] = None
        self.rl_checkpoint_path: Optional[str] = None
    
    def run(self) -> Dict[str, Any]:
        """Run complete staged training pipeline."""
        self.logger.info("Starting staged training: SFT -> RL")
        
        sft_metrics = self._run_sft_phase()
        rl_metrics = self._run_rl_phase()
        
        return {
            "sft_metrics": sft_metrics,
            "rl_metrics": rl_metrics,
            "sft_checkpoint": self.sft_checkpoint_path,
            "final_checkpoint": self.rl_checkpoint_path,
        }
    
    def _run_sft_phase(self) -> Dict[str, Any]:
        model = GemmaWrapper(
            checkpoint=self.cfg.model.base_checkpoint,
            device=self.cfg.device,
            load_in_4bit=self.cfg.model.load_in_4bit,
            load_in_8bit=self.cfg.model.load_in_8bit,
        )
        
        peft_adapter = create_adapter(
            adapter_type=self.cfg.model.peft_type,
            rank=self.cfg.model.peft_rank,
            alpha=self.cfg.model.peft_alpha,
            dropout=self.cfg.model.peft_dropout,
            target_modules=self.cfg.model.peft_target_modules,
        )
        
        train_loader = self.train_dataset.get_dataloader(
            batch_size=self.cfg.training.batch_size, shuffle=True
        )
        val_loader = None
        if self.val_dataset is not None:
            val_loader = self.val_dataset.get_dataloader(
                batch_size=self.cfg.training.batch_size, shuffle=False
            )
        
        trainer = SFTTrainer(
            model=model,
            cfg=self.cfg,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            peft_adapter=peft_adapter,
        )
        
        metrics = trainer.train()
        
        self.sft_checkpoint_path = os.path.join(
            self.cfg.logging.checkpoint_dir, "sft_final"
        )
        trainer.save_checkpoint(self.sft_checkpoint_path)
        
        if peft_adapter is not None:
            trainer.export_lora_checkpoint(
                os.path.join(self.cfg.logging.checkpoint_dir, "sft_lora")
            )
        
        return metrics
    
    def _run_rl_phase(self) -> Dict[str, Any]:
        if self.sft_checkpoint_path is None:
            raise RuntimeError("SFT phase must run before RL phase")
        
        model = GemmaWrapper(
            checkpoint=self.cfg.model.base_checkpoint,
            device=self.cfg.device,
        )
        
        checkpoint_path = Path(self.sft_checkpoint_path)
        if (checkpoint_path / "model").exists():
            model.load(str(checkpoint_path / "model"))
        
        if self.reward_model is None:
            self.reward_model = RewardModel(
                hidden_size=self.cfg.reward.trainable.hidden_size,
                num_layers=self.cfg.reward.trainable.num_layers,
                reward_config=self.cfg.reward,
            )
        
        ref_model = GemmaWrapper(
            checkpoint=self.cfg.model.base_checkpoint,
            device=self.cfg.device,
        )
        ref_model.freeze()
        
        train_loader = self.train_dataset.get_dataloader(
            batch_size=self.cfg.training.batch_size // 2, shuffle=True
        )
        
        original_epochs = self.cfg.training.epochs
        self.cfg.training.epochs = max(1, original_epochs // 2)
        
        trainer = RLTrainer(
            policy_model=model,
            reward_model=self.reward_model,
            tokenizer=self.tokenizer,
            cfg=self.cfg,
            train_dataloader=train_loader,
            ref_model=ref_model,
        )
        
        metrics = trainer.train()
        
        self.cfg.training.epochs = original_epochs
        
        self.rl_checkpoint_path = os.path.join(
            self.cfg.logging.checkpoint_dir, "staged_final"
        )
        trainer.save_checkpoint(self.rl_checkpoint_path)
        
        return metrics
