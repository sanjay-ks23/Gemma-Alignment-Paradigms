"""
SFT (Supervised Fine-Tuning) trainer implementation.

This module provides the SFTTrainer class for standard supervised
fine-tuning with optional PEFT adapter support.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.core.config import ExperimentConfig
from src.models.base_model import ModelWrapper
from src.models.peft_adapters import BaseAdapter
from src.trainers.base_trainer import BaseTrainer


class SFTTrainer(BaseTrainer):
    """
    Supervised Fine-Tuning trainer with PEFT support.
    
    Implements standard next-token prediction training with cross-entropy
    loss. Supports LoRA and QLoRA adapters for parameter-efficient training.
    
    Attributes:
        peft_adapter: Optional PEFT adapter (LoRA/QLoRA).
    
    Example:
        >>> trainer = SFTTrainer(model, cfg, train_loader, val_loader)
        >>> trainer.train()
        >>> trainer.export_lora_checkpoint("./lora_weights")
    """
    
    def __init__(
        self,
        model: ModelWrapper,
        cfg: ExperimentConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        peft_adapter: Optional[BaseAdapter] = None,
    ):
        """
        Initialize the SFT trainer.
        
        Args:
            model: Model wrapper to train.
            cfg: Experiment configuration.
            train_dataloader: Training data loader.
            val_dataloader: Optional validation data loader.
            peft_adapter: Optional PEFT adapter to apply.
        """
        self.peft_adapter = peft_adapter
        
        # Apply PEFT adapter if provided
        if peft_adapter is not None:
            model = peft_adapter.apply(model)
        
        super().__init__(model, cfg, train_dataloader, val_dataloader)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Perform a single SFT training step.
        
        Computes cross-entropy loss for next-token prediction and
        performs backward pass with gradient scaling if enabled.
        
        Args:
            batch: Dictionary with input_ids, attention_mask, labels.
        
        Returns:
            Loss value for this step.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch.get("labels", input_ids.clone())
        
        # Mixed precision forward pass
        with torch.cuda.amp.autocast(
            enabled=self.cfg.training.fp16 or self.cfg.training.bf16,
            dtype=torch.bfloat16 if self.cfg.training.bf16 else torch.float16,
        ):
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = output.loss
            
            # Scale loss for gradient accumulation
            loss = loss / self.cfg.training.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.cfg.training.gradient_accumulation_steps
    
    def export_lora_checkpoint(self, path: str) -> None:
        """
        Export LoRA adapter weights separately.
        
        This saves only the adapter weights, which can be loaded later
        to apply fine-tuning to the base model.
        
        Args:
            path: Directory to save adapter weights.
        """
        if self.peft_adapter is None:
            self.logger.warning("No PEFT adapter to export")
            return
        
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        
        self.peft_adapter.save(str(path_obj))
        self.logger.info(f"LoRA checkpoint exported to {path}")
    
    def merge_and_save(self, path: str) -> None:
        """
        Merge adapter weights into base model and save.
        
        This creates a standalone model with adapter weights merged,
        which can be loaded without PEFT library.
        
        Args:
            path: Directory to save merged model.
        """
        if self.peft_adapter is None:
            self.logger.info("No adapter to merge, saving model directly")
            self.model.save(path)
            return
        
        # Merge adapter weights
        merged_model = self.peft_adapter.remove(self.model)
        merged_model.save(path)
        
        self.logger.info(f"Merged model saved to {path}")
    
    def on_train_start(self) -> None:
        """Called at training start."""
        super().on_train_start()
        
        # Log PEFT info
        if self.peft_adapter is not None:
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            self.logger.info(
                f"PEFT enabled: {trainable:,} trainable params "
                f"({100 * trainable / total:.2f}%)"
            )
        
        # Prepare model for training
        if hasattr(self.model, "prepare_for_training"):
            self.model.prepare_for_training()
