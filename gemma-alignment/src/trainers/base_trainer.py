"""
Base trainer providing common training loop functionality.

This module defines the abstract BaseTrainer class with lifecycle hooks
and shared utilities for all training paradigms.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.core.config import ExperimentConfig
from src.core.utils import get_logger, save_checkpoint, load_checkpoint


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers.
    
    Provides common functionality for training loops including:
    - Optimizer and scheduler setup
    - Gradient accumulation
    - Checkpoint saving/loading
    - Logging and progress tracking
    - Lifecycle hooks for customization
    
    Subclasses must implement train_step() for task-specific training logic.
    
    Attributes:
        model: The model to train.
        cfg: Experiment configuration.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        global_step: Current training step.
        current_epoch: Current epoch number.
    """
    
    def __init__(
        self,
        model: nn.Module,
        cfg: ExperimentConfig,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train.
            cfg: Experiment configuration.
            train_dataloader: Training data loader.
            val_dataloader: Validation data loader.
        """
        self.model = model
        self.cfg = cfg
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        self.device = torch.device(cfg.device if cfg.device != "auto" else 
                                   ("cuda" if torch.cuda.is_available() else "cpu"))
        self.logger = get_logger("gemma-alignment")
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.scaler = None
        if cfg.training.fp16 or cfg.training.bf16:
            self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.fp16)
        
        # Gradient accumulation tracking
        self.accumulated_loss = 0.0
        self.accumulation_steps = 0
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay handling."""
        # Separate parameters that should and shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "LayerNorm" in name or "layernorm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_grouped_params = [
            {"params": decay_params, "weight_decay": self.cfg.training.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        return torch.optim.AdamW(
            optimizer_grouped_params,
            lr=self.cfg.training.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        """Create learning rate scheduler with warmup."""
        if self.train_dataloader is None:
            return None
        
        total_steps = (
            len(self.train_dataloader) 
            * self.cfg.training.epochs 
            // self.cfg.training.gradient_accumulation_steps
        )
        warmup_steps = int(total_steps * self.cfg.training.warmup_ratio)
        
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
            )
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Perform a single training step.
        
        Args:
            batch: Dictionary containing batch data.
        
        Returns:
            Loss value for this step.
        """
        pass
    
    def train(self) -> Dict[str, Any]:
        """
        Run the full training loop.
        
        Returns:
            Dictionary containing training metrics and results.
        """
        self.on_train_start()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(self.cfg.training.epochs):
            self.current_epoch = epoch
            
            # Training epoch
            epoch_loss = self.train_epoch(epoch)
            train_losses.append(epoch_loss)
            
            # Validation
            if self.val_dataloader is not None:
                val_metrics = self.validate()
                val_loss = val_metrics.get("loss", float("inf"))
                val_losses.append(val_loss)
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(
                        os.path.join(self.cfg.logging.checkpoint_dir, "best_model")
                    )
            
            # Save periodic checkpoint
            if (epoch + 1) % max(1, self.cfg.training.epochs // 3) == 0:
                self.save_checkpoint(
                    os.path.join(self.cfg.logging.checkpoint_dir, f"epoch_{epoch + 1}")
                )
            
            # Debug mode: stop after first epoch
            if self.cfg.debug:
                self.logger.info("Debug mode: stopping after first epoch")
                break
        
        self.on_train_end()
        
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": self.best_val_loss,
            "total_steps": self.global_step,
        }
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number.
        
        Returns:
            Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch + 1}/{self.cfg.training.epochs}",
            disable=self.cfg.debug,
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Training step
            loss = self.train_step(batch)
            
            # Gradient accumulation
            self.accumulated_loss += loss
            self.accumulation_steps += 1
            
            if self.accumulation_steps >= self.cfg.training.gradient_accumulation_steps:
                # Gradient clipping
                if self.cfg.training.max_grad_norm > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.training.max_grad_norm,
                    )
                
                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                
                # Logging
                avg_loss = self.accumulated_loss / self.accumulation_steps
                total_loss += avg_loss
                num_batches += 1
                
                self.on_step_end(self.global_step, avg_loss)
                
                # Update progress bar
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                
                # Reset accumulation
                self.accumulated_loss = 0.0
                self.accumulation_steps = 0
                self.global_step += 1
            
            # Debug mode: stop after a few batches
            if self.cfg.debug and batch_idx >= 2:
                self.logger.info("Debug mode: stopping after 3 batches")
                break
        
        return total_loss / max(num_batches, 1)
    
    def validate(self) -> Dict[str, float]:
        """
        Run validation loop.
        
        Returns:
            Dictionary containing validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                output = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch.get("labels"),
                )
                
                if output.loss is not None:
                    total_loss += output.loss.item()
                    num_batches += 1
                
                # Debug mode
                if self.cfg.debug and batch_idx >= 2:
                    break
        
        avg_loss = total_loss / max(num_batches, 1)
        self.logger.info(f"Validation loss: {avg_loss:.4f}")
        
        return {"loss": avg_loss}
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save training checkpoint.
        
        Args:
            path: Directory to save checkpoint.
        """
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.cfg,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Save training state
        torch.save(checkpoint, path_obj / "trainer_state.pt")
        
        # Save model
        if hasattr(self.model, "save"):
            self.model.save(str(path_obj / "model"))
        else:
            torch.save(self.model.state_dict(), path_obj / "model.pt")
        
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """
        Load training checkpoint.
        
        Args:
            path: Directory containing checkpoint.
        """
        path_obj = Path(path)
        
        # Load training state
        checkpoint = torch.load(path_obj / "trainer_state.pt", map_location=self.device)
        
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load model
        if hasattr(self.model, "load"):
            self.model.load(str(path_obj / "model"))
        else:
            self.model.load_state_dict(
                torch.load(path_obj / "model.pt", map_location=self.device)
            )
        
        self.logger.info(f"Checkpoint loaded from {path}")
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to the training device."""
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved
    
    # Lifecycle hooks - override in subclasses as needed
    
    def on_train_start(self) -> None:
        """Called at the start of training."""
        self.logger.info("Training started")
        self.logger.info(f"Total epochs: {self.cfg.training.epochs}")
        self.logger.info(f"Batch size: {self.cfg.training.batch_size}")
        self.logger.info(f"Gradient accumulation: {self.cfg.training.gradient_accumulation_steps}")
    
    def on_train_end(self) -> None:
        """Called at the end of training."""
        self.logger.info(f"Training completed. Total steps: {self.global_step}")
    
    def on_step_end(self, step: int, loss: float) -> None:
        """Called at the end of each training step."""
        if step % self.cfg.logging.log_interval == 0:
            lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info(f"Step {step}: loss={loss:.4f}, lr={lr:.2e}")
