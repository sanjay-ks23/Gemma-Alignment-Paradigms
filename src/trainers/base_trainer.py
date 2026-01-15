"""Base trainer with common training loop functionality."""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.core.config import ExperimentConfig
from src.core.utils import get_logger


class BaseTrainer(ABC):
    """Abstract base class for all trainers."""
    
    def __init__(
        self,
        model: nn.Module,
        cfg: ExperimentConfig,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.cfg = cfg
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        self.device = torch.device(
            cfg.device if cfg.device != "auto" else 
            ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.logger = get_logger("gemma-alignment")
        
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        self.scaler = None
        if cfg.training.fp16 or cfg.training.bf16:
            self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.fp16)
        
        self.accumulated_loss = 0.0
        self.accumulation_steps = 0
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "LayerNorm" in name or "layernorm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        return torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.cfg.training.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.cfg.training.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
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
        """Perform single training step. Returns loss value."""
        pass
    
    def train(self) -> Dict[str, Any]:
        """Run full training loop."""
        self._log_train_config()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(self.cfg.training.epochs):
            self.current_epoch = epoch
            epoch_loss = self._train_epoch(epoch)
            train_losses.append(epoch_loss)
            
            if self.val_dataloader is not None:
                val_metrics = self._validate()
                val_loss = val_metrics.get("loss", float("inf"))
                val_losses.append(val_loss)
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(
                        os.path.join(self.cfg.logging.checkpoint_dir, "best_model")
                    )
            
            if (epoch + 1) % max(1, self.cfg.training.epochs // 3) == 0:
                self.save_checkpoint(
                    os.path.join(self.cfg.logging.checkpoint_dir, f"epoch_{epoch + 1}")
                )
            
            if self.cfg.debug:
                break
        
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": self.best_val_loss,
            "total_steps": self.global_step,
        }
    
    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch + 1}/{self.cfg.training.epochs}",
            disable=self.cfg.debug,
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            batch = self._move_batch_to_device(batch)
            loss = self.train_step(batch)
            
            self.accumulated_loss += loss
            self.accumulation_steps += 1
            
            if self.accumulation_steps >= self.cfg.training.gradient_accumulation_steps:
                if self.cfg.training.max_grad_norm > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.training.max_grad_norm,
                    )
                
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                
                avg_loss = self.accumulated_loss / self.accumulation_steps
                total_loss += avg_loss
                num_batches += 1
                
                if self.global_step % self.cfg.logging.log_interval == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    self.logger.info(f"Step {self.global_step}: loss={avg_loss:.4f}, lr={lr:.2e}")
                
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                
                self.accumulated_loss = 0.0
                self.accumulation_steps = 0
                self.global_step += 1
            
            if self.cfg.debug and batch_idx >= 2:
                break
        
        return total_loss / max(num_batches, 1)
    
    def _validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                batch = self._move_batch_to_device(batch)
                output = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch.get("labels"),
                )
                
                if output.loss is not None:
                    total_loss += output.loss.item()
                    num_batches += 1
                
                if self.cfg.debug and batch_idx >= 2:
                    break
        
        avg_loss = total_loss / max(num_batches, 1)
        self.logger.info(f"Validation loss: {avg_loss:.4f}")
        return {"loss": avg_loss}
    
    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
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
        
        torch.save(checkpoint, path_obj / "trainer_state.pt")
        
        if hasattr(self.model, "save"):
            self.model.save(str(path_obj / "model"))
        else:
            torch.save(self.model.state_dict(), path_obj / "model.pt")
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        path_obj = Path(path)
        checkpoint = torch.load(path_obj / "trainer_state.pt", map_location=self.device)
        
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if hasattr(self.model, "load"):
            self.model.load(str(path_obj / "model"))
        else:
            self.model.load_state_dict(
                torch.load(path_obj / "model.pt", map_location=self.device)
            )
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    
    def _log_train_config(self) -> None:
        self.logger.info(f"Training: epochs={self.cfg.training.epochs}, "
                        f"batch_size={self.cfg.training.batch_size}, "
                        f"accumulation={self.cfg.training.gradient_accumulation_steps}")
