"""SFT trainer with PEFT adapter support."""

from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from src.core.config import ExperimentConfig
from src.models.base_model import ModelWrapper
from src.models.peft_adapters import BaseAdapter
from src.trainers.base_trainer import BaseTrainer


class SFTTrainer(BaseTrainer):
    """Supervised Fine-Tuning trainer with LoRA/QLoRA support."""
    
    def __init__(
        self,
        model: ModelWrapper,
        cfg: ExperimentConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        peft_adapter: Optional[BaseAdapter] = None,
    ):
        self.peft_adapter = peft_adapter
        
        if peft_adapter is not None:
            model = peft_adapter.apply(model)
        
        super().__init__(model, cfg, train_dataloader, val_dataloader)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch.get("labels", input_ids.clone())
        
        with torch.cuda.amp.autocast(
            enabled=self.cfg.training.fp16 or self.cfg.training.bf16,
            dtype=torch.bfloat16 if self.cfg.training.bf16 else torch.float16,
        ):
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = output.loss / self.cfg.training.gradient_accumulation_steps
        
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.cfg.training.gradient_accumulation_steps
    
    def export_lora_checkpoint(self, path: str) -> None:
        """Export adapter weights separately."""
        if self.peft_adapter is None:
            self.logger.warning("No PEFT adapter to export")
            return
        
        Path(path).mkdir(parents=True, exist_ok=True)
        self.peft_adapter.save(path)
    
    def merge_and_save(self, path: str) -> None:
        """Merge adapter weights into base model and save."""
        if self.peft_adapter is None:
            self.model.save(path)
            return
        
        merged_model = self.peft_adapter.remove(self.model)
        merged_model.save(path)
