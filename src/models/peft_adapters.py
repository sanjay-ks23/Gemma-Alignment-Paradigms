"""PEFT adapter implementations for LoRA."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.models.base_model import ModelWrapper


class BaseAdapter(ABC):
    """Abstract base class for PEFT adapters."""
    
    @abstractmethod
    def apply(self, model: ModelWrapper) -> ModelWrapper:
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        pass
    
    @abstractmethod
    def remove(self, model: ModelWrapper) -> ModelWrapper:
        pass


class LoRAAdapter(BaseAdapter):
    """LoRA (Low-Rank Adaptation) adapter."""
    
    def __init__(
        self,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        bias: str = "none",
        task_type: str = "CAUSAL_LM",
    ):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        self.bias = bias
        self.task_type = task_type
        self._peft_model = None
    
    def apply(self, model: ModelWrapper) -> ModelWrapper:
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type=self.task_type,
        )
        
        model.model = get_peft_model(model.model, lora_config)
        self._peft_model = model.model
        
        return model
    
    def save(self, path: str) -> None:
        if self._peft_model is None:
            raise RuntimeError("Adapter not applied")
        
        Path(path).mkdir(parents=True, exist_ok=True)
        self._peft_model.save_pretrained(path)
    
    def load(self, path: str) -> None:
        if self._peft_model is None:
            raise RuntimeError("Adapter not applied")
        
        self._peft_model.load_adapter(path, adapter_name="default")
    
    def remove(self, model: ModelWrapper) -> ModelWrapper:
        if self._peft_model is None:
            return model
        
        model.model = self._peft_model.merge_and_unload()
        self._peft_model = None
        return model
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
        }


def create_adapter(
    adapter_type: str,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
) -> Optional[BaseAdapter]:
    """Factory function to create adapters by type."""
    if adapter_type == "lora":
        return LoRAAdapter(rank=rank, alpha=alpha, dropout=dropout, target_modules=target_modules)
    elif adapter_type == "none":
        return None
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
