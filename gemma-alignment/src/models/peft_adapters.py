"""
PEFT adapter implementations for LoRA and QLoRA.

This module provides adapter classes that can be applied to model wrappers
to enable parameter-efficient fine-tuning.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from src.models.base_model import ModelWrapper


class BaseAdapter(ABC):
    """
    Abstract base class for PEFT adapters.
    
    All adapter implementations should inherit from this class and provide
    methods for applying, saving, loading, and removing adapters.
    """
    
    @abstractmethod
    def apply(self, model: ModelWrapper) -> ModelWrapper:
        """
        Apply the adapter to a model.
        
        Args:
            model: Model wrapper to apply adapter to.
        
        Returns:
            Model with adapter applied.
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save adapter weights to disk.
        
        Args:
            path: Directory path to save adapter.
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load adapter weights from disk.
        
        Args:
            path: Directory path to load adapter from.
        """
        pass
    
    @abstractmethod
    def remove(self, model: ModelWrapper) -> ModelWrapper:
        """
        Remove the adapter from a model.
        
        Args:
            model: Model to remove adapter from.
        
        Returns:
            Model with adapter removed.
        """
        pass


class LoRAAdapter(BaseAdapter):
    """
    LoRA (Low-Rank Adaptation) adapter for parameter-efficient fine-tuning.
    
    LoRA adds trainable low-rank matrices to the model's attention layers,
    allowing efficient fine-tuning with significantly reduced memory.
    
    Attributes:
        rank: Rank of the low-rank matrices.
        alpha: Scaling factor for LoRA.
        dropout: Dropout probability for LoRA layers.
        target_modules: List of module names to apply LoRA to.
    
    Example:
        >>> adapter = LoRAAdapter(rank=8, alpha=16)
        >>> model = adapter.apply(model)
        >>> # Train model
        >>> adapter.save("./lora_checkpoint")
    """
    
    def __init__(
        self,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        bias: str = "none",
        task_type: str = "CAUSAL_LM",
    ):
        """
        Initialize LoRA adapter configuration.
        
        Args:
            rank: Rank of the decomposition matrices (r in LoRA paper).
            alpha: Scaling factor (scales the LoRA weight by alpha/rank).
            dropout: Dropout probability applied to LoRA layers.
            target_modules: Modules to apply LoRA to. Defaults to attention layers.
            bias: Bias training mode ("none", "all", "lora_only").
            task_type: Task type for PEFT library.
        """
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
        """
        Apply LoRA adapter to the model.
        
        Args:
            model: Model wrapper to apply LoRA to.
        
        Returns:
            Model with LoRA applied all frozen base weights.
        """
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type=self.task_type,
        )
        
        # Apply PEFT to the underlying model
        model.model = get_peft_model(model.model, lora_config)
        self._peft_model = model.model
        
        # Print trainable parameters info
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"LoRA applied: {trainable_params:,} trainable / {total_params:,} total "
              f"({100 * trainable_params / total_params:.2f}%)")
        
        return model
    
    def save(self, path: str) -> None:
        """Save LoRA adapter weights."""
        if self._peft_model is None:
            raise RuntimeError("LoRA adapter not applied. Call apply() first.")
        
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        self._peft_model.save_pretrained(path_obj)
    
    def load(self, path: str) -> None:
        """Load LoRA adapter weights."""
        if self._peft_model is None:
            raise RuntimeError("LoRA adapter not applied. Call apply() first.")
        
        from peft import PeftModel
        
        # Load adapter weights
        self._peft_model.load_adapter(path, adapter_name="default")
    
    def remove(self, model: ModelWrapper) -> ModelWrapper:
        """Remove LoRA and merge weights into base model."""
        if self._peft_model is None:
            return model
        
        # Merge LoRA weights into base model
        model.model = self._peft_model.merge_and_unload()
        self._peft_model = None
        
        return model
    
    def get_config(self) -> Dict[str, Any]:
        """Get adapter configuration as dictionary."""
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type,
        }


class QLoRAAdapter(BaseAdapter):
    """
    QLoRA adapter combining 4-bit quantization with LoRA.
    
    QLoRA enables fine-tuning of large models on limited hardware by
    quantizing the base model to 4-bit precision while training LoRA
    adapters in higher precision.
    
    Note:
        The model should be loaded with load_in_4bit=True for QLoRA.
        This adapter assumes the model is already quantized.
    
    Attributes:
        rank: Rank of the LoRA matrices.
        alpha: Scaling factor for LoRA.
        dropout: Dropout probability.
        target_modules: Modules to apply QLoRA to.
    """
    
    def __init__(
        self,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        bias: str = "none",
        task_type: str = "CAUSAL_LM",
    ):
        """
        Initialize QLoRA adapter configuration.
        
        Args:
            rank: Rank of LoRA decomposition.
            alpha: LoRA scaling factor.
            dropout: Dropout probability.
            target_modules: Target modules for LoRA.
            bias: Bias training mode.
            task_type: PEFT task type.
        """
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
        """
        Apply QLoRA adapter to the quantized model.
        
        The model should already be loaded with 4-bit quantization.
        This method adds LoRA adapters on top of the quantized base.
        
        Args:
            model: Quantized model wrapper.
        
        Returns:
            Model with QLoRA applied.
        """
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        
        # Prepare quantized model for training
        model.model = prepare_model_for_kbit_training(model.model)
        
        # Create LoRA config
        lora_config = LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type=self.task_type,
        )
        
        # Apply PEFT
        model.model = get_peft_model(model.model, lora_config)
        self._peft_model = model.model
        
        # Print info
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"QLoRA applied: {trainable_params:,} trainable / {total_params:,} total "
              f"({100 * trainable_params / total_params:.2f}%)")
        
        return model
    
    def save(self, path: str) -> None:
        """Save QLoRA adapter weights."""
        if self._peft_model is None:
            raise RuntimeError("QLoRA adapter not applied. Call apply() first.")
        
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        self._peft_model.save_pretrained(path_obj)
    
    def load(self, path: str) -> None:
        """Load QLoRA adapter weights."""
        if self._peft_model is None:
            raise RuntimeError("QLoRA adapter not applied. Call apply() first.")
        
        self._peft_model.load_adapter(path, adapter_name="default")
    
    def remove(self, model: ModelWrapper) -> ModelWrapper:
        """
        Remove QLoRA adapter.
        
        Note: For quantized models, full merge may not be supported.
        This returns the model with adapter removed but base still quantized.
        """
        if self._peft_model is None:
            return model
        
        # For quantized models, we can't fully merge, so just remove adapter
        try:
            model.model = self._peft_model.merge_and_unload()
        except Exception:
            # If merge fails (common with quantized models), just keep as-is
            pass
        
        self._peft_model = None
        return model
    
    def get_config(self) -> Dict[str, Any]:
        """Get adapter configuration as dictionary."""
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type,
            "quantization": "4bit",
        }


def create_adapter(
    adapter_type: str,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
) -> BaseAdapter:
    """
    Factory function to create adapters by type.
    
    Args:
        adapter_type: Type of adapter ("lora", "qlora", "none").
        rank: LoRA rank.
        alpha: LoRA alpha.
        dropout: Dropout probability.
        target_modules: Target modules for adaptation.
    
    Returns:
        Adapter instance or None for "none" type.
    """
    if adapter_type == "lora":
        return LoRAAdapter(
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
        )
    elif adapter_type == "qlora":
        return QLoRAAdapter(
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
        )
    elif adapter_type == "none":
        return None
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
