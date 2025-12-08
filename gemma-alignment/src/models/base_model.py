"""
Base model wrapper providing a unified interface for language models.

This module defines the abstract ModelWrapper class that all model
implementations should inherit from, ensuring consistent APIs across
different model architectures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class ModelOutput:
    """
    Standardized output from model forward pass.
    
    Attributes:
        logits: Output logits of shape (batch_size, seq_len, vocab_size).
        loss: Optional loss value if labels were provided.
        hidden_states: Optional tuple of hidden states from each layer.
        past_key_values: Optional cached key-value pairs for generation.
    """
    
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]] = None


class ModelWrapper(nn.Module, ABC):
    """
    Abstract base class for language model wrappers.
    
    This class provides a unified interface for different language model
    implementations, abstracting away framework-specific details and
    ensuring consistent APIs for training and inference.
    
    Subclasses must implement the abstract methods to provide model-specific
    functionality.
    
    Attributes:
        config: Model configuration dictionary.
        device: Device the model is on.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model wrapper.
        
        Args:
            config: Optional configuration dictionary.
        """
        super().__init__()
        self.config = config or {}
        self._device = torch.device("cpu")
    
    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return self._device
    
    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ModelOutput:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            labels: Optional labels for loss computation.
            **kwargs: Additional model-specific arguments.
        
        Returns:
            ModelOutput: Standardized output containing logits and optionally loss.
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        do_sample: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            top_k: Top-k sampling.
            do_sample: Whether to sample or use greedy decoding.
            **kwargs: Additional generation arguments.
        
        Returns:
            torch.Tensor: Generated token IDs of shape (batch_size, seq_len + new_tokens).
        """
        pass
    
    def get_trainable_params(self) -> Iterator[nn.Parameter]:
        """
        Get iterator over trainable parameters.
        
        Returns:
            Iterator over parameters with requires_grad=True.
        """
        return (p for p in self.parameters() if p.requires_grad)
    
    def count_trainable_params(self) -> int:
        """
        Count the number of trainable parameters.
        
        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.get_trainable_params())
    
    def count_total_params(self) -> int:
        """
        Count total number of parameters.
        
        Returns:
            Total number of parameters.
        """
        return sum(p.numel() for p in self.parameters())
    
    def freeze(self) -> None:
        """Freeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self) -> None:
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True
    
    def save(self, path: str) -> None:
        """
        Save model weights to disk.
        
        Args:
            path: Path to save the model weights.
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path_obj)
    
    def load(self, path: str, strict: bool = True) -> None:
        """
        Load model weights from disk.
        
        Args:
            path: Path to the saved weights.
            strict: Whether to require exact match of state dict keys.
        """
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.load_state_dict(state_dict, strict=strict)
    
    def to_device(self, device: str) -> "ModelWrapper":
        """
        Move model to specified device.
        
        Args:
            device: Target device ("cuda", "cpu", etc.).
        
        Returns:
            Self for method chaining.
        """
        self._device = torch.device(device)
        return self.to(self._device)
    
    def get_input_embeddings(self) -> Optional[nn.Module]:
        """
        Get the input embedding layer.
        
        Returns:
            Embedding layer or None if not applicable.
        """
        return None
    
    def get_output_embeddings(self) -> Optional[nn.Module]:
        """
        Get the output embedding/LM head layer.
        
        Returns:
            LM head layer or None if not applicable.
        """
        return None
    
    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing to reduce memory usage."""
        pass  # Override in subclass if supported
    
    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        pass  # Override in subclass if supported
