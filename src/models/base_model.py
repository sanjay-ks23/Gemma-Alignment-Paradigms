"""Base model wrapper providing unified interface for language models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class ModelOutput:
    """Standardized output from model forward pass."""
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]] = None


class ModelWrapper(nn.Module, ABC):
    """Abstract base class for language model wrappers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        self._device = torch.device("cpu")
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ModelOutput:
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
        pass
    
    def get_trainable_params(self) -> Iterator[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)
    
    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.get_trainable_params())
    
    def count_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
    
    def save(self, path: str) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path_obj)
    
    def load(self, path: str, strict: bool = True) -> None:
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.load_state_dict(state_dict, strict=strict)
    
    def to_device(self, device: str) -> "ModelWrapper":
        self._device = torch.device(device)
        return self.to(self._device)
    
    def get_input_embeddings(self) -> Optional[nn.Module]:
        return None
    
    def get_output_embeddings(self) -> Optional[nn.Module]:
        return None
    
    def gradient_checkpointing_enable(self) -> None:
        pass
    
    def gradient_checkpointing_disable(self) -> None:
        pass
