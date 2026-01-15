"""Gemma model wrapper."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.models.base_model import ModelOutput, ModelWrapper


class GemmaWrapper(ModelWrapper):
    """Wrapper for Google Gemma 270M model."""
    
    def __init__(
        self,
        checkpoint: str = "google/gemma-3-270m-it",
        device: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        super().__init__(config={"checkpoint": checkpoint})
        
        self.checkpoint = checkpoint
        self._torch_dtype = torch_dtype or torch.bfloat16
        
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)
        
        self.model = self._load_model(checkpoint, trust_remote_code, **kwargs)
    
    def _load_model(self, checkpoint: str, trust_remote_code: bool, **kwargs) -> nn.Module:
        from transformers import AutoModelForCausalLM
        
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=self._torch_dtype,
            trust_remote_code=trust_remote_code,
            device_map="auto" if self._device.type == "cuda" else None,
            **kwargs,
        )
        
        if self._device.type != "cuda":
            model = model.to(self._device)
        
        return model
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> ModelOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        
        return ModelOutput(
            logits=outputs.logits,
            loss=outputs.loss if hasattr(outputs, "loss") else None,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            past_key_values=getattr(outputs, "past_key_values", None),
        )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                top_k=top_k if do_sample else 0,
                do_sample=do_sample,
                pad_token_id=self.model.config.pad_token_id,
                eos_token_id=self.model.config.eos_token_id,
                **kwargs,
            )
    
    def get_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = self.forward(input_ids, attention_mask)
        shift_logits = output.logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        gathered = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        
        if attention_mask is not None:
            gathered = gathered * attention_mask[:, 1:].contiguous()
        
        return gathered
    
    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
    
    def load(self, path: str, strict: bool = True) -> None:
        from transformers import AutoModelForCausalLM
        
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=self._torch_dtype,
            device_map="auto" if self._device.type == "cuda" else None,
        )
    
    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()
    
    def get_output_embeddings(self) -> nn.Module:
        return self.model.get_output_embeddings()
    
    def gradient_checkpointing_enable(self) -> None:
        self.model.gradient_checkpointing_enable()
    
    def prepare_for_training(self) -> None:
        self.model.train()
        self.gradient_checkpointing_enable()
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
