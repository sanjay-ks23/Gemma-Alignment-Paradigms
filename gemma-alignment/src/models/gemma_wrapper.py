"""
Gemma model wrapper for loading and using Google Gemma models.

This module provides the GemmaWrapper class that wraps Hugging Face's
Gemma model implementations, exposing a clean interface for training
and inference.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.models.base_model import ModelOutput, ModelWrapper


class GemmaWrapper(ModelWrapper):
    """
    Wrapper for Google Gemma language models.
    
    This class loads Gemma models from Hugging Face checkpoints and provides
    a unified interface for forward passes and text generation.
    
    Attributes:
        model: The underlying Hugging Face model.
        checkpoint: The model checkpoint name or path.
    
    Example:
        >>> wrapper = GemmaWrapper("google/gemma-3-270m-it")
        >>> output = wrapper(input_ids, attention_mask)
        >>> generated = wrapper.generate(input_ids, max_new_tokens=50)
    """
    
    def __init__(
        self,
        checkpoint: str = "google/gemma-3-270m-it",
        device: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        trust_remote_code: bool = True,
        attn_implementation: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the Gemma wrapper.
        
        Args:
            checkpoint: Model checkpoint name or path.
            device: Device to load model on ("auto", "cuda", "cpu").
            torch_dtype: Data type for model weights.
            load_in_4bit: Whether to load in 4-bit quantization.
            load_in_8bit: Whether to load in 8-bit quantization.
            trust_remote_code: Whether to trust remote code.
            attn_implementation: Attention implementation ("eager", "flash_attention_2").
            **kwargs: Additional arguments for model loading.
        """
        super().__init__(config={"checkpoint": checkpoint})
        
        self.checkpoint = checkpoint
        self._torch_dtype = torch_dtype or torch.bfloat16
        
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(device)
        
        # Load model
        self.model = self._load_model(
            checkpoint=checkpoint,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            **kwargs,
        )
    
    def _load_model(
        self,
        checkpoint: str,
        load_in_4bit: bool,
        load_in_8bit: bool,
        trust_remote_code: bool,
        attn_implementation: Optional[str],
        **kwargs,
    ) -> nn.Module:
        """Load the Gemma model from checkpoint."""
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        
        # Configure quantization if requested
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self._torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Build loading arguments
        load_kwargs = {
            "pretrained_model_name_or_path": checkpoint,
            "torch_dtype": self._torch_dtype,
            "trust_remote_code": trust_remote_code,
            "device_map": "auto" if self._device.type == "cuda" else None,
            **kwargs,
        }
        
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        
        model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        
        # Move to device if not using device_map
        if load_kwargs.get("device_map") is None:
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
        """
        Forward pass through Gemma model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            labels: Optional labels for loss computation.
            output_hidden_states: Whether to return hidden states.
            **kwargs: Additional arguments passed to the model.
        
        Returns:
            ModelOutput with logits and optionally loss and hidden states.
        """
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
        num_return_sequences: int = 1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_p: Nucleus sampling probability threshold.
            top_k: Top-k sampling (0 = disabled).
            do_sample: Whether to use sampling (False = greedy).
            num_return_sequences: Number of sequences to generate per input.
            pad_token_id: Token ID to use for padding.
            eos_token_id: Token ID to stop generation.
            **kwargs: Additional generation arguments.
        
        Returns:
            Generated token IDs of shape (batch_size * num_return_sequences, total_len).
        """
        # Use model's special tokens if not provided
        if pad_token_id is None:
            pad_token_id = self.model.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.model.config.eos_token_id
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                top_k=top_k if do_sample else 0,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **kwargs,
            )
        
        return outputs
    
    def get_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute log probabilities for input sequences.
        
        This is useful for RL training (PPO, DPO) where we need to compute
        the probability of generated sequences.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
        
        Returns:
            Log probabilities of shape (batch_size, seq_len - 1).
        """
        output = self.forward(input_ids, attention_mask)
        logits = output.logits
        
        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        # Compute log probs
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for actual tokens
        gathered_log_probs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask out padding
        if attention_mask is not None:
            mask = attention_mask[:, 1:].contiguous()
            gathered_log_probs = gathered_log_probs * mask
        
        return gathered_log_probs
    
    def save(self, path: str) -> None:
        """Save model to directory."""
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path_obj)
    
    def load(self, path: str, strict: bool = True) -> None:
        """Load model from directory."""
        from transformers import AutoModelForCausalLM
        
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=self._torch_dtype,
            device_map="auto" if self._device.type == "cuda" else None,
        )
    
    def get_input_embeddings(self) -> nn.Module:
        """Get the input embedding layer."""
        return self.model.get_input_embeddings()
    
    def get_output_embeddings(self) -> nn.Module:
        """Get the output LM head layer."""
        return self.model.get_output_embeddings()
    
    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing."""
        self.model.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        self.model.gradient_checkpointing_disable()
    
    def prepare_for_training(self) -> None:
        """
        Prepare model for training.
        
        This enables features like gradient checkpointing and ensures
        proper configuration for training mode.
        """
        self.model.train()
        self.gradient_checkpointing_enable()
        
        # Enable input embeddings to require gradients for gradient checkpointing
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
