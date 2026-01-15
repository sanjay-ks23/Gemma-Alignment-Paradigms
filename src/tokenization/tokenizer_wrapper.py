"""Tokenizer wrapper for Hugging Face tokenizers."""

from typing import Dict, List, Optional, Union

import torch


class TokenizerWrapper:
    """Clean interface over Hugging Face tokenizers."""
    
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
    
    @classmethod
    def from_pretrained(cls, checkpoint: str, **kwargs) -> "TokenizerWrapper":
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, **kwargs)
        return cls(tokenizer)
    
    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size
    
    @property
    def pad_token_id(self) -> int:
        return self._tokenizer.pad_token_id
    
    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.eos_token_id
    
    @property
    def bos_token_id(self) -> Optional[int]:
        return self._tokenizer.bos_token_id
    
    def encode(
        self,
        text: str,
        max_length: int = 512,
        truncation: bool = True,
        add_special_tokens: bool = True,
    ) -> Dict[str, torch.Tensor]:
        encoded = self._tokenizer(
            text,
            max_length=max_length,
            truncation=truncation,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
    
    def encode_batch(
        self,
        texts: List[str],
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        encoded = self._tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def decode_batch(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> List[str]:
        return self._tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        if hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        
        result = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            result += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"
        
        if add_generation_prompt:
            result += "<start_of_turn>model\n"
        
        return result
    
    def __call__(self, text, **kwargs):
        return self._tokenizer(text, **kwargs)
    
    def __len__(self) -> int:
        return len(self._tokenizer)
