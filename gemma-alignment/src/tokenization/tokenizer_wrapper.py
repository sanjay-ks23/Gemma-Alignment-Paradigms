"""
Tokenizer wrapper providing a clean PyTorch-centric interface.

This module wraps Hugging Face tokenizers to provide a simplified,
consistent interface for text encoding and decoding operations.
"""

from typing import Dict, List, Optional, Union

import torch


class TokenizerWrapper:
    """
    A wrapper around Hugging Face tokenizers providing a clean PyTorch interface.
    
    This class encapsulates the Hugging Face tokenizer, exposing only the
    essential methods needed for alignment training while hiding
    implementation details.
    
    Attributes:
        _tokenizer: The underlying HF tokenizer instance.
        pad_token_id: Token ID for padding.
        eos_token_id: Token ID for end of sequence.
        bos_token_id: Token ID for beginning of sequence.
    """
    
    def __init__(self, tokenizer):
        """
        Initialize with an existing tokenizer.
        
        Args:
            tokenizer: A Hugging Face tokenizer instance.
        """
        self._tokenizer = tokenizer
        
        # Ensure padding is configured
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        self.pad_token_id = self._tokenizer.pad_token_id
        self.eos_token_id = self._tokenizer.eos_token_id
        self.bos_token_id = self._tokenizer.bos_token_id
        self.vocab_size = len(self._tokenizer)
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> "TokenizerWrapper":
        """
        Load a tokenizer from a pretrained checkpoint.
        
        Args:
            checkpoint: Model checkpoint name or path (e.g., "google/gemma-3-270m-it").
            trust_remote_code: Whether to trust remote code for custom tokenizers.
            **kwargs: Additional arguments passed to AutoTokenizer.
        
        Returns:
            TokenizerWrapper: Initialized tokenizer wrapper.
        
        Example:
            >>> tokenizer = TokenizerWrapper.from_pretrained("google/gemma-3-270m-it")
            >>> encoded = tokenizer.encode("Hello, world!")
        """
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        
        return cls(tokenizer)
    
    def encode(
        self,
        text: str,
        max_length: int = 512,
        truncation: bool = True,
        padding: bool = False,
        return_tensors: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a single text string.
        
        Args:
            text: Input text to encode.
            max_length: Maximum sequence length.
            truncation: Whether to truncate sequences longer than max_length.
            padding: Whether to pad to max_length.
            return_tensors: Whether to return PyTorch tensors.
        
        Returns:
            Dictionary containing:
            - input_ids: Token ID tensor of shape (seq_len,).
            - attention_mask: Attention mask tensor of shape (seq_len,).
        """
        encoded = self._tokenizer(
            text,
            max_length=max_length,
            truncation=truncation,
            padding="max_length" if padding else False,
            return_tensors="pt" if return_tensors else None,
        )
        
        if return_tensors:
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
            }
        return dict(encoded)
    
    def encode_batch(
        self,
        texts: List[str],
        max_length: int = 512,
        truncation: bool = True,
        padding: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of text strings.
        
        Args:
            texts: List of input texts to encode.
            max_length: Maximum sequence length.
            truncation: Whether to truncate sequences.
            padding: Whether to pad sequences to the same length.
        
        Returns:
            Dictionary containing:
            - input_ids: Token ID tensor of shape (batch_size, seq_len).
            - attention_mask: Attention mask tensor of shape (batch_size, seq_len).
        """
        encoded = self._tokenizer(
            texts,
            max_length=max_length,
            truncation=truncation,
            padding="longest" if padding else False,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
    
    def decode(
        self,
        ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            ids: Token IDs to decode (list or tensor).
            skip_special_tokens: Whether to remove special tokens from output.
        
        Returns:
            Decoded text string.
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
    
    def decode_batch(
        self,
        ids: Union[List[List[int]], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """
        Decode a batch of token ID sequences.
        
        Args:
            ids: Batch of token IDs (list of lists or 2D tensor).
            skip_special_tokens: Whether to remove special tokens.
        
        Returns:
            List of decoded text strings.
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        
        return self._tokenizer.batch_decode(ids, skip_special_tokens=skip_special_tokens)
    
    def get_chat_template(self) -> Optional[str]:
        """
        Get the chat template if available.
        
        Returns:
            Chat template string or None.
        """
        return getattr(self._tokenizer, "chat_template", None)
    
    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> Union[str, Dict[str, torch.Tensor]]:
        """
        Apply chat template to format messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            tokenize: Whether to return tokenized output.
            add_generation_prompt: Whether to add generation prompt.
        
        Returns:
            Formatted string or tokenized dict.
        
        Example:
            >>> messages = [{"role": "user", "content": "Hello!"}]
            >>> formatted = tokenizer.apply_chat_template(messages)
        """
        result = self._tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt" if tokenize else None,
        )
        
        if tokenize:
            return {
                "input_ids": result.squeeze(0) if result.dim() > 1 else result,
            }
        return result
    
    def save_pretrained(self, path: str) -> None:
        """
        Save the tokenizer to disk.
        
        Args:
            path: Directory path to save to.
        """
        self._tokenizer.save_pretrained(path)
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self._tokenizer)
    
    def __call__(self, *args, **kwargs):
        """Passthrough to underlying tokenizer for advanced usage."""
        return self._tokenizer(*args, **kwargs)
