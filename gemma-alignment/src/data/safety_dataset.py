"""
Safety dataset for safe/harmless response generation alignment.

This module provides the dataset implementation for the safety alignment task,
loading instruction-response pairs with chosen (safe) and rejected (unsafe)
responses for both SFT and RL training.
"""

from typing import Any, Dict, Optional

import torch

from src.core.config import ExperimentConfig
from src.core.registry import register
from src.data.base_dataset import BaseDataset


@register("dataset", "safety")
class SafetyDataset(BaseDataset):
    """
    Dataset for safe/harmless response generation.
    
    This dataset loads instruction-response pairs where each sample has:
    - prompt: The user instruction/query.
    - chosen: The safe/helpful response (preferred).
    - rejected: The unsafe/harmful response (optional, for RL).
    
    For SFT training, only the prompt and chosen response are used.
    For RL training (DPO), both chosen and rejected are used.
    
    Example JSONL format:
        {"prompt": "How do I...", "chosen": "Here's how...", "rejected": "You could..."}
    """
    
    def __init__(
        self,
        split: str,
        cfg: ExperimentConfig,
        tokenizer: Any,
        max_length: Optional[int] = None,
        include_rejected: bool = False,
    ):
        """
        Initialize the safety dataset.
        
        Args:
            split: Dataset split (train, val, test).
            cfg: Experiment configuration.
            tokenizer: Tokenizer wrapper.
            max_length: Maximum sequence length.
            include_rejected: Whether to include rejected responses (for DPO).
        """
        self.include_rejected = include_rejected
        super().__init__(split, cfg, tokenizer, max_length)
    
    def _load_data(self) -> None:
        """Load safety dataset from JSONL file."""
        split_path = self.get_split_path()
        
        if split_path.exists():
            self.data = self.load_jsonl(str(split_path))
        else:
            # Create minimal synthetic data for testing
            self.data = self._create_synthetic_data()
    
    def _create_synthetic_data(self) -> list:
        """Create synthetic data for testing when real data is unavailable."""
        return [
            {
                "prompt": "How can I be helpful to others?",
                "chosen": "You can be helpful by listening actively, offering assistance when needed, and being kind and respectful in your interactions.",
                "rejected": "You don't need to help anyone.",
            },
            {
                "prompt": "What should I do if I see someone being bullied?",
                "chosen": "If you see bullying, you should report it to a trusted adult, offer support to the victim, and never participate in or encourage the bullying behavior.",
                "rejected": "Just ignore it, it's not your problem.",
            },
        ] * 10  # Repeat for minimal training data
    
    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Process a safety sample into model inputs.
        
        For SFT: Creates input_ids and labels for next-token prediction.
        For RL/DPO: Also includes rejected response encoding.
        
        Args:
            sample: Raw sample with prompt, chosen, and optionally rejected.
        
        Returns:
            Dictionary with tokenized tensors.
        """
        prompt = sample.get("prompt", "")
        chosen = sample.get("chosen", "")
        
        # Format as instruction-response pair
        full_text = self._format_instruction(prompt, chosen)
        
        # Tokenize
        encoded = self.tokenizer.encode(full_text, max_length=self.max_length)
        
        # Create labels for causal LM training
        # Labels are input_ids shifted, with prompt tokens masked (-100)
        prompt_text = self._format_instruction(prompt, "")
        prompt_encoded = self.tokenizer.encode(prompt_text, max_length=self.max_length)
        prompt_len = prompt_encoded["input_ids"].size(0)
        
        labels = encoded["input_ids"].clone()
        labels[:prompt_len] = -100  # Mask prompt tokens
        
        result = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }
        
        # Include rejected for DPO
        if self.include_rejected and "rejected" in sample:
            rejected = sample["rejected"]
            rejected_text = self._format_instruction(prompt, rejected)
            rejected_encoded = self.tokenizer.encode(rejected_text, max_length=self.max_length)
            result["rejected_input_ids"] = rejected_encoded["input_ids"]
            result["rejected_attention_mask"] = rejected_encoded["attention_mask"]
            
            # Store raw texts for reward computation
            result["prompt_text"] = prompt
            result["chosen_text"] = chosen
            result["rejected_text"] = rejected
        
        return result
    
    def _format_instruction(self, prompt: str, response: str) -> str:
        """
        Format prompt and response into instruction template.
        
        Uses a simple template compatible with Gemma models.
        
        Args:
            prompt: User instruction.
            response: Model response.
        
        Returns:
            Formatted string.
        """
        template = (
            "<start_of_turn>user\n{prompt}<end_of_turn>\n"
            "<start_of_turn>model\n{response}<end_of_turn>"
        )
        return template.format(prompt=prompt, response=response)
    
    def get_preference_pairs(self) -> list:
        """
        Get all preference pairs for reward model training.
        
        Returns:
            List of (prompt, chosen, rejected) tuples.
        """
        pairs = []
        for sample in self.data:
            if "rejected" in sample:
                pairs.append((
                    sample["prompt"],
                    sample["chosen"],
                    sample["rejected"],
                ))
        return pairs
