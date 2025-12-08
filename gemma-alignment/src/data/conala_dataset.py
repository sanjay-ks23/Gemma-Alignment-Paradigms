"""
CoNaLa dataset for code question-answering.

This module provides a stub dataset implementation for the CoNaLa
(Code/Natural Language Challenge) code generation task.
"""

from typing import Any, Dict, Optional

import torch

from src.core.config import ExperimentConfig
from src.core.registry import register
from src.data.base_dataset import BaseDataset


@register("dataset", "conala")
class CoNaLaDataset(BaseDataset):
    """
    Dataset for code question-answering and generation.
    
    This dataset loads natural language queries with corresponding Python
    code snippets for training code generation models.
    
    Expected JSONL format:
        {"intent": "sort a list in reverse", "snippet": "sorted(lst, reverse=True)"}
    
    Note:
        This is a stub implementation. The actual CoNaLa dataset requires
        separate download. See data/README.md for details.
    """
    
    def __init__(
        self,
        split: str,
        cfg: ExperimentConfig,
        tokenizer: Any,
        max_length: Optional[int] = None,
    ):
        """
        Initialize the CoNaLa dataset.
        
        Args:
            split: Dataset split (train, val, test).
            cfg: Experiment configuration.
            tokenizer: Tokenizer wrapper.
            max_length: Maximum sequence length.
        """
        super().__init__(split, cfg, tokenizer, max_length)
    
    def _load_data(self) -> None:
        """Load CoNaLa dataset from JSONL file."""
        split_path = self.get_split_path()
        
        if split_path.exists():
            self.data = self.load_jsonl(str(split_path))
        else:
            # Create minimal synthetic data for testing
            self.data = self._create_synthetic_data()
    
    def _create_synthetic_data(self) -> list:
        """Create synthetic code data for testing."""
        return [
            {
                "intent": "sort a list in descending order",
                "snippet": "sorted(my_list, reverse=True)",
            },
            {
                "intent": "read a file line by line",
                "snippet": "with open('file.txt', 'r') as f:\n    for line in f:\n        print(line)",
            },
            {
                "intent": "get the current date and time",
                "snippet": "from datetime import datetime\ncurrent_time = datetime.now()",
            },
            {
                "intent": "convert a string to lowercase",
                "snippet": "my_string.lower()",
            },
        ] * 5
    
    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Process a CoNaLa sample into model inputs.
        
        Args:
            sample: Raw sample with intent and snippet.
        
        Returns:
            Dictionary with tokenized tensors.
        """
        intent = sample.get("intent", "")
        snippet = sample.get("snippet", "")
        
        # Format for code generation
        full_text = self._format_code_generation(intent, snippet)
        
        # Tokenize
        encoded = self.tokenizer.encode(full_text, max_length=self.max_length)
        
        # Create labels with prompt masking
        prompt_text = self._format_code_generation(intent, "")
        prompt_encoded = self.tokenizer.encode(prompt_text, max_length=self.max_length)
        prompt_len = prompt_encoded["input_ids"].size(0)
        
        labels = encoded["input_ids"].clone()
        labels[:prompt_len] = -100
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }
    
    def _format_code_generation(self, intent: str, snippet: str) -> str:
        """Format intent and code snippet for the model."""
        template = (
            "<start_of_turn>user\n"
            "Write Python code to: {intent}<end_of_turn>\n"
            "<start_of_turn>model\n```python\n{snippet}\n```<end_of_turn>"
        )
        return template.format(intent=intent, snippet=snippet)
