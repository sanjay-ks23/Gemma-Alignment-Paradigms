"""
Clinical dataset for medical conversation summarization.

This module provides a stub dataset implementation for the MTS-Dialog
clinical summarization task.
"""

from typing import Any, Dict, Optional

import torch

from src.core.config import ExperimentConfig
from src.core.registry import register
from src.data.base_dataset import BaseDataset


@register("dataset", "clinical")
class ClinicalDataset(BaseDataset):
    """
    Dataset for clinical conversation summarization.
    
    This dataset loads doctor-patient dialogues with corresponding summaries
    for training summarization models in the medical domain.
    
    Expected JSONL format:
        {"dialogue": "Doctor: ... Patient: ...", "summary": "The patient..."}
    
    Note:
        This is a stub implementation. The actual MTS-Dialog dataset requires
        separate download and preprocessing. See data/README.md for details.
    """
    
    def __init__(
        self,
        split: str,
        cfg: ExperimentConfig,
        tokenizer: Any,
        max_length: Optional[int] = None,
    ):
        """
        Initialize the clinical dataset.
        
        Args:
            split: Dataset split (train, val, test).
            cfg: Experiment configuration.
            tokenizer: Tokenizer wrapper.
            max_length: Maximum sequence length.
        """
        super().__init__(split, cfg, tokenizer, max_length)
    
    def _load_data(self) -> None:
        """Load clinical dataset from JSONL file."""
        split_path = self.get_split_path()
        
        if split_path.exists():
            self.data = self.load_jsonl(str(split_path))
        else:
            # Create minimal synthetic data for testing
            self.data = self._create_synthetic_data()
    
    def _create_synthetic_data(self) -> list:
        """Create synthetic clinical data for testing."""
        return [
            {
                "dialogue": (
                    "Doctor: What brings you in today?\n"
                    "Patient: I've been having headaches for about a week.\n"
                    "Doctor: Can you describe the pain?\n"
                    "Patient: It's a dull ache, mostly in my forehead."
                ),
                "summary": (
                    "Patient presents with one week of dull frontal headaches. "
                    "No additional symptoms reported."
                ),
            },
            {
                "dialogue": (
                    "Doctor: How are you feeling today?\n"
                    "Patient: I have a cough that won't go away.\n"
                    "Doctor: How long have you had it?\n"
                    "Patient: About two weeks now."
                ),
                "summary": (
                    "Patient reports persistent cough for two weeks. "
                    "Further evaluation recommended."
                ),
            },
        ] * 10
    
    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Process a clinical sample into model inputs.
        
        Args:
            sample: Raw sample with dialogue and summary.
        
        Returns:
            Dictionary with tokenized tensors.
        """
        dialogue = sample.get("dialogue", "")
        summary = sample.get("summary", "")
        
        # Format for summarization
        full_text = self._format_summarization(dialogue, summary)
        
        # Tokenize
        encoded = self.tokenizer.encode(full_text, max_length=self.max_length)
        
        # Create labels with prompt masking
        prompt_text = self._format_summarization(dialogue, "")
        prompt_encoded = self.tokenizer.encode(prompt_text, max_length=self.max_length)
        prompt_len = prompt_encoded["input_ids"].size(0)
        
        labels = encoded["input_ids"].clone()
        labels[:prompt_len] = -100
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }
    
    def _format_summarization(self, dialogue: str, summary: str) -> str:
        """Format dialogue and summary for the model."""
        template = (
            "<start_of_turn>user\n"
            "Summarize the following clinical dialogue:\n\n"
            "{dialogue}<end_of_turn>\n"
            "<start_of_turn>model\n{summary}<end_of_turn>"
        )
        return template.format(dialogue=dialogue, summary=summary)
