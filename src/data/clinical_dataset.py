"""Clinical summarization dataset."""

from typing import Any, Dict, Optional

import torch

from src.core.config import ExperimentConfig
from src.core.registry import register
from src.data.base_dataset import BaseDataset


@register("dataset", "clinical")
class ClinicalDataset(BaseDataset):
    """Dataset for clinical conversation summarization."""
    
    def __init__(
        self,
        split: str,
        cfg: ExperimentConfig,
        tokenizer: Any,
        max_length: Optional[int] = None,
        include_rejected: bool = False,
    ):
        self.include_rejected = include_rejected
        super().__init__(split, cfg, tokenizer, max_length)
    
    def _load_data(self) -> None:
        split_path = self.get_split_path()
        
        if split_path.exists():
            self.data = self.load_jsonl(str(split_path))
        else:
            self.data = self._create_synthetic_data()
    
    def _create_synthetic_data(self) -> list:
        return [
            {
                "dialogue": "Doctor: What brings you in today?\nPatient: I've had a headache for three days.",
                "summary": "Patient presents with persistent headache lasting three days.",
            },
            {
                "dialogue": "Doctor: Any other symptoms?\nPatient: Some nausea and light sensitivity.",
                "summary": "Patient reports associated nausea and photophobia.",
            },
        ] * 10
    
    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        dialogue = sample.get("dialogue", "")
        summary = sample.get("summary", "")
        
        full_text = self._format_instruction(dialogue, summary)
        encoded = self.tokenizer.encode(full_text, max_length=self.max_length)
        
        prompt_text = self._format_instruction(dialogue, "")
        prompt_encoded = self.tokenizer.encode(prompt_text, max_length=self.max_length)
        prompt_len = prompt_encoded["input_ids"].size(0)
        
        labels = encoded["input_ids"].clone()
        labels[:prompt_len] = -100
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }
    
    def _format_instruction(self, dialogue: str, summary: str) -> str:
        return (
            f"<start_of_turn>user\nSummarize this medical dialogue:\n{dialogue}<end_of_turn>\n"
            f"<start_of_turn>model\n{summary}<end_of_turn>"
        )
