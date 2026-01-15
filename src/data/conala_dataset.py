"""CoNaLa code Q&A dataset."""

from typing import Any, Dict, Optional

import torch

from src.core.config import ExperimentConfig
from src.core.registry import register
from src.data.base_dataset import BaseDataset


@register("dataset", "conala")
class CoNaLaDataset(BaseDataset):
    """Dataset for natural language to code generation."""
    
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
                "intent": "sort a list of dictionaries by key",
                "snippet": "sorted(my_list, key=lambda x: x['key'])",
            },
            {
                "intent": "reverse a string",
                "snippet": "my_string[::-1]",
            },
        ] * 10
    
    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        intent = sample.get("intent", "")
        snippet = sample.get("snippet", "")
        
        full_text = self._format_instruction(intent, snippet)
        encoded = self.tokenizer.encode(full_text, max_length=self.max_length)
        
        prompt_text = self._format_instruction(intent, "")
        prompt_encoded = self.tokenizer.encode(prompt_text, max_length=self.max_length)
        prompt_len = prompt_encoded["input_ids"].size(0)
        
        labels = encoded["input_ids"].clone()
        labels[:prompt_len] = -100
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }
    
    def _format_instruction(self, intent: str, snippet: str) -> str:
        return (
            f"<start_of_turn>user\nWrite Python code: {intent}<end_of_turn>\n"
            f"<start_of_turn>model\n{snippet}<end_of_turn>"
        )
