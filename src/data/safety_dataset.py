"""Safety dataset for safe/harmless response generation alignment."""

from typing import Any, Dict, Optional

import torch

from src.core.config import ExperimentConfig
from src.core.registry import register
from src.data.base_dataset import BaseDataset


@register("dataset", "safety")
class SafetyDataset(BaseDataset):
    """Dataset for safe/harmless response generation."""
    
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
                "prompt": "How can I be helpful to others?",
                "chosen": "You can be helpful by listening actively, offering assistance when needed, and being kind and respectful in your interactions.",
                "rejected": "You don't need to help anyone.",
            },
            {
                "prompt": "What should I do if I see someone being bullied?",
                "chosen": "If you see bullying, you should report it to a trusted adult, offer support to the victim, and never participate in or encourage the bullying behavior.",
                "rejected": "Just ignore it, it's not your problem.",
            },
        ] * 10
    
    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        prompt = sample.get("prompt", "")
        chosen = sample.get("chosen", "")
        
        full_text = self._format_instruction(prompt, chosen)
        encoded = self.tokenizer.encode(full_text, max_length=self.max_length)
        
        prompt_text = self._format_instruction(prompt, "")
        prompt_encoded = self.tokenizer.encode(prompt_text, max_length=self.max_length)
        prompt_len = prompt_encoded["input_ids"].size(0)
        
        labels = encoded["input_ids"].clone()
        labels[:prompt_len] = -100
        
        result = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }
        
        if self.include_rejected and "rejected" in sample:
            rejected = sample["rejected"]
            rejected_text = self._format_instruction(prompt, rejected)
            rejected_encoded = self.tokenizer.encode(rejected_text, max_length=self.max_length)
            result["rejected_input_ids"] = rejected_encoded["input_ids"]
            result["rejected_attention_mask"] = rejected_encoded["attention_mask"]
            result["prompt_text"] = prompt
            result["chosen_text"] = chosen
            result["rejected_text"] = rejected
        
        return result
    
    def _format_instruction(self, prompt: str, response: str) -> str:
        return (
            f"<start_of_turn>user\n{prompt}<end_of_turn>\n"
            f"<start_of_turn>model\n{response}<end_of_turn>"
        )
    
    def get_preference_pairs(self) -> list:
        return [
            (s["prompt"], s["chosen"], s["rejected"])
            for s in self.data if "rejected" in s
        ]
