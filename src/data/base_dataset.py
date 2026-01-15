"""Base dataset class for alignment tasks."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
from torch.utils.data import Dataset

from src.core.config import ExperimentConfig


class BaseDataset(Dataset, ABC):
    """Abstract base class for alignment datasets."""
    
    def __init__(
        self,
        split: str,
        cfg: ExperimentConfig,
        tokenizer: Any,
        max_length: Optional[int] = None,
    ):
        self.split = split
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.max_length = max_length or cfg.training.max_seq_length
        self.data: List[Dict[str, Any]] = []
        self._load_data()
    
    @abstractmethod
    def _load_data(self) -> None:
        pass
    
    @abstractmethod
    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._process_sample(self.data[idx])
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for idx in range(len(self)):
            yield self[idx]
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not batch:
            return {}
        
        collated = {}
        for key in batch[0].keys():
            values = [sample[key] for sample in batch]
            
            if not isinstance(values[0], torch.Tensor):
                collated[key] = values
                continue
            
            if values[0].dim() == 1:
                max_len = max(v.size(0) for v in values)
                padded = []
                for v in values:
                    if v.size(0) < max_len:
                        pad_value = -100 if key == "labels" else 0
                        padding = torch.full((max_len - v.size(0),), pad_value, dtype=v.dtype)
                        v = torch.cat([v, padding])
                    padded.append(v)
                collated[key] = torch.stack(padded)
            else:
                collated[key] = torch.stack(values)
        
        return collated
    
    def get_dataloader(
        self,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=torch.cuda.is_available(),
        )
    
    @classmethod
    def load_jsonl(cls, path: str) -> List[Dict[str, Any]]:
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def get_split_path(self) -> Path:
        return Path(self.cfg.dataset_path) / self.cfg.task / f"{self.split}.jsonl"
