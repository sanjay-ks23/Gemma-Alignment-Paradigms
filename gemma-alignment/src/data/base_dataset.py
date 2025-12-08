"""
Base dataset class providing common functionality for all alignment datasets.

This module defines the abstract interface and shared utilities for dataset
implementations used in SFT and RL training.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
from torch.utils.data import Dataset

from src.core.config import ExperimentConfig


class BaseDataset(Dataset, ABC):
    """
    Abstract base class for alignment datasets.
    
    All dataset implementations should inherit from this class and implement
    the abstract methods to provide task-specific data loading and processing.
    
    Attributes:
        split: Dataset split (train, val, test).
        cfg: Experiment configuration.
        tokenizer: Tokenizer wrapper for encoding text.
        data: List of loaded data samples.
        max_length: Maximum sequence length for tokenization.
    """
    
    def __init__(
        self,
        split: str,
        cfg: ExperimentConfig,
        tokenizer: Any,
        max_length: Optional[int] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            split: Dataset split to load (train, val, test).
            cfg: Experiment configuration.
            tokenizer: Tokenizer wrapper for encoding text.
            max_length: Maximum sequence length. Defaults to cfg.training.max_seq_length.
        """
        self.split = split
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.max_length = max_length or cfg.training.max_seq_length
        self.data: List[Dict[str, Any]] = []
        
        self._load_data()
    
    @abstractmethod
    def _load_data(self) -> None:
        """
        Load data from disk into self.data.
        
        This method should read the appropriate split file and populate
        self.data with a list of dictionaries containing the raw data.
        """
        pass
    
    @abstractmethod
    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single sample into model inputs.
        
        Args:
            sample: Raw data sample from self.data.
        
        Returns:
            Dictionary containing tokenized model inputs with keys:
            - input_ids: Token IDs tensor.
            - attention_mask: Attention mask tensor.
            - labels: Label tensor (for SFT) or reference text (for RL).
        """
        pass
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single processed sample by index.
        
        Args:
            idx: Sample index.
        
        Returns:
            Dictionary with tokenized model inputs.
        """
        sample = self.data[idx]
        return self._process_sample(sample)
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over all samples."""
        for idx in range(len(self)):
            yield self[idx]
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples into batched tensors.
        
        This method handles padding and stacking of variable-length sequences.
        
        Args:
            batch: List of sample dictionaries from __getitem__.
        
        Returns:
            Dictionary with batched and padded tensors.
        """
        if not batch:
            return {}
        
        collated = {}
        keys = batch[0].keys()
        
        for key in keys:
            values = [sample[key] for sample in batch]
            
            # Skip non-tensor values
            if not isinstance(values[0], torch.Tensor):
                collated[key] = values
                continue
            
            # Find max length in batch
            if values[0].dim() == 1:
                max_len = max(v.size(0) for v in values)
                
                # Pad sequences to max length
                padded = []
                for v in values:
                    if v.size(0) < max_len:
                        # Pad with 0 for input_ids/attention_mask, -100 for labels
                        pad_value = -100 if key == "labels" else 0
                        padding = torch.full(
                            (max_len - v.size(0),),
                            pad_value,
                            dtype=v.dtype,
                        )
                        v = torch.cat([v, padding])
                    padded.append(v)
                
                collated[key] = torch.stack(padded)
            else:
                # For multi-dimensional tensors, just stack
                collated[key] = torch.stack(values)
        
        return collated
    
    def get_dataloader(
        self,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:
        """
        Create a DataLoader for this dataset.
        
        Args:
            batch_size: Batch size.
            shuffle: Whether to shuffle the data.
            num_workers: Number of worker processes.
        
        Returns:
            DataLoader instance.
        """
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
        """
        Load data from a JSONL file.
        
        Args:
            path: Path to JSONL file.
        
        Returns:
            List of dictionaries, one per line.
        """
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def get_split_path(self) -> Path:
        """
        Get the path to the split file for this dataset.
        
        Returns:
            Path to the split file.
        """
        base_path = Path(self.cfg.dataset_path)
        task_path = base_path / self.cfg.task
        return task_path / f"{self.split}.jsonl"
