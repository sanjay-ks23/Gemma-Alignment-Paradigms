"""
Tests for dataset classes.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from src.core.config import ExperimentConfig
from src.data.base_dataset import BaseDataset
from src.data.safety_dataset import SafetyDataset


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
    
    def encode(self, text: str, max_length: int = 512) -> dict:
        # Simple mock encoding
        token_count = min(len(text.split()) * 2, max_length)
        return {
            "input_ids": torch.randint(2, self.vocab_size, (token_count,)),
            "attention_mask": torch.ones(token_count, dtype=torch.long),
        }
    
    def encode_batch(self, texts: list, max_length: int = 512) -> dict:
        encoded = [self.encode(t, max_length) for t in texts]
        max_len = max(e["input_ids"].size(0) for e in encoded)
        
        input_ids = torch.zeros(len(texts), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(texts), max_len, dtype=torch.long)
        
        for i, e in enumerate(encoded):
            seq_len = e["input_ids"].size(0)
            input_ids[i, :seq_len] = e["input_ids"]
            attention_mask[i, :seq_len] = e["attention_mask"]
        
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class TestBaseDataset:
    """Tests for BaseDataset functionality."""
    
    def test_load_jsonl(self, tmp_path):
        """Test JSONL loading utility."""
        data = [
            {"prompt": "Hello", "response": "Hi there"},
            {"prompt": "How are you?", "response": "I'm good"},
        ]
        
        jsonl_file = tmp_path / "test.jsonl"
        with open(jsonl_file, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        
        loaded = BaseDataset.load_jsonl(str(jsonl_file))
        
        assert len(loaded) == 2
        assert loaded[0]["prompt"] == "Hello"
        assert loaded[1]["response"] == "I'm good"
    
    def test_collate_fn_padding(self):
        """Test that collate_fn properly pads sequences."""
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([1, 2, 3]),
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([4, 5]),
            },
        ]
        
        collated = BaseDataset.collate_fn(batch)
        
        assert collated["input_ids"].shape == (2, 3)
        assert collated["attention_mask"].shape == (2, 3)
        assert collated["labels"].shape == (2, 3)
        
        # Check padding
        assert collated["input_ids"][1, 2] == 0  # Padded with 0
        assert collated["labels"][1, 2] == -100  # Labels padded with -100


class TestSafetyDataset:
    """Tests for SafetyDataset."""
    
    @pytest.fixture
    def sample_data_dir(self, tmp_path):
        """Create sample safety dataset."""
        safety_dir = tmp_path / "safety"
        safety_dir.mkdir()
        
        data = [
            {
                "prompt": "How can I help others?",
                "chosen": "You can volunteer or donate.",
                "rejected": "Don't help anyone.",
            },
            {
                "prompt": "What is kindness?",
                "chosen": "Kindness is being nice to others.",
                "rejected": "Kindness is weakness.",
            },
        ]
        
        for split in ["train", "val", "test"]:
            split_file = safety_dir / f"{split}.jsonl"
            with open(split_file, "w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
        
        return tmp_path
    
    def test_dataset_length(self, sample_data_dir):
        """Test that dataset reports correct length."""
        cfg = ExperimentConfig()
        cfg.dataset_path = str(sample_data_dir)
        cfg.task = "safety"
        
        tokenizer = MockTokenizer()
        dataset = SafetyDataset("train", cfg, tokenizer)
        
        assert len(dataset) == 2
    
    def test_dataset_item_schema(self, sample_data_dir):
        """Test that items have expected keys."""
        cfg = ExperimentConfig()
        cfg.dataset_path = str(sample_data_dir)
        cfg.task = "safety"
        
        tokenizer = MockTokenizer()
        dataset = SafetyDataset("train", cfg, tokenizer)
        
        item = dataset[0]
        
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)
    
    def test_dataset_with_rejected(self, sample_data_dir):
        """Test dataset includes rejected when requested."""
        cfg = ExperimentConfig()
        cfg.dataset_path = str(sample_data_dir)
        cfg.task = "safety"
        
        tokenizer = MockTokenizer()
        dataset = SafetyDataset("train", cfg, tokenizer, include_rejected=True)
        
        item = dataset[0]
        
        assert "rejected_input_ids" in item
        assert "rejected_attention_mask" in item
    
    def test_get_preference_pairs(self, sample_data_dir):
        """Test extraction of preference pairs."""
        cfg = ExperimentConfig()
        cfg.dataset_path = str(sample_data_dir)
        cfg.task = "safety"
        
        tokenizer = MockTokenizer()
        dataset = SafetyDataset("train", cfg, tokenizer)
        
        pairs = dataset.get_preference_pairs()
        
        assert len(pairs) == 2
        prompt, chosen, rejected = pairs[0]
        assert prompt == "How can I help others?"
        assert "volunteer" in chosen
    
    def test_synthetic_data_fallback(self, tmp_path):
        """Test that synthetic data is created when file doesn't exist."""
        cfg = ExperimentConfig()
        cfg.dataset_path = str(tmp_path)  # No data files here
        cfg.task = "safety"
        
        tokenizer = MockTokenizer()
        dataset = SafetyDataset("train", cfg, tokenizer)
        
        # Should have synthetic data
        assert len(dataset) > 0
    
    def test_dataloader_creation(self, sample_data_dir):
        """Test DataLoader creation."""
        cfg = ExperimentConfig()
        cfg.dataset_path = str(sample_data_dir)
        cfg.task = "safety"
        
        tokenizer = MockTokenizer()
        dataset = SafetyDataset("train", cfg, tokenizer)
        
        loader = dataset.get_dataloader(batch_size=2, shuffle=False)
        
        batch = next(iter(loader))
        
        assert "input_ids" in batch
        assert batch["input_ids"].shape[0] == 2
