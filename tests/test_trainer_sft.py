"""
Tests for SFT trainer.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.core.config import ExperimentConfig
from src.trainers.sft_trainer import SFTTrainer


class MockModel(nn.Module):
    """Mock model for testing."""
    
    def __init__(self, vocab_size: int = 1000, hidden_size: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.config = MagicMock()
        self.config.pad_token_id = 0
        self.config.eos_token_id = 1
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        embeddings = self.embedding(input_ids)
        logits = self.linear(embeddings)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        output = MagicMock()
        output.logits = logits
        output.loss = loss
        return output
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
    
    def load(self, path):
        self.load_state_dict(
            torch.load(os.path.join(path, "model.pt"), weights_only=True)
        )
    
    def prepare_for_training(self):
        self.train()
    
    def gradient_checkpointing_enable(self):
        pass


class MockDataLoader:
    """Mock dataloader for testing."""
    
    def __init__(self, batch_size: int = 4, num_batches: int = 3):
        self.batch_size = batch_size
        self.num_batches = num_batches
    
    def __iter__(self):
        for _ in range(self.num_batches):
            yield {
                "input_ids": torch.randint(2, 100, (self.batch_size, 32)),
                "attention_mask": torch.ones(self.batch_size, 32, dtype=torch.long),
                "labels": torch.randint(2, 100, (self.batch_size, 32)),
            }
    
    def __len__(self):
        return self.num_batches


class TestSFTTrainer:
    """Tests for SFTTrainer."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        cfg = ExperimentConfig()
        cfg.training.epochs = 1
        cfg.training.batch_size = 4
        cfg.training.gradient_accumulation_steps = 1
        cfg.training.learning_rate = 1e-4
        cfg.training.fp16 = False
        cfg.training.bf16 = False
        cfg.logging.log_interval = 1
        cfg.debug = True
        cfg.device = "cpu"
        return cfg
    
    @pytest.fixture
    def trainer(self, config, tmp_path):
        """Create trainer instance."""
        config.logging.output_dir = str(tmp_path / "outputs")
        config.logging.checkpoint_dir = str(tmp_path / "checkpoints")
        
        model = MockModel()
        train_loader = MockDataLoader()
        val_loader = MockDataLoader(num_batches=1)
        
        return SFTTrainer(
            model=model,
            cfg=config,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
        )
    
    def test_train_step_returns_loss(self, trainer):
        """Test that train_step returns a float loss."""
        batch = {
            "input_ids": torch.randint(2, 100, (4, 32)),
            "attention_mask": torch.ones(4, 32, dtype=torch.long),
            "labels": torch.randint(2, 100, (4, 32)),
        }
        
        loss = trainer.train_step(batch)
        
        assert isinstance(loss, float)
        assert not torch.isnan(torch.tensor(loss))
        assert not torch.isinf(torch.tensor(loss))
    
    def test_train_step_loss_is_finite(self, trainer):
        """Test that loss is finite for valid input."""
        batch = {
            "input_ids": torch.randint(2, 100, (4, 32)),
            "attention_mask": torch.ones(4, 32, dtype=torch.long),
            "labels": torch.randint(2, 100, (4, 32)),
        }
        
        for _ in range(3):
            loss = trainer.train_step(batch)
            assert loss < float("inf")
            assert loss > 0
    
    def test_train_runs_to_completion(self, trainer):
        """Test that training loop completes."""
        results = trainer.train()
        
        assert "train_losses" in results
        assert len(results["train_losses"]) > 0
        assert results["total_steps"] > 0
    
    def test_save_checkpoint(self, trainer, tmp_path):
        """Test checkpoint saving."""
        checkpoint_path = tmp_path / "test_checkpoint"
        
        trainer.save_checkpoint(str(checkpoint_path))
        
        assert checkpoint_path.exists()
        assert (checkpoint_path / "trainer_state.pt").exists()
        assert (checkpoint_path / "model").exists()
    
    def test_load_checkpoint(self, trainer, tmp_path):
        """Test checkpoint loading."""
        checkpoint_path = tmp_path / "test_checkpoint"
        
        # Train a bit and save
        trainer.global_step = 100
        trainer.current_epoch = 2
        trainer.save_checkpoint(str(checkpoint_path))
        
        # Reset and load
        trainer.global_step = 0
        trainer.current_epoch = 0
        trainer.load_checkpoint(str(checkpoint_path))
        
        assert trainer.global_step == 100
        assert trainer.current_epoch == 2


class TestSFTTrainerWithPEFT:
    """Tests for SFTTrainer with PEFT adapters."""
    
    def test_export_lora_checkpoint_without_adapter(self, tmp_path):
        """Test export_lora_checkpoint when no adapter is set."""
        cfg = ExperimentConfig()
        cfg.debug = True
        cfg.device = "cpu"
        cfg.logging.checkpoint_dir = str(tmp_path)
        
        model = MockModel()
        train_loader = MockDataLoader()
        
        trainer = SFTTrainer(
            model=model,
            cfg=cfg,
            train_dataloader=train_loader,
            peft_adapter=None,
        )
        
        # Should not raise, just log warning
        trainer.export_lora_checkpoint(str(tmp_path / "lora"))
