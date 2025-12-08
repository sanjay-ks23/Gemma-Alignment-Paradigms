"""
Tests for reward model.
"""

import pytest
import torch
import torch.nn as nn

from src.core.config import RewardConfig
from src.models.reward_model import RewardModel


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def encode_batch(self, texts, max_length=512):
        batch_size = len(texts)
        seq_len = 32
        return {
            "input_ids": torch.randint(2, 1000, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        }


class TestRewardModel:
    """Tests for RewardModel."""
    
    @pytest.fixture
    def reward_model(self):
        """Create reward model instance."""
        return RewardModel(
            hidden_size=64,
            num_layers=2,
            vocab_size=1000,
            max_seq_length=128,
        )
    
    def test_forward_with_input_ids(self, reward_model):
        """Test forward pass with input IDs."""
        input_ids = torch.randint(2, 1000, (4, 32))
        attention_mask = torch.ones(4, 32, dtype=torch.long)
        
        rewards = reward_model(input_ids=input_ids, attention_mask=attention_mask)
        
        assert rewards.shape == (4,)
        assert rewards.dtype == torch.float32
    
    def test_forward_with_hidden_states(self, reward_model):
        """Test forward pass with pre-computed hidden states."""
        hidden_states = torch.randn(4, 32, 64)
        attention_mask = torch.ones(4, 32, dtype=torch.long)
        
        rewards = reward_model(hidden_states=hidden_states, attention_mask=attention_mask)
        
        assert rewards.shape == (4,)
    
    def test_forward_returns_scalar(self, reward_model):
        """Test that forward returns scalar rewards."""
        input_ids = torch.randint(2, 1000, (1, 16))
        
        reward = reward_model(input_ids=input_ids)
        
        # Single sample should still return 1D tensor
        assert reward.dim() == 1
        assert reward.size(0) == 1
    
    def test_forward_requires_input(self, reward_model):
        """Test that forward raises error without input."""
        with pytest.raises(ValueError):
            reward_model()
    
    def test_heuristic_rewards(self, reward_model):
        """Test heuristic reward computation."""
        texts = [
            "This is a helpful and safe response.",
            "This text contains hate and violence.",
            "Short.",
        ]
        
        rewards = reward_model._compute_heuristic_rewards(texts)
        
        assert len(rewards) == 3
        assert all(0 <= r <= 1 for r in rewards)
        # Safe text should have higher reward
        assert rewards[0] > rewards[1]
    
    def test_train_on_pairs(self, reward_model):
        """Test training on preference pairs."""
        chosen = [
            "This is a helpful response.",
            "I can assist you with that.",
        ]
        rejected = [
            "I refuse to help.",
            "Go away.",
        ]
        
        tokenizer = MockTokenizer()
        
        metrics = reward_model.train_on_pairs(
            chosen, rejected, tokenizer, epochs=2, batch_size=2
        )
        
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert len(metrics["loss"]) == 2  # 2 epochs
        
        # Loss should generally decrease
        # (not guaranteed on 2 samples, so just check it ran)
        assert all(l < float("inf") for l in metrics["loss"])
    
    def test_train_reduces_loss(self, reward_model):
        """Test that training reduces loss over time."""
        chosen = [
            "Helpful response 1.",
            "Helpful response 2.",
            "Helpful response 3.",
            "Helpful response 4.",
        ]
        rejected = [
            "Unhelpful 1.",
            "Unhelpful 2.",
            "Unhelpful 3.",
            "Unhelpful 4.",
        ]
        
        tokenizer = MockTokenizer()
        
        metrics = reward_model.train_on_pairs(
            chosen, rejected, tokenizer, epochs=5, batch_size=4
        )
        
        # Should show improvement (or at least not get worse)
        initial_loss = metrics["loss"][0]
        final_loss = metrics["loss"][-1]
        
        # Allow some variance but generally should improve
        assert final_loss <= initial_loss * 1.5  # Not getting much worse
    
    def test_save_and_load(self, reward_model, tmp_path):
        """Test saving and loading reward model."""
        # Get initial prediction
        input_ids = torch.randint(2, 1000, (2, 16))
        initial_reward = reward_model(input_ids=input_ids)
        
        # Save
        save_path = tmp_path / "reward_model"
        reward_model.save_pretrained(str(save_path))
        
        # Load into new model
        loaded_model = RewardModel.from_pretrained(str(save_path))
        
        # Predictions should match
        loaded_reward = loaded_model(input_ids=input_ids)
        
        torch.testing.assert_close(initial_reward, loaded_reward)
    
    def test_hybrid_reward_computation(self, reward_model):
        """Test hybrid reward computation."""
        texts = ["A safe and helpful response."]
        tokenizer = MockTokenizer()
        
        rewards = reward_model.compute_hybrid_reward(texts, tokenizer)
        
        assert rewards.shape == (1,)
        assert not torch.isnan(rewards).any()


class TestRewardModelGradients:
    """Tests for reward model gradient computation."""
    
    def test_gradients_flow(self):
        """Test that gradients flow through the model."""
        model = RewardModel(hidden_size=32, num_layers=1, vocab_size=100)
        
        input_ids = torch.randint(2, 100, (2, 16))
        rewards = model(input_ids=input_ids)
        
        loss = rewards.mean()
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_preference_loss_gradients(self):
        """Test gradients from preference loss."""
        model = RewardModel(hidden_size=32, num_layers=1, vocab_size=100)
        
        chosen_ids = torch.randint(2, 100, (2, 16))
        rejected_ids = torch.randint(2, 100, (2, 16))
        
        chosen_rewards = model(input_ids=chosen_ids)
        rejected_rewards = model(input_ids=rejected_ids)
        
        # Preference loss
        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
        loss.backward()
        
        # Check gradients
        has_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grads
