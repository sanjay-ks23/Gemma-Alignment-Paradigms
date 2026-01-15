"""
Tests for configuration loading and validation.
"""

import os
import tempfile
from pathlib import Path

import pytest

from src.core.config import (
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    RLConfig,
    RewardConfig,
    LoggingConfig,
    load_config,
    save_config,
)


class TestConfigDataclasses:
    """Tests for configuration dataclasses."""
    
    def test_experiment_config_defaults(self):
        """Test that ExperimentConfig has sensible defaults."""
        cfg = ExperimentConfig()
        
        assert cfg.task == "safety"
        assert cfg.seed == 42
        assert cfg.device == "auto"
        assert cfg.debug is False
    
    def test_model_config_defaults(self):
        """Test ModelConfig defaults."""
        cfg = ModelConfig()
        
        assert cfg.base_checkpoint == "google/gemma-3-270m-it"
        assert cfg.size == "270m"
        assert cfg.peft_type == "lora"
        assert cfg.peft_rank == 8
        assert cfg.peft_alpha == 16
    
    def test_training_config_defaults(self):
        """Test TrainingConfig defaults."""
        cfg = TrainingConfig()
        
        assert cfg.mode == "sft"
        assert cfg.epochs == 3
        assert cfg.batch_size == 4
        assert cfg.learning_rate == 5e-5
    
    def test_rl_config_defaults(self):
        """Test RLConfig defaults."""
        cfg = RLConfig()
        
        assert cfg.algorithm == "ppo"
        assert cfg.ppo_clip == 0.2
        assert cfg.rollout_size == 64


class TestConfigLoading:
    """Tests for loading configs from YAML files."""
    
    def test_load_simple_config(self, tmp_path):
        """Test loading a simple config file."""
        config_content = """
task: safety
seed: 123
device: cpu
model:
  base_checkpoint: google/gemma-3-270m-it
  size: 270m
  peft_type: lora
training:
  mode: sft
  epochs: 2
  batch_size: 8
"""
        config_file = tmp_path / "test_config.yml"
        config_file.write_text(config_content)
        
        cfg = load_config(str(config_file))
        
        assert cfg.task == "safety"
        assert cfg.seed == 123
        assert cfg.device == "cpu"
        assert cfg.model.base_checkpoint == "google/gemma-3-270m-it"
        assert cfg.training.mode == "sft"
        assert cfg.training.epochs == 2
        assert cfg.training.batch_size == 8
    
    def test_load_config_with_defaults(self, tmp_path):
        """Test loading config that overrides defaults."""
        defaults_content = """
task: safety
seed: 42
model:
  base_checkpoint: google/gemma-3-270m-it
  peft_type: lora
training:
  mode: sft
  epochs: 3
"""
        main_content = """
seed: 100
training:
  epochs: 5
"""
        defaults_file = tmp_path / "defaults.yml"
        defaults_file.write_text(defaults_content)
        
        main_file = tmp_path / "main.yml"
        main_file.write_text(main_content)
        
        cfg = load_config(str(main_file), defaults_path=str(defaults_file))
        
        # Should override
        assert cfg.seed == 100
        assert cfg.training.epochs == 5
        
        # Should inherit from defaults
        assert cfg.task == "safety"
    
    def test_load_config_file_not_found(self):
        """Test that missing config raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yml")
    
    def test_load_rl_config(self, tmp_path):
        """Test loading RL-specific config."""
        config_content = """
task: safety
training:
  mode: rl
rl:
  algorithm: ppo
  ppo_clip: 0.1
  rollout_size: 32
  kl_coef: 0.05
"""
        config_file = tmp_path / "rl_config.yml"
        config_file.write_text(config_content)
        
        cfg = load_config(str(config_file))
        
        assert cfg.training.mode == "rl"
        assert cfg.rl.algorithm == "ppo"
        assert cfg.rl.ppo_clip == 0.1
        assert cfg.rl.rollout_size == 32
        assert cfg.rl.kl_coef == 0.05


class TestConfigSaving:
    """Tests for saving configs to YAML files."""
    
    def test_save_and_reload_config(self, tmp_path):
        """Test that saved config can be reloaded."""
        original = ExperimentConfig(
            task="clinical",
            seed=999,
            device="cuda",
        )
        original.model.peft_rank = 16
        original.training.epochs = 5
        
        save_path = tmp_path / "saved_config.yml"
        save_config(original, str(save_path))
        
        loaded = load_config(str(save_path))
        
        assert loaded.task == "clinical"
        assert loaded.seed == 999
        assert loaded.device == "cuda"
