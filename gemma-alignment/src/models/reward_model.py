"""
Reward model for RL-based alignment training.

This module provides the RewardModel class that computes scalar rewards
for model outputs, supporting both trainable and heuristic approaches.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.config import RewardConfig


class RewardModel(nn.Module):
    """
    Reward model for computing scalar rewards from text or embeddings.
    
    The reward model supports three modes:
    - Trainable: Neural network trained on preference pairs.
    - Heuristic: Rule-based rewards (toxicity, length, repetition).
    - Hybrid: Combination of trainable and heuristic signals.
    
    Architecture:
    - Transformer encoder (if from_scratch) or frozen encoder backbone.
    - Mean pooling over sequence.
    - MLP head producing scalar reward.
    
    Example:
        >>> reward_model = RewardModel(hidden_size=768, num_layers=3)
        >>> rewards = reward_model(hidden_states)  # Returns scalar per sample
        >>> reward_model.train_on_pairs(chosen_texts, rejected_texts, tokenizer)
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_layers: int = 3,
        vocab_size: int = 256000,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        reward_config: Optional[RewardConfig] = None,
    ):
        """
        Initialize the reward model.
        
        Args:
            hidden_size: Hidden dimension size.
            num_layers: Number of transformer layers (if building encoder).
            vocab_size: Vocabulary size for embedding layer.
            max_seq_length: Maximum sequence length.
            dropout: Dropout probability.
            reward_config: Configuration for reward computation.
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.config = reward_config or RewardConfig()
        
        # Token embedding (for when processing raw tokens)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_seq_length, hidden_size)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Reward head: pooled features -> scalar
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _create_positional_encoding(
        self, max_len: int, d_model: int
    ) -> torch.Tensor:
        """Create sinusoidal positional encodings."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute reward scores.
        
        Can accept either raw token IDs or pre-computed hidden states.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            hidden_states: Pre-computed hidden states of shape (batch_size, seq_len, hidden_size).
        
        Returns:
            Reward scores of shape (batch_size,).
        """
        if hidden_states is None:
            if input_ids is None:
                raise ValueError("Either input_ids or hidden_states must be provided")
            
            # Embed tokens
            hidden_states = self.embedding(input_ids)
            
            # Add positional encoding
            seq_len = hidden_states.size(1)
            pos_enc = self.pos_encoding[:, :seq_len, :].to(hidden_states.device)
            hidden_states = hidden_states + pos_enc
        
        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to transformer format (True = masked)
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None
        
        # Encode
        encoded = self.encoder(hidden_states, src_key_padding_mask=src_key_padding_mask)
        
        # Mean pooling over sequence
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = encoded.mean(dim=1)
        
        # Compute reward
        reward = self.reward_head(pooled).squeeze(-1)
        
        return reward
    
    def compute_hybrid_reward(
        self,
        texts: List[str],
        tokenizer: Any,
        model_rewards: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute hybrid rewards combining trainable and heuristic signals.
        
        Args:
            texts: List of generated texts.
            tokenizer: Tokenizer for encoding.
            model_rewards: Pre-computed model rewards (optional).
        
        Returns:
            Combined reward scores.
        """
        device = next(self.parameters()).device
        
        # Compute trainable reward if not provided
        if model_rewards is None:
            encoded = tokenizer.encode_batch(texts)
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            model_rewards = self.forward(input_ids, attention_mask)
        
        # Compute heuristic rewards
        heuristic_rewards = self._compute_heuristic_rewards(texts)
        heuristic_rewards = torch.tensor(heuristic_rewards, device=device)
        
        # Combine with configured weights
        alpha = self.config.alpha
        beta = self.config.beta
        
        combined = alpha * heuristic_rewards + beta * model_rewards
        
        # Normalize and clip
        if self.config.normalize_rewards:
            combined = (combined - combined.mean()) / (combined.std() + 1e-8)
        
        combined = torch.clamp(combined, -self.config.clip_rewards, self.config.clip_rewards)
        
        return combined
    
    def _compute_heuristic_rewards(self, texts: List[str]) -> List[float]:
        """
        Compute rule-based heuristic rewards.
        
        Includes:
        - Toxicity penalty (keyword-based approximation).
        - Length penalty/bonus.
        - Repetition penalty.
        
        Args:
            texts: List of generated texts.
        
        Returns:
            List of heuristic reward scores.
        """
        heuristic_config = self.config.heuristic
        rewards = []
        
        # Simple toxicity keywords (placeholder - use actual classifier in production)
        toxic_keywords = {
            "hate", "kill", "stupid", "idiot", "die", "attack",
            "violent", "harm", "hurt", "terrible", "awful",
        }
        
        for text in texts:
            reward = 1.0
            text_lower = text.lower()
            words = text_lower.split()
            
            # Toxicity penalty
            toxic_count = sum(1 for word in words if word in toxic_keywords)
            toxicity_score = toxic_count / max(len(words), 1)
            if toxicity_score > heuristic_config.toxicity_threshold:
                reward -= heuristic_config.toxicity_weight * toxicity_score
            
            # Length penalty (penalize very short or very long responses)
            length = len(text)
            if length < 20:
                reward -= heuristic_config.length_penalty * (20 - length) / 20
            elif length > 1000:
                reward -= heuristic_config.length_penalty * min((length - 1000) / 1000, 0.5)
            
            # Repetition penalty
            if len(words) > 5:
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < 0.5:  # High repetition
                    reward -= heuristic_config.repetition_penalty * (0.5 - unique_ratio)
            
            rewards.append(max(0.0, reward))
        
        return rewards
    
    def train_on_pairs(
        self,
        chosen_texts: List[str],
        rejected_texts: List[str],
        tokenizer: Any,
        epochs: int = 3,
        lr: float = 2e-5,
        batch_size: int = 8,
    ) -> Dict[str, List[float]]:
        """
        Train the reward model on preference pairs.
        
        Uses a contrastive loss where chosen responses should receive
        higher rewards than rejected responses.
        
        Args:
            chosen_texts: List of preferred responses.
            rejected_texts: List of rejected responses.
            tokenizer: Tokenizer for encoding texts.
            epochs: Number of training epochs.
            lr: Learning rate.
            batch_size: Training batch size.
        
        Returns:
            Dictionary with training metrics (losses, accuracies).
        """
        assert len(chosen_texts) == len(rejected_texts), "Must have equal pairs"
        
        device = next(self.parameters()).device
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        
        self.train()
        
        metrics = {"loss": [], "accuracy": []}
        n_samples = len(chosen_texts)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            
            # Shuffle data
            indices = torch.randperm(n_samples).tolist()
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                
                # Encode batch
                chosen_batch = [chosen_texts[j] for j in batch_indices]
                rejected_batch = [rejected_texts[j] for j in batch_indices]
                
                chosen_encoded = tokenizer.encode_batch(chosen_batch)
                rejected_encoded = tokenizer.encode_batch(rejected_batch)
                
                chosen_ids = chosen_encoded["input_ids"].to(device)
                chosen_mask = chosen_encoded["attention_mask"].to(device)
                rejected_ids = rejected_encoded["input_ids"].to(device)
                rejected_mask = rejected_encoded["attention_mask"].to(device)
                
                # Forward pass
                chosen_rewards = self.forward(chosen_ids, chosen_mask)
                rejected_rewards = self.forward(rejected_ids, rejected_mask)
                
                # Preference loss: chosen should be higher than rejected
                # Using log-sigmoid loss similar to DPO
                loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                
                # Metrics
                epoch_loss += loss.item() * len(batch_indices)
                epoch_correct += (chosen_rewards > rejected_rewards).sum().item()
            
            avg_loss = epoch_loss / n_samples
            accuracy = epoch_correct / n_samples
            metrics["loss"].append(avg_loss)
            metrics["accuracy"].append(accuracy)
            
            print(f"Epoch {epoch + 1}/{epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")
        
        return metrics
    
    def save_pretrained(self, path: str) -> None:
        """Save reward model to disk."""
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "reward_model.pt"))
        
        # Save config
        import json
        config_dict = {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config_dict, f)
    
    @classmethod
    def from_pretrained(cls, path: str) -> "RewardModel":
        """Load reward model from disk."""
        import json
        import os
        
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        
        model = cls(**config)
        state_dict = torch.load(
            os.path.join(path, "reward_model.pt"),
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        
        return model
