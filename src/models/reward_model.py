"""Reward model for RL-based alignment training."""

import json
import math
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.config import RewardConfig


class RewardModel(nn.Module):
    """Reward model computing scalar rewards from text or embeddings."""
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_layers: int = 3,
        vocab_size: int = 256000,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        reward_config: Optional[RewardConfig] = None,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.config = reward_config or RewardConfig()
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = self._create_positional_encoding(max_seq_length, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )
        
        self._init_weights()
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def _init_weights(self) -> None:
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
        """Compute reward scores."""
        if hidden_states is None:
            if input_ids is None:
                raise ValueError("Either input_ids or hidden_states must be provided")
            
            hidden_states = self.embedding(input_ids)
            seq_len = hidden_states.size(1)
            pos_enc = self.pos_encoding[:, :seq_len, :].to(hidden_states.device)
            hidden_states = hidden_states + pos_enc
        
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
        
        encoded = self.encoder(hidden_states, src_key_padding_mask=src_key_padding_mask)
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = encoded.mean(dim=1)
        
        return self.reward_head(pooled).squeeze(-1)
    
    def compute_hybrid_reward(
        self,
        texts: List[str],
        tokenizer: Any,
        model_rewards: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute hybrid rewards combining trainable and heuristic signals."""
        device = next(self.parameters()).device
        
        if model_rewards is None:
            encoded = tokenizer.encode_batch(texts)
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            model_rewards = self.forward(input_ids, attention_mask)
        
        heuristic_rewards = torch.tensor(
            self._compute_heuristic_rewards(texts), device=device
        )
        
        combined = self.config.alpha * heuristic_rewards + self.config.beta * model_rewards
        
        if self.config.normalize_rewards:
            combined = (combined - combined.mean()) / (combined.std() + 1e-8)
        
        return torch.clamp(combined, -self.config.clip_rewards, self.config.clip_rewards)
    
    def _compute_heuristic_rewards(self, texts: List[str]) -> List[float]:
        heuristic_config = self.config.heuristic
        toxic_keywords = {
            "hate", "kill", "stupid", "idiot", "die", "attack",
            "violent", "harm", "hurt", "terrible", "awful",
        }
        
        rewards = []
        for text in texts:
            reward = 1.0
            text_lower = text.lower()
            words = text_lower.split()
            
            toxic_count = sum(1 for word in words if word in toxic_keywords)
            toxicity_score = toxic_count / max(len(words), 1)
            if toxicity_score > heuristic_config.toxicity_threshold:
                reward -= heuristic_config.toxicity_weight * toxicity_score
            
            if len(text) < 20:
                reward -= heuristic_config.length_penalty * (20 - len(text)) / 20
            elif len(text) > 1000:
                reward -= heuristic_config.length_penalty * min((len(text) - 1000) / 1000, 0.5)
            
            if len(words) > 5:
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < 0.5:
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
        """Train reward model on preference pairs."""
        assert len(chosen_texts) == len(rejected_texts)
        
        device = next(self.parameters()).device
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        
        self.train()
        metrics = {"loss": [], "accuracy": []}
        n_samples = len(chosen_texts)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            indices = torch.randperm(n_samples).tolist()
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                
                chosen_batch = [chosen_texts[j] for j in batch_indices]
                rejected_batch = [rejected_texts[j] for j in batch_indices]
                
                chosen_encoded = tokenizer.encode_batch(chosen_batch)
                rejected_encoded = tokenizer.encode_batch(rejected_batch)
                
                chosen_ids = chosen_encoded["input_ids"].to(device)
                chosen_mask = chosen_encoded["attention_mask"].to(device)
                rejected_ids = rejected_encoded["input_ids"].to(device)
                rejected_mask = rejected_encoded["attention_mask"].to(device)
                
                chosen_rewards = self.forward(chosen_ids, chosen_mask)
                rejected_rewards = self.forward(rejected_ids, rejected_mask)
                
                loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item() * len(batch_indices)
                epoch_correct += (chosen_rewards > rejected_rewards).sum().item()
            
            metrics["loss"].append(epoch_loss / n_samples)
            metrics["accuracy"].append(epoch_correct / n_samples)
        
        return metrics
    
    def save_pretrained(self, path: str) -> None:
        """Save reward model to disk."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "reward_model.pt"))
        
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({"hidden_size": self.hidden_size, "num_layers": self.num_layers}, f)
    
    @classmethod
    def from_pretrained(cls, path: str) -> "RewardModel":
        """Load reward model from disk."""
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        
        model = cls(**config)
        model.load_state_dict(torch.load(
            os.path.join(path, "reward_model.pt"),
            map_location="cpu",
            weights_only=True,
        ))
        return model
