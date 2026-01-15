"""RL trainer with PPO, GRPO, and DPO implementations."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.core.config import ExperimentConfig
from src.core.utils import get_logger
from src.models.base_model import ModelWrapper
from src.models.reward_model import RewardModel
from src.tokenization.tokenizer_wrapper import TokenizerWrapper
from src.trainers.base_trainer import BaseTrainer


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data during RL training."""
    prompts: List[str] = field(default_factory=list)
    prompt_ids: Optional[torch.Tensor] = None
    generations: List[str] = field(default_factory=list)
    generation_ids: Optional[torch.Tensor] = None
    attention_masks: Optional[torch.Tensor] = None
    old_log_probs: Optional[torch.Tensor] = None
    rewards: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None
    
    def __len__(self) -> int:
        return len(self.prompts)
    
    def to_device(self, device: torch.device) -> "RolloutBuffer":
        for attr in ["prompt_ids", "generation_ids", "attention_masks", 
                     "old_log_probs", "rewards", "advantages"]:
            tensor = getattr(self, attr)
            if tensor is not None:
                setattr(self, attr, tensor.to(device))
        return self


class RLTrainer(BaseTrainer):
    """RL trainer supporting PPO, GRPO, and DPO algorithms."""
    
    def __init__(
        self,
        policy_model: ModelWrapper,
        reward_model: RewardModel,
        tokenizer: TokenizerWrapper,
        cfg: ExperimentConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        ref_model: Optional[ModelWrapper] = None,
    ):
        super().__init__(policy_model, cfg, train_dataloader, val_dataloader)
        
        self.reward_model = reward_model.to(self.device)
        self.reward_model.eval()
        self.tokenizer = tokenizer
        self.ref_model = ref_model
        self.logger = get_logger("gemma-alignment")
        
        if self.ref_model is not None:
            self.ref_model = self.ref_model.to(self.device)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
        
        self.rl_cfg = cfg.rl
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        algorithm = self.rl_cfg.algorithm
        
        if algorithm == "ppo":
            return self._ppo_step(batch)
        elif algorithm == "grpo":
            return self._grpo_step(batch)
        elif algorithm == "dpo":
            return self._dpo_step(batch)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _ppo_step(self, batch: Dict[str, torch.Tensor]) -> float:
        rollouts = self._collect_rollouts(batch)
        total_loss = 0.0
        
        for _ in range(self.rl_cfg.num_ppo_epochs):
            current_log_probs = self._compute_log_probs(
                rollouts.generation_ids, rollouts.attention_masks
            )
            
            log_prob_sum = current_log_probs.sum(dim=-1)
            old_log_sum = rollouts.old_log_probs.sum(dim=-1)
            ratio = torch.exp(log_prob_sum - old_log_sum)
            
            surr1 = ratio * rollouts.advantages
            surr2 = torch.clamp(
                ratio, 1 - self.rl_cfg.ppo_clip, 1 + self.rl_cfg.ppo_clip
            ) * rollouts.advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -self.rl_cfg.entropy_coeff * (-current_log_probs.mean())
            
            loss = policy_loss + entropy_loss
            scaled_loss = loss / self.cfg.training.gradient_accumulation_steps
            scaled_loss.backward()
            
            total_loss += loss.item()
        
        return total_loss / self.rl_cfg.num_ppo_epochs
    
    def _grpo_step(self, batch: Dict[str, torch.Tensor]) -> float:
        rollouts = self._collect_rollouts(batch)
        
        current_log_probs = self._compute_log_probs(
            rollouts.generation_ids, rollouts.attention_masks
        )
        
        log_prob_sum = current_log_probs.sum(dim=-1)
        old_log_sum = rollouts.old_log_probs.sum(dim=-1)
        
        rewards = rollouts.rewards
        group_advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        ratio = torch.exp(log_prob_sum - old_log_sum)
        kl_penalty = (ratio - 1) ** 2
        
        loss = -(ratio * group_advantages).mean() + 0.05 * kl_penalty.mean()
        
        scaled_loss = loss / self.cfg.training.gradient_accumulation_steps
        scaled_loss.backward()
        
        return loss.item()
    
    def _dpo_step(self, batch: Dict[str, torch.Tensor]) -> float:
        chosen_ids = batch["input_ids"]
        chosen_mask = batch["attention_mask"]
        rejected_ids = batch.get("rejected_input_ids")
        rejected_mask = batch.get("rejected_attention_mask")
        
        if rejected_ids is None:
            return 0.0
        
        beta = self.rl_cfg.dpo_beta
        
        chosen_log_probs = self._compute_log_probs(chosen_ids, chosen_mask)
        rejected_log_probs = self._compute_log_probs(rejected_ids, rejected_mask)
        
        chosen_mask_shifted = chosen_mask[:, 1:].float()
        rejected_mask_shifted = rejected_mask[:, 1:].float()
        
        chosen_log_sum = (chosen_log_probs * chosen_mask_shifted).sum(dim=-1)
        rejected_log_sum = (rejected_log_probs * rejected_mask_shifted).sum(dim=-1)
        
        if self.ref_model is not None:
            with torch.no_grad():
                ref_chosen = self._compute_log_probs(
                    chosen_ids, chosen_mask, model=self.ref_model
                )
                ref_rejected = self._compute_log_probs(
                    rejected_ids, rejected_mask, model=self.ref_model
                )
                ref_chosen_sum = (ref_chosen * chosen_mask_shifted).sum(dim=-1)
                ref_rejected_sum = (ref_rejected * rejected_mask_shifted).sum(dim=-1)
        else:
            ref_chosen_sum = torch.zeros_like(chosen_log_sum)
            ref_rejected_sum = torch.zeros_like(rejected_log_sum)
        
        pi_logratios = chosen_log_sum - rejected_log_sum
        ref_logratios = ref_chosen_sum - ref_rejected_sum
        
        logits = beta * (pi_logratios - ref_logratios)
        loss = -F.logsigmoid(logits).mean()
        
        scaled_loss = loss / self.cfg.training.gradient_accumulation_steps
        scaled_loss.backward()
        
        return loss.item()
    
    def _collect_rollouts(self, batch: Dict[str, torch.Tensor]) -> RolloutBuffer:
        self.model.eval()
        
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
            )
        
        generations = self.tokenizer.decode_batch(
            generated_ids[:, input_ids.size(1):], skip_special_tokens=True
        )
        prompts = self.tokenizer.decode_batch(input_ids, skip_special_tokens=True)
        
        self.model.train()
        log_probs = self._compute_log_probs(generated_ids, attention_mask)
        
        rewards = self._compute_rewards(generations)
        
        if self.ref_model is not None:
            with torch.no_grad():
                ref_log_probs = self._compute_log_probs(
                    generated_ids, attention_mask, model=self.ref_model
                )
            kl = (log_probs - ref_log_probs).sum(dim=-1)
            rewards = rewards - self.rl_cfg.kl_coef * kl
        
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        return RolloutBuffer(
            prompts=prompts,
            prompt_ids=input_ids,
            generations=generations,
            generation_ids=generated_ids,
            attention_masks=torch.ones_like(generated_ids),
            old_log_probs=log_probs.detach(),
            rewards=rewards,
            advantages=advantages,
        ).to_device(self.device)
    
    def _compute_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        if model is None:
            model = self.model
        
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        
        log_probs = F.log_softmax(logits, dim=-1)
        return torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    
    def _compute_rewards(self, generations: List[str]) -> torch.Tensor:
        encoded = self.tokenizer.encode_batch(generations)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        with torch.no_grad():
            if self.cfg.reward.type == "hybrid":
                return self.reward_model.compute_hybrid_reward(generations, self.tokenizer)
            return self.reward_model(input_ids, attention_mask)
