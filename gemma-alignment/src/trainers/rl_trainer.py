"""
Reinforcement Learning trainer with PPO, GRPO, and DPO implementations.

This module provides the RLTrainer class that implements multiple RL
algorithms for alignment training using a reward model.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.core.config import ExperimentConfig
from src.core.utils import get_logger
from src.models.base_model import ModelWrapper
from src.models.reward_model import RewardModel
from src.tokenization.tokenizer_wrapper import TokenizerWrapper
from src.trainers.base_trainer import BaseTrainer


@dataclass
class RolloutBuffer:
    """
    Buffer for storing rollout data during RL training.
    
    Stores generated sequences, log probabilities, rewards, and computed
    advantages for policy gradient updates.
    
    Attributes:
        prompts: Original prompt texts.
        prompt_ids: Tokenized prompt IDs.
        generations: Generated response texts.
        generation_ids: Tokenized generation IDs.
        old_log_probs: Log probabilities from the policy that generated responses.
        rewards: Reward scores for each generation.
        advantages: Computed advantages (rewards - baseline).
        values: Value estimates (if using actor-critic).
        returns: Discounted returns.
    """
    
    prompts: List[str] = field(default_factory=list)
    prompt_ids: Optional[torch.Tensor] = None
    generations: List[str] = field(default_factory=list)
    generation_ids: Optional[torch.Tensor] = None
    attention_masks: Optional[torch.Tensor] = None
    old_log_probs: Optional[torch.Tensor] = None
    rewards: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None
    values: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None
    
    def __len__(self) -> int:
        return len(self.prompts)
    
    def to_device(self, device: torch.device) -> "RolloutBuffer":
        """Move all tensors to specified device."""
        if self.prompt_ids is not None:
            self.prompt_ids = self.prompt_ids.to(device)
        if self.generation_ids is not None:
            self.generation_ids = self.generation_ids.to(device)
        if self.attention_masks is not None:
            self.attention_masks = self.attention_masks.to(device)
        if self.old_log_probs is not None:
            self.old_log_probs = self.old_log_probs.to(device)
        if self.rewards is not None:
            self.rewards = self.rewards.to(device)
        if self.advantages is not None:
            self.advantages = self.advantages.to(device)
        if self.values is not None:
            self.values = self.values.to(device)
        if self.returns is not None:
            self.returns = self.returns.to(device)
        return self


class RLTrainer(BaseTrainer):
    """
    Reinforcement Learning trainer supporting PPO, GRPO, and DPO.
    
    Implements:
    - PPO (Proximal Policy Optimization): Clipped surrogate objective.
    - GRPO (Group Relative Policy Optimization): Trust region variant.
    - DPO (Direct Preference Optimization): Reward-free preference learning.
    
    Attributes:
        policy_model: The model being optimized.
        reward_model: Reward model for computing rewards.
        ref_model: Reference model for KL penalty (optional).
        tokenizer: Tokenizer for encoding/decoding.
    
    Example:
        >>> trainer = RLTrainer(policy_model, reward_model, tokenizer, cfg, dataset)
        >>> trainer.train()
    """
    
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
        """
        Initialize the RL trainer.
        
        Args:
            policy_model: Model to optimize with RL.
            reward_model: Reward model for scoring generations.
            tokenizer: Tokenizer for text processing.
            cfg: Experiment configuration.
            train_dataloader: Training data loader.
            val_dataloader: Optional validation data loader.
            ref_model: Optional reference model for KL penalty.
        """
        super().__init__(policy_model, cfg, train_dataloader, val_dataloader)
        
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.ref_model = ref_model
        self.logger = get_logger("gemma-alignment")
        
        # Move reward model to device
        self.reward_model = self.reward_model.to(self.device)
        self.reward_model.eval()
        
        # Freeze reference model if provided
        if self.ref_model is not None:
            self.ref_model = self.ref_model.to(self.device)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
        
        # RL-specific config
        self.rl_cfg = cfg.rl
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Perform one RL training step based on configured algorithm.
        
        Args:
            batch: Training batch data.
        
        Returns:
            Loss value.
        """
        algorithm = self.rl_cfg.algorithm
        
        if algorithm == "ppo":
            return self._ppo_train_step(batch)
        elif algorithm == "grpo":
            return self._grpo_train_step(batch)
        elif algorithm == "dpo":
            return self._dpo_train_step(batch)
        else:
            raise ValueError(f"Unknown RL algorithm: {algorithm}")
    
    def _ppo_train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        PPO training step.
        
        Collects rollouts, computes advantages, and updates policy
        using clipped surrogate objective.
        """
        # Collect rollouts
        rollouts = self.collect_rollouts(batch)
        
        # PPO update
        total_loss = 0.0
        
        for _ in range(self.rl_cfg.num_ppo_epochs):
            loss = self.ppo_step(rollouts)
            total_loss += loss
        
        return total_loss / self.rl_cfg.num_ppo_epochs
    
    def _grpo_train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        GRPO training step.
        
        Similar to PPO but with group-relative advantage computation
        and different trust region approach.
        """
        rollouts = self.collect_rollouts(batch)
        loss = self.grpo_step(rollouts)
        return loss
    
    def _dpo_train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        DPO training step.
        
        Uses preference pairs directly without explicit reward model.
        """
        # DPO uses preference pairs from the batch
        chosen_ids = batch["input_ids"]
        chosen_mask = batch["attention_mask"]
        rejected_ids = batch.get("rejected_input_ids")
        rejected_mask = batch.get("rejected_attention_mask")
        
        if rejected_ids is None:
            self.logger.warning("DPO requires rejected samples in batch")
            return 0.0
        
        loss = self.dpo_loss(
            chosen_ids, chosen_mask,
            rejected_ids, rejected_mask,
        )
        
        # Backward pass
        scaled_loss = loss / self.cfg.training.gradient_accumulation_steps
        scaled_loss.backward()
        
        return loss.item()
    
    def collect_rollouts(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> RolloutBuffer:
        """
        Collect rollouts by generating from the current policy.
        
        Args:
            batch: Batch containing prompts.
        
        Returns:
            RolloutBuffer with generated sequences and computed rewards.
        """
        self.model.eval()
        
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        # Generate responses
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
            )
        
        # Decode generations
        generations = self.tokenizer.decode_batch(
            generated_ids[:, input_ids.size(1):],
            skip_special_tokens=True,
        )
        prompts = self.tokenizer.decode_batch(input_ids, skip_special_tokens=True)
        
        # Compute log probs for generated sequences
        self.model.train()
        log_probs = self._compute_sequence_log_probs(generated_ids, attention_mask)
        
        # Compute rewards
        rewards = self.compute_rewards(generations, prompts)
        
        # Compute KL penalty if reference model available
        if self.ref_model is not None:
            with torch.no_grad():
                ref_log_probs = self._compute_sequence_log_probs(
                    generated_ids, attention_mask, model=self.ref_model
                )
            kl = (log_probs - ref_log_probs).sum(dim=-1)
            rewards = rewards - self.rl_cfg.kl_coef * kl
        
        # Compute advantages
        advantages = self.compute_advantages(rewards)
        
        # Create attention mask for full sequence
        full_mask = torch.ones_like(generated_ids)
        
        buffer = RolloutBuffer(
            prompts=prompts,
            prompt_ids=input_ids,
            generations=generations,
            generation_ids=generated_ids,
            attention_masks=full_mask,
            old_log_probs=log_probs.detach(),
            rewards=rewards,
            advantages=advantages,
            returns=rewards,  # Simplified: returns = rewards
        )
        
        return buffer.to_device(self.device)
    
    def _compute_sequence_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Compute log probabilities for a sequence."""
        if model is None:
            model = self.model
        
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        
        log_probs = F.log_softmax(logits, dim=-1)
        gathered = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1))
        
        return gathered.squeeze(-1)
    
    def compute_rewards(
        self,
        generations: List[str],
        references: List[str],
    ) -> torch.Tensor:
        """
        Compute rewards for generated sequences.
        
        Uses the reward model to score generations, optionally combined
        with heuristic signals.
        
        Args:
            generations: List of generated texts.
            references: List of reference prompts.
        
        Returns:
            Tensor of reward scores.
        """
        # Encode generations
        encoded = self.tokenizer.encode_batch(generations)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # Get rewards from model
        with torch.no_grad():
            if self.cfg.reward.type == "hybrid":
                rewards = self.reward_model.compute_hybrid_reward(
                    generations, self.tokenizer
                )
            else:
                rewards = self.reward_model(input_ids, attention_mask)
        
        return rewards
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute advantages for policy gradient.
        
        Uses simple reward normalization if values not provided.
        
        Args:
            rewards: Reward tensor of shape (batch_size,).
            values: Optional value estimates for GAE.
        
        Returns:
            Advantage tensor.
        """
        if values is None:
            # Simple reward normalization
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            # GAE (Generalized Advantage Estimation)
            # Simplified single-step version
            advantages = rewards - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def ppo_step(self, rollouts: RolloutBuffer) -> float:
        """
        Perform one PPO update step.
        
        Computes clipped surrogate objective and updates policy.
        
        Args:
            rollouts: Buffer containing rollout data.
        
        Returns:
            Loss value.
        """
        # Compute current log probs
        current_log_probs = self._compute_sequence_log_probs(
            rollouts.generation_ids,
            rollouts.attention_masks,
        )
        
        # Compute ratio
        ratio = torch.exp(current_log_probs - rollouts.old_log_probs)
        
        # Sum over sequence
        ratio = ratio.sum(dim=-1)
        old_log_sum = rollouts.old_log_probs.sum(dim=-1)
        new_log_sum = current_log_probs.sum(dim=-1)
        ratio = torch.exp(new_log_sum - old_log_sum)
        
        # Clipped surrogate objective
        advantages = rollouts.advantages
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio, 1 - self.rl_cfg.ppo_clip, 1 + self.rl_cfg.ppo_clip
        ) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Entropy bonus
        # Approximate entropy from log probs
        entropy = -current_log_probs.mean()
        entropy_loss = -self.rl_cfg.entropy_coeff * entropy
        
        total_loss = policy_loss + entropy_loss
        
        # Backward
        scaled_loss = total_loss / self.cfg.training.gradient_accumulation_steps
        scaled_loss.backward()
        
        return total_loss.item()
    
    def grpo_step(self, rollouts: RolloutBuffer) -> float:
        """
        Perform one GRPO (Group Relative Policy Optimization) update.
        
        GRPO computes advantages relative to other samples in the batch,
        providing a natural baseline without value function.
        
        Args:
            rollouts: Buffer containing rollout data.
        
        Returns:
            Loss value.
        """
        # Compute current log probs
        current_log_probs = self._compute_sequence_log_probs(
            rollouts.generation_ids,
            rollouts.attention_masks,
        )
        
        # Sum log probs over sequence
        log_prob_sum = current_log_probs.sum(dim=-1)
        old_log_prob_sum = rollouts.old_log_probs.sum(dim=-1)
        
        # Group-relative advantages (rewards relative to batch mean)
        rewards = rollouts.rewards
        group_baseline = rewards.mean()
        group_advantages = rewards - group_baseline
        group_advantages = group_advantages / (group_advantages.std() + 1e-8)
        
        # Policy gradient with trust region
        ratio = torch.exp(log_prob_sum - old_log_prob_sum)
        
        # Soft trust region (no hard clipping, use KL penalty instead)
        kl_penalty = (ratio - 1) ** 2  # Approximate KL
        
        policy_loss = -(ratio * group_advantages).mean()
        trust_loss = 0.5 * kl_penalty.mean()
        
        total_loss = policy_loss + 0.1 * trust_loss
        
        # Backward
        scaled_loss = total_loss / self.cfg.training.gradient_accumulation_steps
        scaled_loss.backward()
        
        return total_loss.item()
    
    def dpo_loss(
        self,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DPO (Direct Preference Optimization) loss.
        
        DPO directly optimizes the preference model without explicit rewards,
        using a closed-form solution from the reward modeling objective.
        
        Loss = -log(sigmoid(beta * (log_pi(chosen) - log_pi(rejected) 
                                   - log_ref(chosen) + log_ref(rejected))))
        
        Args:
            chosen_ids: Token IDs for preferred responses.
            chosen_mask: Attention mask for preferred.
            rejected_ids: Token IDs for rejected responses.
            rejected_mask: Attention mask for rejected.
        
        Returns:
            DPO loss tensor.
        """
        beta = self.rl_cfg.dpo_beta
        
        # Compute policy log probs
        chosen_log_probs = self._compute_sequence_log_probs(chosen_ids, chosen_mask)
        rejected_log_probs = self._compute_sequence_log_probs(rejected_ids, rejected_mask)
        
        # Sum over sequence (using mask)
        chosen_mask_shifted = chosen_mask[:, 1:].float()
        rejected_mask_shifted = rejected_mask[:, 1:].float()
        
        chosen_log_sum = (chosen_log_probs * chosen_mask_shifted).sum(dim=-1)
        rejected_log_sum = (rejected_log_probs * rejected_mask_shifted).sum(dim=-1)
        
        # Compute reference log probs if available
        if self.ref_model is not None:
            with torch.no_grad():
                ref_chosen = self._compute_sequence_log_probs(
                    chosen_ids, chosen_mask, model=self.ref_model
                )
                ref_rejected = self._compute_sequence_log_probs(
                    rejected_ids, rejected_mask, model=self.ref_model
                )
                
                ref_chosen_sum = (ref_chosen * chosen_mask_shifted).sum(dim=-1)
                ref_rejected_sum = (ref_rejected * rejected_mask_shifted).sum(dim=-1)
        else:
            # Without reference model, use implicit uniform reference
            ref_chosen_sum = torch.zeros_like(chosen_log_sum)
            ref_rejected_sum = torch.zeros_like(rejected_log_sum)
        
        # DPO loss
        pi_logratios = chosen_log_sum - rejected_log_sum
        ref_logratios = ref_chosen_sum - ref_rejected_sum
        
        logits = beta * (pi_logratios - ref_logratios)
        loss = -F.logsigmoid(logits).mean()
        
        return loss
    
    def on_train_start(self) -> None:
        """Called at training start."""
        super().on_train_start()
        self.logger.info(f"RL Algorithm: {self.rl_cfg.algorithm}")
        self.logger.info(f"Rollout size: {self.rl_cfg.rollout_size}")
        if self.rl_cfg.algorithm == "ppo":
            self.logger.info(f"PPO clip: {self.rl_cfg.ppo_clip}")
        elif self.rl_cfg.algorithm == "dpo":
            self.logger.info(f"DPO beta: {self.rl_cfg.dpo_beta}")
