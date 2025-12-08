# Design Notes: Gemma Alignment Paradigms

## Overview

This document provides technical design decisions, compute cost estimates, and hyperparameter guidance for the Gemma alignment experiments.

## Architecture Decisions

### 1. PyTorch-First Approach

The codebase implements training loops directly in PyTorch rather than using high-level libraries:

- **SFT Trainer**: Custom training loop with gradient accumulation, mixed precision, and lifecycle hooks
- **RL Trainer**: Hand-implemented PPO, GRPO, and DPO algorithms
- **Reward Model**: Custom transformer encoder with preference training

This provides full control over algorithmic details and enables easy experimentation.

### 2. PEFT Integration

We use the HuggingFace PEFT library for LoRA/QLoRA, wrapped in our adapter classes:

- `LoRAAdapter`: Standard LoRA with configurable rank and target modules
- `QLoRAAdapter`: 4-bit quantized base with LoRA adapters

The wrappers provide a clean interface for applying, saving, and removing adapters.

### 3. Reward Model Design

The hybrid reward model combines:

1. **Heuristic signals**: Toxicity detection, length penalty, repetition penalty
2. **Trainable component**: 3-layer transformer encoder trained on preference pairs

Formula: `reward = alpha * heuristic_score + beta * model_score`

Default: `alpha = 0.5, beta = 0.5`

---

## Compute Cost Estimates

### Memory Requirements

| Configuration | Model Size | VRAM (Training) | VRAM (Inference) |
|--------------|------------|-----------------|------------------|
| SFT Full | 270M | ~8 GB | ~2 GB |
| SFT + LoRA | 270M | ~4-6 GB | ~2 GB |
| SFT + QLoRA | 1B | ~8-10 GB | ~4 GB |
| PPO | 270M | ~8 GB | N/A |
| Staged | 270M | ~8 GB | ~2 GB |

### Training Time Estimates

For batch_size=4, gradient_accumulation=4, 1000 training samples:

| Configuration | Time/Epoch (A100) | Time/Epoch (RTX 3090) |
|--------------|-------------------|----------------------|
| SFT + LoRA 270M | ~3 min | ~5 min |
| SFT + QLoRA 1B | ~10 min | ~15 min |
| PPO 270M | ~15 min | ~25 min |
| Staged 270M | ~20 min | ~35 min |

---

## Hyperparameter Guidance

### SFT Training

```yaml
training:
  learning_rate: 5e-5      # Good starting point
  warmup_ratio: 0.1        # 10% warmup
  weight_decay: 0.01       # Standard regularization
  max_grad_norm: 1.0       # Gradient clipping
```

For LoRA:
```yaml
model:
  peft_rank: 8             # Start small, increase if underfitting
  peft_alpha: 16           # Typically 2x rank
  peft_dropout: 0.1        # Regularization
```

For QLoRA:
```yaml
model:
  peft_rank: 16            # Can go higher with quantization
  peft_alpha: 32
  load_in_4bit: true
training:
  learning_rate: 2e-4      # Higher LR for QLoRA
```

### RL Training

PPO:
```yaml
rl:
  algorithm: ppo
  ppo_clip: 0.2            # Standard PPO clip
  entropy_coeff: 0.01      # Exploration bonus
  kl_coef: 0.1             # KL penalty for base model
  num_ppo_epochs: 4        # Updates per rollout
  rollout_size: 64         # Samples before update
```

DPO:
```yaml
rl:
  algorithm: dpo
  dpo_beta: 0.1            # KL constraint strength
```

### Reward Model

```yaml
reward:
  trainable:
    hidden_size: 768       # Match model hidden size
    num_layers: 3          # Small is sufficient
    learning_rate: 2e-5    # Slower than policy
  heuristic:
    toxicity_weight: 0.5   # Penalty for toxic content
```

---

## When to Use 270M vs 1B

### Use 270M (google/gemma-3-270m-it)

- Initial experiments and debugging
- Safety alignment tasks (sufficient for guardrails)
- Limited compute resources
- Rapid iteration and prototyping
- RL experiments (lower memory for rollouts)

### Use 1B (google/gemma-3-1b-it)

- Production deployment
- Tasks requiring more world knowledge
- Code generation (CoNaLa)
- Clinical summarization (nuanced language)
- Final comparative evaluation

---

## Evaluation Metrics by Task

### Safety/Harmlessness
- **Primary**: Harmlessness rate (keyword-based)
- **Secondary**: Safety precision/recall
- **Human**: Safety ratings (1-5 scale)

### Clinical Summarization
- **Primary**: ROUGE-L, BERTScore F1
- **Secondary**: Fact-F1 (factual accuracy)
- **Human**: Clinical accuracy, completeness

### Code Q&A
- **Primary**: CodeBLEU, exact match
- **Secondary**: Syntax validity, pass rate
- **Human**: Correctness, Pythonic style

---

## Statistical Comparison

For comparing paradigms, we use:

1. **Paired t-test**: Between-paradigm comparison on same samples
2. **Bootstrap CI**: 95% confidence intervals for metric differences
3. **Effect size**: Cohen's d for practical significance

Minimum recommended sample sizes:
- Automatic evaluation: 500+ samples
- Human evaluation: 200 samples with 2+ annotators

---

## Troubleshooting

### Out of Memory
- Reduce batch_size
- Increase gradient_accumulation_steps
- Use QLoRA for 1B model
- Enable gradient checkpointing

### Training Instability
- Lower learning_rate
- Increase warmup_ratio
- Check gradient norms in logs
- Reduce ppo_clip for RL

### Poor Reward Signal
- Increase heuristic weight (alpha)
- Train reward model longer
- Add more preference pairs
- Check reward normalization
