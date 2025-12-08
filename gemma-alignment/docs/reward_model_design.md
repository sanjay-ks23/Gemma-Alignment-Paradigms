# Reward Model Design

## Overview

The reward model is a critical component for RL-based alignment. It provides scalar reward signals that guide policy optimization toward desired behaviors.

## Architecture

### Trainable Component

```
Input Tokens -> Embedding -> Transformer Encoder -> Mean Pooling -> MLP -> Scalar Reward
```

Configuration:
- Hidden size: 768 (default)
- Transformer layers: 3
- Attention heads: 8
- FFN ratio: 4x
- Parameters: ~2-10M

### MLP Head

```
Pooled [768] -> Linear [768] -> ReLU -> Dropout -> Linear [384] -> ReLU -> Dropout -> Linear [1]
```

## Reward Computation Strategies

### 1. Heuristic Rewards

Fast, interpretable signals:

```python
reward = 1.0

# Toxicity penalty
toxicity = count_toxic_keywords(text) / word_count
if toxicity > threshold:
    reward -= toxicity_weight * toxicity

# Length penalty
if len(text) < min_length:
    reward -= length_penalty
elif len(text) > max_length:
    reward -= length_penalty * 0.5

# Repetition penalty
unique_ratio = unique_words / total_words
if unique_ratio < 0.5:
    reward -= repetition_penalty * (0.5 - unique_ratio)
```

Pros:
- No training required
- Interpretable
- Fast inference

Cons:
- Limited expressivity
- May miss subtle patterns

### 2. Trainable Rewards

Neural network trained on preference data:

Training objective (Bradley-Terry):
```
L = -log(sigmoid(r(chosen) - r(rejected)))
```

Training process:
1. Collect preference pairs (chosen, rejected)
2. Forward both through reward model
3. Compute contrastive loss
4. Update parameters

Pros:
- Learns complex patterns
- Adapts to task

Cons:
- Requires preference data
- May overfit

### 3. Hybrid Rewards

Combines both approaches:

```
reward = alpha * heuristic_reward + beta * model_reward
```

Where:
- `alpha + beta = 1` (normalized)
- Default: `alpha = 0.5, beta = 0.5`

Benefits:
- Heuristics provide stable baseline
- Model captures nuanced preferences
- Robust to reward model errors

## Training Data

### Data Collection

For safety task, preference pairs come from:
1. HH-RLHF dataset (chosen/rejected pairs)
2. Synthetic negatives (perturbed safe responses)
3. Model-generated samples with human labels

### Data Format

```json
{
  "prompt": "How do I...",
  "chosen": "Here's a safe way to...",
  "rejected": "You could try this dangerous..."
}
```

### Data Preparation Script

```bash
python scripts/prepare_reward_data.py \
    --input data/datasets/safety/train.jsonl \
    --output data/reward_training/pairs.jsonl \
    --num_synthetic 1000
```

## Reward Normalization

To stabilize RL training:

1. **Per-batch normalization**:
   ```python
   rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
   ```

2. **Clipping**:
   ```python
   rewards = torch.clamp(rewards, -clip_value, clip_value)
   ```

3. **Running statistics** (optional):
   Track exponential moving average of mean/std

## Evaluation

### Intrinsic Metrics

- Training accuracy (chosen > rejected)
- Validation accuracy
- Loss convergence

### Extrinsic Metrics

- Correlation with human ratings
- RL training stability
- Final policy quality

## Implementation Notes

### Training Loop

```python
for epoch in range(epochs):
    for chosen, rejected in dataloader:
        chosen_reward = reward_model(chosen)
        rejected_reward = reward_model(rejected)
        
        loss = -log_sigmoid(chosen_reward - rejected_reward).mean()
        
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(parameters, 1.0)
        optimizer.step()
```

### Inference

```python
def compute_rewards(generations, tokenizer, reward_model):
    encoded = tokenizer.encode_batch(generations)
    with torch.no_grad():
        rewards = reward_model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"]
        )
    return rewards
```

## Hyperparameter Recommendations

| Parameter | Default | Range |
|-----------|---------|-------|
| hidden_size | 768 | 256-1024 |
| num_layers | 3 | 2-6 |
| learning_rate | 2e-5 | 1e-5 - 5e-5 |
| batch_size | 8 | 4-16 |
| epochs | 3 | 2-5 |
| dropout | 0.1 | 0.05-0.2 |

## Common Issues

### Reward Hacking

When policy exploits reward model weaknesses:
- Increase heuristic weight
- Add diversity penalties
- Regularize with KL to base model

### Reward Collapse

When all samples get similar rewards:
- Check normalization
- Verify preference labels
- Increase model capacity

### Training Instability

- Lower learning rate
- Increase warmup
- Add gradient clipping
