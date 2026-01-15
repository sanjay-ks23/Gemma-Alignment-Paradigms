# Gemma Alignment Paradigms

**High-performance PyTorch framework for comparative LLM alignment on Google Gemma 3 270M.**

This repository focuses on the technical challenges of **alignment stability**, **reward hacking mitigation**, and **efficiency** across Supervised Fine-Tuning (SFT), Reinforcement Learning (RL), and Staged training pipelines.

---

## Technical Core

### 1. Hybrid Reward System
To prevent common **reward hacking** issues where models exploit specific tokens for high scores without improved behavior, this implementation uses a dual-signal approach:
- **Learnable Transformer Encoder**: Captures high-dimensional semantic preferences.
- **Hard Heuristic Constraints**: Penalizes toxicity, repetition, and degenerate length patterns via configurable alpha/beta weights.

### 2. Multi-Paradigm Alignment
- **PPO/GRPO**: Implements clipped objectives and group-relative rewards for stable policy updates.
- **DPO**: Direct policy optimization from preference pairs, bypassing the explicit reward model.
- **Staged SFT → RL**: Demonstrates that building a strong semantic baseline through SFT significantly improves the sample efficiency and stability of subsequent RL refinement.

### 3. Resource-Efficient Architecture
- **PEFT/LoRA Integration**: Optimized for single-GPU training (6-8GB VRAM) by targeting specific attention modules (`q_proj`, `v_proj`).
- **Standardized Evaluation**: Uniform benchmarking across Safety (Harmlessness), Clinical (Summarization), and Code (NL→Python) tasks.

---

## Quick Start

```bash
pip install -e ".[dev]"

# SFT Warmup
python -m src.experiments.runner --config configs/sft_lora_270m.yml

# RL / PPO Refinement
python -m src.experiments.runner --config configs/ppo_rl_270m.yml

# Orchestrated Staged Pipeline
python -m src.experiments.runner --config configs/staged_sft_then_ppo.yml
```

---

## File Structure

- `src/trainers/`: Hand-coded PPO, GRPO, DPO, and SFT loops.
- `src/models/`: Gemma wrappers and Hybrid Reward Model.
- `src/data/`: Task-specific safety, clinical, and code datasets.
- `tests/`: Gradient flow and logic verification for all training modes.

---

## Contact & Citation
Developed by [Sanjay K. Saravanan](https://github.com/sanjay-ks23). Licensed under MIT.
