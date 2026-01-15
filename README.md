# Gemma Alignment Paradigms

<div align="center">

**A Comparative Study of Alignment Training Methods for Google Gemma 3**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Overview

This repository implements a **PyTorch-first comparative study** of three alignment paradigms for fine-tuning Google Gemma 3 (270M) language models:

| Paradigm | Method | Description |
|----------|--------|-------------|
| **SFT** | Supervised Fine-Tuning + LoRA | Parameter-efficient instruction tuning |
| **RL** | PPO / GRPO / DPO | Reinforcement learning from human feedback |
| **Staged** | SFT → RL Pipeline | Two-phase training combining both approaches |

The codebase is designed for **reproducibility**, **modularity**, and **extensibility**, featuring hand-coded training loops with custom abstractions over Hugging Face components.

---

## Key Features

- **Pure PyTorch Training Loops** - Full control over optimization with gradient accumulation, mixed precision, and checkpointing
- **LoRA Adapters** - Memory-efficient fine-tuning using low-rank adaptation
- **Multiple RL Algorithms** - PPO (Proximal Policy Optimization), GRPO (Group Relative Policy Optimization), DPO (Direct Preference Optimization)
- **Hybrid Reward Model** - Combines trainable transformer encoder with heuristic signals (toxicity, length, repetition)
- **Multi-Task Support** - Safety (harmless responses), Clinical (medical summarization), Code (NL→Python)
- **Evaluation Suite** - ROUGE, BERTScore, harmlessness metrics with human evaluation manifest generation

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.1+ (for GPU training)
- 6-8 GB VRAM (for Gemma 270M with LoRA)

### Setup

```bash
# Clone repository
git clone https://github.com/sanjay-ks23/Gemma-Alignment-Paradigms.git
cd Gemma-Alignment-Paradigms

# Install dependencies
pip install -e ".[dev]"

# Login to Hugging Face (required for Gemma access)
huggingface-cli login
```

### Model Access

Gemma models require license acceptance:

1. Visit [google/gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it)
2. Accept the license agreement
3. Authenticate with `huggingface-cli login`

---

## Quick Start

### Training

```bash
# Supervised Fine-Tuning with LoRA
python -m src.experiments.runner --config configs/sft_lora_270m.yml

# Reinforcement Learning with PPO
python -m src.experiments.runner --config configs/ppo_rl_270m.yml

# Staged Pipeline (SFT → RL)
python -m src.experiments.runner --config configs/staged_sft_then_ppo.yml

# Debug mode (fast iteration with 2-3 batches)
python -m src.experiments.runner --config configs/sft_lora_270m.yml --debug
```

### Inference Demo

```bash
# Start Flask API server
python -m src.api.serve_demo --checkpoint outputs/best_model --port 5000
```

---

## Project Structure

```
Gemma-Alignment-Paradigms/
│
├── configs/                    # YAML experiment configurations
│   ├── defaults.yml            # Base configuration with all options
│   ├── sft_lora_270m.yml       # SFT + LoRA training
│   ├── ppo_rl_270m.yml         # PPO reinforcement learning
│   └── staged_sft_then_ppo.yml # Two-phase SFT → PPO
│
├── src/
│   ├── core/                   # Configuration, registry, utilities
│   │   ├── config.py           # Dataclass-based config management
│   │   ├── registry.py         # Dynamic component registration
│   │   └── utils.py            # Logging, seeding, checkpointing
│   │
│   ├── data/                   # Dataset implementations
│   │   ├── base_dataset.py     # Abstract dataset class
│   │   ├── safety_dataset.py   # Harmless response generation
│   │   ├── clinical_dataset.py # Medical conversation summarization
│   │   └── conala_dataset.py   # Natural language → Python code
│   │
│   ├── models/                 # Model wrappers and adapters
│   │   ├── base_model.py       # Abstract model interface
│   │   ├── gemma_wrapper.py    # Gemma model wrapper
│   │   ├── peft_adapters.py    # LoRA adapter implementation
│   │   └── reward_model.py     # Hybrid reward model
│   │
│   ├── trainers/               # Training implementations
│   │   ├── base_trainer.py     # Common training loop logic
│   │   ├── sft_trainer.py      # Supervised fine-tuning
│   │   ├── rl_trainer.py       # PPO, GRPO, DPO algorithms
│   │   └── staged_trainer.py   # SFT → RL pipeline orchestration
│   │
│   ├── evaluation/             # Metrics and evaluation
│   │   ├── evaluator.py        # Evaluation harness
│   │   ├── metrics.py          # ROUGE, BERTScore, harmlessness
│   │   └── human_eval_manifest.py # Human annotation generation
│   │
│   ├── experiments/            # CLI experiment runner
│   │   └── runner.py           # Main entry point
│   │
│   └── api/                    # Inference API
│       └── serve_demo.py       # Flask demo server
│
├── data/
│   ├── datasets/               # Dataset storage
│   │   └── safety/             # Safety task data (train/val/test.jsonl)
│   └── download_datasets.py    # Dataset download script
│
├── tests/                      # Unit tests
│   ├── test_config.py
│   ├── test_dataset.py
│   ├── test_reward_model.py
│   └── test_trainer_sft.py
│
├── Dockerfile                  # CUDA 12.1 training container
├── pyproject.toml              # Package configuration
├── requirements.txt            # Pinned dependencies
└── run_experiment.sh           # Training wrapper script
```

---

## Configuration

Experiments are configured via YAML files. Key parameters:

```yaml
# configs/sft_lora_270m.yml
model:
  base_checkpoint: google/gemma-3-270m-it
  peft_type: lora           # lora | none
  peft_rank: 8              # LoRA rank (higher = more params)
  peft_alpha: 16            # LoRA scaling factor

training:
  mode: sft                 # sft | rl | staged
  epochs: 3
  batch_size: 4
  learning_rate: 5e-5
  gradient_accumulation_steps: 4
  max_seq_length: 512
  bf16: true                # Use bfloat16 precision

rl:
  algorithm: ppo            # ppo | grpo | dpo
  ppo_clip: 0.2
  kl_coef: 0.1              # KL penalty coefficient
```

See `configs/defaults.yml` for all available options.

---

## Training Paradigms

### 1. Supervised Fine-Tuning (SFT)

Standard instruction tuning with cross-entropy loss on (prompt, response) pairs.

```python
# Uses LoRA for parameter efficiency
# Only trains ~0.1% of parameters
python -m src.experiments.runner --config configs/sft_lora_270m.yml
```

### 2. Reinforcement Learning (RL)

Three algorithms available:

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **PPO** | Proximal Policy Optimization with clipped objective | General RLHF |
| **GRPO** | Group-relative rewards without value function | Simpler alternative to PPO |
| **DPO** | Direct Preference Optimization from preference pairs | No reward model needed |

```python
# Configure in config YAML
rl:
  algorithm: ppo  # or grpo, dpo
```

### 3. Staged Training

Two-phase approach: SFT warmup followed by RL refinement.

```python
python -m src.experiments.runner --config configs/staged_sft_then_ppo.yml
```

---

## Evaluation

### Automated Metrics

| Task | Metrics |
|------|---------|
| Safety | Harmlessness rate, ROUGE |
| Clinical | ROUGE, BERTScore |
| Code | BLEU, Exact Match |

### Human Evaluation

Generate annotation manifests for human evaluation:

```python
from src.evaluation.human_eval_manifest import generate_manifest

generate_manifest(
    prompts, references, sft_outputs, rl_outputs, staged_outputs,
    output_path="eval_manifest.csv",
    num_samples=200
)
```

---

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Quality

```bash
# Linting
ruff check src/

# Formatting
black src/

# Type checking
mypy src/
```

### Docker

```bash
# Build container
docker build -t gemma-alignment .

# Run training
docker run --gpus all -v $(pwd):/workspace gemma-alignment \
    python -m src.experiments.runner --config configs/sft_lora_270m.yml
```

---

## Hardware Requirements

| Configuration | VRAM | Training Time (3 epochs) |
|---------------|------|--------------------------|
| Gemma 270M + LoRA | 6-8 GB | ~2 hours on A100 |
| Gemma 270M Full | 12-16 GB | ~4 hours on A100 |

---

## Citation

```bibtex
@misc{gemma-alignment-paradigms,
  author = {Sanjay K. Saravanan},
  title = {Gemma Alignment Paradigms: A Comparative Study},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/sanjay-ks23/Gemma-Alignment-Paradigms}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Google Gemma](https://ai.google.dev/gemma) for the base models
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) for model loading
- [PEFT](https://github.com/huggingface/peft) for LoRA implementation
