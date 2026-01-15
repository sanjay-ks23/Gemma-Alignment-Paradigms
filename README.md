# Gemma Alignment

PyTorch implementation for alignment training on Google Gemma 3 270M.

## Paradigms

- **SFT**: Supervised fine-tuning with LoRA
- **RL**: PPO, GRPO, DPO algorithms
- **Staged**: SFT followed by RL

## Setup

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# SFT with LoRA
python -m src.experiments.runner --config configs/sft_lora_270m.yml

# PPO RL
python -m src.experiments.runner --config configs/ppo_rl_270m.yml

# Staged pipeline
python -m src.experiments.runner --config configs/staged_sft_then_ppo.yml

# Debug mode
python -m src.experiments.runner --config configs/sft_lora_270m.yml --debug
```

## Structure

```
src/
├── core/         Config, utils
├── data/         Datasets
├── models/       Gemma wrapper, LoRA, reward model
├── trainers/     SFT, RL, staged trainers
├── evaluation/   Metrics
└── experiments/  CLI runner
```

## Model Access

1. Accept license at [google/gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it)
2. Run `huggingface-cli login`

## Testing

```bash
pytest tests/ -v
```
