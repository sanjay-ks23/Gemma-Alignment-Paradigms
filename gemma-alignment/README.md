# Gemma Alignment Paradigms

A production-grade PyTorch-first implementation of alignment training paradigms for Google Gemma 3 models (270M and 1B), supporting comparative studies across three approaches:

1. **SFT + PEFT**: Supervised Fine-Tuning with LoRA/QLoRA adapters
2. **RL-only**: Reinforcement Learning using PPO, GRPO, or DPO algorithms
3. **Staged Pipeline**: SFT followed by RL refinement

## Features

- PyTorch-native training loops (no black-box library dependencies)
- Modular OOP architecture with clean abstractions
- Configuration-driven experiments via YAML
- Multiple RL algorithms: PPO, GRPO, DPO
- Hybrid reward model (trainable + heuristic)
- Automatic and human evaluation support
- Full reproducibility with seeds and Docker

## Quick Start

### Installation

```bash
# Clone and enter repository
cd gemma-alignment

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

### Download Datasets

```bash
python data/download_datasets.py --dataset safety --splits train,val,test
```

### Run Debug Training (CPU/Local)

Quick validation run on synthetic data:

```bash
python -m src.experiments.runner --config configs/sft_lora_270m.yml --debug
```

### Run Full Training (Single GPU)

```bash
# SFT with LoRA on 270M model
python -m src.experiments.runner --config configs/sft_lora_270m.yml

# RL with PPO on 270M model  
python -m src.experiments.runner --config configs/ppo_rl_270m.yml

# Staged: SFT -> PPO
python -m src.experiments.runner --config configs/staged_sft_then_ppo.yml

# QLoRA on 1B model (requires more VRAM)
python -m src.experiments.runner --config configs/qlora_1b.yml
```

### Using run_experiment.sh

```bash
chmod +x run_experiment.sh

# With dataset download
./run_experiment.sh configs/sft_lora_270m.yml --download

# Debug mode
./run_experiment.sh configs/sft_lora_270m.yml --debug
```

## Project Structure

```
gemma-alignment/
├── configs/              # YAML experiment configurations
├── data/                 # Dataset download scripts and storage
├── src/
│   ├── core/            # Configuration, registry, utilities
│   ├── data/            # Dataset classes
│   ├── tokenization/    # Tokenizer wrapper
│   ├── models/          # Model wrappers, PEFT, reward model
│   ├── trainers/        # SFT, RL, Staged trainers
│   ├── evaluation/      # Metrics and evaluation harness
│   ├── experiments/     # CLI runner
│   └── api/             # Demo inference server
├── tests/               # Unit tests
├── docs/                # Design documentation
└── benchmarks/          # Evaluation outputs
```

## Configuration

All experiments are configured via YAML files. See `configs/defaults.yml` for all options.

Key configuration sections:

```yaml
task: safety              # safety | clinical | conala
model:
  base_checkpoint: google/gemma-3-270m-it
  peft_type: lora         # lora | qlora | none
  peft_rank: 8
training:
  mode: sft               # sft | rl | staged
  epochs: 3
  batch_size: 4
rl:
  algorithm: ppo          # ppo | grpo | dpo
reward:
  type: hybrid            # heuristic | trainable | hybrid
```

## Evaluation

### Automatic Evaluation

```python
from src.evaluation.evaluator import Evaluator
from src.core.config import load_config

cfg = load_config("configs/sft_lora_270m.yml")
evaluator = Evaluator(cfg, tokenizer)
results = evaluator.evaluate(model, test_dataset, split="test")
```

### Human Evaluation

Generate manifest for human annotators:

```python
from src.evaluation.human_eval_manifest import generate_manifest

generate_manifest(
    prompts, references,
    sft_outputs, rl_outputs, staged_outputs,
    output_path="human_eval.csv",
    num_samples=200
)
```

## Inference Server

Run a demo server for interactive testing:

```bash
python src/api/serve_demo.py --checkpoint ./checkpoints/sft_lora_270m --port 5000
```

API endpoints:
- `GET /health` - Health check
- `POST /generate` - Generate text from prompt
- `POST /batch_generate` - Batch generation

## Docker

```bash
# Build image
docker build -t gemma-alignment .

# Run training
docker run --gpus all -v $(pwd)/outputs:/app/outputs gemma-alignment \
    --config configs/sft_lora_270m.yml
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_config.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Requirements

- Python 3.10+
- PyTorch 2.1+
- CUDA 12.1 (for GPU training)
- 4-8GB VRAM for 270M model
- 12-16GB VRAM for 1B model (8-10GB with QLoRA)

## Model Access

Gemma models require approval from Google. Accept the license at:
- [google/gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it)
- [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it)

Then authenticate with Hugging Face:
```bash
huggingface-cli login
```

## License

MIT License. See LICENSE for details.

Dataset licenses:
- Anthropic HH-RLHF: MIT
- MTS-Dialog: Research use only
- CoNaLa: CC BY-SA 4.0
