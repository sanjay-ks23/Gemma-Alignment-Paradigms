# Dataset Documentation

This directory contains datasets for alignment experiments.

## Directory Structure

```
datasets/
├── safety/          # Safe/Harmless Response Generation
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── clinical/        # Clinical Conversation Summarization
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
└── conala/          # Code Q&A
    ├── train.jsonl
    ├── val.jsonl
    └── test.jsonl
```

## Dataset Sources and Licensing

### Safety Dataset (Default)

**Source**: Anthropic HH-RLHF (Helpful and Harmless)
- Repository: https://huggingface.co/datasets/Anthropic/hh-rlhf
- License: MIT
- Citation:
  ```
  @article{bai2022training,
    title={Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback},
    author={Bai, Yuntao and others},
    journal={arXiv preprint arXiv:2204.05862},
    year={2022}
  }
  ```

**Format** (JSONL):
```json
{"prompt": "User question", "chosen": "Safe response", "rejected": "Unsafe response"}
```

### Clinical Dataset

**Source**: MTS-Dialog (Medical Transcription Summarization)
- Repository: https://github.com/abachaa/MTS-Dialog
- License: Research use only
- Note: Requires manual download due to licensing restrictions

**Format** (JSONL):
```json
{"dialogue": "Doctor: ... Patient: ...", "summary": "Clinical summary"}
```

### CoNaLa Dataset

**Source**: CoNaLa (Code/Natural Language Challenge)
- Repository: https://huggingface.co/datasets/neulab/conala
- License: CC BY-SA 4.0
- Citation:
  ```
  @inproceedings{yin2018mining,
    title={Mining Source Code from Stack Overflow},
    author={Yin, Pengcheng and others},
    booktitle={ACL},
    year={2018}
  }
  ```

**Format** (JSONL):
```json
{"intent": "Natural language description", "snippet": "Python code"}
```

## Downloading Datasets

Run the download script to prepare datasets:

```bash
# Download safety dataset (default)
python download_datasets.py --dataset safety --splits train,val,test

# Download all datasets
python download_datasets.py --dataset all

# Download to custom directory
python download_datasets.py --dataset safety --output-dir /path/to/output
```

## Data Processing

All datasets are converted to JSONL format with task-specific fields:

- **Safety**: prompt, chosen, rejected
- **Clinical**: dialogue, summary
- **CoNaLa**: intent, snippet

The download script handles:
1. Downloading from Hugging Face Hub
2. Preprocessing and formatting
3. Creating train/val/test splits
4. Generating synthetic data if download fails

## Usage Notes

1. The safety dataset includes preference pairs (chosen/rejected) for both SFT and RL training.
2. Clinical dataset requires manual download from the original source.
3. All datasets have synthetic fallbacks for testing without network access.
4. Default split sizes: train (5000), val (500), test (500).

## Adding Custom Datasets

To add a new dataset:

1. Create a new dataset class in `src/data/` inheriting from `BaseDataset`
2. Register with `@register("dataset", "name")`
3. Add download logic to `download_datasets.py`
4. Update this README with licensing information
