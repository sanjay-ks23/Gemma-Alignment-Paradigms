"""Human evaluation manifest generator."""

import csv
import random
from pathlib import Path
from typing import List


def generate_manifest(
    prompts: List[str],
    references: List[str],
    sft_outputs: List[str],
    rl_outputs: List[str],
    staged_outputs: List[str],
    output_path: str,
    num_samples: int = 200,
    seed: int = 42,
) -> str:
    """Generate CSV manifest for human evaluation."""
    assert len(prompts) == len(references) == len(sft_outputs) == len(rl_outputs) == len(staged_outputs)
    
    random.seed(seed)
    indices = random.sample(range(len(prompts)), min(num_samples, len(prompts)))
    
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "prompt", "reference", "sft_output", "rl_output", "staged_output",
            "sft_rating", "rl_rating", "staged_rating", "preferred", "notes"
        ])
        
        for i, idx in enumerate(indices):
            writer.writerow([
                i + 1,
                prompts[idx][:1000],
                references[idx][:1000],
                sft_outputs[idx][:1000],
                rl_outputs[idx][:1000],
                staged_outputs[idx][:1000],
                "", "", "", "", ""
            ])
    
    return str(output)


def load_manifest(path: str) -> list:
    """Load manifest CSV file."""
    with open(path, "r", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def compute_agreement(annotations_a: list, annotations_b: list, column: str = "preferred") -> float:
    """Compute inter-annotator agreement."""
    assert len(annotations_a) == len(annotations_b)
    return sum(1 for a, b in zip(annotations_a, annotations_b) if a[column] == b[column]) / len(annotations_a)
