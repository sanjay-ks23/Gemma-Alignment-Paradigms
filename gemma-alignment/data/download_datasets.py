#!/usr/bin/env python3
"""
Dataset download and preparation script.

This script downloads and prepares datasets for the alignment experiments.
It supports the safety, clinical, and conala datasets.

Usage:
    python download_datasets.py --dataset safety --splits train,val,test
    python download_datasets.py --dataset all
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


DATASETS_DIR = Path(__file__).parent / "datasets"


def download_safety_dataset(splits: List[str], output_dir: Path) -> None:
    """
    Download and prepare the safety alignment dataset.
    
    Uses Anthropic's HH-RLHF dataset from Hugging Face.
    License: MIT
    
    Args:
        splits: List of splits to download (train, val, test).
        output_dir: Directory to save the prepared data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not HAS_DATASETS:
        print("Warning: 'datasets' library not installed. Creating synthetic data.")
        _create_synthetic_safety_data(splits, output_dir)
        return
    
    print("Downloading Anthropic HH-RLHF dataset...")
    
    try:
        # Load the harmless subset
        dataset = load_dataset("Anthropic/hh-rlhf", "harmless-base")
        
        # Map splits
        split_mapping = {
            "train": "train",
            "val": "test",  # Use test as validation since HH-RLHF has no val
            "test": "test",
        }
        
        for split in splits:
            hf_split = split_mapping.get(split, split)
            if hf_split not in dataset:
                print(f"Warning: Split '{hf_split}' not found in dataset.")
                continue
            
            data = dataset[hf_split]
            output_path = output_dir / f"{split}.jsonl"
            
            # Process and save samples
            processed = []
            for item in data:
                # Parse the chosen and rejected from the format
                chosen = item.get("chosen", "")
                rejected = item.get("rejected", "")
                
                # Extract prompt from chosen (format: "Human: ... Assistant: ...")
                prompt = _extract_prompt(chosen)
                chosen_response = _extract_response(chosen)
                rejected_response = _extract_response(rejected)
                
                if prompt and chosen_response:
                    processed.append({
                        "prompt": prompt,
                        "chosen": chosen_response,
                        "rejected": rejected_response,
                    })
            
            # Limit size for practical training
            if split == "train":
                processed = processed[:5000]  # 5K for training
            else:
                processed = processed[:500]   # 500 for val/test
            
            _save_jsonl(processed, output_path)
            print(f"Saved {len(processed)} samples to {output_path}")
            
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Falling back to synthetic data.")
        _create_synthetic_safety_data(splits, output_dir)


def download_clinical_dataset(splits: List[str], output_dir: Path) -> None:
    """
    Download and prepare the clinical summarization dataset.
    
    Note: MTS-Dialog requires manual download from the original source.
    This creates stub files with synthetic data for testing.
    
    Args:
        splits: List of splits to prepare.
        output_dir: Directory to save the prepared data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Note: MTS-Dialog requires manual download.")
    print("Creating synthetic clinical data for testing...")
    
    synthetic_data = [
        {
            "dialogue": (
                "Doctor: What brings you in today?\n"
                "Patient: I've been having persistent headaches.\n"
                "Doctor: How long have you had them?\n"
                "Patient: About two weeks now.\n"
                "Doctor: On a scale of 1-10, how severe is the pain?\n"
                "Patient: Usually around 6 or 7."
            ),
            "summary": (
                "Patient presents with a two-week history of persistent headaches "
                "rated 6-7/10 in severity. No additional symptoms reported at this time."
            ),
        },
        {
            "dialogue": (
                "Doctor: How are you feeling today?\n"
                "Patient: I have a cough that won't go away.\n"
                "Doctor: Is it a dry cough or productive?\n"
                "Patient: Mostly dry, but sometimes I cough up phlegm.\n"
                "Doctor: Any fever or shortness of breath?\n"
                "Patient: No fever, but I do get winded easily."
            ),
            "summary": (
                "Patient reports persistent cough, primarily dry with occasional "
                "productive episodes. Denies fever but reports dyspnea on exertion."
            ),
        },
    ]
    
    for split in splits:
        output_path = output_dir / f"{split}.jsonl"
        
        # Vary data size by split
        multiplier = 50 if split == "train" else 10
        data = synthetic_data * multiplier
        
        _save_jsonl(data, output_path)
        print(f"Saved {len(data)} synthetic samples to {output_path}")


def download_conala_dataset(splits: List[str], output_dir: Path) -> None:
    """
    Download and prepare the CoNaLa code generation dataset.
    
    Uses the CoNaLa dataset from Hugging Face.
    License: CC BY-SA 4.0
    
    Args:
        splits: List of splits to download.
        output_dir: Directory to save the prepared data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not HAS_DATASETS:
        print("Warning: 'datasets' library not installed. Creating synthetic data.")
        _create_synthetic_conala_data(splits, output_dir)
        return
    
    print("Downloading CoNaLa dataset...")
    
    try:
        dataset = load_dataset("neulab/conala")
        
        split_mapping = {
            "train": "train",
            "val": "test",
            "test": "test",
        }
        
        for split in splits:
            hf_split = split_mapping.get(split, split)
            if hf_split not in dataset:
                print(f"Warning: Split '{hf_split}' not found.")
                continue
            
            data = dataset[hf_split]
            output_path = output_dir / f"{split}.jsonl"
            
            processed = []
            for item in data:
                intent = item.get("intent", "")
                snippet = item.get("snippet", "")
                
                if intent and snippet:
                    processed.append({
                        "intent": intent,
                        "snippet": snippet,
                    })
            
            if split == "train":
                processed = processed[:2000]
            else:
                processed = processed[:200]
            
            _save_jsonl(processed, output_path)
            print(f"Saved {len(processed)} samples to {output_path}")
            
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Falling back to synthetic data.")
        _create_synthetic_conala_data(splits, output_dir)


def _extract_prompt(text: str) -> str:
    """Extract the human prompt from HH-RLHF format."""
    if "Human:" not in text:
        return ""
    
    # Get the first human turn
    parts = text.split("Human:")
    if len(parts) < 2:
        return ""
    
    human_turn = parts[1].split("Assistant:")[0].strip()
    return human_turn


def _extract_response(text: str) -> str:
    """Extract the assistant response from HH-RLHF format."""
    if "Assistant:" not in text:
        return ""
    
    # Get the last assistant turn
    parts = text.split("Assistant:")
    if len(parts) < 2:
        return ""
    
    # Get the last response
    response = parts[-1].split("Human:")[0].strip()
    return response


def _save_jsonl(data: list, path: Path) -> None:
    """Save data to JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _create_synthetic_safety_data(splits: List[str], output_dir: Path) -> None:
    """Create synthetic safety data for testing."""
    synthetic = [
        {
            "prompt": "How can I be more helpful to my community?",
            "chosen": "You can volunteer, donate to local causes, or simply be kind to your neighbors. Community involvement strengthens social bonds.",
            "rejected": "Communities don't matter, just focus on yourself.",
        },
        {
            "prompt": "What should I do if I see someone in distress?",
            "chosen": "Check if they need help, offer support, and if it's an emergency, call emergency services. Always prioritize safety.",
            "rejected": "Ignore them and walk away.",
        },
        {
            "prompt": "How do I handle a disagreement with a friend?",
            "chosen": "Listen to their perspective, express your feelings calmly, and work together to find common ground. Healthy communication is key.",
            "rejected": "Cut them out of your life immediately.",
        },
    ]
    
    for split in splits:
        multiplier = 100 if split == "train" else 20
        data = synthetic * multiplier
        
        output_path = output_dir / f"{split}.jsonl"
        _save_jsonl(data, output_path)
        print(f"Created {len(data)} synthetic samples at {output_path}")


def _create_synthetic_conala_data(splits: List[str], output_dir: Path) -> None:
    """Create synthetic CoNaLa data for testing."""
    synthetic = [
        {"intent": "sort a list in descending order", "snippet": "sorted(my_list, reverse=True)"},
        {"intent": "read all lines from a file", "snippet": "with open('file.txt') as f:\n    lines = f.readlines()"},
        {"intent": "get current timestamp", "snippet": "import time\ntimestamp = time.time()"},
        {"intent": "join list elements with comma", "snippet": "','.join(my_list)"},
        {"intent": "check if key exists in dictionary", "snippet": "'key' in my_dict"},
    ]
    
    for split in splits:
        multiplier = 100 if split == "train" else 20
        data = synthetic * multiplier
        
        output_path = output_dir / f"{split}.jsonl"
        _save_jsonl(data, output_path)
        print(f"Created {len(data)} synthetic samples at {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download alignment datasets")
    parser.add_argument(
        "--dataset",
        choices=["safety", "clinical", "conala", "all"],
        default="safety",
        help="Dataset to download",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated list of splits to download",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/datasets/<dataset>)",
    )
    args = parser.parse_args()
    
    splits = [s.strip() for s in args.splits.split(",")]
    
    datasets_to_download = []
    if args.dataset == "all":
        datasets_to_download = ["safety", "clinical", "conala"]
    else:
        datasets_to_download = [args.dataset]
    
    for dataset in datasets_to_download:
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = DATASETS_DIR / dataset
        
        print(f"\n{'='*50}")
        print(f"Preparing {dataset} dataset")
        print(f"{'='*50}")
        
        if dataset == "safety":
            download_safety_dataset(splits, output_dir)
        elif dataset == "clinical":
            download_clinical_dataset(splits, output_dir)
        elif dataset == "conala":
            download_conala_dataset(splits, output_dir)
    
    print("\nDataset preparation complete!")


if __name__ == "__main__":
    main()
