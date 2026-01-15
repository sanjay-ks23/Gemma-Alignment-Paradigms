"""Evaluation harness for model assessment."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from src.core.config import ExperimentConfig
from src.core.utils import get_logger
from src.data.base_dataset import BaseDataset
from src.models.base_model import ModelWrapper
from src.tokenization.tokenizer_wrapper import TokenizerWrapper
from src.evaluation import metrics


class Evaluator:
    """Evaluation harness for alignment experiments."""
    
    def __init__(self, cfg: ExperimentConfig, tokenizer: TokenizerWrapper):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.logger = get_logger("gemma-alignment")
        self.device = torch.device(
            cfg.device if cfg.device != "auto" else 
            ("cuda" if torch.cuda.is_available() else "cpu")
        )
    
    def evaluate(
        self,
        model: ModelWrapper,
        dataset: BaseDataset,
        split: str = "val",
        max_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """Run evaluation on dataset."""
        model.eval()
        
        predictions = []
        references = []
        
        dataloader = dataset.get_dataloader(
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
        )
        
        total_samples = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", disable=self.cfg.debug):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=256,
                    do_sample=False,
                )
                
                generated = self.tokenizer.decode_batch(
                    outputs[:, input_ids.size(1):],
                    skip_special_tokens=True,
                )
                predictions.extend(generated)
                
                if "labels" in batch:
                    labels = batch["labels"]
                    for i in range(labels.size(0)):
                        valid_ids = labels[i][labels[i] != -100]
                        references.append(self.tokenizer.decode(valid_ids.tolist()))
                
                total_samples += len(generated)
                if max_samples and total_samples >= max_samples:
                    break
        
        results = self._compute_task_metrics(predictions, references)
        
        self.logger.info(f"Evaluation results ({split}):")
        for name, value in results.items():
            self.logger.info(f"  {name}: {value:.4f}")
        
        return results
    
    def _compute_task_metrics(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        results = {}
        task = self.cfg.task
        
        if task == "safety":
            results["harmlessness"] = metrics.compute_harmlessness(predictions)
            if references:
                results.update(metrics.compute_rouge(predictions, references))
        
        elif task == "clinical":
            if references:
                results.update(metrics.compute_rouge(predictions, references))
                results.update(metrics.compute_bertscore(predictions, references))
        
        elif task == "conala":
            if references:
                results.update(metrics.compute_codebleu(predictions, references))
                exact_matches = sum(
                    1 for p, r in zip(predictions, references) if p.strip() == r.strip()
                )
                results["exact_match"] = exact_matches / len(predictions)
        
        results["avg_length"] = sum(len(p.split()) for p in predictions) / len(predictions)
        
        return results
