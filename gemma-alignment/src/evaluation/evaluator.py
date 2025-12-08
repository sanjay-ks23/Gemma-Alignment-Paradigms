"""
Evaluation harness for model assessment.

This module provides the Evaluator class that coordinates automatic
and human evaluation of aligned models.
"""

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
    """
    Evaluation harness for alignment experiments.
    
    Performs automatic evaluation using task-specific metrics and
    generates manifests for human evaluation.
    
    Attributes:
        cfg: Experiment configuration.
        tokenizer: Tokenizer for encoding/decoding.
    
    Example:
        >>> evaluator = Evaluator(cfg, tokenizer)
        >>> results = evaluator.evaluate(model, dataset, split="test")
        >>> print(results["rouge_l"], results["harmlessness"])
    """
    
    def __init__(
        self,
        cfg: ExperimentConfig,
        tokenizer: TokenizerWrapper,
    ):
        """
        Initialize the evaluator.
        
        Args:
            cfg: Experiment configuration.
            tokenizer: Tokenizer wrapper.
        """
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
        """
        Run evaluation on a dataset split.
        
        Args:
            model: Model to evaluate.
            dataset: Dataset containing evaluation samples.
            split: Split name for logging.
            max_samples: Maximum samples to evaluate (None = all).
        
        Returns:
            Dictionary of metric names to values.
        """
        self.logger.info(f"Evaluating on {split} split...")
        
        model.eval()
        
        predictions = []
        references = []
        prompts = []
        
        # Generate predictions
        dataloader = dataset.get_dataloader(
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
        )
        
        total_samples = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Generate
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=256,
                    do_sample=False,  # Greedy for evaluation
                )
                
                # Decode
                generated_texts = self.tokenizer.decode_batch(
                    outputs[:, input_ids.size(1):],
                    skip_special_tokens=True,
                )
                prompt_texts = self.tokenizer.decode_batch(
                    input_ids, skip_special_tokens=True
                )
                
                predictions.extend(generated_texts)
                prompts.extend(prompt_texts)
                
                # Get references if available
                if "labels" in batch:
                    labels = batch["labels"]
                    # Filter out -100 padding
                    ref_texts = []
                    for i in range(labels.size(0)):
                        valid_ids = labels[i][labels[i] != -100]
                        ref_texts.append(
                            self.tokenizer.decode(valid_ids.tolist())
                        )
                    references.extend(ref_texts)
                
                total_samples += len(generated_texts)
                if max_samples and total_samples >= max_samples:
                    break
        
        # Compute metrics based on task
        results = self._compute_task_metrics(predictions, references, prompts)
        
        # Log results
        self.logger.info(f"Evaluation results ({split}):")
        for name, value in results.items():
            self.logger.info(f"  {name}: {value:.4f}")
        
        return results
    
    def _compute_task_metrics(
        self,
        predictions: List[str],
        references: List[str],
        prompts: List[str],
    ) -> Dict[str, float]:
        """
        Compute task-specific metrics.
        
        Args:
            predictions: Model predictions.
            references: Reference texts.
            prompts: Original prompts.
        
        Returns:
            Dictionary of metrics.
        """
        results = {}
        
        task = self.cfg.task
        
        if task == "safety":
            # Harmlessness metrics
            harmlessness = metrics.compute_harmlessness(predictions)
            results["harmlessness"] = harmlessness
            
            # ROUGE if references available
            if references:
                rouge = metrics.compute_rouge(predictions, references)
                results.update(rouge)
        
        elif task == "clinical":
            # Summarization metrics
            if references:
                rouge = metrics.compute_rouge(predictions, references)
                results.update(rouge)
                
                bertscore = metrics.compute_bertscore(predictions, references)
                results.update(bertscore)
        
        elif task == "conala":
            # Code generation metrics
            if references:
                codebleu = metrics.compute_codebleu(predictions, references)
                results.update(codebleu)
                
                # Exact match
                exact_matches = sum(
                    1 for p, r in zip(predictions, references)
                    if p.strip() == r.strip()
                )
                results["exact_match"] = exact_matches / len(predictions)
        
        # Per-sample length stats
        avg_length = sum(len(p.split()) for p in predictions) / len(predictions)
        results["avg_length"] = avg_length
        
        return results
    
    def run_statistical_tests(
        self,
        results: Dict[str, List[float]],
        metric_name: str = "harmlessness",
        num_bootstrap: int = 1000,
    ) -> Dict[str, float]:
        """
        Run statistical significance tests between methods.
        
        Uses bootstrap resampling to compute confidence intervals
        and p-values.
        
        Args:
            results: Dictionary mapping method names to lists of scores.
            metric_name: Metric to compare.
            num_bootstrap: Number of bootstrap samples.
        
        Returns:
            Dictionary with statistical test results.
        """
        import numpy as np
        from scipy import stats
        
        method_names = list(results.keys())
        if len(method_names) < 2:
            return {}
        
        test_results = {}
        
        for i, method1 in enumerate(method_names):
            for method2 in method_names[i + 1:]:
                scores1 = np.array(results[method1])
                scores2 = np.array(results[method2])
                
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(scores1, scores2)
                
                # Bootstrap confidence interval for difference
                diffs = []
                n = len(scores1)
                for _ in range(num_bootstrap):
                    idx = np.random.choice(n, n, replace=True)
                    boot_diff = scores1[idx].mean() - scores2[idx].mean()
                    diffs.append(boot_diff)
                
                ci_lower = np.percentile(diffs, 2.5)
                ci_upper = np.percentile(diffs, 97.5)
                
                key = f"{method1}_vs_{method2}"
                test_results[key] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "mean_diff": scores1.mean() - scores2.mean(),
                    "ci_95": (ci_lower, ci_upper),
                }
        
        return test_results
    
    def compare_paradigms(
        self,
        models: Dict[str, ModelWrapper],
        dataset: BaseDataset,
        output_dir: str,
    ) -> Dict[str, Any]:
        """
        Compare multiple alignment paradigms.
        
        Evaluates each model and generates comparison report.
        
        Args:
            models: Dictionary of paradigm names to models.
            dataset: Evaluation dataset.
            output_dir: Directory for output files.
        
        Returns:
            Comparison results.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        all_predictions = {}
        
        for name, model in models.items():
            self.logger.info(f"\nEvaluating {name}...")
            results = self.evaluate(model, dataset, split="test")
            all_results[name] = results
        
        # Statistical comparison
        if len(models) >= 2:
            # Convert to per-sample scores for statistical testing
            # (simplified - in practice, store per-sample metrics)
            stat_results = {}
            for metric in ["harmlessness", "rouge_l"]:
                paradigm_scores = {}
                for name, results in all_results.items():
                    if metric in results:
                        # Simulate per-sample scores (placeholder)
                        paradigm_scores[name] = [results[metric]] * 100
                
                if len(paradigm_scores) >= 2:
                    stat_results[metric] = self.run_statistical_tests(
                        paradigm_scores, metric
                    )
        
        return {
            "metrics": all_results,
            "statistical_tests": stat_results,
        }
