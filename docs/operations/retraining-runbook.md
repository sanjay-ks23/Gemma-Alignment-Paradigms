# Retraining Workflow Runbook

Use this runbook to retrain the Gemma 3n therapeutic model whenever new data is ingested, evaluation metrics drift, or safety policies change.

## 1. Trigger Criteria

- New counselling transcripts or policy updates require model refresh.
- Evaluation metrics (BLEU, ROUGE, empathy scores) fall below thresholds.
- Compliance review mandates updated system prompts or guardrails.

## 2. Preparation

- [ ] Complete ingestion per [Ingestion Runbook](ingestion-runbook.md) and confirm version alignment.
- [ ] Ensure `WANDB_API_KEY` is set if experiment tracking is required.
- [ ] Verify GPU capacity or adjust configs for CPU-only training (smaller batch sizes, shorter sequence lengths).

## 3. Environment Setup

```bash
cd "Curriculum Learning"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Configuration Review

- Update `configs/model_config.yaml` with new model paths or quantisation hints.
- Adjust `configs/training_config.yaml` for epoch counts, batch sizes, or gradient accumulation.
- Update `configs/lora_config.yaml` if target modules change between releases.

## 5. Supervised Fine-Tuning (Curriculum Recommended)

```bash
python scripts/train_curriculum.py \
  --output_dir ./checkpoints/curriculum_2024_10 \
  --experiment_name gemma3n_curriculum_2024_10 \
  --curriculum_strategy therapeutic \
  --target_dataset_size 2500 \
  --wandb_project gemma-therapeutic  # optional
```

Monitor progress via console logs or Weights & Biases. Early stopping is available via `training.training_args`.

## 6. Direct Preference Optimisation (Optional)

```bash
python scripts/preference_tuning.py \
  --output_dir ./checkpoints/dpo_2024_10 \
  --sft_model_path ./checkpoints/curriculum_2024_10/latest \
  --experiment_name gemma3n_dpo_2024_10
```

Expect higher latency; schedule during low usage windows.

## 7. Adapter Merge / Export

```bash
python scripts/merge_adapters.py \
  --adapter_path ./checkpoints/dpo_2024_10/latest \
  --base_model google/gemma-3n-E2B-it \
  --output_dir ../model-artifacts/gemma-3n-2024_10 \
  --test_generation
```

- `--test_generation` performs a smoke test to confirm merged weights generate coherent text.
- Store outputs under a versioned directory (mount location referenced by `GEMMA_MODEL_PATH`).

## 8. Evaluation & Sign-Off

1. Run automated metrics:
   ```bash
   python src/eval.py \
     --model_path ../model-artifacts/gemma-3n-2024_10 \
     --dataset_path ./data/validation \
     --output_path ./eval_reports/2024_10
   ```
2. Conduct human evaluation on a curated set of scenarios, especially high-risk cases.
3. Update release notes with metrics, qualitative feedback, and known limitations.

## 9. Deployment

- Update `.env` to point `GEMMA_MODEL_PATH` and `GEMMA_ADAPTER_PATH` to the new directory.
- Restart the backend container (`docker compose up -d --force-recreate backend`).
- Warm the model by issuing a few `/api/chat` requests to pre-populate caches.

## 10. Post-Deployment Monitoring

- Watch latency, GPU utilisation, and error rates for the first 24 hours.
- Confirm retrieval alignment by spot-checking Graph RAG context logs.
- Keep the previous model version available for rapid rollback.

## 11. Documentation & Handover

- Log the retraining in the change management system with dataset versions, training configs, and evaluation results.
- Notify compliance stakeholders and customer support of the rollout timeline.
- Archive training logs and checkpoints to secure storage.

Adhering to this runbook ensures reproducible, auditable retraining cycles aligned with both technical and regulatory requirements.
