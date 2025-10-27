# Configuration Reference

This catalogue lists the key configuration surfaces that drive the Gemma 3n therapeutic platform. Use it as a checklist when promoting changes across environments.

## Environment Variables

> Copy `.env.example` to `.env` and adjust values per environment.

| Variable | Default | Description |
| --- | --- | --- |
| `BACKEND_PORT` | `5000` | Host port bound to the Flask API container. |
| `FRONTEND_PORT` | `3000` | Host port for the static frontend (Nginx). |
| `NEO4J_BOLT_PORT` | `7687` | Neo4j Bolt driver port exposed to clients. |
| `NEO4J_HTTP_PORT` | `7474` | Neo4j Browser / HTTP endpoint. |
| `QDRANT_HTTP_PORT` | `6333` | Qdrant REST API port. |
| `QDRANT_GRPC_PORT` | `6334` | Qdrant gRPC port. |
| `NEO4J_AUTH` | `neo4j/local123` | Credentials in `user/password` form; change immediately for production. |
| `GEMMA_BASE_PATH` | `/models` | Directory inside the backend container containing base model files. |
| `GEMMA_MODEL_PATH` | `/models/gemma-3n` | Directory containing the merged/adapter-aware Gemma model. |
| `GEMMA_ADAPTER_PATH` | `/models/gemma-3n` | LoRA adapter directory (can diverge from base path for adapter-only deployments). |
| `GEMMA_OFFLOAD_DIR` | `/offload` | Scratch space for CPU offloading when running in 8-bit. |
| `HOST_MODEL_DIR` | `./model-artifacts` | Host directory mounted read-only into `/models`. |
| `BITSANDBYTES_FORCE_CPU` | `1` | Forces bitsandbytes to CPU-only kernels; set to `0` when GPUs are available. |
| `FRONTEND_API_URL` | `http://backend:5000/api` | Base URL used by frontend assets to reach the API through the Compose network. |
| `WANDB_API_KEY` | _empty_ | Enables Weights & Biases experiment tracking when set. |
| `HF_TOKEN` | _empty_ | Hugging Face hub token for gated models or dataset downloads. |
| `PIP_EXTRA_INDEX_URL` | `https://download.pytorch.org/whl/cpu` | Optional extra index used during backend image builds (passed as build arg). |

## Application Configuration Files

| File | Purpose | Key Fields |
| --- | --- | --- |
| `Curriculum Learning/configs/model_config.yaml` | Base model + tokenizer configuration | `model.name`, `model.device_map`, `tokenizer.padding_side`, `quantization.*` |
| `Curriculum Learning/configs/training_config.yaml` | Default training arguments | `training.num_train_epochs`, `training.per_device_train_batch_size`, `training.gradient_accumulation_steps` |
| `Curriculum Learning/configs/lora_config.yaml` | LoRA adapter configuration | `lora.r`, `lora.target_modules`, `lora_dropout`, stage-specific overrides |
| `Curriculum Learning/configs/dpo_config.yaml` (if present) | Preference optimisation settings | `beta`, `loss_type`, `per_device_train_batch_size` |
| `Curriculum Learning/scripts/*.py` | CLI entrypoints that combine configs | Flags such as `--curriculum_strategy`, `--output_dir`, `--merge_final_model` |

> Use `ConfigManager` (`src/utils.py`) to programmatically load and cache YAML configurations. It ensures directories exist before writing and centralises validation.

## Logging & Telemetry

| Component | Configuration | Notes |
| --- | --- | --- |
| Flask backend | Standard Python logging; inherits `LOGLEVEL` (set via environment variable if needed). | Log files written to stdout by Gunicorn for container aggregation. |
| Curriculum training | `Logger` in `utils.py` writes timestamped `.log` files under `logs/` and streams to console. | Filenames include module name and timestamp. |
| Experiment tracking | Controlled by `training_config['wandb']`. | When `WANDB_API_KEY` is provided, metrics, configs, and artefacts are pushed to W&B. |

## Graph & Vector Schema Settings

| Location | Description |
| --- | --- |
| `docs/setup/neo4j.md` | Indexes, constraints, and APOC settings for Neo4j. |
| `docs/setup/vector-store.md` | Default Qdrant collection names, payload schema, and shard configuration. |

## Prompting & Chat Handler

- `TherapeuticChatHandler.system_prompt` contains the baseline counselling persona. Update this string to reflect policy changes.
- Environment overrides (`FLASK_APP_PATH`, `FLASK_TEMPLATE_PATH`, `FLASK_STATIC_PATH`) can be used to inject different templates without modifying code.
- Generation parameters (temperature, top_p, top_k, repetition penalty) are exposed via the `/api/chat` payload—see [API documentation](api/README.md).

Keep this reference current whenever you add new configuration knobs; consistency reduces surprises during deployments and incident response.
