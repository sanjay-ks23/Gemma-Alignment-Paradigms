# Gemma 3n Therapeutic Platform

This repository combines a curriculum-learning fine-tuning pipeline for Google's Gemma 3n model with a production-ready Flask API that delivers contextual, safety-aware therapeutic conversations. Retrieval-Augmented Generation (RAG) augments the model with knowledge graph insights and vector-based exemplars, while documentation and automation streamline operations.

## Quick Start (Docker Compose)

1. **Clone & prepare environment variables**
   ```bash
   cp .env.example .env
   # edit GEMMA_* paths, NEO4J_AUTH, and other values as needed
   ```

2. **Populate required assets**
   - Place Gemma 3n base weights and LoRA adapters under `model-artifacts/` (see [backend setup](docs/setup/backend.md)).
   - Copy your frontend build into `frontend/dist/` or keep the placeholder for API-only usage.

3. **Launch the stack**
   ```bash
   docker compose up --build
   ```

4. **Validate services**
   ```bash
   curl http://localhost:5000/api/health
   open http://localhost:3000        # Frontend (optional)
   open http://localhost:7474        # Neo4j Browser (credentials from .env)
   ```

Stop the stack with `docker compose down` (add `-v` to clear volumes).

## Local Development

- **Backend**: Follow [docs/setup/backend.md](docs/setup/backend.md) to install dependencies, configure environment variables, and run the Flask server with Gunicorn or the built-in dev server.
- **Fine-Tuning**: The `Curriculum Learning/` package provides scripts for SFT, DPO, and curriculum training. Review its `README.md` for dataset details and command examples.
- **Frontend**: Build your SPA and copy the static output into `frontend/dist/`. Guidance lives in [docs/setup/frontend.md](docs/setup/frontend.md).

## Documentation

The full documentation suite (architecture, Graph RAG flow, deployment, configuration, safety, API, and runbooks) lives under [`docs/`](docs/README.md). Highlights:

- [Architecture Overview](docs/architecture.md)
- [Graph RAG Data Flow](docs/graph-rag.md)
- [Deployment Guide](docs/deployment.md)
- [Safety & Compliance for Minors in India](docs/safety-compliance.md)
- [OpenAPI Specification](docs/api/openapi.yaml)
- Operational runbooks for [ingestion](docs/operations/ingestion-runbook.md), [retraining](docs/operations/retraining-runbook.md), and [monitoring](docs/operations/monitoring-alerting.md)

## Key Directories

| Path | Description |
| --- | --- |
| `Curriculum Learning/` | Model training pipeline (data preprocessing, curriculum strategies, evaluation). |
| `Flask chat app/` | Flask backend, chat handler, and model loader (now environment-configurable). |
| `frontend/dist/` | Static frontend assets served by the Nginx container. |
| `model-artifacts/` | Placeholder for Gemma base weights and LoRA adapters mounted into the backend container. |
| `docs/` | Comprehensive system documentation suite. |

## Safety Note

This project targets mental health support scenarios. Review and comply with the [Safety & Compliance](docs/safety-compliance.md) guidelines—especially when serving minors in India. Always maintain human oversight for high-risk conversations and adhere to mandatory reporting laws.

## Contributing

- Keep documentation and configuration references in sync with code changes.
- Avoid committing proprietary model weights—use environment variables and volume mounts instead.
- Run formatting/linting tools corresponding to modified modules before submitting changes.

For issues or enhancement ideas, open a ticket and reference the relevant documentation section to speed up review.
