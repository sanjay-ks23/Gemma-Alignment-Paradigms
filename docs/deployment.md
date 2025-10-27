# Deployment Guide

This guide explains how to deploy the complete Gemma 3n therapeutic platform using the provided Docker Compose stack. It also outlines environment variables, persistent volumes, and production hardening tips.

## Prerequisites

- Docker Engine 24+ and Docker Compose Plugin 2.20+ (or Docker Desktop with Compose).
- Access to Gemma 3n base weights and fine-tuned LoRA adapters mounted under `model-artifacts/`.
- Optional: NVIDIA Container Toolkit for GPU acceleration (set `--gpus` in production orchestrators).

## Repository Layout Relevant to Deployment

```
.
├── docker-compose.yml
├── .env.example
├── Flask chat app/
│   └── Dockerfile
├── frontend/
│   └── dist/              # Static build artefacts served by Nginx
└── model-artifacts/       # Mounted read-only into the backend container
```

## Docker Compose Stack

```yaml
services:
  backend:  # Flask API + Gemma 3n loader
  neo4j:    # Knowledge graph with APOC + GDS plugins
  vector-store:  # Qdrant vector database
  frontend:  # Static UI served via nginx:alpine
```

### Bring the Stack Online

1. Copy the sample environment file and edit as needed:
   ```bash
   cp .env.example .env
   # Update GEMMA_* paths to the location of your model weights
   ```

2. Populate host directories:
   - `model-artifacts/`: place or mount the Gemma base model and LoRA adapters (`gemma-3n`, tokenizer files, `adapter_model.safetensors`, etc.).
   - `frontend/dist/`: output of your SPA build (e.g. React/Vite) including `index.html`.

3. Start the stack:
   ```bash
   docker compose up --build
   ```
   The backend image installs requirements from `Flask chat app/requirements.txt` and exposes Flask via Gunicorn on port `BACKEND_PORT` (default `5000`).

4. Verify health:
   ```bash
   curl http://localhost:5000/api/health
   open http://localhost:3000            # or FRONTEND_PORT value
   neo4j-browser http://localhost:7474   # credentials from NEO4J_AUTH
   ```

### Important Volumes

| Volume | Purpose | Default Mount |
| --- | --- | --- |
| `backend_offload` | Disk space for bitsandbytes CPU offloading | `/offload` inside backend |
| `backend_cache` | Hugging Face cache for tokenizer/model shards | `/cache/huggingface` |
| `neo4j_data` | Neo4j database storage | `/data` |
| `neo4j_logs` | Neo4j logs | `/logs` |
| `qdrant_data` | Qdrant storage | `/qdrant/storage` |

Volumes persist across container restarts. Clear them cautiously (`docker volume rm ...`) when rotating datasets or performing clean installs.

### Environment Variable Highlights

See [Configuration Reference](configuration.md) for a complete list. Common values:

| Variable | Description |
| --- | --- |
| `GEMMA_BASE_PATH`, `GEMMA_MODEL_PATH`, `GEMMA_ADAPTER_PATH` | Control which weights are mounted and loaded by `GemmaModelLoader`. |
| `NEO4J_AUTH` | Credentials in the form `username/password`. Change defaults before production use. |
| `FRONTEND_API_URL` | Used by the frontend Nginx container to route API calls. |
| `PIP_EXTRA_INDEX_URL` | Optional CPU-only PyTorch wheels index consumed during backend image build. |

### GPU Deployment Notes

- Install `nvidia-container-toolkit` and run Docker with `--gpus all` (Compose: `deploy.resources.reservations.devices`).
- Set `BITSANDBYTES_FORCE_CPU=0` to enable 8-bit GPU kernels.
- Adjust `backend_offload` volume size if running in mixed CPU/GPU mode.

### Production Hardening Checklist

- Terminate TLS at a reverse proxy (e.g. Traefik, NGINX) or managed load balancer in front of the frontend + backend.
- Configure authentication/authorisation in the backend if exposing publicly (API keys, OAuth or session tokens).
- Move secrets to a vault or orchestrator-level secret store instead of `.env` files.
- Enable Neo4j role-based access control and restrict APOC procedures as needed.
- Configure Qdrant snapshots and Neo4j backups for disaster recovery.
- Integrate the services with your central logging/monitoring (see [Monitoring & Alerting](operations/monitoring-alerting.md)).

## Tear Down

```bash
docker compose down
# Optionally remove volumes if you want a clean slate
docker compose down -v
```

Keep track of container image tags when promoting to higher environments to guarantee reproducibility.
