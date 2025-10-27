# Backend Setup Guide

This guide covers local and containerised setup of the Flask backend that exposes the therapeutic chat API.

## Prerequisites

- Python 3.11+
- `pip` and (optional) `virtualenv`
- Access to Gemma model weights + LoRA adapters
- (Optional) GPU with CUDA 12+ and NVIDIA drivers for accelerated inference

## Local Development Setup

1. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies** from the backend directory:
   ```bash
   cd "Flask chat app"
   pip install -r requirements.txt
   pip install gunicorn  # optional for local production-style serving
   ```

3. **Set environment variables**:
   ```bash
   export FLASK_APP_PATH=$(pwd)
   export GEMMA_BASE_PATH=/absolute/path/to/model-artifacts
   export GEMMA_MODEL_PATH=$GEMMA_BASE_PATH/gemma-3n
   export GEMMA_ADAPTER_PATH=$GEMMA_BASE_PATH/gemma-3n
   export GEMMA_OFFLOAD_DIR=$GEMMA_BASE_PATH/offload
   export BITSANDBYTES_FORCE_CPU=1  # set to 0 when CUDA is available
   ```

4. **Run the application**:
   ```bash
   python app.py           # Development server
   # or
   gunicorn --workers 2 --threads 4 --bind 0.0.0.0:5000 app:app
   ```

5. **Verify**:
   ```bash
   curl http://localhost:5000/api/health
   ```

## Container Build & Run

Use the Dockerfile in `Flask chat app/`:

```bash
docker build \
  --build-arg PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
  -t gemma-backend ./"Flask chat app"

docker run --rm -p 5000:5000 \
  -e GEMMA_BASE_PATH=/models \
  -e GEMMA_MODEL_PATH=/models/gemma-3n \
  -e GEMMA_ADAPTER_PATH=/models/gemma-3n \
  -e BITSANDBYTES_FORCE_CPU=1 \
  -v $(pwd)/model-artifacts:/models:ro \
  gemma-backend
```

The container entrypoint uses Gunicorn and honours the same environment variables as the local setup.

## Model Asset Expectations

The backend expects the following files under `GEMMA_MODEL_PATH` (customise as needed):

- `config.json`, `tokenizer_config.json`, `tokenizer.model`, `special_tokens_map.json`
- `adapter_model.safetensors` (for LoRA adapters) or merged `pytorch_model.bin`
- Optional `chat_template.jinja` loaded by `TherapeuticChatHandler`

If adapters live in a different directory than the base model, set `GEMMA_ADAPTER_PATH` accordingly.

## Development Tips

- Use the `/api/history` endpoint to inspect conversation state during debugging.
- Toggle `debug=True` in `app.run(...)` only for local testing; it is disabled by default to avoid double model initialisation.
- When editing prompts or safety templates, update the environment variables instead of hard-coding paths.
- For CPU-only machines, set `BITSANDBYTES_FORCE_CPU=1` and lower `max_length` in request payloads to reduce latency.

Refer to the [API documentation](../api/README.md) for request/response schemas and optional parameters.
