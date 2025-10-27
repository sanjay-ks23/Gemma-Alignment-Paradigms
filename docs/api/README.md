# API Guide

The therapeutic chatbot exposes a RESTful API via the Flask backend. All endpoints are namespaced under `/api`. Refer to the accompanying [OpenAPI specification](openapi.yaml) for machine-readable details.

## Base URL

- Local: `http://localhost:5000`
- Docker Compose network: `http://backend:5000`
- Production: `https://<your-domain>`

## Endpoints Overview

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Service readiness check; returns model metadata and conversation counters. |
| `POST` | `/api/chat` | Generates a therapeutic response for a user message. |
| `POST` | `/api/clear` | Clears the active conversation history. |
| `GET` | `/api/history` | Retrieves the current conversation history. |
| `POST` | `/api/history` | Replaces the conversation history (session restore). |
| `GET` | `/api/info` | Returns model and application metadata. |

## Example: Generate a Response

```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
        "message": "I have been anxious about exams lately.",
        "temperature": 0.65,
        "max_length": 400,
        "use_history": true
      }'
```

Response:

```json
{
  "response": "It sounds like the pressure has been heavy lately...",
  "conversation_stats": {
    "total_messages": 2,
    "user_messages": 1,
    "model_messages": 1
  },
  "timestamp": null
}
```

## Error Handling

Errors follow the structure defined by `ErrorResponse` in the OpenAPI spec. Example:

```json
{
  "error": "Missing 'message' in request body"
}
```

HTTP status codes:

- `400` – Validation errors (missing parameters, empty message).
- `500` – Unexpected backend failure; inspect logs.
- `503` – Model not initialised; wait for warm-up or check deployment.

## Authentication & Rate Limiting (Future Work)

The current implementation does not enforce authentication. Before production launch:

- Add API keys, OAuth, or JWT validation middlewares.
- Implement rate limiting (Flask-Limiter, API gateway, or edge proxy).
- Emit audit logs for every API call (user ID, session ID, retrieval plan).

## SDK / Client Usage

- **JavaScript** – Use `fetch` or Axios with the `FRONTEND_API_URL` environment variable configured at build time.
- **Python** – Simple wrapper:
  ```python
  import requests

  def chat(message: str, api_base: str = "http://localhost:5000"):
      resp = requests.post(f"{api_base}/api/chat", json={"message": message})
      resp.raise_for_status()
      return resp.json()
  ```

For complete schema definitions and examples, open the OpenAPI file in Swagger UI, Stoplight Studio, or VS Code's REST client extension.
