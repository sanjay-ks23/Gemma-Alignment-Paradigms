# Frontend Setup Guide

The frontend provides the therapeutic chat interface that communicates with the Flask backend. It is deployed as static assets served by Nginx within the Docker Compose stack.

## Project Structure

```
frontend/
└── dist/
    └── index.html  # Build artefact copied here before deployment
```

The repository ships with an empty `dist/.gitkeep` placeholder. Replace this directory with the output of your build pipeline (e.g. React, Vue, Svelte, or plain HTML/JS).

## Build Recommendations

1. **Choose a Framework** – React (Vite), Next.js static export, or any SPA framework.
2. **Expose API URL** – Use build-time environment variables to point to the backend:
   ```bash
   VITE_API_URL=http://localhost:5000/api npm run build
   # or for create-react-app
   REACT_APP_API_BASE=http://localhost:5000/api npm run build
   ```
3. **Output Directory** – Configure the bundler to emit assets into `frontend/dist` (e.g. Vite: `build.outDir = "dist"`).

## Local Preview

Before copying artefacts into `frontend/dist`, verify the frontend locally:

```bash
npm install
npm run dev
```

Ensure cross-origin requests target the backend by configuring the API base URL and enabling CORS (already enabled in `app.py`).

## Deployment via Docker Compose

When `frontend/dist` contains the compiled assets, `docker-compose.yml` mounts it read-only into the Nginx container:

```yaml
frontend:
  image: nginx:alpine
  volumes:
    - ./frontend/dist:/usr/share/nginx/html:ro
  environment:
    - API_BASE_URL=${FRONTEND_API_URL:-http://backend:5000/api}
```

> **Note:** Nginx cannot inject environment variables into static files. Ensure your build embeds the correct API URL or use a small client-side configuration script that reads `window.API_BASE_URL` exposed via HTML `<script>` tags.

## Static Configuration Snippet

Embed a lightweight runtime configuration file if you want to keep builds environment-agnostic:

```html
<!-- frontend/dist/config.js -->
<script>
  window.APP_CONFIG = {
    apiBaseUrl: "${API_BASE_URL}" // templated by CI/CD before deployment
  };
</script>
```

Update the Flask backend to read the header `X-Session-Id` or similar identifiers emitted by the frontend for observability.

## Accessibility & Safeguards

- Display clear disclaimers about the chatbot's limitations, especially for crisis scenarios.
- Provide quick links to helplines (e.g. CHILDLINE 1098, local emergency numbers).
- Implement session timeouts and consent reminders for minor users in India (see [Safety & Compliance](../safety-compliance.md)).

Once the assets are in place, restarting the Compose stack will serve the latest UI instantly.
