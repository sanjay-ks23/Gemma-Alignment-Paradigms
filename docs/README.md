# System Documentation Suite

Welcome to the consolidated documentation for the Gemma 3n therapeutic fine-tuning and deployment platform. The files in this directory describe the end-to-end system architecture, operational workflows, deployment guidance, and governance controls required to run the application safely and reliably.

## How to Use This Documentation

The documentation set is organised by topic. Start with the architecture overview to understand the big picture, then follow the setup and deployment guides to bring each subsystem online. Operational runbooks and compliance notes ensure day-two readiness.

### At-a-Glance Index

| Domain | Documents |
| --- | --- |
| Architecture & Data Flow | [Architecture Overview](architecture.md) · [Graph RAG Flow](graph-rag.md) |
| Deployment | [Docker & Environment Guide](deployment.md) |
| Setup Guides | [Backend](setup/backend.md) · [Neo4j](setup/neo4j.md) · [Vector Store](setup/vector-store.md) · [Frontend](setup/frontend.md) |
| Configuration Reference | [Configuration Catalogue](configuration.md) |
| Safety & Compliance | [Safeguards for Minors in India](safety-compliance.md) |
| API Documentation | [Endpoint Guide](api/README.md) · [OpenAPI Specification](api/openapi.yaml) |
| Operations | [Ingestion Pipeline Runbook](operations/ingestion-runbook.md) · [Retraining Workflow](operations/retraining-runbook.md) · [Monitoring & Alerting](operations/monitoring-alerting.md) |

## Conventions

- **Mermaid** diagrams are used for component and flow visualisations. Render them locally with tools such as `mmdc` or within Markdown viewers that support Mermaid.
- Shell snippets default to bash (`$` prompt). Replace placeholders (e.g. `<...>`) before executing commands.
- File paths are relative to the repository root unless otherwise noted.
- Environment variables referenced in the documentation are defined in [`.env.example`](../.env.example).

## Related Resources

- `Curriculum Learning/README.md` provides deep technical detail on the model training approach and datasets.
- `Flask chat app/` contains the production Flask application that exposes the conversational API documented here.

For questions or clarifications, raise an issue in the repository so updates can be tracked and reviewed. Feedback on documentation gaps is especially welcome.
