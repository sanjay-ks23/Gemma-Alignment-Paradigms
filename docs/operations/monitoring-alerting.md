# Monitoring & Alerting Guide

Reliable therapeutic assistance depends on proactive observability. This guide outlines the signals, tooling, and alert thresholds recommended for the Gemma 3n platform.

## Observability Stack Overview

| Layer | Signals | Suggested Tools |
| --- | --- | --- |
| Flask Backend | Request latency, 4xx/5xx rates, generation duration, retrieval failures | Prometheus + Grafana, OpenTelemetry exporters |
| Neo4j | Query latency, transaction errors, heap utilisation, page cache hit rate | Neo4j Ops Manager, Prometheus JMX exporter |
| Qdrant | Search latency, payload read/write throughput, segment count | Built-in `/metrics` endpoint scraped by Prometheus |
| Model Runtime | GPU/CPU usage, memory footprint, number of active sessions | NVIDIA DCGM exporter, `torch.cuda` telemetry, custom stats via `/api/health` |

## Instrumentation Checklist

- Enable Flask request logging (Gunicorn access log) and forward to central log aggregation (ELK, Loki, etc.).
- Expose custom metrics by integrating `prometheus_flask_exporter` or OpenTelemetry instrumentation within the backend.
- Configure Neo4j to emit metrics via `dbms.track_query_cpu_time=true` and expose JMX.
- For Qdrant, scrape `http://vector-store:6333/metrics` to capture ANN performance metrics.
- Ship container logs to your log platform (e.g. Fluent Bit + CloudWatch or Stackdriver).

## Key Metrics & Thresholds

| Metric | Target / Alert Threshold | Action |
| --- | --- | --- |
| Backend P95 latency | < 4s (normal), alert at > 8s for 5 minutes | Investigate retrieval/model slowness, scale horizontally. |
| Backend 5xx rate | < 1%, alert at > 5% for 2 consecutive minutes | Inspect logs for stack traces, verify dependency health. |
| Graph query latency | < 300 ms, alert at > 1s sustained | Optimise Cypher queries, review indexes, check Neo4j load. |
| Qdrant search latency | < 200 ms, alert at > 500 ms sustained | Rebuild HNSW index, scale replicas, review hardware. |
| GPU memory utilisation | < 80%, alert at > 95% for 1 minute | Investigate memory leaks, reduce `max_length`, scale GPU capacity. |
| Safety escalation backlog | 0 outstanding, alert if > 1 for > 2 minutes | Engage human moderators immediately. |

## Alert Routing

- Integrate Prometheus Alertmanager or your APM's alerting to notify on-call clinicians and SREs.
- Route safety-critical alerts (self-harm, abuse detection) to a dedicated escalation channel with 24/7 coverage.
- Configure synthetic checks (heartbeat pings) to verify `/api/health` availability from multiple regions.

## Dashboards

Create dashboards with the following panels:

1. **Conversation Funnel** – Requests per minute, user sentiment (if captured), escalation counts.
2. **Retrieval Insights** – Graph vs vector hit ratios, average context window size, fallback occurrences.
3. **Model Performance** – Generation duration histogram, token output rate, temperature usage distribution.
4. **Infrastructure Health** – CPU/GPU usage, container restarts, volume capacity.

## Logging Standards

- Include request IDs (`X-Request-Id`) and user session identifiers in logs for traceability.
- Redact personal data before logs leave the secure boundary.
- Structure logs in JSON for machine parsing (e.g. `python-json-logger`).

## Incident Response

1. Acknowledge alerts in the on-call tool and assess severity.
2. Consult runbooks: [Ingestion](ingestion-runbook.md), [Retraining](retraining-runbook.md), and this guide.
3. If user impact is high, communicate status on public channels as per incident policy.
4. Escalate to legal/compliance teams for incidents involving minors.
5. Document root cause and remediation steps in the incident tracker.

## Continuous Improvement

- Conduct monthly game days simulating high-risk scenarios and service degradation.
- Review alert noise quarterly; adjust thresholds to minimise fatigue.
- Incorporate telemetry insights into roadmap prioritisation (e.g. improving retrieval or scaling GPU resources).

Robust monitoring and disciplined alerting keep the therapeutic experience safe, compliant, and responsive.
