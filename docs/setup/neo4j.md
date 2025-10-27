# Neo4j Setup Guide

Neo4j stores the knowledge graph used in the Graph RAG pipeline. Follow this guide to provision indexes, enable plugins, and manage data loads.

## Provisioning

### Using Docker Compose (recommended)

The provided [`docker-compose.yml`](../../docker-compose.yml) spins up Neo4j 5.x with APOC and Graph Data Science plugins enabled. Update `.env` with a secure `NEO4J_AUTH` value before launching.

```bash
# Start only the Neo4j service
NEO4J_AUTH=neo4j/StrongPassword docker compose up neo4j
```

### Manual Docker Run

```bash
docker run -it --rm \
  --name gemma-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/StrongPassword \
  -e NEO4J_PLUGINS='["apoc","graph-data-science"]' \
  -v neo4j_data:/data \
  -v neo4j_logs:/logs \
  neo4j:5.23
```

## Initial Configuration

1. Visit `http://localhost:7474` and log in using the credentials configured in `NEO4J_AUTH`.
2. Change the default password on first login.
3. Enable Bloom or AuraDS as needed for advanced visualisation.

## Schema & Constraints

Run the following Cypher snippets to prepare the schema. Adjust names to match your taxonomy.

```cypher
// Example labels and constraints
CREATE CONSTRAINT client_id IF NOT EXISTS
FOR (c:ClientProfile) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT symptom_id IF NOT EXISTS
FOR (s:Symptom) REQUIRE s.code IS UNIQUE;

CREATE CONSTRAINT intervention_id IF NOT EXISTS
FOR (i:Intervention) REQUIRE i.id IS UNIQUE;

CREATE INDEX IF NOT EXISTS FOR (p:Policy) ON (p.tag);
```

## APOC Configuration

APOC procedures are pre-enabled in the Docker stack. If running Neo4j elsewhere, update `neo4j.conf`:

```
apoc.export.file.enabled=true
apoc.import.file.enabled=true
apoc.trigger.enabled=true
apoc.uuid.enabled=true
```

Restart the database after updating the configuration.

## Data Loading Workflow

1. Export graph facts from the ingestion pipeline (see [Ingestion Runbook](../operations/ingestion-runbook.md)).
2. Load CSV, JSON, or Cypher files using APOC:
   ```cypher
   CALL apoc.load.json("file:///graph_payload.json") YIELD value
   WITH value
   MERGE (c:ClientProfile {id: value.client.id})
   SET c += value.client.properties
   ...
   ```
3. Version each load using semantic version tags (`version`, `ingested_at`) to support rollbacks.

## Backups & Maintenance

- Run periodic `neo4j-admin dump` jobs and store snapshots securely.
- Monitor heap usage and adjust JVM flags (`NEO4J_dbms_memory_heap_initial__size`) for large graphs.
- Prune outdated relationships using scheduled APOC jobs.

## Security

- Restrict bolt/HTTP ports to trusted networks or private subnets in production.
- Create least-privilege roles for application, ingestion, and analyst workloads.
- Use TLS certificates for encrypted connections (`dbms.ssl.policy.bolt.enabled=true`).

Neo4j is central to the hybrid RAG strategy. Keep schema, indexes, and data quality in sync with retraining cycles to avoid stale or inconsistent context injection.
