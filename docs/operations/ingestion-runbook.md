# Ingestion Pipeline Runbook

This runbook governs the end-to-end ingestion of conversational datasets, graph knowledge, and vector embeddings. Follow these steps whenever new data is sourced or existing corpora are refreshed.

## 1. Pre-Flight Checklist

- [ ] Source files validated (checksums, schema, licensing)
- [ ] Data protection assessment completed (PII handling, consent records)
- [ ] Sufficient storage available for intermediate artefacts
- [ ] Neo4j and Qdrant services reachable from ingestion environment

## 2. Load Raw Datasets

```bash
cd "Curriculum Learning"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### CounselChat (Hugging Face)
```python
from src.data_preprocessing import CounselChatProcessor
processor = CounselChatProcessor()
raw_dataset = processor.load_raw_data()
conversations = processor.process_conversations(raw_dataset)
```

### AnnoMI (CSV)
```python
from src.data_preprocessing import AnnoMIProcessor
processor = AnnoMIProcessor(csv_path="/path/to/AnnoMI-full.csv")
raw_df = processor.load_raw_data()
conversations = processor.process_conversations(raw_df)
```

## 3. Persist Formatted Datasets

```python
from src.data_preprocessing import DatasetManager
manager = DatasetManager(tokenizer_name="google/gemma-3n-E2B-it")
manager.save_conversations_to_disk(conversations, output_dir="./data/processed/2024_10")
```

Outputs include Hugging Face dataset files (`.arrow`) plus metadata JSON for downstream jobs.

## 4. Embed & Push to Vector Store

1. Generate sentence embeddings (e.g. using the fine-tuned tokenizer or an auxiliary model).
2. Use Qdrant client to upsert points:
   ```python
   from qdrant_client import QdrantClient
   client = QdrantClient(host="localhost", port=6333)
   client.upsert(
       collection_name="therapeutic_examples",
       points=[...]
   )
   ```
3. Tag each point with `version`, `source`, and `risk_level`.

## 5. Load Graph Facts into Neo4j

1. Produce Cypher files or JSON payloads containing nodes and relationships.
2. Import via APOC:
   ```cypher
   CALL apoc.load.json("file:///graph/2024_10_payload.json") YIELD value
   MERGE (c:ClientProfile {id: value.client.id})
   SET c += value.client.properties, c.version = value.version
   ...
   ```
3. Record the ingestion batch ID and timestamp in a dedicated `IngestionLog` node for audit trails.

## 6. Validation

- **Dataset Counts:** Compare number of conversations before/after filtering. Expectation ranges documented in `Curriculum Learning/README.md`.
- **Graph Consistency:** Run Cypher smoke tests (e.g. ensure every `Intervention` links to at least one `Symptom`).
- **Vector Recall:** Issue sample queries and manually inspect retrieved passages for topical relevance.
- **Version Alignment:** Confirm the same version tag is present across datasets, graph, and vector points.

## 7. Post-Ingestion Tasks

- Update data catalog entries and notify stakeholders of new version availability.
- Trigger downstream retraining workflow ([Retraining Runbook](retraining-runbook.md)) if required.
- Archive source files and intermediate outputs to cold storage with retention policies.

## 8. Rollback Procedure

If validation fails:

1. Remove the offending vector entries using version tags.
2. Revert Neo4j nodes/relationships introduced in the batch (use `DETACH DELETE` on version-specific nodes or maintain rollback scripts).
3. Restore prior datasets from backup.

## 9. Documentation

Log the ingestion in your change management system, including:

- Data sources and versions
- Validation outcomes
- Issues encountered + resolutions
- Tickets raised for follow-up actions

Stay disciplined with this runbook to ensure retrieval quality remains high and compliant.
