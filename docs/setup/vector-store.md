# Vector Store Setup Guide

The vector store powers semantic retrieval over counselling transcripts, safety policies, and exemplar responses. The default deployment uses [Qdrant](https://qdrant.tech/), but any ANN-compatible service can be substituted with minor client adjustments.

## Provisioning Qdrant

### Docker Compose

`docker-compose.yml` provisions Qdrant with persistent storage by default. Verify host ports in `.env`:

```bash
QDRANT_HTTP_PORT=6333
QDRANT_GRPC_PORT=6334
```

Start the service:

```bash
docker compose up vector-store
```

### Manual Docker Run

```bash
docker run -it --rm \
  --name gemma-qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v qdrant_data:/qdrant/storage \
  qdrant/qdrant:latest
```

## Collections & Payload Schema

Define consistent collection names in your ingestion scripts. Recommended structure:

| Collection | Purpose | Suggested Distance |
| --- | --- | --- |
| `therapeutic_examples` | High-quality therapist responses indexed by topic | Cosine |
| `safety_protocols` | Crisis management passages and policy excerpts | Dot |
| `faq_embeddings` | Frequently asked questions and canned support replies | Cosine |

Sample payload:

```json
{
  "conversation_id": "counselchat_1024",
  "speaker": "assistant",
  "topic": "grief",
  "risk_level": "medium",
  "version": "2024.10",
  "source": "CounselChat"
}
```

Store `version` to help the backend avoid mixing inconsistent data during rollouts.

## Creating a Collection via API

```bash
curl -X PUT "http://localhost:6333/collections/therapeutic_examples" \
  -H "Content-Type: application/json" \
  -d '{
        "vectors": {
          "size": 1024,
          "distance": "Cosine"
        },
        "shard_number": 1,
        "replication_factor": 1
      }'
```

Bulk insert points using the REST API or gRPC client libraries. Embeddings can be produced with the fine-tuned tokenizer or an external sentence transformer.

## Maintenance

- Schedule periodic snapshots using Qdrant's `/collections/{name}/snapshots` endpoint; replicate to blob storage for DR.
- Run `OPTIMIZE` jobs during low-traffic windows to compact segments (see Qdrant documentation).
- Monitor latency and recall; adjust `hnsw_config` parameters if queries become slow at scale.

## Alternative Providers

If you prefer managed services (e.g. Pinecone, Weaviate Cloud, Milvus), implement a thin adapter mirroring the methods consumed by the backend retrieval module:

- `query_embeddings(text, top_k)`
- `upsert_points(embeddings, payload)`
- `delete_version(version_tag)`

Document provider-specific environment variables separately and ensure they are surfaced in [configuration.md](../configuration.md).
