# Memory-Efficient On-Premises RAG with LogosDB

This guide explains how to build a retrieval-augmented generation (RAG) system that runs entirely on your hardware, with predictable memory usage that scales with query patterns rather than dataset size.

## Overview

A typical RAG pipeline has three components:

1. **Embedding Model** — Converts text to vectors (runs locally with sentence-transformers)
2. **Vector Store** — Stores and searches vectors (LogosDB with memory-mapped storage)
3. **LLM** — Generates responses (local models via llama.cpp, Ollama, or similar)

This guide focuses on component #2: using LogosDB for memory-efficient storage and retrieval.

## RAM Model

LogosDB uses `mmap()` for zero-copy access to vector data. This means:

| Metric | Formula | Example (1M × 384-dim) |
|--------|---------|------------------------|
| Vector data on disk | `N × dim × 4 bytes` | ~1.5 GB |
| HNSW index overhead | ~10-20% of vector data | ~150-300 MB |
| **Peak query RAM** | Depends on touched pages | **<100 MB typical** |
| Max theoretical RAM | Index + hot vectors | ~300 MB + query working set |

The key insight: **RAM usage depends on query patterns, not database size.**

### How mmap Works

When you search:
1. OS loads touched index pages from disk into RAM
2. Frequently accessed pages stay cached (fast)
3. Cold pages are evicted by OS when memory pressure rises
4. No explicit loading/unloading needed

### Reducing Memory Further

| Technique | Impact | Implementation |
|-----------|--------|----------------|
| Smaller embeddings | 4× reduction for 96-dim vs 384-dim | Use `all-MiniLM-L6-v2` (384d) or quantize further |
| Time sharding | Only hot shards stay in cache | One DB per week/month |
| External quantization | 4× reduction for int8 vs float32 | Quantize embeddings before storage |
| Reduce HNSW M | Lower memory, slightly worse recall | Set `M=8` instead of default `M=16` |

## Complete RAG Example

```python
"""Minimal on-prem RAG with LogosDB."""
import numpy as np
import logosdb
from sentence_transformers import SentenceTransformer

# 1. Load embedding model (runs locally)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dim = model.get_sentence_embedding_dimension()

# 2. Create LogosDB (memory-mapped, grows with data)
db = logosdb.DB("/var/lib/myapp/knowledge", dim=dim, distance=logosdb.DIST_COSINE)

# 3. Index documents (embed offline, then put)
documents = [
    "LogosDB is a memory-efficient vector database.",
    "HNSW enables fast approximate nearest neighbor search.",
    "Memory mapping provides zero-copy access to large files.",
]
for text in documents:
    embedding = model.encode(text)
    # Cosine distance auto-normalizes; no manual normalization needed
    db.put(embedding, text=text)

# 4. Query (only touched pages load into RAM)
query = "What is a fast vector search algorithm?"
query_embedding = model.encode(query)

hits = db.search(query_embedding, top_k=3)
for h in hits:
    print(f"{h.score:.4f}  {h.text}")
```

Output:
```
0.9234  HNSW enables fast approximate nearest neighbor search.
0.8912  LogosDB is a memory-efficient vector database.
0.8543  Memory mapping provides zero-copy access to large files.
```

## Architecture Patterns

### Pattern 1: Embed Offline, Query Online

**Ingestion (batch job):**
```python
# Runs once, can be slow
for batch in chunks(all_documents, size=1000):
    embeddings = model.encode(batch)
    for text, vec in zip(batch, embeddings):
        db.put(vec, text=text)
```

**Query (online, low latency):**
```python
# Runs per user query, must be fast
hits = db.search(query_embedding, top_k=5)
```

### Pattern 2: Time-Shard for Infinite Retention

Instead of one giant database, use time-based shards:

```
/data/knowledge/
  ├── 2025-01/
  │     ├── vectors.bin
  │     └── hnsw.idx
  ├── 2025-02/
  │     └── ...
  └── 2025-03/
        └── ...
```

Query recent + relevant historical shards:

```python
# Query current month + last month (others cold on disk)
for shard in ["2025-03", "2025-02"]:
    db = logosdb.DB(f"/data/knowledge/{shard}", dim=dim)
    hits = db.search(query_embedding, top_k=10)
    all_results.extend(hits)

# Rerank and take top_k globally
top_results = sorted(all_results, key=lambda h: h.score, reverse=True)[:5]
```

Benefits:
- Old data stays cold on disk (OS evicts pages)
- Can drop old shards entirely (compliance/retention)
- Parallel query across shards possible

### Pattern 3: With LlamaIndex or LangChain

```python
# LlamaIndex integration
from logosdb import LogosDBIndex
from llama_index.core import Document

store = LogosDBIndex(uri="/data/knowledge", dim=384, use_cosine=True)

docs = [Document(text="..."), Document(text="...")]
# Add with embeddings pre-computed
store.add(docs)

# Query
from llama_index.core.vector_stores import VectorStoreQuery
query = VectorStoreQuery(query_embedding=embedding, similarity_top_k=5)
results = store.query(query)
```

See [framework integrations](../README.md) for details.

## Sizing Guide

| Vectors | Dim | Disk Size | Index RAM (typical) | Query RAM (typical) |
|---------|-----|-----------|---------------------|---------------------|
| 10K | 384 | 15 MB | 2 MB | <10 MB |
| 100K | 384 | 153 MB | 15 MB | <20 MB |
| 1M | 384 | 1.5 GB | 150 MB | <100 MB |
| 10M | 384 | 15 GB | 1.5 GB | <200 MB |

**Notes:**
- Disk size = vectors + HNSW index (~10-20% overhead)
- Index RAM grows with dataset (HNSW graph structure)
- Query RAM depends on working set (how many distinct vectors you touch)

## External Quantization

For even smaller footprints, quantize embeddings before storage:

```python
# Example: int8 quantization (4× smaller than float32)
import numpy as np

def quantize_fp32_to_int8(x: np.ndarray) -> np.ndarray:
    """Quantize float32 embeddings to int8."""
    # Scale to [-128, 127] range
    scale = 127.0 / np.abs(x).max()
    quantized = (x * scale).astype(np.int8)
    return quantized, scale

def dequantize_int8_to_fp32(quantized: np.ndarray, scale: float) -> np.ndarray:
    """Dequantize back to float32 for search."""
    return quantized.astype(np.float32) / scale

# Store quantized (requires LogosDB with custom preprocessing)
# This is a pattern, not built-in — implement in your wrapper
```

**Tradeoffs:**
- int8: ~1% recall drop, 4× smaller
- binary: ~10% recall drop, 32× smaller (use dedicated binary indexes)

LogosDB stores float32 natively; quantization is an application-layer concern.

## Deployment Checklist

- [ ] Embedding model runs locally (sentence-transformers, onnx, etc.)
- [ ] LogosDB on fast local storage (SSD, not network mount)
- [ ] HNSW parameters tuned for your data (M, ef_construction)
- [ ] Optional: time-sharding for >10M vectors
- [ ] Optional: quantization for edge devices
- [ ] Monitoring: track RSS vs dataset size over time

## Further Reading

- [Basic Usage Example](../examples/python/basic_usage.py)
- [Memory-Efficient RAG Example](../examples/python/memory_efficient_rag.py)
- [LangChain Integration](../python/logosdb/langchain.py)
- [LlamaIndex Integration](../python/logosdb/llamaindex.py)
- HNSW paper: [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320)
