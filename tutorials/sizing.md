# LogosDB Sizing Guide

Memory and disk space estimator for planning large-scale deployments.

## Overview

LogosDB uses three main components that consume storage and memory:

1. **Vector file** (`vectors.bin`) - Fixed-stride binary storage with mmap
2. **HNSW index** (`hnsw.idx`) - Approximate nearest neighbor graph structure
3. **Metadata** (`metadata.jsonl`) - Text and timestamps (variable size)

## Storage Formulas

### Vector File Size

| Precision | Bytes/Dim | Formula |
|-----------|-----------|---------|
| float32 (default) | 4 | `32 + N × dim × 4` |
| float16 | 2 | `32 + N × dim × 2` |
| int8 | 1 | `32 + N × dim × 1` |

The 32-byte header contains: magic (4), version (4), dim (4), dtype (4), n_rows (8), scale (4), reserved (4).

### HNSW Index Size

Per the [hnswlib documentation](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md):

```
Index size ≈ N × M × 9 bytes
```

Where:
- `N` = number of vectors
- `M` = graph connectivity (default 16, range 2-100)
- `9 bytes` = average of 8-10 bytes per link

The factor `M × 9` represents the per-element graph overhead. Higher M = better recall, more RAM.

### Metadata Size

Variable based on text content:
- Per row: JSON object with `text` (string) and `timestamp` (ISO 8601)
- Typical: 100-500 bytes per document depending on text length
- Estimate: `N × avg_text_bytes × 1.2` (JSON overhead)

## Baseline Sizing Tables

### float32 (Default) - Common Embedding Dimensions

| Vectors | Dim | Vector File | HNSW Index (M=16) | Total Disk | Index RAM |
|---------|-----|-------------|-------------------|------------|-----------|
| 100K | 384 | 153 MB | 14 MB | ~167 MB | ~14 MB |
| 100K | 768 | 306 MB | 14 MB | ~320 MB | ~14 MB |
| 100K | 1024 | 410 MB | 14 MB | ~424 MB | ~14 MB |
| 100K | 4096 | 1.6 GB | 14 MB | ~1.6 GB | ~14 MB |
| 1M | 384 | 1.5 GB | 144 MB | ~1.6 GB | ~144 MB |
| 1M | 768 | 3.0 GB | 144 MB | ~3.1 GB | ~144 MB |
| 1M | 1024 | 4.0 GB | 144 MB | ~4.1 GB | ~144 MB |
| 1M | 4096 | 16.0 GB | 144 MB | ~16.1 GB | ~144 MB |
| 10M | 384 | 15.3 GB | 1.4 GB | ~16.7 GB | ~1.4 GB |
| 10M | 768 | 30.6 GB | 1.4 GB | ~32.0 GB | ~1.4 GB |
| 10M | 1024 | 40.9 GB | 1.4 GB | ~42.3 GB | ~1.4 GB |
| 10M | 4096 | 163.8 GB | 1.4 GB | ~165.2 GB | ~1.4 GB |

### HNSW M Parameter Impact (1M vectors, dim=768)

| M Value | Index Size | Recall* | Build Time |
|---------|------------|---------|------------|
| 8 | 72 MB | ~0.85 | 2x faster |
| 16 (default) | 144 MB | ~0.95 | baseline |
| 32 | 288 MB | ~0.98 | 2x slower |
| 64 | 576 MB | ~0.99 | 4x slower |

*Approximate recall @ ef=50; actual depends on data distribution.

### Reduced Precision Comparison (1M vectors, dim=768)

| Precision | Vector File | vs float32 |
|-----------|-------------|------------|
| float32 | 3.0 GB | 1.0× (baseline) |
| float16 | 1.5 GB | 0.5× |
| int8 | 0.8 GB | 0.25× |

Note: LogosDB supports float16/int8 storage via `StorageDtype` in the C++ API. Python bindings currently expose float32 primarily.

## RAM Usage Model

### Key Insight

LogosDB uses `mmap()` for zero-copy access:

| Component | Behavior | Typical RAM |
|-----------|----------|-------------|
| Vector file | Page-cached by OS | Only touched pages |
| HNSW index | Must reside in RAM | Full index size |
| Query working set | Temporary | ~1-10 MB |

**Formula:**
```
Typical RSS = HNSW_index_size + (query_pattern_dependent_cache)
           ≈ N × M × 9 + (working_set_vectors × dim × 4)
```

### Query Pattern Examples (1M × 768-dim, M=16)

| Query Pattern | Vectors Touched | Query RAM | Total RSS |
|---------------|-----------------|-----------|-----------|
| Single query, random | ~500 | 2 MB | ~146 MB |
| Single query, similar docs | ~200 | 1 MB | ~145 MB |
| Batch 100 queries | ~5000 | 15 MB | ~159 MB |
| Full scan (rare) | 1M | 3 GB | ~3.1 GB |

The OS page cache keeps frequently accessed vectors in RAM; cold vectors stay on disk.

## Operational Planning

### SSD Requirements

| Dataset | Dim | Total Disk | IOPS | Notes |
|---------|-----|------------|------|-------|
| 1M | 768 | ~3.1 GB | Low | Fits on any SSD |
| 10M | 768 | ~32 GB | Medium | Standard NVMe OK |
| 100M | 768 | ~320 GB | High | Fast NVMe recommended |

### RAM Planning

Plan for at least the HNSW index size plus working set:

```python
# Conservative estimate
min_ram_gb = (N * M * 9) / (1024**3) + 0.5  # +0.5 GB working set

# Example: 10M vectors, M=16
min_ram_gb = (10_000_000 * 16 * 9) / (1024**3) + 0.5
           = 1.34 + 0.5
           ≈ 2 GB
```

### Time-Sharding for Scale

For infinite retention without infinite RAM, use time-based shards:

```
/data/knowledge/
  ├── 2025-01/          # Cold (disk only)
  ├── 2025-02/          # Cold (disk only)
  ├── 2025-03/          # Warm (partially cached)
  └── 2025-04/          # Hot (mostly in RAM)
```

Query only relevant shards; old data stays cold on disk.

## Using the Sizing Calculator

A Python utility is provided for custom calculations:

```bash
python -m logosdb.sizing --rows 10_000_000 --dim 1024 --m 16
```

Output:
```
LogosDB Size Estimate
=====================
Input:
  Rows: 10,000,000
  Dimensions: 1,024
  HNSW M: 16
  Precision: float32

Storage:
  Vector file:     40.96 GB
  HNSW index:       1.44 GB
  Metadata (est):   2.00 GB  (200 bytes/doc)
  Total disk:      44.40 GB

Memory:
  Index RAM (required): 1.44 GB
  Query RAM (typical):  <200 MB
```

Programmatic usage:

```python
from logosdb.sizing import estimate_size

est = estimate_size(
    n_rows=1_000_000,
    dim=768,
    m=16,
    dtype='float32',
    avg_text_bytes=200
)

print(f"Total disk: {est.total_disk_gb:.2f} GB")
print(f"Index RAM: {est.index_ram_gb:.2f} GB")
```

## Comparison with Other Databases

| Database | 1M×768 Storage | Query RAM | Notes |
|----------|----------------|-----------|-------|
| LogosDB | ~3.1 GB | ~150 MB | mmap, HNSW index in RAM only |
| ChromaDB | ~3.5 GB | ~1.5 GB | Python + SQLite overhead |
| pgvector | ~3.0 GB | ~3.0 GB | Full table in shared_buffers |
| Faiss (HNSW) | ~3.1 GB | ~3.1 GB | Loads everything into RAM |

LogosDB's mmap approach provides the lowest query RAM footprint for large datasets.

## See Also

- [RAG On-Premises Guide](rag-on-prem.md) - Memory-efficient deployment patterns
- [HNSW Parameters](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md) - Tuning M and ef_construction
- `python -m logosdb.sizing --help` - CLI calculator
