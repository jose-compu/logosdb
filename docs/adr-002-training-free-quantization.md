# ADR-002: Training-Free Quantization for HNSW Vector Storage

**Status:** Proposed  
**Date:** 2026-04-29  
**Milestone:** 0.7.0  
**Related Issues:** [#25](https://github.com/jose-compu/logosdb/issues/25), [#26](https://github.com/jose-compu/logosdb/issues/26)

## Summary

This document describes the architecture for integrating **training-free vector quantization** (specifically float16 and int8 reduced-precision storage) into LogosDB's append-only storage layer while maintaining compatibility with the existing HNSW graph-based search.

## Context

LogosDB stores vectors in a memory-mapped file (`vectors.bin`) using float32 format. For large RAG corpora (e.g., 10M vectors × 384 dimensions), this consumes approximately 15 GB of disk space and significant RAM when memory-mapped. Reduced-precision storage can shrink this footprint by 2× (float16) or 4× (int8).

## Decision Drivers

1. **Memory efficiency**: Smaller on-disk footprint reduces mmap pressure
2. **Search quality**: Quantization error must not significantly degrade HNSW recall
3. **Backward compatibility**: Existing float32 databases must remain usable
4. **Streaming ingestion**: New vectors can arrive without rebuilding the index
5. **No training data**: Unlike product quantization, training-free methods work with any embedding model

## Decision

We will implement **storage-time quantization with search-time dequantization**:

1. Store vectors in reduced precision (float16 or int8) on disk
2. Dequantize to float32 when loading into HNSW
3. Keep HNSW working in float32 space (hnswlib's native format)

This approach prioritizes implementation simplicity and search accuracy over maximum memory savings during queries.

## Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Dequantize at search** (chosen) | Simple, accurate, no hnswlib changes | Peak memory still float32 | Accepted |
| Native int8 HNSW distances | Lower peak memory, faster distance calc | Requires hnswlib modifications, quantization-aware search | Rejected for v1 |
| Product Quantization (PQ) | Higher compression (10-20×) | Requires training, complex IVF integration | Out of scope |
| TurboQuant-style methods | Near-lossless compression | Complex, research-stage for embeddings | Future work |

## Implementation

### Storage Format v2

The `StorageHeader` is extended to include a `dtype` field indicating the precision:

```cpp
enum StorageDtype {
    DTYPE_FLOAT32 = 0,  // 4 bytes per dimension
    DTYPE_FLOAT16 = 1,  // 2 bytes per dimension  
    DTYPE_INT8    = 2,  // 1 byte per dimension + scale
};

struct StorageHeader {
    uint32_t magic    = 0x4C4F474F;  // "LOGO"
    uint32_t version  = 2;            // bumped to 2
    uint32_t dim      = 0;
    uint32_t dtype    = 0;            // NEW: StorageDtype value
    uint64_t n_rows   = 0;
    float    scale    = 1.0f;         // NEW: for int8 dequantization
};
```

### int8 Quantization Strategy

For int8 storage, we use **per-vector scale quantization** rather than per-dimension or global scale:

- Each vector is quantized independently: `int8_i = round(float32_i / scale * 127)`
- Scale is stored per-vector (implied: `scale = max(abs(vector)) / 127`)
- This preserves relative magnitudes within each vector better than global scaling

However, for storage efficiency in v1, we use a **global scale** stored in the header, computed as the maximum absolute value seen across all stored vectors. This is simpler but slightly less accurate for outlier vectors.

### API Changes

```cpp
// C API additions
#define LOGOSDB_DTYPE_FLOAT32  0
#define LOGOSDB_DTYPE_FLOAT16  1
#define LOGOSDB_DTYPE_INT8     2

void logosdb_options_set_dtype(logosdb_options_t * opts, int dtype);

// C++ API additions
struct Options {
    int dim = 0;
    int dtype = LOGOSDB_DTYPE_FLOAT32;  // NEW
    // ... other options
};
```

### Search Path

```
Storage (int8/float16) → mmap → dequantize → float32[] → hnswlib::search
                              ↑
                         on-demand during search or bulk load
```

## Failure Modes and Mitigations

| Risk | Mitigation |
|------|------------|
| Quantization error degrades recall | Document `ef_search` tuning; users can increase for quantized indexes |
| Distribution shift | Global scale may become suboptimal; recommend re-indexing if embedding model changes |
| Backward compatibility | Version 1 files load as float32; version 2 files require new code |
| int8 overflow | Clamp to [-127, 127] range, reserve -128 for future use |

## Non-Claims

LogosDB **does not** implement the following (despite related research):

1. **TurboQuant** (arXiv:2504.19874): This ADR discusses quantization approaches but LogosDB does not implement TurboQuant's specific algorithms.
2. **Product Quantization (PQ)**: No clustering, no codebooks, no IVF structures.
3. **Binary embeddings**: No Hamming distance support in HNSW.
4. **Learned quantization**: No neural network-based compression.

## References

1. [TurboQuant: Succinct Floating-Point Arithmetic for Neural Network Inference](https://arxiv.org/html/2504.19874v1) — Shows training-free quantization is feasible for inference-side use, though this ADR implements simpler scalar quantization.
2. [HNSW: Efficient and robust approximate nearest neighbor search](https://arxiv.org/abs/1603.09320) — The underlying graph algorithm.
3. [Scalar Quantization for Search](https://qdrant.tech/articles/scalar-quantization/) — Practical guide to int8 quantization in vector search.

## Future Work

- Per-vector scale for better int8 accuracy
- SIMD-accelerated dequantization (AVX2/NEON)
- Native int8 distance functions in hnswlib (no dequantization needed)
- Automatic dtype selection based on vector statistics
