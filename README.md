[![CI](https://github.com/jose-compu/logosdb/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jose-compu/logosdb/actions/workflows/ci.yml)
[![Python](https://github.com/jose-compu/logosdb/actions/workflows/python.yml/badge.svg?branch=main)](https://github.com/jose-compu/logosdb/actions/workflows/python.yml)
[![PyPI](https://img.shields.io/pypi/v/logosdb.svg)](https://pypi.org/project/logosdb/)
[![Python versions](https://img.shields.io/pypi/pyversions/logosdb.svg)](https://pypi.org/project/logosdb/)

LogosDB is a fast semantic vector database written in C/C++ that provides approximate nearest-neighbor search over embedding vectors with associated text metadata.

Authors: Jose ([@jose-compu](https://github.com/jose-compu))

# Features

  * Vectors and metadata are stored as flat binary files, memory-mapped for zero-copy reads.
  * Approximate nearest-neighbor search via [HNSW](https://arxiv.org/abs/1603.09320) (hnswlib), O(log n) query time.
  * Each vector row carries optional text and ISO 8601 timestamp metadata (JSONL sidecar).
  * The basic operations are `Put(embedding, text, timestamp)` and `Search(query, top_k)`.
  * Bulk vector access for direct tensor construction (e.g. loading into GPU memory).
  * Thread-safe writes via internal mutex; concurrent reads are lock-free.
  * Crash recovery: HNSW index is automatically backfilled from the append-only vector store on open.
  * Scales to millions of vectors.

# Documentation

The public interface is in `include/logosdb/logosdb.h`. Callers should not include or rely on the details of any other header files in this package. Those internal APIs may be changed without warning.

Guide to header files:

* **include/logosdb/logosdb.h**: Main interface to the DB. Start here. Contains:
  - C API with opaque handles and `errptr` convention (RocksDB/LevelDB style)
  - C++ convenience wrapper (`logosdb::DB`) with RAII and exceptions
  - `logosdb::Options` for HNSW tuning parameters
  - `logosdb::SearchHit` result struct

# Limitations

  * This is not a general-purpose vector database. It is purpose-built for embedding-based memory retrieval in LLM inference ([funes.cpp](../funes.cpp)).
  * Only a single process (possibly multi-threaded) can access a particular database at a time.
  * There is no client-server support built into the library. An application that needs such support will have to wrap their own server around the library.
  * Vectors must be L2-normalized before insertion (inner-product similarity is used).
  * Embedding generation is external — the caller provides pre-computed float vectors.

# Getting the Source

```bash
git clone --recurse-submodules <repository-url>
cd logosdb
```

# Building

This project supports [CMake](https://cmake.org/) out of the box.

Quick start:

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build .
```

This builds:

| Target | Description |
|--------|-------------|
| `logosdb` | Static library (`liblogosdb.a`) |
| `logosdb-cli` | Command-line tool for put, search, info |
| `logosdb-bench` | Benchmark: HNSW vs brute-force, with ChromaDB comparison |
| `logosdb-test` | Unit tests |

# Usage (C API)

```c
#include <logosdb/logosdb.h>

char *err = NULL;
logosdb_options_t *opts = logosdb_options_create();
logosdb_options_set_dim(opts, 2048);

logosdb_t *db = logosdb_open("/tmp/mydb", opts, &err);
logosdb_options_destroy(opts);

float vec[2048] = { /* ... */ };
logosdb_put(db, vec, 2048, "My commute is 42 minutes",
            "2025-06-25T10:00:00Z", &err);

logosdb_search_result_t *res = logosdb_search(db, query_vec, 2048, 5, &err);
for (int i = 0; i < logosdb_result_count(res); i++) {
    printf("#%d score=%.4f text=%s\n", i,
           logosdb_result_score(res, i),
           logosdb_result_text(res, i));
}
logosdb_result_free(res);
logosdb_close(db);
```

# Usage (C++ wrapper)

```cpp
#include <logosdb/logosdb.h>

logosdb::DB db("/tmp/mydb", {.dim = 2048});
db.put(embedding, "My commute is 42 minutes", "2025-06-25T10:00:00Z");

auto results = db.search(query, 5);
for (auto &r : results) {
    printf("id=%llu score=%.4f text=%s\n", r.id, r.score, r.text.c_str());
}
```

# Python

LogosDB ships Python bindings built with [pybind11](https://pybind11.readthedocs.io/) and [scikit-build-core](https://scikit-build-core.readthedocs.io/).

Install from PyPI (binary wheels provided for Linux x86_64/aarch64 and macOS x86_64/arm64 on CPython 3.9–3.13):

```bash
pip install logosdb
```

Or build from source in a clone:

```bash
pip install .
```

Usage:

```python
import numpy as np
import logosdb

db = logosdb.DB("/tmp/mydb", dim=128)

v = np.random.randn(128).astype(np.float32)
v /= np.linalg.norm(v)

rid = db.put(v, text="hello", timestamp="2025-06-25T10:00:00Z")
# rid is the row id for this vector (often 0 for the first insert).
hits = db.search(v, top_k=5)
print(hits[0].text, hits[0].score)

# Replace an existing row: first arg is that row's id (must still be "live").
# update() tombstones the old row and appends a new one; it returns the NEW id.
v2 = np.random.randn(128).astype(np.float32)
v2 /= np.linalg.norm(v2)
new_id = db.update(rid, v2, text="replaced")

# Tombstone a row by id (excluded from search). count() includes deleted rows;
# count_live() does not.
db.delete(new_id)

# After a delete, that id cannot be updated again — insert a fresh row with put().
# zero-copy bulk view over mmap-backed storage (shape: (count, dim), read-only)
vectors = db.raw_vectors()

print(db.count(), db.count_live(), db.dim)
```

Run the Python tests and examples:

```bash
pip install ".[test]"
pytest tests/python/

python examples/python/basic_usage.py

# sentence-transformers demo (optional heavy dep)
pip install ".[examples]"
python examples/python/sentence_transformers_demo.py
```

# CLI

```bash
# Database info
logosdb-cli info /tmp/mydb

# Search with a binary query vector file
logosdb-cli search /tmp/mydb --query-file q.bin --top-k 5
```

# Performance

Here is a performance report from the included `logosdb-bench` program. The results are somewhat noisy, but should be enough to get a ballpark performance estimate.

## Setup

We use databases with 1K, 10K, and 100K vectors. Each vector has 2048 dimensions (matching typical LLM embedding sizes). Vectors are L2-normalized random unit vectors.

    LogosDB:    version 0.2.1
    CPU:        Apple M-series (ARM64)
    Dim:        2048
    HNSW M:     16, ef_construction: 200, ef_search: 50

## Write performance

    put (1K vectors):    ~50 µs/op   (~20,000 inserts/sec)
    put (10K vectors):   ~80 µs/op   (~12,500 inserts/sec)
    put (100K vectors):  ~120 µs/op  (~8,300 inserts/sec)

Each "op" above corresponds to a write of a single vector + metadata + HNSW index update.

## Search performance

    HNSW top-5 (1K):     ~0.1 ms/query
    HNSW top-5 (10K):    ~0.3 ms/query
    HNSW top-5 (100K):   ~1.2 ms/query

    Brute-force top-5 (1K):    ~0.3 ms/query
    Brute-force top-5 (10K):   ~2.5 ms/query
    Brute-force top-5 (100K):  ~25 ms/query

HNSW maintains sub-linear scaling while brute-force grows linearly with database size. At 100K vectors, HNSW is roughly 20x faster.

## Benchmark vs ChromaDB

```bash
logosdb-bench --dim 2048 --counts 1000,10000,100000
```

| Metric | ChromaDB | LogosDB |
|--------|----------|---------|
| Language | Python + C (hnswlib) | Pure C/C++ |
| Search algorithm | HNSW | HNSW (same hnswlib) |
| Storage | SQLite + Parquet | Binary mmap + JSONL |
| Startup overhead | Python runtime + deps | Zero (linked library) |
| Embedding generation | Built-in (Sentence Transformers) | External (caller provides vectors) |
| Target use case | General-purpose vector store | Embedded LLM inference memory |
| Search latency (100K, dim=2048) | ~5-10 ms | ~1-3 ms |
| Memory footprint (100K, dim=2048) | ~1.5 GB (Python + SQLite) | ~800 MB (mmap) |
| Cold start | ~2-5 s (Python imports) | <10 ms |
| Dependencies | Python, NumPy, SQLite, hnswlib | hnswlib (header-only, vendored) |

LogosDB uses the same HNSW implementation as ChromaDB (hnswlib) but eliminates Python overhead, SQLite serialization, and Sentence Transformer coupling. The result is a leaner library optimized for the single use case of embedded semantic memory for LLM inference.

# Repository contents

    include/logosdb/logosdb.h     Public C/C++ API (start here)
    src/logosdb.cpp               Core engine: wires storage + index + metadata
    src/storage.h / storage.cpp   Fixed-stride binary vector file with mmap
    src/metadata.h / metadata.cpp Append-only JSONL text + timestamp store
    src/hnsw_index.h / .cpp       Thin wrapper around hnswlib
    tools/logosdb-cli.cpp         Command-line interface
    tools/logosdb-bench.cpp       Benchmark tool
    tests/test_basic.cpp          C++ unit tests
    tests/python/test_smoke.py    Python smoke tests (pytest)
    python/src/bindings.cpp       pybind11 Python bindings
    python/logosdb/               Python package (logosdb._core + stubs)
    examples/python/              Python usage examples
    pyproject.toml                Python build/config (scikit-build-core)
    third_party/hnswlib/          Vendored hnswlib (header-only)
    CHANGELOG                     Release history
    LICENSE                       MIT license text

# License

MIT — see [LICENSE](LICENSE) for the full text.
