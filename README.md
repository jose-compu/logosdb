<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/9bc8a125-3a6b-4e83-b2e9-0e32bd315ff3" />

[![CI](https://github.com/jose-compu/logosdb/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jose-compu/logosdb/actions/workflows/ci.yml)
[![Python](https://github.com/jose-compu/logosdb/actions/workflows/python.yml/badge.svg?branch=main)](https://github.com/jose-compu/logosdb/actions/workflows/python.yml)
[![PyPI](https://img.shields.io/pypi/v/logosdb.svg)](https://pypi.org/project/logosdb/)
[![Python versions](https://img.shields.io/pypi/pyversions/logosdb.svg)](https://pypi.org/project/logosdb/)

LogosDB is a fast semantic vector database written in C/C++ that provides approximate nearest-neighbor search over embedding vectors with associated text metadata.

Authors: Jose ([@jose-compu](https://github.com/jose-compu))

**Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md) for build instructions, style guide, and PR workflow.

# Features

  * Vectors and metadata are stored as flat binary files, memory-mapped for zero-copy reads.
  * Approximate nearest-neighbor search via [HNSW](https://arxiv.org/abs/1603.09320) (hnswlib), O(log n) query time.
  * Each vector row carries optional text and ISO 8601 timestamp metadata (JSONL sidecar).
  * The basic operations are `Put(embedding, text, timestamp)` and `Search(query, top_k)`.
  * **Timestamp range filtering**: search within a time window (e.g., "last 24 hours").
  * **Multiple distance metrics**: inner product, cosine similarity (auto-normalized), or L2 Euclidean.
  * Bulk vector access for direct tensor construction (e.g. loading into GPU memory).
  * Thread-safe API on a single open handle: an internal mutex serializes operations that touch the index and metadata (including search).
  * Crash recovery: HNSW index is automatically backfilled from the append-only vector store on open.
  * Scales to millions of vectors.
  * **Framework integrations**: LangChain and LlamaIndex VectorStore adapters.
  * **MCP server**: first-class [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) integration via `logosdb-mcp-server`.

# Documentation

The public interface is in `include/logosdb/logosdb.h`. Callers should not include or rely on the details of any other header files in this package. Those internal APIs may be changed without warning.

**Planning a deployment?** See [docs/sizing.md](docs/sizing.md) for disk/RAM estimates based on N×dim, or run `python -m logosdb.sizing --rows 1_000_000 --dim 768`.

Guide to header files:

* **include/logosdb/logosdb.h**: Main interface to the DB. Start here. Contains:
  - C API with opaque handles and `errptr` convention (RocksDB/LevelDB style)
  - C++ convenience wrapper (`logosdb::DB`) with RAII and exceptions
  - `logosdb::Options` for HNSW tuning and distance metric selection
  - `logosdb::SearchHit` result struct
  - `logosdb_search_ts_range()` for timestamp-filtered search

# Limitations

  * This is not a general-purpose vector database. It is purpose-built for embedding-based memory retrieval in LLM inference ([funes.cpp](../funes.cpp)).
  * Only a single process (possibly multi-threaded) can access a particular database at a time.
  * There is no client-server support built into the library. An application that needs such support will have to wrap their own server around the library.
  * For inner-product distance (`LOGOSDB_DIST_IP`, the default), vectors must be L2-normalized before insertion. Use `LOGOSDB_DIST_COSINE` for automatic normalization.
  * Embedding generation is external — the caller provides pre-computed float vectors.

# Roadmap

Work is tracked in [open issues](https://github.com/jose-compu/logosdb/issues?q=is%3Aissue+is%3Aopen). Version bumps follow [Semantic Versioning](https://semver.org/) with these project conventions while the library is **pre-1.0** (`0.x.y`):

| Bump | When |
|------|------|
| **Patch** (`0.x.Z`) | Bug fixes, security fixes, documentation, CI/build-only changes, and internal tooling that does **not** alter the supported contract of the public C API in [`include/logosdb/logosdb.h`](include/logosdb/logosdb.h) or the stable surface of language bindings. |
| **Minor** (`0.Y.z`) | Backward-compatible additions: new functions or options, new storage or index behaviors behind explicit settings, new integrations, new platforms, and additive file-format upgrades with automatic or guided migration. |
| **Major `1.0.0`** | First **stable** release: a documented compatibility promise for the public C API and for on-disk formats (including upgrade paths). Intended when the **Path to 1.0.0** criteria (below) are satisfied so downstream authors can depend on semver semantics without surprise breakage across patch/minor lines. |

Target versions and issue assignment match **[GitHub milestones](https://github.com/jose-compu/logosdb/milestones)** (each open roadmap issue has exactly one milestone).

## Milestones → issues

### **SHIPPED** [0.7.7](https://github.com/jose-compu/logosdb/milestone/7) — patch (`0.x.Z`) **SHIPPED**

  * [#74](https://github.com/jose-compu/logosdb/issues/74) — security: harden encoding and injection surfaces across MCP, CLI, and integrations
  * [#9](https://github.com/jose-compu/logosdb/issues/9) — libFuzzer JSON harness in CI; local ASan/UBSan builds possible; full-tree sanitizer CI not enabled (vendored hnswlib noise)

### **SHIPPED** [0.7.8](https://github.com/jose-compu/logosdb/milestone/14) — patch (`0.x.Z`) **SHIPPED**

  * [#94](https://github.com/jose-compu/logosdb/issues/94) — feat(mcp): expose timestamp-range search (`search_ts_range` / `from_ts`, `to_ts`)
  * [#95](https://github.com/jose-compu/logosdb/issues/95) — docs: document Claude Code slash commands (/index, /search, /forget) in main README
  * [#96](https://github.com/jose-compu/logosdb/issues/96) — feat(mcp) or docs: semantic delete / forget-by-query vs ID-only `logosdb_delete`

### **SHIPPED** [0.8.0](https://github.com/jose-compu/logosdb/milestone/5) — minor — operations and durability **SHIPPED**

  * [#88](https://github.com/jose-compu/logosdb/issues/88) — DB doctor/upgrade: compatibility checks and guided migrations
  * [#83](https://github.com/jose-compu/logosdb/issues/83) — Snapshots and backup: consistent point-in-time export/restore
  * [#82](https://github.com/jose-compu/logosdb/issues/82) — Metrics and observability: expose query/ingest/index health counters
  * [#81](https://github.com/jose-compu/logosdb/issues/81) — Compaction/vacuum: reclaim space from tombstones and fragmented files

### [0.9.0](https://github.com/jose-compu/logosdb/milestone/8) — minor — throughput and scale

  * [#87](https://github.com/jose-compu/logosdb/issues/87) — Streaming import/export for very large corpora
  * [#80](https://github.com/jose-compu/logosdb/issues/80) — Batch ingest v2: high-throughput put_batch with WAL-aware commit

### [0.10.0](https://github.com/jose-compu/logosdb/milestone/9) — minor — search and metadata

  * [#85](https://github.com/jose-compu/logosdb/issues/85) — Hybrid retrieval mode: ANN score + lexical score fusion
  * [#84](https://github.com/jose-compu/logosdb/issues/84) — Filter API v2: structured metadata predicates beyond timestamp range

### [0.11.0](https://github.com/jose-compu/logosdb/milestone/10) — minor — multi-tenancy and tooling

  * [#86](https://github.com/jose-compu/logosdb/issues/86) — Multi-tenant namespaces: quota and isolation within one DB root
  * [#89](https://github.com/jose-compu/logosdb/issues/89) — Recall benchmarking utility for HNSW tuning

### [0.12.0](https://github.com/jose-compu/logosdb/milestone/11) — minor — integrations

  * [#78](https://github.com/jose-compu/logosdb/issues/78) — Feature: Codex plugin marketplace integration for LogosDB

### [0.13.0](https://github.com/jose-compu/logosdb/milestone/12) — minor — platform

  * [#10](https://github.com/jose-compu/logosdb/issues/10) — Windows support: abstract POSIX file I/O behind a portable layer

### [1.0.0](https://github.com/jose-compu/logosdb/milestone/13) — stable release tag

No roadmap issues are assigned only to this milestone: it marks publishing **`1.0.0`** once the criteria below are satisfied; implementation work stays on the milestones above until shipped.

## Path to 1.0.0

`1.0.0` is **not** a single mega-release of every open issue; it is the milestone where the project commits to stable semver for the public API and supported persistence story. Practically, **1.0.0** is targeted when:

  * The public C API in `include/logosdb/logosdb.h` is treated as stable: breaking changes only in future major versions, with migration notes.
  * On-disk layout and version negotiation are documented, with doctor/upgrade guidance ([#88](https://github.com/jose-compu/logosdb/issues/88)) and a credible backup/restore path ([#83](https://github.com/jose-compu/logosdb/issues/83)).
  * The security baseline in [#74](https://github.com/jose-compu/logosdb/issues/74) is addressed for MCP, CLI, and bundled integrations.
  * Tier-1 platforms for the release are explicit (first-class Windows ([#10](https://github.com/jose-compu/logosdb/issues/10)) or a documented Unix-only 1.0, decided at release time).

Features such as hybrid retrieval ([#85](https://github.com/jose-compu/logosdb/issues/85)), filter predicates ([#84](https://github.com/jose-compu/logosdb/issues/84)), or multi-tenant namespaces ([#86](https://github.com/jose-compu/logosdb/issues/86)) can ship in **0.x** minors as they land; they do not all block **1.0.0** unless adopted into the stable API surface.

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

float vec[2048] = { /* ... unnormalized vector ... */ };

// L2-normalize for inner-product distance (returns 0 on success, -1 if zero norm)
if (logosdb_l2_normalize(vec, 2048) == 0) {
    logosdb_put(db, vec, 2048, "My commute is 42 minutes",
                "2025-06-25T10:00:00Z", &err);
}

logosdb_search_result_t *res = logosdb_search(db, query_vec, 2048, 5, &err);
for (int i = 0; i < logosdb_result_count(res); i++) {
    printf("#%d score=%.4f text=%s\n", i,
           logosdb_result_score(res, i),
           logosdb_result_text(res, i));
}
logosdb_result_free(res);
logosdb_close(db);
```

## Search with timestamp range filter

```c
#include <logosdb/logosdb.h>

char *err = NULL;
logosdb_options_t *opts = logosdb_options_create();
logosdb_options_set_dim(opts, 2048);

logosdb_t *db = logosdb_open("/tmp/mydb", opts, &err);

// Search for top-5 matches within the last 24 hours
logosdb_search_result_t *res = logosdb_search_ts_range(
    db, query_vec, 2048, 5,
    "2025-04-21T10:00:00Z",  // from (inclusive), NULL for no lower bound
    "2025-04-22T10:00:00Z",  // to (inclusive), NULL for no upper bound
    50,                      // candidate_k: internal fetch multiplier (10x top_k recommended)
    &err);

for (int i = 0; i < logosdb_result_count(res); i++) {
    printf("#%d score=%.4f ts=%s text=%s\n", i,
           logosdb_result_score(res, i),
           logosdb_result_timestamp(res, i),
           logosdb_result_text(res, i));
}
logosdb_result_free(res);
logosdb_close(db);
```

## Using different distance metrics

```c
#include <logosdb/logosdb.h>

char *err = NULL;
logosdb_options_t *opts = logosdb_options_create();
logosdb_options_set_dim(opts, 2048);

// Use cosine similarity (automatically normalizes vectors)
logosdb_options_set_distance(opts, LOGOSDB_DIST_COSINE);

// Or use L2 Euclidean distance
// logosdb_options_set_distance(opts, LOGOSDB_DIST_L2);

// Default is LOGOSDB_DIST_IP (inner product on L2-normalized vectors)

logosdb_t *db = logosdb_open("/tmp/mydb", opts, &err);
logosdb_options_destroy(opts);

// For cosine: vectors are automatically normalized on put/search
float vec[2048] = { /* ... unnormalized vector ... */ };
logosdb_put(db, vec, 2048, "entry", "2025-04-22T10:00:00Z", &err);

logosdb_close(db);
```

# Usage (C++ wrapper)

```cpp
#include <logosdb/logosdb.h>
#include <vector>

// Basic usage with default inner-product distance
logosdb::DB db("/tmp/mydb", {.dim = 2048});

// L2-normalize your vectors before insertion (required for inner-product distance)
std::vector<float> embedding = load_some_vector();  // unnormalized
if (logosdb::l2_normalize(embedding)) {
    db.put(embedding, "My commute is 42 minutes", "2025-06-25T10:00:00Z");
}

// Or use l2_normalized() to get a normalized copy
auto normalized = logosdb::l2_normalized(query);
auto results = db.search(normalized, 5);
for (auto &r : results) {
    printf("id=%llu score=%.4f text=%s\n", r.id, r.score, r.text.c_str());
}
```

## C++: Search with timestamp filter

```cpp
#include <logosdb/logosdb.h>

logosdb::DB db("/tmp/mydb", {.dim = 2048});

// Search within a time window
auto results = db.search_ts_range(
    query, 5,
    "2025-04-21T00:00:00Z",  // from timestamp
    "2025-04-22T00:00:00Z",  // to timestamp
    50);                     // candidate_k (optional, defaults to 10x top_k)

for (auto &r : results) {
    printf("id=%llu score=%.4f ts=%s\n", r.id, r.score, r.timestamp.c_str());
}
```

## C++: Using cosine or L2 distance

```cpp
#include <logosdb/logosdb.h>

// Cosine similarity - vectors are automatically normalized
logosdb::DB db("/tmp/mydb", {.dim = 2048, .distance = LOGOSDB_DIST_COSINE});

// Put unnormalized vectors - they will be normalized automatically
db.put(unnormalized_embedding, "entry", "2025-04-22T10:00:00Z");

auto results = db.search(query, 5);
// scores are cosine similarities in [0, 1]

// L2 Euclidean distance
// logosdb::DB db("/tmp/mydb", {.dim = 2048, .distance = LOGOSDB_DIST_L2});
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
import logosdb
from sentence_transformers import SentenceTransformer

# Local Hugging Face embeddings model (runs on your machine)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dim = model.get_sentence_embedding_dimension()

# Use cosine distance so LogosDB auto-normalizes vectors
db = logosdb.DB("/tmp/agent_memory", dim=dim, distance=logosdb.DIST_COSINE)

# Three learnings captured by an AI agent
learnings = [
    ("Retrying API calls with exponential backoff reduced transient failures by 42%.", "2026-05-06T09:00:00Z"),
    ("Splitting long tasks into smaller batches improved throughput and lowered memory spikes.", "2026-05-06T09:05:00Z"),
    ("Adding idempotency keys prevented duplicate writes during network retries.", "2026-05-06T09:10:00Z"),
]

for text, ts in learnings:
    emb = model.encode(text).astype("float32")
    db.put(emb, text=text, timestamp=ts)

# Ask a natural-language question
question = "How can we avoid duplicate writes when retries happen?"
q_emb = model.encode(question).astype("float32")

hits = db.search(q_emb, top_k=3)
for h in hits:
    print(f"{h.score:.4f}  {h.text}")
```

## Python: Using cosine distance (no manual normalization needed)

```python
import numpy as np
import logosdb

# With cosine distance, vectors are automatically normalized
db = logosdb.DB("/tmp/mydb", dim=128, distance=logosdb.DIST_COSINE)

# No need to normalize - just put raw vectors
v = np.random.randn(128).astype(np.float32)
rid = db.put(v, text="unnormalized vector", timestamp="2025-04-22T10:00:00Z")

# Search also works with unnormalized queries
query = np.random.randn(128).astype(np.float32)
hits = db.search(query, top_k=5)
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

## Memory-Efficient On-Prem RAG

LogosDB is designed for memory-efficient retrieval-augmented generation (RAG) that runs entirely on your hardware.

### RAM Model

LogosDB uses `mmap()` for zero-copy access. Your RAM usage scales with **query patterns**, not dataset size:

| Dataset | Dim | Disk | Typical Query RAM |
|---------|-----|------|-------------------|
| 100K | 384 | 153 MB | <20 MB |
| 1M | 384 | 1.5 GB | <100 MB |
| 10M | 384 | 15 GB | <200 MB |

The OS caches hot index pages; cold data stays on disk. No explicit loading/unloading needed.

### Quick RAG Example

```python
import numpy as np
import logosdb
from sentence_transformers import SentenceTransformer

# 1. Load model (runs locally)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dim = model.get_sentence_embedding_dimension()

# 2. Create DB with cosine distance (auto-normalizes)
db = logosdb.DB("/data/knowledge", dim=dim, distance=logosdb.DIST_COSINE)

# 3. Index documents
for text in documents:
    emb = model.encode(text)
    db.put(emb, text=text)  # Auto-normalized with cosine distance

# 4. Query (only touched pages load into RAM)
query_emb = model.encode("What is HNSW?")
for hit in db.search(query_emb, top_k=3):
    print(f"{hit.score:.4f}  {hit.text}")
```

See [docs/rag-on-prem.md](docs/rag-on-prem.md) for complete guide including:
- Time-sharding for infinite retention
- External quantization patterns
- Architecture patterns for production

See [docs/sizing.md](docs/sizing.md) for detailed disk/RAM formulas and the `python -m logosdb.sizing` calculator.

Run the memory-efficient RAG example:

```bash
pip install ".[examples]"
python examples/python/memory_efficient_rag.py
```

## Python: LlamaIndex VectorStore

```bash
pip install 'logosdb[llama-index]'
```

```python
from logosdb import LogosDBIndex
from llama_index.core import Document
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import VectorStoreQuery
import numpy as np

# Create the vector store
db = LogosDBIndex(uri="/tmp/mydb", dim=128)

# Add nodes with pre-computed embeddings
node = TextNode(
    text="My commute is 42 minutes",
    embedding=np.random.randn(128).astype(np.float32).tolist(),
    metadata={"timestamp": "2025-04-28T10:00:00Z"}
)
db.add([node])

# Query
query_emb = np.random.randn(128).astype(np.float32).tolist()
query = VectorStoreQuery(query_embedding=query_emb, similarity_top_k=5)
results = db.query(query)

for node, score in zip(results.nodes, results.similarities):
    print(f"Score: {score:.4f}, Text: {node.text}")

# Timestamp range filtering
results = db.query(query, ts_from="2025-04-01T00:00:00Z", ts_to="2025-04-30T23:59:59Z")
```

The `LogosDBIndex` class implements LlamaIndex's `VectorStore` interface, supporting:

- `add(nodes)` - Add nodes with embeddings
- `delete(node_id)` - Delete by node ID
- `query(VectorStoreQuery)` - Similarity search by vector
- `count()` / `len(store)` - Number of live documents
- Timestamp filtering via `ts_from` and `ts_to` kwargs

# CLI

```bash
# Database info
logosdb-cli info /tmp/mydb

# Search with a binary query vector file
logosdb-cli search /tmp/mydb --query-file q.bin --top-k 5
```

# Node.js

## MCP Server — Claude Code integration

`logosdb-mcp-server` is a [Model Context Protocol](https://modelcontextprotocol.io/) server that
exposes LogosDB to **Claude Code** (and any other MCP client) over stdio.  It lets Claude index
files, persist knowledge across sessions, and do semantic search without leaving the conversation.

### Claude Code: complete recipe

Follow these steps in order. Claude Code spawns the MCP server with **working directory = your project root** (the folder you opened); paths in `mcp.json` and `LOGOSDB_PATH` are resolved relative to that cwd unless you use absolute paths.

#### 1. Prerequisites

- **Node.js** 18 or newer on your `PATH` (`node -v`, `npm -v`). For the published package you also need **`npx`**.
- **Claude Code** installed and signed in ([overview](https://docs.anthropic.com/en/docs/claude-code/overview)).
- Decide **where the DB lives**: `LOGOSDB_PATH` (default `./.logosdb`) is created under the project root; add the directory to `.gitignore` if you do not want it committed.

#### 2. Install the server (pick one)

**Option A — This repository (local `node`, no `npx`).** From the **repository root**:

```bash
npm install
```

That installs the `mcp` workspace and runs **`prepare`**, which compiles TypeScript to [`mcp/dist/index.js`](mcp/dist/index.js). After you change MCP sources, run `npm run mcp:build` at the root or `npm run build` inside `mcp/`.

**Option B — Any other project (published [`logosdb-mcp-server`](https://www.npmjs.com/package/logosdb-mcp-server)).** In your app’s root:

```bash
npm install logosdb-mcp-server
```

You can still invoke it without a project dependency via `npx -y logosdb-mcp-server` (see Option C).

#### 3. Register the MCP server in Claude Code

Project-local config lives in **`.claude/mcp.json`** at the repository root. User-wide config is **`~/.claude.json`** (same `mcpServers` shape). Only one definition named `logosdb` is needed.

**If you use Option A (this clone):** the repo already contains [`.claude/mcp.json`](.claude/mcp.json):

```json
{
  "mcpServers": {
    "logosdb": {
      "command": "node",
      "args": ["./mcp/dist/index.js"],
      "env": {
        "LOGOSDB_PATH": "./.logosdb"
      }
    }
  }
}
```

Open **this folder** as the Claude Code project so `./mcp/dist/index.js` resolves. If the client’s cwd is **not** the repo root, set `"args"` to an **absolute** path to `mcp/dist/index.js` (see [Installing locally — Option C](#installing-locally-without-npx)).

**If you use Option B or prefer always-latest from npm (Option C):** create or merge `.claude/mcp.json`:

```json
{
  "mcpServers": {
    "logosdb": {
      "command": "npx",
      "args": ["-y", "logosdb-mcp-server"],
      "env": {
        "LOGOSDB_PATH": "./.logosdb"
      }
    }
  }
}
```

Default embeddings are **local Transformers.js** (no API keys); first run may download model weights. Optional **Ollama / OpenAI / Voyage** env vars are documented in [`mcp/README.md`](mcp/README.md#configure).

#### 4. Path rules for `logosdb_index_file`

The server only indexes files that resolve under **`process.cwd()`** or **`LOGOSDB_INDEX_ROOT`** (if set). Symlinks that escape those roots are rejected. Keep indexing paths inside the project you opened, or set `LOGOSDB_INDEX_ROOT` to an absolute allowed root (details in [`mcp/README.md` — Path confinement](mcp/README.md#path-confinement-logosdb_index_file)).

#### 5. Reload Claude Code and verify

- **Restart** Claude Code or reload MCP configuration after editing `mcp.json`.
- The **logosdb** server starts on **first tool use** (stdio).
- Ask the agent to run **`logosdb_list`** (or check the MCP tools panel). You should see namespaces (possibly empty) rather than a spawn error.

#### 6. Index, search, delete

- **Natural language:** e.g. “Index `src/` into the `code` namespace”, “Search the codebase for JWT validation”.
- **Slash commands (optional):** if `.claude/commands/` is present in the project (this repo ships them), use `/index`, `/search`, `/forget` — see the table below.
- **Snippet memory:** the agent can call **`logosdb_index`** for short text; **`logosdb_index_file`** chunks files and directories.

Use **one embedding backend consistently** per namespace on disk (do not change model dimension on existing data). See [Environment variables](#environment-variables) and [`mcp/README.md`](mcp/README.md).

#### 7. Optional: agent instructions

Copy the **`CLAUDE.md`** template block from [Agent instructions (`CLAUDE.md` and similar)](#agent-instructions-claudemd-and-similar) into your project so the agent indexes and searches without being reminded every session.

#### 8. Troubleshooting

| Symptom | What to try |
|--------|----------------|
| MCP fails to start / “Cannot find module” | From the same cwd Claude uses, run `node ./mcp/dist/index.js` (Option A) or `npx -y logosdb-mcp-server` (Option C) in a terminal; fix missing `npm install` or broken path. |
| Relative `./mcp/dist/index.js` not found | Use an **absolute** `args` path to `index.js`, or open the correct folder as the project root. |
| `logosdb_index_file` rejects a path | Path is outside cwd / `LOGOSDB_INDEX_ROOT`; use a path inside the project or set `LOGOSDB_INDEX_ROOT`. |
| Search looks wrong after changing model | New embeddings must use a **fresh** `LOGOSDB_PATH` or new namespace; dimensions must match. |

**Google Antigravity:** same stdio MCP pattern (`node` + local script, or `npx` + package). Step-by-step: [`mcp/README.md` — Google Antigravity](mcp/README.md#google-antigravity).

### Claude Code slash commands

This repo ships **project slash commands** under [.claude/commands/](.claude/commands/) (in addition to [`.claude/mcp.json`](.claude/mcp.json) for the MCP server):

| Command | Role |
|---------|------|
| `/index` | Index a **file or directory** via `logosdb_index_file` with **`incremental: true`** (new/changed files only; `commands/index.md`) |
| `/search` | Semantic search; optional ISO `ts_from` / `ts_to` on the MCP tool (`commands/search.md`) |
| `/forget` | Delete by row `id` **or** by natural-language `query` (`commands/forget.md`) |

### Agent instructions (`CLAUDE.md` and similar)

The MCP server does not index the repository by itself: **the agent must call the tools** (or you rely on slash commands / hooks). To make LogosDB feel automatic without typing `/index` every time, add a block like the following to your project’s **`CLAUDE.md`** (or any instructions file your agent reads on every session). Adjust namespaces and paths to your repo.

````markdown
## LogosDB (semantic memory via MCP)

The **logosdb** MCP server is configured. Data lives on disk under `LOGOSDB_PATH` (see `.claude/mcp.json`); it **persists across sessions**.

**Namespaces:** Use separate namespaces for different concerns (e.g. `code` for `src/`, `docs` for `docs/`, `decisions` for short architectural notes). Search only the namespace that matches the user’s task.

**When starting substantive work on this codebase:**
1. If the user has not indexed recently and you need broad code context, call **`logosdb_index_file`** with **`incremental: true`** on the smallest useful path (e.g. `src/` or a package directory), not the whole monorepo unless asked.
2. Before answering “where is X implemented?” or similar, call **`logosdb_search`** with a tight natural-language `query`, `namespace` set appropriately, and `top_k` between **3** and **8**. Do not paste entire trees into the chat—retrieve, then read only the cited files.
3. For “what did we decide recently?” style questions, use **`logosdb_search`** with optional **`ts_from` / `ts_to`** (ISO 8601 inclusive bounds) on the `decisions` or `docs` namespace when timestamps matter.
4. When the user states a durable fact worth remembering (API contract, policy, workaround), call **`logosdb_index`** into the right namespace with concise text (timestamps are stored automatically; optional **`metadata`** can label the source).

**After large refactors or dependency upgrades:** Re-run **`logosdb_index_file`** with **`incremental: true`** on affected paths so search stays aligned without duplicating chunks for unchanged files.

**Deletion:** Use **`logosdb_delete`** with **`id`** from a prior search hit, or with **`query`** + optional **`match_rank`** / **`search_top_k`** to remove a semantically matched row when the user asks to forget something.
````

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `LOGOSDB_PATH` | `./.logosdb` | Root directory for all namespace databases |
| `EMBEDDING_PROVIDER` | *(local)* | Omit for Transformers.js on-device; or `ollama`, `openai`, `voyage` |
| `TRANSFORMERS_MODEL` | `Xenova/all-MiniLM-L6-v2` | Local embedding model (bundled MCP path) |
| `OLLAMA_*` | — | See `mcp/README.md` when using Ollama |
| `OPENAI_API_KEY` | — | Required when `EMBEDDING_PROVIDER=openai` |
| `VOYAGE_API_KEY` | — | Required when `EMBEDDING_PROVIDER=voyage` |
| `LOGOSDB_CHUNK_SIZE` | `800` | Target characters per chunk for file indexing |

Voyage AI (`voyage-3`, dim=1024) is Anthropic's recommended cloud embedding model:

```json
"env": {
  "LOGOSDB_PATH": "./.logosdb",
  "EMBEDDING_PROVIDER": "voyage",
  "VOYAGE_API_KEY": "<your-voyage-api-key>"
}
```

### Available tools

| Tool | Description |
|---|---|
| `logosdb_index` | Embed and store a text snippet in a namespace |
| `logosdb_index_file` | Chunk, embed, and store a file or tree; optional **`incremental: true`** (skip unchanged, replace changed, prune deleted under a directory) |
| `logosdb_search` | Semantic search; optional `ts_from` / `ts_to` (ISO 8601) for timestamp-window filter |
| `logosdb_list` | List all namespaces |
| `logosdb_info` | Stats for a namespace (count, dimension, path) |
| `logosdb_delete` | Delete by row `id`, or by natural-language `query` (`search_top_k`, `match_rank`) |

### Installing locally (without npx)

**Option A — global CLI on your machine**

```bash
npm install -g logosdb-mcp-server
```

In `.claude/mcp.json`, point the server at the global binary:

```json
"logosdb": {
  "command": "logosdb-mcp-server",
  "args": [],
  "env": { "LOGOSDB_PATH": "./.logosdb" }
}
```

Ensure the directory containing the global npm binaries is on your `PATH` when Claude Code spawns the process (same shell you use for `npm install -g`).

**Option B — project-local `node_modules` (no global install)**

From your app repo:

```bash
npm install logosdb-mcp-server
```

Then use `npx` so the binary resolves from `./node_modules`:

```json
"logosdb": {
  "command": "npx",
  "args": ["-y", "logosdb-mcp-server"],
  "env": { "LOGOSDB_PATH": "./.logosdb" }
}
```

Omit `-y` if you prefer a fixed local install only. You can also call the entry script explicitly:

```json
"command": "node",
"args": ["./node_modules/logosdb-mcp-server/dist/index.js"],
"env": { "LOGOSDB_PATH": "./.logosdb" }
```

**Option C — development build from a LogosDB clone**

```bash
cd mcp && npm install && npm run build
```

This repository’s checked-in [`.claude/mcp.json`](.claude/mcp.json) uses a **relative** path `./mcp/dist/index.js` after **`npm install` from the repo root** (workspace `prepare`). For **other** clients or if the process cwd is not the repo root, use an **absolute** path to `mcp/dist/index.js`:

```json
"logosdb": {
  "command": "node",
  "args": ["/absolute/path/to/logosdb/mcp/dist/index.js"],
  "env": { "LOGOSDB_PATH": "./.logosdb" }
}
```

The same `env` block (`LOGOSDB_PATH`, embedding variables) applies to every option. Restart Claude Code or reload MCP config after editing `.claude/mcp.json`.

# Performance

Here is a performance report from the included `logosdb-bench` program. The results are somewhat noisy, but should be enough to get a ballpark performance estimate.

## Setup

We use databases with 1K, 10K, and 100K vectors. Each vector has 2048 dimensions (matching typical LLM embedding sizes). Vectors are L2-normalized random unit vectors.

*(Report below was produced with an older `logosdb-bench` binary labeled 0.5.0; scaling and relative HNSW vs brute-force behavior are still representative of current builds.)*

    LogosDB:    version 0.8.0
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
    package.json / package-lock.json   Private npm workspace (MCP + `nodejs` logosdb); `npm install` at repo root only
    mcp/                          MCP server (logosdb-mcp-server npm package)
    .claude/mcp.json              Claude Code MCP config (local `node ./mcp/dist/index.js`)
    .claude/commands/             Slash command prompts (/index, /search, /forget)
    CHANGELOG                     Release history
    LICENSE                       MIT license text

# Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Building from source
- Running tests and benchmarks
- Code style and PR workflow

Please review our [Code of Conduct](CODE_OF_CONDUCT.md) and [Security Policy](SECURITY.md).

# License

MIT — see [LICENSE](LICENSE) for the full text.
