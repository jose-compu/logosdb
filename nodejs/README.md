# LogosDB - Node.js Bindings

Fast semantic vector database (HNSW + mmap) for Node.js. Zero-copy memory-mapped storage with local-only deployment.

## Features

- **Memory-efficient**: Uses mmap() — RAM scales with queries, not data size
- **Fast**: HNSW approximate nearest neighbor (O(log n) queries)
- **Local-only**: No cloud dependencies, data never leaves your machine
- **TypeScript support**: Full type definitions included
- **Cross-platform**: Linux, macOS, Windows (x64, arm64)

## Installation

```bash
npm install logosdb
```

Prebuilt binaries are provided for common platforms. If a prebuilt binary is not available, npm will compile from source (requires Python, C++ compiler, and CMake).

## Quick Start

```javascript
const { DB, DIST_COSINE } = require('logosdb');

// Create database
const db = new DB('/tmp/mydb', {
  dim: 384,              // Vector dimension
  distance: DIST_COSINE  // Auto-normalizes vectors
});

// Insert documents with embeddings
const embedding = [0.1, 0.2, /* ... 384 floats ... */];
const id = db.put(embedding, 'My document text', '2025-01-01T00:00:00Z');

// Search
const query = [0.15, 0.25, /* ... */];
const hits = db.search(query, 5);

for (const hit of hits) {
  console.log(`${hit.score.toFixed(4)}  ${hit.text}`);
}

// Close
db.close();
```

## API

### `new DB(path, options)`

Create a new database instance.

**Options:**
- `dim` (number): Vector dimension (default: 128)
- `maxElements` (number): Maximum capacity (default: 1,000,000)
- `efConstruction` (number): HNSW build quality (default: 200)
- `M` (number): HNSW graph degree (default: 16)
- `efSearch` (number): HNSW search width (default: 50)
- `distance` (number): Distance metric (`DIST_IP`, `DIST_COSINE`, `DIST_L2`)

### `db.put(embedding, text?, timestamp?)`

Insert a vector. Returns the assigned row ID.

### `db.search(queryEmbedding, topK?)`

Search for similar vectors. Returns array of `SearchHit` objects.

### `db.searchTsRange(queryEmbedding, options)`

Search with timestamp filter.

**Options:**
- `topK` (number): Number of results
- `tsFrom` (string): Start timestamp (ISO 8601)
- `tsTo` (string): End timestamp (ISO 8601)
- `candidateK` (number): Internal multiplier for filtering

### `db.update(id, embedding, text?, timestamp?)`

Update a row (marks old as deleted, creates new). Returns new ID.

### `db.delete(id)`

Delete a row by ID.

### `db.count()` / `db.countLive()`

Get total/live row counts.

### `db.close()`

Close the database.

## Distance Metrics

- `DIST_IP` (0): Inner product (default, requires L2-normalized vectors)
- `DIST_COSINE` (1): Cosine similarity (auto-normalizes)
- `DIST_L2` (2): Euclidean distance

## Memory Model

LogosDB uses memory-mapped files:

| Dataset | Disk | Typical Query RAM |
|---------|------|-------------------|
| 100K × 384-dim | 153 MB | <20 MB |
| 1M × 384-dim | 1.5 GB | <100 MB |
| 10M × 384-dim | 15 GB | <200 MB |

RAM scales with query patterns, not dataset size.

## TypeScript

```typescript
import { DB, SearchHit, DIST_COSINE } from 'logosdb';

const db = new DB('/tmp/mydb', { dim: 384, distance: DIST_COSINE });
const hits: SearchHit[] = db.search(embedding, 5);
```

## Building from Source

```bash
npm install --build-from-source
```

Requirements:
- Python 3.x
- C++17 compiler (GCC, Clang, MSVC)
- CMake 3.15+
- Node.js 16+

## License

MIT — see [LICENSE](../LICENSE)

## Links

- GitHub: https://github.com/jose-compu/logosdb
- Issues: https://github.com/jose-compu/logosdb/issues
- Python bindings: `pip install logosdb`
