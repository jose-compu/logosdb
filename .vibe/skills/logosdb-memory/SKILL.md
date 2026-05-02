---
name: logosdb-memory
description: Semantic memory for Mistral Vibe — index code/docs, search across sessions, forget outdated knowledge using LogosDB + Mistral embeddings.
license: MIT
compatibility: Python 3.9+
user-invocable: true
allowed-tools:
  - bash
  - read_file
  - grep
---

# LogosDB Memory Skill

Gives Vibe persistent semantic memory across sessions by indexing your project
into a local LogosDB vector store and performing sub-millisecond semantic search.

## Prerequisites

```bash
pip install 'logosdb[mistral]'   # or: pip install logosdb requests numpy
export MISTRAL_API_KEY="..."
```

## Setup

Add `LOGOSDB_PATH` to your environment (or `.vibe/.env`) to set where databases
are stored (default: `./.logosdb`):

```bash
LOGOSDB_PATH=./.logosdb
MISTRAL_API_KEY=your_key_here
```

## Slash commands (inside Vibe)

Three focused slash commands are available once this skill is enabled:

| Command | Example |
|---|---|
| `/ldb-index` | `/ldb-index ./src --namespace=backend` |
| `/ldb-search` | `/ldb-search "JWT validation" --namespace=backend` |
| `/ldb-forget` | `/ldb-forget --namespace=backend --id=42` |

### Example session

```
$ cd myproject && vibe

> /ldb-index ./src --namespace=backend
Indexed 42 files into 'backend' collection

> Find where we handle JWT validation
Searching... Found 3 matches:
  1. src/auth/jwt.ts (score: 0.94)
  2. src/middleware/auth.ts (score: 0.87)
  3. src/utils/token.ts (score: 0.72)

> Show me the first one
[Vibe displays src/auth/jwt.ts with explanation]

> /ldb-search "retry logic" --namespace=backend --top-k=3
Searching... Found 3 matches:
  1. src/utils/retry.ts (score: 0.91)
  2. src/api/client.ts (score: 0.83)
  3. src/jobs/queue.ts (score: 0.77)

> /ldb-forget --namespace=backend --id=42
Deleted 1 entry from 'backend' namespace.
```

Natural-language queries (`> Find where we...`) also work without a slash command
— Vibe calls `logosdb-vibe search` automatically when it understands you are
asking about the indexed codebase.

## Standalone CLI

```bash
logosdb-vibe index ./src --namespace code
logosdb-vibe search "retry logic" --namespace code --top-k 5
logosdb-vibe forget --namespace code --id 42
logosdb-vibe info --namespace code
logosdb-vibe list
```

## Python API (inside a Vibe tool or script)

```python
from logosdb.vibe import VibeMemory

mem = VibeMemory(uri="./.logosdb")        # reads MISTRAL_API_KEY from env

# Index an entire directory
mem.index("./src", namespace="code")

# Search
results = mem.search("where is JWT validated?", namespace="code", top_k=5)
for r in results:
    print(r["score"], r["file"], r["text"][:120])

# Forget by semantic match
mem.forget(namespace="code", query="deprecated auth module")

# Forget by ID (from search result)
mem.forget(namespace="code", memory_id=42)

# Stats
print(mem.info())
```

## Namespaces

Use separate namespaces to keep concerns isolated:

| Namespace | Content |
|---|---|
| `code`  | Source files (`./src`, `./lib`) |
| `docs`  | Markdown, RST (`./docs`) |
| `tests` | Test files (`./tests`) |
| `notes` | Free-form session notes |

## MCP alternative

Vibe also supports MCP servers. To use the same LogosDB server as Claude Code,
see `.vibe/config.toml` in this project for the `[[mcp_servers]]` configuration.
