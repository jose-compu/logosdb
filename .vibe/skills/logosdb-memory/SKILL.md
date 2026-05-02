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

## Slash command usage (inside Vibe)

```
/logosdb-memory index ./src --namespace code
/logosdb-memory search "where is JWT validated" --namespace code --top-k 5
/logosdb-memory forget --namespace code --query "old feature"
/logosdb-memory info
/logosdb-memory list
```

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
