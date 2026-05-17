# codex-semantic-memory

Local-first semantic memory for [Codex](https://github.com/openai/codex).
Powered by [LogosDB](https://github.com/jose-compu/logosdb) — vector search, zero cloud, stdio MCP.

## What it does automatically

| Event | Action |
|---|---|
| Session start | Background-indexes the project directory (incremental, honours `.gitignore`) |
| Every user prompt | Semantic search → injects relevant memories silently |
| After each response | Persists a turn record (Q + A + key reasoning) to memory |

All automatic actions are **silent** — nothing is printed to the console.

## Prerequisites

```bash
npm install -g logosdb-mcp-server   # or: npx is used as a fallback
```

## Install

### Option 1 — home-local (recommended, available across all projects)

```bash
mkdir -p ~/.codex/plugins
cp -R /path/to/codex-semantic-memory ~/.codex/plugins/codex-semantic-memory
```

Add to `~/.agents/plugins/marketplace.json`:

```json
{
  "plugins": [
    {
      "name": "codex-semantic-memory",
      "source": { "source": "local", "path": "./plugins/codex-semantic-memory" },
      "policy": { "installation": "AVAILABLE", "authentication": "NONE" },
      "category": "Productivity"
    }
  ]
}
```

Then restart Codex and install via `/plugins`.

### Option 2 — repo-local

Copy the folder into `<repo-root>/plugins/codex-semantic-memory` and add the same
entry to `<repo-root>/.agents/plugins/marketplace.json` with `"path": "./plugins/codex-semantic-memory"`.

## Memory scope

Controlled by `LOGOS_MEMORY_MODE` (default: `global`).

| Mode | Behaviour |
|---|---|
| `global` (default) | Turn records → `codex-global` namespace, shared across all projects. Source files → `codex-proj-<name>`. Context search covers both. |
| `project` | All storage and search isolated to `codex-proj-<name>`. Each repo is independent. |

```bash
# project-only memory
export LOGOS_MEMORY_MODE=project
```

## Explicit commands

| Command | Description |
|---|---|
| `/memory-recall <query>` | Semantic search past memories |
| `/memory-remember <text>` | Explicitly store text to memory |
| `/memory-index [path]` | Re-index project (or given path) |
| `/memory-status` | Show namespace, mode, vector counts |

## Namespaces

| Namespace | Purpose |
|---|---|
| `codex-global` | Cross-project turn records (default store in global mode) |
| `codex-proj-<name>` | Project-specific source-file chunks; auto-derived from cwd basename |

## DB location

| Mode | Path |
|---|---|
| `global` | `~/.codex/.logosdb` |
| `project` | `<project-root>/.logosdb` |

## Configuration

| Env var | Default | Description |
|---|---|---|
| `LOGOS_MEMORY_MODE` | `global` | `global` or `project` |
| `LOGOSDB_PATH` | Set by wrapper | Override DB path directly |
| `LOGOSDB_QUOTA_MAX_VECTORS` | `0` (unlimited) | Max vectors per namespace |
| `LOGOSDB_QUOTA_MAX_NAMESPACES` | `0` (unlimited) | Max namespace count |
