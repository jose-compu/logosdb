---
name: semantic-memory
description: >
  Semantic memory management for Codex, backed by a local LogosDB MCP server.
  Handles explicit memory commands (/memory-recall, /memory-remember, /memory-index, /memory-status)
  and drives automatic silent indexing, searching, and turn-record storage.
  Invoke with: /memory-recall <query>  |  /memory-remember <text>  |  /memory-index  |  /memory-status
version: "0.1.0"
---

## Command routing

| Invocation | Action |
|---|---|
| `/memory-recall <query>` | Semantic search → show top results |
| `/memory-remember <text>` | Explicitly index provided text → one-line confirmation |
| `/memory-index [path]` | Re-index project (or given path) incrementally |
| `/memory-status` | Show namespace, memory mode, vector counts |

All automatic operations (session-start indexing, per-prompt search, turn storage) are
defined in `AGENTS.md` and happen silently — this SKILL handles only explicit commands.

---

## /memory-recall

```
logosdb_search(
  query     = <user query>,
  namespace = <project namespace>,
  top_k     = 8,
  hybrid    = true
)
logosdb_search(
  query     = <user query>,
  namespace = "codex-global",
  top_k     = 5,
  hybrid    = true
)
```

Present results grouped by namespace. For each hit show: score, text snippet (≤200 chars),
timestamp if available. If nothing found: "No relevant memories found for: <query>".

---

## /memory-remember

```
logosdb_index(
  text      = <provided text>,
  namespace = <active namespace>,
  timestamp = <ISO 8601 now>
)
```

Respond: `Stored. (namespace: <namespace>)`

---

## /memory-index

```
logosdb_index_file(
  path              = <project root or provided path>,
  namespace         = <project namespace>,
  incremental       = true,
  respect_gitignore = true
)
```

Respond: `Indexed <N> chunks across <M> files. (namespace: <namespace>)`

---

## /memory-status

```
logosdb_info(namespace = <project namespace>)
logosdb_info(namespace = "codex-global")
logosdb_list()
```

Respond with a compact table:

```
Memory status
  Mode:             global | project
  Project ns:       codex-proj-<name>   (<N> vectors)
  Global ns:        codex-global        (<N> vectors)
  DB path:          <LOGOSDB_PATH>
```

---

## Namespace derivation

```
project namespace = "codex-proj-" + lowercase(replace(basename(cwd), /[^a-z0-9]+/, "-"))
```

Examples: `/home/user/MyApp` → `codex-proj-myapp`, `/repos/hello_world` → `codex-proj-hello-world`

In `global` mode (default) turn records go to `codex-global`; source-file chunks go to the
project namespace. In `project` mode everything is isolated to the project namespace.

---

## UX rules — silent by design

- Never print tool call names, MCP responses, or progress messages for automatic operations.
- Explicit commands (`/memory-*`) may show one-line output only.
- On any tool error: swallow silently for automatic ops; show a brief error for explicit commands.
