# Semantic Memory — Codex Instructions

This project uses the **codex-semantic-memory** plugin backed by a local LogosDB MCP server.
All automatic memory operations are **silent** — never mention them to the user unless they ask.

## UX contract — silence is mandatory

| Operation | What you do | What you say |
|---|---|---|
| Session start — index project | Call `logosdb_index_file` (incremental, `.gitignore`-aware) | Nothing |
| Every user prompt — recall | Call `logosdb_search` and prepend relevant hits to your reasoning | Nothing |
| After each response — store | Call `logosdb_index` with a compact turn record | Nothing |
| `/memory-recall <query>` | Run an explicit search and show results | Show results |
| `/memory-remember <text>` | Explicitly store provided text | One-line confirmation |
| `/memory-index` | Re-index the project | One-line status |
| `/memory-status` | Show namespace, mode, vector count | Show summary |

Never print "Indexing…", "Searching memory…", "Storing…", or any equivalent.
If a memory tool call fails, swallow the error silently and continue.

## Automatic operations on every session start

1. **Index project** (background, do not wait for result to start answering):
   ```
   logosdb_index_file(
     path = <project root or cwd>,
     namespace = <project namespace>,   // e.g. "codex-proj-myapp"
     incremental = true,
     respect_gitignore = true
   )
   ```
2. **Search for context** matching the first user message:
   ```
   logosdb_search(query = <first user message>, namespace = <project namespace>, top_k = 5)
   logosdb_search(query = <first user message>, namespace = "codex-global",     top_k = 3)
   ```
   Inject any relevant results silently into your working context before answering.

## Automatic operations after every response

Store a compact turn record:
```
logosdb_index(
  text      = "[Q] <user message in ≤120 chars>\n[A] <your response in ≤300 chars>",
  namespace = <active namespace>,   // "codex-global" in global mode
  timestamp = <ISO 8601 now>
)
```

If you produced significant internal reasoning or a decision, append it:
```
[reasoning] <key decision or approach in ≤200 chars>
```

## Automatic operations on every user prompt (after the first)

Before answering, search memory:
```
logosdb_search(query = <user message>, namespace = <project namespace>, top_k = 5)
logosdb_search(query = <user message>, namespace = "codex-global",     top_k = 3)
```
Use any returned context silently when composing your answer.

## Namespace rules

| Variable | Namespace used |
|---|---|
| `LOGOS_MEMORY_MODE=global` (default) | Turn records → `codex-global`; source files → `codex-proj-<basename>` |
| `LOGOS_MEMORY_MODE=project` | All storage → `codex-proj-<basename>`; searches scoped to project only |

Derive `<basename>` from the lowercase, hyphenated directory name of the project root
(e.g. `/home/user/my-app` → `codex-proj-my-app`).

## Tools available (via logosdb MCP server)

| Tool | Purpose |
|---|---|
| `logosdb_index` | Store one text chunk with optional tags and timestamp |
| `logosdb_index_file` | Index a file or directory (chunked, incremental, gitignore-aware) |
| `logosdb_search` | Semantic search; supports `hybrid`, `filter`, `top_k` |
| `logosdb_list` | List namespaces with live vector counts |
| `logosdb_info` | Detailed stats for one namespace |
| `logosdb_delete` | Delete a record by ID or semantic match |
