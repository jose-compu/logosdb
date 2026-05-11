# logosdb-mcp-server

MCP server that exposes [LogosDB](https://github.com/jose-compu/logosdb) semantic search to
**Claude Code**, **Google Antigravity**, and any other [Model Context Protocol](https://modelcontextprotocol.io/) client.

## Install

```bash
npm install -g logosdb-mcp-server
```

Or run without installing via `npx -y logosdb-mcp-server`.

## Configure

### Default: local embeddings (Transformers.js)

If you omit `EMBEDDING_PROVIDER`, the server uses **[Transformers.js](https://github.com/xenova/transformers.js)** (`@xenova/transformers`) with a small on-device model. **No API keys.** The first run may download weights (cache under the standard Transformers.js cache directory).

Add to `.claude/mcp.json` in your project (or `~/.claude.json` for global use):

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

Default model: `Xenova/all-MiniLM-L6-v2` (384 dimensions). Override:

```json
"env": {
  "LOGOSDB_PATH": "./.logosdb",
  "TRANSFORMERS_MODEL": "Xenova/all-MiniLM-L6-v2",
  "TRANSFORMERS_EMBEDDING_DIM": "384"
}
```

If you switch models, set `TRANSFORMERS_EMBEDDING_DIM` / `EMBEDDING_DIM` to the **exact** output size of that model.

### Path confinement (`logosdb_index_file`)

Indexing resolves paths with `realpath` and only allows files under **`process.cwd()`** or, if set, **`LOGOSDB_INDEX_ROOT`** (absolute directory). Symlinks that escape those roots are rejected. Optional: set **`EMBEDDING_FETCH_TIMEOUT_MS`** (milliseconds, capped at 600000) for Ollama/OpenAI/Voyage HTTP calls; default is 120000.

### Local HTTP: Ollama

Run [Ollama](https://ollama.com) with an embedding model (e.g. `nomic-embed-text`), then:

```json
"env": {
  "LOGOSDB_PATH": "./.logosdb",
  "EMBEDDING_PROVIDER": "ollama",
  "OLLAMA_HOST": "http://127.0.0.1:11434",
  "OLLAMA_EMBED_MODEL": "nomic-embed-text",
  "OLLAMA_EMBEDDING_DIM": "768"
}
```

### Cloud (opt-in): OpenAI

```json
"env": {
  "LOGOSDB_PATH": "./.logosdb",
  "EMBEDDING_PROVIDER": "openai",
  "OPENAI_API_KEY": "<your-openai-api-key>"
}
```

### Cloud (opt-in): Voyage AI (dim=1024)

```json
"env": {
  "LOGOSDB_PATH": "./.logosdb",
  "EMBEDDING_PROVIDER": "voyage",
  "VOYAGE_API_KEY": "<your-voyage-api-key>"
}
```

## Google Antigravity

[Google Antigravity](https://codelabs.developers.google.com/google-workspace-mcp-antigravity) is an agentic IDE stack that can load **MCP servers** over **stdio** — the same pattern as Claude Code. **`logosdb-mcp-server`** is published as [`logosdb-mcp-server`](https://www.npmjs.com/package/logosdb-mcp-server) on npm and speaks standard MCP tools (`logosdb_index`, `logosdb_index_file`, `logosdb_search`, `logosdb_list`, `logosdb_info`, `logosdb_delete`). See the [MCP specification](https://modelcontextprotocol.io).

### Where to configure

Exact menu labels change between Antigravity builds. In general:

1. Open the **Agent** (or AI) panel.
2. Use **Manage MCP servers**, **MCP settings**, or **View raw config** (wording may differ).
3. Add a stdio server whose **command** is `npx` and **args** include `-y` and `logosdb-mcp-server`, with **environment variables** for `LOGOSDB_PATH` and embedding options.

If Antigravity exposes a raw JSON file (`mcpServers` or project-level MCP config), use the same JSON shape as below. Confirm the file path in your Antigravity version if the agent cannot start the server.

**Requirements:** [Node.js](https://nodejs.org/) on your `PATH` (for `npx`) and a writable `LOGOSDB_PATH` directory.

### Recommended: local embeddings (no API keys)

Matches the default elsewhere in this README — no `OPENAI_API_KEY` required:

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

### Optional: cloud embeddings

**OpenAI:**

```json
{
  "mcpServers": {
    "logosdb": {
      "command": "npx",
      "args": ["-y", "logosdb-mcp-server"],
      "env": {
        "LOGOSDB_PATH": "./.logosdb",
        "EMBEDDING_PROVIDER": "openai",
        "OPENAI_API_KEY": "<your-openai-api-key>"
      }
    }
  }
}
```

**Voyage AI** (`EMBEDDING_PROVIDER=voyage`, `VOYAGE_API_KEY`): see the [Configure](#configure) section above.

**Ollama** (local HTTP): set `EMBEDDING_PROVIDER=ollama` and the `OLLAMA_*` variables from [Local HTTP: Ollama](#local-http-ollama).

### Further reading

- Google codelab — [Google Workspace MCP servers in Antigravity](https://codelabs.developers.google.com/google-workspace-mcp-antigravity)
- Community walkthrough — [How to use MCP servers in Antigravity](https://antigravity.codes/blog/antigravity-mcp-tutorial)

## Slash commands

Three custom slash commands are included in `.claude/commands/` and are available as soon as you open Claude Code in this project:

| Command | Example |
|---|---|
| `/index` | `/index ./src --namespace=backend` |
| `/search` | `/search "JWT validation" --namespace=backend` |
| `/forget` | `/forget --namespace=backend --id=42` |

## Example session

Once configured, start Claude Code in your project:

```
$ cd myproject && claude

> /index ./src --namespace=backend
Indexed 42 files into 'backend' collection

> Find where we handle JWT validation
Searching... Found 3 matches:
  1. src/auth/jwt.ts (score: 0.94)
  2. src/middleware/auth.ts (score: 0.87)
  3. src/utils/token.ts (score: 0.72)

> Show me the first one
[Claude displays src/auth/jwt.ts with explanation]

---

> Index the src/ directory so you can search it semantically

● logosdb_index_file({ path: "./src/auth/jwt.ts", namespace: "code" })
  ✓ { indexed: 4, namespace: "code", file: "./src/auth/jwt.ts" }
● logosdb_index_file({ path: "./src/middleware/auth.ts", namespace: "code" })
  ✓ { indexed: 2, namespace: "code", file: "./src/middleware/auth.ts" }
  … (42 files indexed)

I've indexed the src/ directory into the "code" namespace — 42 files, 187 chunks.

> Find where we validate JWT tokens

● logosdb_search({ query: "JWT token validation", namespace: "code", top_k: 5 })
  ✓ {
      results: [
        { score: 0.9431, file: "src/auth/jwt.ts",         text: "export function verifyToken(token) {…" },
        { score: 0.8712, file: "src/middleware/auth.ts",   text: "const decoded = jwt.verify(req.headers…" },
        { score: 0.8204, file: "src/utils/token.ts",       text: "function decodePayload(token: string)…"  },
      ]
    }

The JWT validation logic lives in three places:
1. **src/auth/jwt.ts** — `verifyToken()` is the main entry point (score 0.94)
2. **src/middleware/auth.ts** — Express middleware that calls `verifyToken` on every request
3. **src/utils/token.ts** — low-level payload decoder used by both

> Remember that we decided to use RS256 instead of HS256 going forward

● logosdb_index({ text: "Decision: use RS256 (asymmetric) instead of HS256 for JWT signing. Decided 2026-05-02.", namespace: "decisions" })
  ✓ { id: 0, indexed: true, namespace: "decisions" }

Noted. I'll keep that in the "decisions" namespace for future sessions.
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LOGOSDB_PATH` | `./.logosdb` | Root directory for all namespace databases |
| `EMBEDDING_PROVIDER` | *(unset → local)* | `transformers` / `local` / `auto` (default), `ollama`, `openai`, `voyage` |
| `TRANSFORMERS_MODEL` | `Xenova/all-MiniLM-L6-v2` | Hugging Face id for Transformers.js |
| `TRANSFORMERS_EMBEDDING_DIM` | `384` | Must match model output (or set `EMBEDDING_DIM`) |
| `EMBEDDING_DIM` | — | Optional global override for expected vector length |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | When `EMBEDDING_PROVIDER=ollama` |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model name |
| `OLLAMA_EMBEDDING_DIM` | `768` | Must match the model’s embedding size |
| `OPENAI_API_KEY` | — | Required when `EMBEDDING_PROVIDER=openai` |
| `VOYAGE_API_KEY` | — | Required when `EMBEDDING_PROVIDER=voyage` |
| `LOGOSDB_CHUNK_SIZE` | `800` | Target characters per chunk when indexing files |

**Important:** Use one embedding backend consistently for a given namespace on disk. Mixing dimensions or models on the same `LOGOSDB_PATH` namespace will produce invalid search results.

## Tools

| Tool | Inputs | Description |
|---|---|---|
| `logosdb_index` | `text`, `namespace`, `metadata?` | Embed and store a text snippet |
| `logosdb_index_file` | `path`, `namespace`, `chunk_size?` | Chunk, embed, and store a file |
| `logosdb_search` | `query`, `namespace`, `top_k?`, `ts_from?`, `ts_to?`, `candidate_k?` | Semantic search; optional inclusive ISO 8601 time window (maps to `search_ts_range`) |
| `logosdb_list` | — | List all namespaces |
| `logosdb_info` | `namespace` | Stats: count, live count, dimension |
| `logosdb_delete` | `namespace`, `id?` **or** `query?`, `search_top_k?`, `match_rank?` | Delete by row id, or embed `query` and delete the `match_rank` hit (default 0) among `search_top_k` neighbors |

Timestamp-filtered search matches the core library: when `ts_from` and/or `ts_to` are set, results are drawn from that window; `candidate_k` defaults to `10 × top_k` if omitted.

## Development

```bash
npm install --ignore-scripts   # skip native addon build (use linked logosdb or published wheel)
npm run build                  # tsc → dist/
npm test                       # path / control-character regression tests (no DB required)
npm start                      # run server directly
```

## License

MIT
