# logosdb-mcp-server

MCP server that exposes [LogosDB](https://github.com/jose-compu/logosdb) semantic search to
**Claude Code** and any other [Model Context Protocol](https://modelcontextprotocol.io/) client.

## Install

```bash
npm install -g logosdb-mcp-server
```

Or run without installing via `npx -y logosdb-mcp-server`.

## Configure

Add to `.claude/mcp.json` in your project (or `~/.claude.json` for global use):

```json
{
  "mcpServers": {
    "logosdb": {
      "command": "npx",
      "args": ["-y", "logosdb-mcp-server"],
      "env": {
        "LOGOSDB_PATH": "./.logosdb",
        "OPENAI_API_KEY": "<your-openai-api-key>"
      }
    }
  }
}
```

For Voyage AI embeddings (Anthropic-recommended, dim=1024):

```json
"env": {
  "LOGOSDB_PATH": "./.logosdb",
  "EMBEDDING_PROVIDER": "voyage",
  "VOYAGE_API_KEY": "<your-voyage-api-key>"
}
```

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
| `EMBEDDING_PROVIDER` | `openai` | `openai` or `voyage` |
| `OPENAI_API_KEY` | — | Required when provider is `openai` |
| `VOYAGE_API_KEY` | — | Required when provider is `voyage` |
| `LOGOSDB_CHUNK_SIZE` | `800` | Target characters per chunk when indexing files |

## Tools

| Tool | Inputs | Description |
|---|---|---|
| `logosdb_index` | `text`, `namespace`, `metadata?` | Embed and store a text snippet |
| `logosdb_index_file` | `path`, `namespace`, `chunk_size?` | Chunk, embed, and store a file |
| `logosdb_search` | `query`, `namespace`, `top_k?` | Semantic search |
| `logosdb_list` | — | List all namespaces |
| `logosdb_info` | `namespace` | Stats: count, live count, dimension |
| `logosdb_delete` | `namespace`, `id` | Delete an entry by row ID |

## Development

```bash
npm install --ignore-scripts   # skip native addon build
npm run build                  # tsc → dist/
npm start                      # run server directly
```

## License

MIT
