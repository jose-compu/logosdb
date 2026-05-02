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
