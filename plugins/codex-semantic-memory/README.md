# codex-semantic-memory

Local semantic memory for [Codex](https://developers.openai.com/codex). The plugin talks to [logosdb-mcp-server](https://www.npmjs.com/package/logosdb-mcp-server) over stdio. Indexes and turn records stay on disk.

Requires **logosdb-mcp-server 1.4.7** (LogosDB 1.7.0 search filters, batch get, and structure-aware indexing). 1.4.7 is in this repository and is not on npm until it is published. Until then, install it from the repo (see below). A global binary older than 1.4.7 is ignored.

## What it indexes

`logosdb_index_file` is called with `chunking: "auto"` and `chunk_context: "header"`.

| File | Split |
|---|---|
| Code (`.ts`, `.py`, `.go`, `.rs`, …) | Function and class boundaries. A unit longer than about 50 lines becomes a sliding window with about 10 lines of overlap. |
| Markdown and RST | Headings. Small sections are merged. A long section falls back to paragraphs. |
| Other text | Paragraphs. A long paragraph breaks on a sentence when it can, with about 100 characters of overlap. |

Each stored row is labeled with the file path, line range, and the function, class, or heading when there is one. The embedded string gets the same facts in a short header. Details: [`mcp/README.md`](../../mcp/README.md#chunking-strategies).

Search can take `allow` and `deny` row-id lists (at most 4096) and `min_score`. `logosdb_search_batch` runs several queries at once. `logosdb_get_many` returns embeddings for up to 32 ids. The silent recall path does not set a score cutoff.

## Prerequisites

Node.js 18 or newer, `bash`, and `git` (so the wrapper can find the project root).

```bash
# from a checkout of this repository, until 1.4.7 is on npm
npm install -g ./mcp
logosdb-mcp-server --version   # 1.4.7

# after publish
npm install -g logosdb-mcp-server@1.4.7
```

## Install in Codex

The catalog for this repo is [`.agents/plugins/marketplace.json`](../../.agents/plugins/marketplace.json).

From the repository root:

```bash
codex plugin marketplace add .
```

Or from GitHub:

```bash
codex plugin marketplace add jose-compu/logosdb
```

Restart Codex, then install **LogosDB Memory** from the marketplace named `logosdb`. To enable it for one trusted project, add this to `.codex/config.toml`:

```toml
[plugins."codex-semantic-memory@logosdb"]
enabled = true
```

A personal copy:

```bash
mkdir -p ~/.codex/plugins
cp -R plugins/codex-semantic-memory ~/.codex/plugins/codex-semantic-memory
```

Point `~/.agents/plugins/marketplace.json` at that directory with a `./` path relative to your home directory, then restart Codex.

Without the plugin, the same server can be registered directly:

```toml
[mcp_servers.logosdb]
command = "bash"
args = ["scripts/logosdb-mcp-wrap.sh"]
cwd = "/absolute/path/to/plugins/codex-semantic-memory"
```

The wrapper sets `LOGOSDB_INDEX_ROOT` to the git root of the process working directory, which is what allows `logosdb_index_file` to read the project when the server is not started inside it. Set `LOGOSDB_INDEX_ROOT` yourself to override that.

## Memory scope

`LOGOS_MEMORY_MODE` defaults to `global`.

| Mode | Behaviour |
|---|---|
| `global` | Turn records go to `codex-global`. Source files go to `codex-proj-<name>`. Search covers both. Database: `~/.codex/.logosdb`. |
| `project` | Everything stays in `codex-proj-<name>`. Database: `<project>/.logosdb`. |

```bash
export LOGOS_MEMORY_MODE=project
```

`LOGOSDB_PATH`, when set, replaces that choice.

## Commands

| Command | Description |
|---|---|
| `/memory-recall <query>` | Search past memories |
| `/memory-remember <text>` | Store text |
| `/memory-index [path]` | Re-index with `chunking: "auto"` |
| `/memory-status` | Namespace, mode, vector counts |

Automatic indexing, recall, and turn storage stay silent. The rules are in `AGENTS.md` and `skills/semantic-memory/SKILL.md`.
