Search LogosDB for semantically similar content.

Arguments: $ARGUMENTS

Parse the arguments:
- Everything before any flag is the search query
- `--namespace=<name>` or `-n <name>` sets the collection (default: "code")
- `--top-k=<n>` or `-k <n>` sets the number of results (default: 5)
- Optional ISO 8601 timestamp window (inclusive), same as MCP `ts_from` / `ts_to`:
  - `--from-ts=<iso>` or `--ts-from=<iso>` → pass as `ts_from`
  - `--to-ts=<iso>` or `--ts-to=<iso>` → pass as `ts_to`
- Optional `--candidate-k=<n>` → pass as `candidate_k` when using a timestamp window (default: 10 × top_k)

Call the `logosdb_search` tool with the query, namespace, top_k, and when bounds are set also ts_from, ts_to, and candidate_k if provided.

Present results in exactly this format:
Searching... Found {N} matches:
  1. {file_path} (score: {score})
  2. {file_path} (score: {score})
  ...

Extract the file path from the [file:...] prefix in the result text.
If no results, respond: No matches found in '{namespace}' namespace.
