---
name: ldb-search
description: Semantic search over an indexed LogosDB namespace.
user-invocable: true
allowed-tools:
  - bash
---

Search LogosDB using the logosdb-vibe CLI.

Parse the user's arguments:
- Everything before any flag is the search query (quote it)
- `--namespace=<name>` or `-n <name>` sets the collection (default: `code`)
- `--top-k=<n>` or `-k <n>` sets result count (default: 5)

Run exactly:
```
logosdb-vibe search "<query>" --namespace <namespace> --top-k <top_k>
```

The CLI returns JSON. Parse it and present results in this format:
Searching... Found {N} matches:
  1. {file} (score: {score})
  2. {file} (score: {score})
  ...

Extract the file path from the `file` field in each result.
If the list is empty, respond: No matches found in '{namespace}' namespace.
