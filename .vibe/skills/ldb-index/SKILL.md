---
name: ldb-index
description: Index a file or directory into LogosDB for semantic search across Vibe sessions.
user-invocable: true
allowed-tools:
  - bash
---

Index files into LogosDB using the logosdb-vibe CLI.

Parse the user's arguments:
- First positional argument is the path to index (file or directory)
- `--namespace=<name>` or `-n <name>` sets the collection (default: `code`)

Run exactly:
```
logosdb-vibe index <path> --namespace <namespace>
```

Then respond in this exact format (no extra prose):
Indexed {files} files into '{namespace}' collection
