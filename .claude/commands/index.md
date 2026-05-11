Index a file or directory into LogosDB for semantic search.

Arguments: $ARGUMENTS

Parse the arguments:
- First positional argument is the path to index (file or directory)
- `--namespace=<name>` or `-n <name>` sets the collection name (default: "code")

Call the `logosdb_index_file` tool with the resolved path, namespace, and **`incremental: true`**.

**Incremental indexing:** only files that are new or changed since the last run (same namespace) are embedded again; for changed files the tool removes the previous chunk rows first. Unchanged files are skipped. For a **directory** path, files that were indexed before but are no longer present under that tree are removed from the store.

When done, respond in exactly this format (no extra prose), substituting values from the tool result (`indexed`, `indexed_files`, `skipped_files`, `namespace`):
Indexed {indexed} chunks ({indexed_files} files updated, {skipped_files} skipped) into '{namespace}' collection
