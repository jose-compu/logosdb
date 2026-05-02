Index a file or directory into LogosDB for semantic search.

Arguments: $ARGUMENTS

Parse the arguments:
- First positional argument is the path to index (file or directory)
- `--namespace=<name>` or `-n <name>` sets the collection name (default: "code")

Call the `logosdb_index_file` tool with the resolved path and namespace.

When done, respond in exactly this format (no extra prose):
Indexed {files} files into '{namespace}' collection
