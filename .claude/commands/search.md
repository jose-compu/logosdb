Search LogosDB for semantically similar content.

Arguments: $ARGUMENTS

Parse the arguments:
- Everything before any flag is the search query
- `--namespace=<name>` or `-n <name>` sets the collection (default: "code")
- `--top-k=<n>` or `-k <n>` sets the number of results (default: 5)

Call the `logosdb_search` tool with the query, namespace, and top_k.

Present results in exactly this format:
Searching... Found {N} matches:
  1. {file_path} (score: {score})
  2. {file_path} (score: {score})
  ...

Extract the file path from the [file:...] prefix in the result text.
If no results, respond: No matches found in '{namespace}' namespace.
