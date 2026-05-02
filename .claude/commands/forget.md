Delete an entry from LogosDB by row ID.

Arguments: $ARGUMENTS

Parse the arguments:
- `--namespace=<name>` or `-n <name>` sets the collection (default: "code")
- `--id=<number>` or positional number is the row ID to delete

Call the `logosdb_delete` tool with the namespace and id.

Respond: Deleted entry {id} from '{namespace}' namespace.
