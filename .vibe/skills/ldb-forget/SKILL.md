---
name: ldb-forget
description: Delete an entry from LogosDB by row ID or by semantic query match.
user-invocable: true
allowed-tools:
  - bash
---

Delete entries from LogosDB using the logosdb-vibe CLI.

Parse the user's arguments:
- `--namespace=<name>` or `-n <name>` sets the collection (default: `code`)
- `--id=<number>` deletes by row ID
- `--query=<text>` or `-q <text>` deletes the closest semantic match

Run exactly one of:
```
logosdb-vibe forget --namespace <namespace> --id <id>
logosdb-vibe forget --namespace <namespace> --query "<query>"
```

Then respond: Deleted {n} entr{y/ies} from '{namespace}' namespace.
