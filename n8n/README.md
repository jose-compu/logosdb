# n8n-nodes-logosdb

Community **n8n** nodes for **[LogosDB](https://github.com/jose-compu/logosdb)** — insert, search, delete, and database info against a local LogosDB directory (via the **`logosdb`** native npm package).

## Monorepo layout

This package lives under the LogosDB repo **`n8n/`**. The root **`package.json`** lists **`n8n`** as an npm workspace so **`logosdb`** resolves from **`../nodejs/`** after `npm install` at the repo root. See the main **[CONTRIBUTING.md](../CONTRIBUTING.md)** for workspace install and publish order.

## Development

From the **repository root**:

```bash
npm install
npm run build -w n8n-nodes-logosdb
```

From **`n8n/`** alone (after dependencies are installed):

```bash
npm run build
npm run lint
```

## Publishing

Published to npm as **`n8n-nodes-logosdb`**. Release the **`logosdb`** native package first so **`^`** semver dependencies resolve on the registry.
