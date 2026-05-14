/**
 * Namespace registry — one LogosDB instance per namespace, all rooted at LOGOSDB_PATH.
 *
 * Native `logosdb` is loaded lazily so the MCP process can complete initialize + tools/list
 * even when prebuilds are missing (clear error on first tool call instead of "0 tools").
 */

import * as fs from 'fs';
import * as path from 'path';

type LogosDBCtor = new (
  p: string,
  opts?: { dim?: number; maxElements?: number; distance?: number },
) => LogosDB;

let nativeCache: { DB: LogosDBCtor; DIST_COSINE: number } | null = null;
let nativeLoadError: Error | null = null;

function getNative(): { DB: LogosDBCtor; DIST_COSINE: number } {
  if (nativeLoadError) throw nativeLoadError;
  if (nativeCache) return nativeCache;
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const mod = require('logosdb') as { DB: LogosDBCtor; DIST_COSINE: number };
    nativeCache = { DB: mod.DB, DIST_COSINE: mod.DIST_COSINE };
    return nativeCache;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    nativeLoadError = err instanceof Error ? err : new Error(msg);
    process.stderr.write(
      `[logosdb-mcp] Native dependency "logosdb" failed to load (${msg}). ` +
        'Install logosdb@^0.7.12 (N-API prebuilds for this OS/arch; from source needs C++17). See nodejs/README.md.\n',
    );
    throw nativeLoadError;
  }
}

export interface SearchHit {
  id: number;
  score: number;
  text: string | null;
  timestamp: string | null;
}

interface LogosDB {
  put(embedding: number[], text?: string, timestamp?: string): number;
  putBatch(
    embeddings: number[],
    n: number,
    texts?: (string | null | undefined)[],
    timestamps?: (string | null | undefined)[],
  ): number[];
  search(queryEmbedding: number[], topK?: number): SearchHit[];
  searchTsRange(
    queryEmbedding: number[],
    options?: { topK?: number; tsFrom?: string; tsTo?: string; candidateK?: number },
  ): SearchHit[];
  delete(id: number): void;
  count(): number;
  countLive(): number;
  dim(): number;
  close(): void;
}

export class NamespaceStore {
  private readonly root: string;
  private readonly dbs = new Map<string, LogosDB>();

  constructor(root: string) {
    this.root = root;
    fs.mkdirSync(root, { recursive: true });
  }

  private nsPath(namespace: string): string {
    // Sanitise: only allow alphanumeric, hyphen, underscore, dot
    if (!/^[a-zA-Z0-9_\-\.]+$/.test(namespace)) {
      throw new Error(`Invalid namespace: "${namespace}". Use [a-z A-Z 0-9 _ - .] only.`);
    }
    return path.join(this.root, namespace);
  }

  open(namespace: string, dim: number): LogosDB {
    if (this.dbs.has(namespace)) return this.dbs.get(namespace)!;

    const { DB, DIST_COSINE } = getNative();
    const p = this.nsPath(namespace);
    fs.mkdirSync(p, { recursive: true });

    const db = new DB(p, { dim, distance: DIST_COSINE });
    this.dbs.set(namespace, db);
    return db;
  }

  get(namespace: string): LogosDB {
    const db = this.dbs.get(namespace);
    if (!db) throw new Error(`Namespace "${namespace}" is not open. Index something first.`);
    return db;
  }

  list(): string[] {
    try {
      return fs.readdirSync(this.root).filter((entry) => {
        const stat = fs.statSync(path.join(this.root, entry));
        return stat.isDirectory();
      });
    } catch {
      return [];
    }
  }

  /**
   * Return live vector count and dim for every currently-open namespace.
   * Namespaces that have never been opened return no entry — callers can
   * present them as "not yet accessed" without forcing a DB open.
   */
  openStatsSnapshot(): Record<string, { countLive: number; dim: number }> {
    const out: Record<string, { countLive: number; dim: number }> = {};
    for (const [ns, db] of this.dbs.entries()) {
      out[ns] = { countLive: db.countLive(), dim: db.dim() };
    }
    return out;
  }

  closeAll(): void {
    for (const db of this.dbs.values()) db.close();
    this.dbs.clear();
  }
}
