/**
 * Namespace registry — one LogosDB instance per namespace, all rooted at LOGOSDB_PATH.
 */

import * as fs from 'fs';
import * as path from 'path';

// eslint-disable-next-line @typescript-eslint/no-require-imports
const { DB, DIST_COSINE } = require('logosdb') as {
  DB: new (p: string, opts?: { dim?: number; maxElements?: number; distance?: number }) => LogosDB;
  DIST_COSINE: number;
};

export interface SearchHit {
  id: number;
  score: number;
  text: string | null;
  timestamp: string | null;
}

interface LogosDB {
  put(embedding: number[], text?: string, timestamp?: string): number;
  search(queryEmbedding: number[], topK?: number): SearchHit[];
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

  closeAll(): void {
    for (const db of this.dbs.values()) db.close();
    this.dbs.clear();
  }
}
