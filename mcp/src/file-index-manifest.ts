/**
 * Per-namespace JSON manifest for incremental `logosdb_index_file`:
 * maps absolute file path → last indexed mtime/size/chunk_size and row ids for deletion on change.
 */

import * as fs from 'fs';
import * as path from 'path';

export const MANIFEST_VERSION = 1 as const;

export type FileIndexEntry = {
  mtimeMs: number;
  size: number;
  chunkSize: number;
  /** Chunking strategy used when this file was indexed ("auto" | "line" | "section" | "legacy"). */
  chunkMode?: string;
  ids: number[];
};

export type FileIndexManifest = {
  version: typeof MANIFEST_VERSION;
  files: Record<string, FileIndexEntry>;
};

export function manifestPath(logosdbPath: string, namespace: string): string {
  const safeNs = namespace.replace(/[^a-zA-Z0-9_\-.]/g, '_');
  return path.join(logosdbPath, '_logosdb_mcp_manifests', `${safeNs}.json`);
}

function emptyManifest(): FileIndexManifest {
  return { version: MANIFEST_VERSION, files: {} };
}

function sanitizeEntry(raw: unknown): FileIndexEntry | null {
  if (!raw || typeof raw !== 'object') return null;
  const o = raw as Record<string, unknown>;
  const mtimeMs = typeof o.mtimeMs === 'number' && Number.isFinite(o.mtimeMs) ? o.mtimeMs : NaN;
  const size = typeof o.size === 'number' && Number.isFinite(o.size) && o.size >= 0 ? o.size : NaN;
  const chunkSize =
    typeof o.chunkSize === 'number' && Number.isFinite(o.chunkSize) && o.chunkSize > 0
      ? o.chunkSize
      : NaN;
  const chunkMode = typeof o.chunkMode === 'string' ? o.chunkMode : undefined;
  const idsRaw = o.ids;
  const ids: number[] = [];
  if (Array.isArray(idsRaw)) {
    for (const x of idsRaw) {
      const n = typeof x === 'number' ? x : parseInt(String(x), 10);
      if (Number.isFinite(n)) ids.push(n);
    }
  }
  if (!Number.isFinite(mtimeMs) || !Number.isFinite(size) || !Number.isFinite(chunkSize))
    return null;
  // Only include chunkMode when present — omitting it keeps backward-compatible
  // deep-equality for entries written before the field was introduced.
  return chunkMode !== undefined
    ? { mtimeMs, size, chunkSize, chunkMode, ids }
    : { mtimeMs, size, chunkSize, ids };
}

export function loadManifest(filePath: string): FileIndexManifest {
  try {
    const raw = fs.readFileSync(filePath, 'utf-8');
    const j = JSON.parse(raw) as { version?: unknown; files?: unknown };
    if (j?.version !== MANIFEST_VERSION || !j.files || typeof j.files !== 'object')
      return emptyManifest();
    const files: Record<string, FileIndexEntry> = {};
    for (const [k, v] of Object.entries(j.files as Record<string, unknown>)) {
      const ent = sanitizeEntry(v);
      if (ent) files[k] = ent;
    }
    return { version: MANIFEST_VERSION, files };
  } catch {
    return emptyManifest();
  }
}

export function saveManifest(filePath: string, manifest: FileIndexManifest): void {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const tmp = `${filePath}.${process.pid}.${Date.now()}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(manifest, null, 0), 'utf-8');
  fs.renameSync(tmp, filePath);
}

/** True if `file` is `root` or a file inside directory `root` (both absolute, real paths). */
export function isUnderRoot(file: string, root: string): boolean {
  const rel = path.relative(root, file);
  if (rel === '') return true;
  return !rel.startsWith('..') && !path.isAbsolute(rel);
}

export interface LogosDbLike {
  delete(id: number): void;
}

/**
 * Remove manifest entries (and delete stored rows) for paths under `rootDir` that are not in `currentFiles`.
 */
export function pruneRemovedPaths(
  rootDir: string,
  currentFiles: Set<string>,
  manifest: FileIndexManifest,
  db: LogosDbLike,
): number {
  let pruned = 0;
  for (const fileKey of Object.keys(manifest.files)) {
    if (!isUnderRoot(fileKey, rootDir)) continue;
    if (currentFiles.has(fileKey)) continue;
    const ent = manifest.files[fileKey];
    if (ent?.ids?.length) {
      for (const id of ent.ids) {
        try {
          db.delete(id);
        } catch {
          /* row may already be gone */
        }
      }
    }
    delete manifest.files[fileKey];
    pruned++;
  }
  return pruned;
}
