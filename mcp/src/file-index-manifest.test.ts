import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { describe, it } from 'node:test';
import {
  isUnderRoot,
  loadManifest,
  MANIFEST_VERSION,
  pruneRemovedPaths,
  saveManifest,
  type FileIndexManifest,
} from './file-index-manifest.js';

describe('file-index-manifest', () => {
  it('isUnderRoot', () => {
    assert.equal(isUnderRoot('/a/b/c', '/a/b'), true);
    assert.equal(isUnderRoot('/a/b', '/a/b'), true);
    assert.equal(isUnderRoot('/a/c', '/a/b'), false);
    assert.equal(isUnderRoot('/ab', '/a'), false);
  });

  it('loadManifest tolerates missing file', () => {
    const m = loadManifest(path.join(os.tmpdir(), `missing-manifest-${Date.now()}.json`));
    assert.equal(m.version, MANIFEST_VERSION);
    assert.deepEqual(m.files, {});
  });

  it('saveManifest round-trip', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'logosdb-manifest-'));
    const p = path.join(dir, 'ns.json');
    const m: FileIndexManifest = {
      version: MANIFEST_VERSION,
      files: { '/x/a.ts': { mtimeMs: 1, size: 2, chunkSize: 800, ids: [1, 2, 3] } },
    };
    saveManifest(p, m);
    const m2 = loadManifest(p);
    assert.deepEqual(m2.files['/x/a.ts'], m.files['/x/a.ts']);
    fs.rmSync(dir, { recursive: true });
  });

  it('pruneRemovedPaths deletes stale keys and calls db.delete', () => {
    const deleted: number[] = [];
    const db = {
      delete(id: number) {
        deleted.push(id);
      },
    };
    const manifest: FileIndexManifest = {
      version: MANIFEST_VERSION,
      files: {
        '/proj/a.ts': { mtimeMs: 1, size: 1, chunkSize: 800, ids: [10] },
        '/proj/b.ts': { mtimeMs: 1, size: 1, chunkSize: 800, ids: [20, 21] },
        '/other/x.ts': { mtimeMs: 1, size: 1, chunkSize: 800, ids: [99] },
      },
    };
    const n = pruneRemovedPaths('/proj', new Set(['/proj/a.ts']), manifest, db);
    assert.equal(n, 1);
    assert.ok(!manifest.files['/proj/b.ts']);
    assert.ok(manifest.files['/proj/a.ts']);
    assert.ok(manifest.files['/other/x.ts']);
    assert.deepEqual(deleted.sort((a, b) => a - b), [20, 21]);
  });
});
