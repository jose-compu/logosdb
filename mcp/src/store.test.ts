import { describe, it, after } from 'node:test';
import assert from 'node:assert/strict';
import * as os from 'node:os';
import * as path from 'node:path';
import * as fs from 'node:fs';

import { NamespaceStore } from './store.js';

// ── openStatsSnapshot ────────────────────────────────────────────────────────
// Tests use mock DB injection to avoid loading the native logosdb addon.

describe('NamespaceStore.openStatsSnapshot', () => {
  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'logosdb-store-snap-'));

  after(() => {
    fs.rmSync(tmpRoot, { recursive: true, force: true });
  });

  it('returns empty object when no namespace has been opened', () => {
    const store = new NamespaceStore(tmpRoot);
    assert.deepEqual(store.openStatsSnapshot(), {});
  });

  it('returns countLive and dim for every open namespace', () => {
    const store = new NamespaceStore(tmpRoot);

    // Inject mock DB instances directly into the private Map — avoids loading native addon.
    const mockA = { countLive: () => 42, dim: () => 384 };
    const mockB = { countLive: () => 7, dim: () => 768 };
    (store as unknown as { dbs: Map<string, typeof mockA> }).dbs.set('ns-alpha', mockA);
    (store as unknown as { dbs: Map<string, typeof mockB> }).dbs.set('ns-beta', mockB);

    assert.deepEqual(store.openStatsSnapshot(), {
      'ns-alpha': { countLive: 42, dim: 384 },
      'ns-beta': { countLive: 7, dim: 768 },
    });
  });

  it('reflects updated counts after mock is mutated', () => {
    const store = new NamespaceStore(tmpRoot);
    let live = 10;
    const mockDb = { countLive: () => live, dim: () => 128 };
    (store as unknown as { dbs: Map<string, typeof mockDb> }).dbs.set('mutable', mockDb);

    assert.equal(store.openStatsSnapshot()['mutable']?.countLive, 10);
    live = 999;
    assert.equal(store.openStatsSnapshot()['mutable']?.countLive, 999);
  });

  it('does not include namespaces that exist on disk but have not been opened', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'logosdb-store-disk-'));
    try {
      // A namespace directory exists on disk but was never opened via store.open()
      fs.mkdirSync(path.join(root, 'disk-only-ns'));
      const store = new NamespaceStore(root);
      const snap = store.openStatsSnapshot();
      assert.equal(Object.keys(snap).length, 0, 'disk-only namespaces must not appear in snapshot');
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  });
});

// ── list ─────────────────────────────────────────────────────────────────────

describe('NamespaceStore.list', () => {
  it('returns only directory entries — not files', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'logosdb-store-list-'));
    try {
      fs.mkdirSync(path.join(root, 'ns-1'));
      fs.mkdirSync(path.join(root, 'ns-2'));
      fs.writeFileSync(path.join(root, 'manifest.json'), '{}');
      fs.writeFileSync(path.join(root, '.logosdb-index'), '');

      const store = new NamespaceStore(root);
      const ns = store.list();
      assert.ok(ns.includes('ns-1'), 'should list ns-1');
      assert.ok(ns.includes('ns-2'), 'should list ns-2');
      assert.ok(!ns.includes('manifest.json'), 'should not list files');
      assert.ok(!ns.includes('.logosdb-index'), 'should not list files');
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  });

  it('returns empty array when root does not exist', () => {
    const store = new NamespaceStore(path.join(os.tmpdir(), 'logosdb-nonexistent-root-' + Date.now()));
    // The store constructor creates the root, so we need to remove it after creation
    // to simulate a missing root for the list() call.
    const rootPath = (store as unknown as { root: string }).root;
    fs.rmSync(rootPath, { recursive: true, force: true });
    assert.deepEqual(store.list(), []);
  });

  it('returns sorted or at minimum all directory entries', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'logosdb-store-multi-'));
    try {
      ['bravo', 'alpha', 'charlie', 'delta'].forEach((n) =>
        fs.mkdirSync(path.join(root, n)),
      );
      const store = new NamespaceStore(root);
      const ns = store.list();
      assert.equal(ns.length, 4);
      ['alpha', 'bravo', 'charlie', 'delta'].forEach((n) =>
        assert.ok(ns.includes(n), `should include ${n}`),
      );
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  });
});
