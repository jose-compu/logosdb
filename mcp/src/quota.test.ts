import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';

import {
  loadQuotaConfig,
  checkVectorQuota,
  checkNsQuota,
  QuotaExceededError,
} from './quota.js';

// ── loadQuotaConfig ───────────────────────────────────────────────────────────

describe('loadQuotaConfig', () => {
  let savedVec: string | undefined;
  let savedNs: string | undefined;

  before(() => {
    savedVec = process.env.LOGOSDB_QUOTA_MAX_VECTORS;
    savedNs = process.env.LOGOSDB_QUOTA_MAX_NAMESPACES;
  });

  after(() => {
    if (savedVec === undefined) delete process.env.LOGOSDB_QUOTA_MAX_VECTORS;
    else process.env.LOGOSDB_QUOTA_MAX_VECTORS = savedVec;
    if (savedNs === undefined) delete process.env.LOGOSDB_QUOTA_MAX_NAMESPACES;
    else process.env.LOGOSDB_QUOTA_MAX_NAMESPACES = savedNs;
  });

  it('defaults to 0 (unlimited) when env vars are absent', () => {
    delete process.env.LOGOSDB_QUOTA_MAX_VECTORS;
    delete process.env.LOGOSDB_QUOTA_MAX_NAMESPACES;
    const cfg = loadQuotaConfig();
    assert.equal(cfg.maxVectorsPerNs, 0);
    assert.equal(cfg.maxNamespaces, 0);
  });

  it('reads positive integer values from env vars', () => {
    process.env.LOGOSDB_QUOTA_MAX_VECTORS = '5000';
    process.env.LOGOSDB_QUOTA_MAX_NAMESPACES = '10';
    const cfg = loadQuotaConfig();
    assert.equal(cfg.maxVectorsPerNs, 5000);
    assert.equal(cfg.maxNamespaces, 10);
  });

  it('clamps negative values to 0', () => {
    process.env.LOGOSDB_QUOTA_MAX_VECTORS = '-1';
    process.env.LOGOSDB_QUOTA_MAX_NAMESPACES = '-99';
    const cfg = loadQuotaConfig();
    assert.equal(cfg.maxVectorsPerNs, 0);
    assert.equal(cfg.maxNamespaces, 0);
  });

  it('clamps non-numeric values to 0', () => {
    process.env.LOGOSDB_QUOTA_MAX_VECTORS = 'unlimited';
    process.env.LOGOSDB_QUOTA_MAX_NAMESPACES = '';
    const cfg = loadQuotaConfig();
    assert.equal(cfg.maxVectorsPerNs, 0);
    assert.equal(cfg.maxNamespaces, 0);
  });

  it('treats explicit "0" string as 0 (not as undefined/default confusion)', () => {
    process.env.LOGOSDB_QUOTA_MAX_VECTORS = '0';
    process.env.LOGOSDB_QUOTA_MAX_NAMESPACES = '0';
    const cfg = loadQuotaConfig();
    assert.equal(cfg.maxVectorsPerNs, 0);
    assert.equal(cfg.maxNamespaces, 0);
  });

  it('truncates float strings via parseInt (no rounding surprises)', () => {
    process.env.LOGOSDB_QUOTA_MAX_VECTORS = '2.9';
    process.env.LOGOSDB_QUOTA_MAX_NAMESPACES = '9.1';
    const cfg = loadQuotaConfig();
    assert.equal(cfg.maxVectorsPerNs, 2);
    assert.equal(cfg.maxNamespaces, 9);
  });
});

// ── checkVectorQuota ──────────────────────────────────────────────────────────

describe('checkVectorQuota', () => {
  const ns = 'test-ns';

  it('does nothing when limit is 0 (unlimited)', () => {
    assert.doesNotThrow(() =>
      checkVectorQuota(999999, 999999, ns, { maxVectorsPerNs: 0, maxNamespaces: 0 }),
    );
  });

  it('allows insert when under limit', () => {
    assert.doesNotThrow(() =>
      checkVectorQuota(40, 10, ns, { maxVectorsPerNs: 100, maxNamespaces: 0 }),
    );
  });

  it('allows insert that exactly reaches the limit', () => {
    assert.doesNotThrow(() =>
      checkVectorQuota(90, 10, ns, { maxVectorsPerNs: 100, maxNamespaces: 0 }),
    );
  });

  it('throws QuotaExceededError when limit would be exceeded', () => {
    assert.throws(
      () => checkVectorQuota(95, 10, ns, { maxVectorsPerNs: 100, maxNamespaces: 0 }),
      QuotaExceededError,
    );
  });

  it('thrown error has kind=vectors and correct namespace', () => {
    try {
      checkVectorQuota(95, 10, ns, { maxVectorsPerNs: 100, maxNamespaces: 0 });
      assert.fail('expected throw');
    } catch (e) {
      assert.ok(e instanceof QuotaExceededError);
      assert.equal(e.kind, 'vectors');
      assert.equal(e.namespace, ns);
    }
  });

  it('throws when adding even 1 vector over limit', () => {
    assert.throws(
      () => checkVectorQuota(100, 1, ns, { maxVectorsPerNs: 100, maxNamespaces: 0 }),
      QuotaExceededError,
    );
  });

  it('adding 0 vectors never throws even when already at the limit', () => {
    assert.doesNotThrow(() =>
      checkVectorQuota(100, 0, ns, { maxVectorsPerNs: 100, maxNamespaces: 0 }),
    );
  });

  it('QuotaExceededError is instanceof Error', () => {
    try {
      checkVectorQuota(100, 1, ns, { maxVectorsPerNs: 100, maxNamespaces: 0 });
      assert.fail('expected throw');
    } catch (e) {
      assert.ok(e instanceof Error, 'should be an Error');
      assert.ok(e instanceof QuotaExceededError, 'should be a QuotaExceededError');
    }
  });

  it('QuotaExceededError.name is "QuotaExceededError"', () => {
    try {
      checkVectorQuota(95, 10, ns, { maxVectorsPerNs: 100, maxNamespaces: 0 });
      assert.fail('expected throw');
    } catch (e) {
      assert.ok(e instanceof QuotaExceededError);
      assert.equal(e.name, 'QuotaExceededError');
    }
  });

  it('QuotaExceededError message mentions namespace, current count, and limit', () => {
    try {
      checkVectorQuota(95, 10, 'my-ns', { maxVectorsPerNs: 100, maxNamespaces: 0 });
      assert.fail('expected throw');
    } catch (e) {
      assert.ok(e instanceof QuotaExceededError);
      assert.ok(e.message.includes('my-ns'), 'message should contain namespace');
      assert.ok(e.message.includes('100'), 'message should contain the limit');
    }
  });
});

// ── checkNsQuota ──────────────────────────────────────────────────────────────

describe('checkNsQuota', () => {
  const existing = ['ns-a', 'ns-b', 'ns-c'];

  it('does nothing when limit is 0 (unlimited)', () => {
    assert.doesNotThrow(() =>
      checkNsQuota(existing, 'ns-new', { maxVectorsPerNs: 0, maxNamespaces: 0 }),
    );
  });

  it('allows creating new namespace when under limit', () => {
    assert.doesNotThrow(() =>
      checkNsQuota(existing, 'ns-new', { maxVectorsPerNs: 0, maxNamespaces: 5 }),
    );
  });

  it('always allows opening an existing namespace regardless of limit', () => {
    assert.doesNotThrow(() =>
      checkNsQuota(existing, 'ns-a', { maxVectorsPerNs: 0, maxNamespaces: 1 }),
    );
  });

  it('throws when namespace count is at the limit and a new one is requested', () => {
    assert.throws(
      () => checkNsQuota(existing, 'ns-new', { maxVectorsPerNs: 0, maxNamespaces: 3 }),
      QuotaExceededError,
    );
  });

  it('thrown error has kind=namespaces', () => {
    try {
      checkNsQuota(existing, 'ns-new', { maxVectorsPerNs: 0, maxNamespaces: 3 });
      assert.fail('expected throw');
    } catch (e) {
      assert.ok(e instanceof QuotaExceededError);
      assert.equal(e.kind, 'namespaces');
      assert.equal(e.namespace, 'ns-new');
    }
  });

  it('allows when count is exactly one below the limit', () => {
    assert.doesNotThrow(() =>
      checkNsQuota(existing, 'ns-new', { maxVectorsPerNs: 0, maxNamespaces: 4 }),
    );
  });

  it('throws when count equals the limit', () => {
    assert.throws(
      () => checkNsQuota(existing, 'ns-d', { maxVectorsPerNs: 0, maxNamespaces: 3 }),
      QuotaExceededError,
    );
  });

  it('allows creating the very first namespace (empty existing list)', () => {
    assert.doesNotThrow(() =>
      checkNsQuota([], 'first-ns', { maxVectorsPerNs: 0, maxNamespaces: 1 }),
    );
  });

  it('rejects creating second namespace when limit is 1 and one already exists', () => {
    assert.throws(
      () => checkNsQuota(['only-ns'], 'second-ns', { maxVectorsPerNs: 0, maxNamespaces: 1 }),
      QuotaExceededError,
    );
  });

  it('QuotaExceededError message mentions namespace name and limit', () => {
    try {
      checkNsQuota(['a', 'b'], 'c', { maxVectorsPerNs: 0, maxNamespaces: 2 });
      assert.fail('expected throw');
    } catch (e) {
      assert.ok(e instanceof QuotaExceededError);
      assert.ok(e.message.includes('"c"'), 'message should contain the new namespace');
      assert.ok(e.message.includes('2'), 'message should contain the limit');
    }
  });

  it('is case-sensitive: "NS-A" is distinct from "ns-a"', () => {
    // "NS-A" is not in existing ["ns-a", "ns-b", "ns-c"] → treated as new namespace
    assert.doesNotThrow(() =>
      checkNsQuota(existing, 'NS-A', { maxVectorsPerNs: 0, maxNamespaces: 5 }),
    );
    assert.throws(
      () => checkNsQuota(existing, 'NS-A', { maxVectorsPerNs: 0, maxNamespaces: 3 }),
      QuotaExceededError,
    );
  });
});
