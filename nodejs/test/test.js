/**
 * LogosDB Node.js bindings tests
 */

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const os = require('os');
const {
  DB,
  DIST_IP,
  DIST_COSINE,
  DIST_L2,
  DTYPE_FLOAT32,
  DTYPE_FLOAT16,
  DTYPE_INT8,
  VERSION,
  normalizeL2,
  compact,
} = require('../lib');

// Helper to generate random unit vector
function randomUnitVector(dim, seed = 0) {
  const vec = new Array(dim);
  let sum = 0;

  // Simple pseudo-random for deterministic tests
  let state = seed;
  for (let i = 0; i < dim; i++) {
    state = (state * 9301 + 49297) % 233280;
    const rnd = state / 233280;
    vec[i] = rnd * 2 - 1;  // -1 to 1
    sum += vec[i] * vec[i];
  }

  // Normalize
  const norm = Math.sqrt(sum);
  return vec.map(v => v / norm);
}

// Helper to create temp directory
function createTempDir(prefix = 'logosdb-test-') {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

// Helper to cleanup
function cleanup(dir) {
  try {
    fs.rmSync(dir, { recursive: true, force: true });
  } catch (e) {
    // Ignore cleanup errors
  }
}

describe('LogosDB Node.js', function() {
  this.timeout(10000);  // 10 second timeout for tests

  describe('Constants', () => {
    it('should export distance constants', () => {
      assert.strictEqual(typeof DIST_IP, 'number');
      assert.strictEqual(typeof DIST_COSINE, 'number');
      assert.strictEqual(typeof DIST_L2, 'number');
    });

    it('should export version', () => {
      assert.strictEqual(typeof VERSION, 'string');
      assert.ok(VERSION.match(/^\d+\.\d+\.\d+/));
      assert.strictEqual(VERSION, '1.8.0');
    });
  });

  describe('normalizeL2', () => {
    it('returns a unit vector', () => {
      const out = normalizeL2([3, 4]);
      assert.strictEqual(out.length, 2);
      assert.ok(Math.abs(out[0] - 0.6) < 1e-9);
      assert.ok(Math.abs(out[1] - 0.8) < 1e-9);
    });

    it('rejects empty and zero vectors', () => {
      assert.throws(() => normalizeL2([]), /non-empty/);
      assert.throws(() => normalizeL2([0, 0]), /zero-length/);
    });
  });

  describe('DB Creation', () => {
    let dbPath;

    afterEach(() => {
      if (dbPath) cleanup(dbPath);
    });

    it('should create a new database', () => {
      dbPath = createTempDir();
      const db = new DB(dbPath, { dim: 128 });
      assert.ok(db);
      assert.strictEqual(db.dim(), 128);
      db.close();
    });

    it('should use default options', () => {
      dbPath = createTempDir();
      const db = new DB(dbPath);
      assert.strictEqual(db.dim(), 128);  // default
      db.close();
    });

    it('should support custom options', () => {
      dbPath = createTempDir();
      const db = new DB(dbPath, {
        dim: 256,
        maxElements: 10000,
        efConstruction: 100,
        M: 8,
        efSearch: 30,
        distance: DIST_COSINE
      });
      assert.strictEqual(db.dim(), 256);
      db.close();
    });
  });

  describe('Basic Operations', () => {
    let db, dbPath;

    beforeEach(() => {
      dbPath = createTempDir();
      db = new DB(dbPath, { dim: 64 });
    });

    afterEach(() => {
      if (db) db.close();
      cleanup(dbPath);
    });

    it('should put and return an id', () => {
      const vec = randomUnitVector(64, 1);
      const id = db.put(vec, 'test text', '2025-01-01T00:00:00Z');
      assert.strictEqual(typeof id, 'number');
      assert.ok(id >= 0);
    });

    it('should put without text and timestamp', () => {
      const vec = randomUnitVector(64, 2);
      const id = db.put(vec);
      assert.strictEqual(typeof id, 'number');
    });

    it('should count documents', () => {
      assert.strictEqual(db.count(), 0);
      assert.strictEqual(db.countLive(), 0);

      db.put(randomUnitVector(64, 3));
      assert.strictEqual(db.count(), 1);
      assert.strictEqual(db.countLive(), 1);
    });
  });

  describe('Search', () => {
    let db, dbPath;

    beforeEach(() => {
      dbPath = createTempDir();
      db = new DB(dbPath, { dim: 64, distance: DIST_COSINE });
    });

    afterEach(() => {
      if (db) db.close();
      cleanup(dbPath);
    });

    it('should search and return hits', () => {
      const vec1 = randomUnitVector(64, 10);
      const vec2 = randomUnitVector(64, 11);
      const vec3 = randomUnitVector(64, 12);

      db.put(vec1, 'first');
      db.put(vec2, 'second');
      db.put(vec3, 'third');

      const hits = db.search(vec1, 3);
      assert.ok(Array.isArray(hits));
      assert.ok(hits.length <= 3);

      if (hits.length > 0) {
        assert.ok(hits[0].id >= 0);
        assert.ok(hits[0].score > 0);
        assert.ok(typeof hits[0].text === 'string' || hits[0].text === null);
      }
    });

    it('should find similar vectors', () => {
      // Use same seed to get identical vectors
      const vec = randomUnitVector(64, 20);

      db.put(vec, 'original');

      // Search with same vector
      const hits = db.search(vec, 1);
      assert.strictEqual(hits.length, 1);
      assert.strictEqual(hits[0].text, 'original');
      assert.ok(hits[0].score > 0.99, 'Self-similarity should be ~1.0');
    });

    it('search(vec, topK) unchanged with numeric second arg', () => {
      const vec = randomUnitVector(64, 21);
      db.put(vec, 'n');
      const hits = db.search(vec, 3);
      assert.ok(hits.length >= 1);
    });

    it('search with allow/deny/minScore filter object', () => {
      const q = randomUnitVector(64, 22);
      const near = randomUnitVector(64, 23);
      const far = randomUnitVector(64, 24);
      const idNear = db.put(near, 'near');
      db.put(far, 'far');
      const hits = db.search(q, { k: 5, deny: [idNear] });
      for (const h of hits) {
        assert.notStrictEqual(h.id, idNear);
      }
    });

    it('searchBatch pads short rows with null and NaN', () => {
      const v0 = randomUnitVector(64, 25);
      const v1 = randomUnitVector(64, 26);
      db.put(v0);
      db.put(v1);
      const batch = db.searchBatch([v0, v1], 4, { minScore: 2.0 });
      assert.strictEqual(batch.ids.length, 2);
      assert.strictEqual(batch.scores.length, 2);
      for (let qi = 0; qi < 2; qi++) {
        for (let kj = 0; kj < 4; kj++) {
          if (batch.ids[qi][kj] === null) {
            assert.ok(Number.isNaN(batch.scores[qi][kj]));
          }
        }
      }
    });

    it('getMany matches get for live ids', () => {
      const v0 = randomUnitVector(64, 27);
      const v1 = randomUnitVector(64, 28);
      const id0 = db.put(v0);
      const id1 = db.put(v1);
      const many = db.getMany([id0, id1]);
      const row0 = db.get(id0);
      const row1 = db.get(id1);
      assert.strictEqual(many.length, 2);
      assert.strictEqual(many[0].length, 64);
      for (let d = 0; d < 64; d++) {
        assert.ok(Math.abs(many[0][d] - row0.embedding[d]) < 1e-4);
        assert.ok(Math.abs(many[1][d] - row1.embedding[d]) < 1e-4);
      }
      assert.ok(row0.embedding instanceof Float32Array);
    });
  });

  describe('Timestamp Range Search', () => {
    let db, dbPath;

    beforeEach(() => {
      dbPath = createTempDir();
      db = new DB(dbPath, { dim: 32, distance: DIST_COSINE });
    });

    afterEach(() => {
      if (db) db.close();
      cleanup(dbPath);
    });

    it('should search with timestamp filter', () => {
      const vec = randomUnitVector(32, 30);

      db.put(vec, 'early', '2025-01-01T00:00:00Z');
      db.put(vec, 'mid', '2025-01-15T00:00:00Z');
      db.put(vec, 'late', '2025-02-01T00:00:00Z');

      const hits = db.searchTsRange(vec, {
        topK: 10,
        tsFrom: '2025-01-01T00:00:00Z',
        tsTo: '2025-01-31T23:59:59Z'
      });

      assert.ok(hits.length >= 2);
      // Should find early and mid, but not late
      const texts = hits.map(h => h.text);
      assert.ok(texts.includes('early'));
      assert.ok(texts.includes('mid'));
    });
  });

  describe('Get by id', () => {
    let db, dbPath;

    beforeEach(() => {
      dbPath = createTempDir();
      db = new DB(dbPath, { dim: 32, distance: DIST_COSINE });
    });

    afterEach(() => {
      if (db) db.close();
      cleanup(dbPath);
    });

    it('should fetch embedding and metadata', () => {
      const vec = randomUnitVector(32, 70);
      const id = db.put(vec, 'hello', '2025-03-01T00:00:00Z');
      const row = db.get(id);
      assert.strictEqual(row.id, id);
      assert.strictEqual(row.text, 'hello');
      assert.strictEqual(row.timestamp, '2025-03-01T00:00:00Z');
      assert.strictEqual(row.embedding.length, 32);
      let dot = 0;
      for (let i = 0; i < 32; i++) dot += row.embedding[i] * vec[i];
      assert.ok(dot > 0.99);
    });

    it('should throw for deleted and out-of-range ids', () => {
      const id = db.put(randomUnitVector(32, 71), 'x');
      db.delete(id);
      assert.throws(() => db.get(id), /deleted|out of range/i);
      assert.throws(() => db.get(999), /out of range/i);
    });
  });

  describe('Update and Delete', () => {
    let db, dbPath;

    beforeEach(() => {
      dbPath = createTempDir();
      db = new DB(dbPath, { dim: 32, distance: DIST_COSINE });
    });

    afterEach(() => {
      if (db) db.close();
      cleanup(dbPath);
    });

    it('should update a row', () => {
      const vec = randomUnitVector(32, 40);
      const id = db.put(vec, 'original');

      const newVec = randomUnitVector(32, 41);
      const newId = db.update(id, newVec, 'updated');

      assert.strictEqual(typeof newId, 'number');
      assert.ok(newId > id);
      assert.strictEqual(db.countLive(), 1);  // Old marked deleted, new added
      assert.strictEqual(db.count(), 2);  // Total includes both
    });

    it('should delete a row', () => {
      const vec = randomUnitVector(32, 50);
      const id = db.put(vec, 'to delete');

      assert.strictEqual(db.countLive(), 1);

      db.delete(id);

      assert.strictEqual(db.countLive(), 0);
      assert.strictEqual(db.count(), 1);  // Still counted but marked deleted
    });
  });

  describe('Error Handling', () => {
    let dbPath;

    afterEach(() => {
      cleanup(dbPath);
    });

    it('should throw on dimension mismatch', () => {
      dbPath = createTempDir();
      const db = new DB(dbPath, { dim: 64 });

      assert.throws(() => {
        db.put(randomUnitVector(32));  // Wrong dimension
      }, /dimension/);

      db.close();
    });

    it('should throw on closed database access', () => {
      dbPath = createTempDir();
      const db = new DB(dbPath, { dim: 64 });
      db.close();

      assert.throws(() => {
        db.put(randomUnitVector(64));
      }, /closed/);
    });
  });

  describe('storage dtype', () => {
    let dbPath;

    afterEach(() => {
      cleanup(dbPath);
    });

    it('float16 create, search, reopen, and compact keep dtype', () => {
      dbPath = createTempDir();
      const dim = 16;
      const db = new DB(dbPath, { dim, dtype: DTYPE_FLOAT16 });
      const v = randomUnitVector(dim, 77);
      db.put(v, 'fp16');
      const hits = db.search(v, 1);
      assert.strictEqual(hits[0].id, 0);
      assert.strictEqual(db.stats().storage_dtype, DTYPE_FLOAT16);
      db.close();

      const db2 = new DB(dbPath, { dim, dtype: DTYPE_FLOAT16 });
      assert.strictEqual(db2.countLive(), 1);
      db2.close();

      const dst = createTempDir('logosdb-fp16-compact-');
      compact(dbPath, dst);
      const db3 = new DB(dst, { dim, dtype: DTYPE_FLOAT16 });
      assert.strictEqual(db3.stats().storage_dtype, DTYPE_FLOAT16);
      db3.close();
      cleanup(dst);
    });

    it('reopen with wrong dtype throws', () => {
      dbPath = createTempDir();
      const dim = 8;
      const db = new DB(dbPath, { dim, dtype: DTYPE_FLOAT16 });
      db.put(randomUnitVector(dim, 1));
      db.close();

      assert.throws(
        () => new DB(dbPath, { dim, dtype: DTYPE_FLOAT32 }),
        /dtype mismatch/i
      );
    });

    it('int8 create and search', () => {
      dbPath = createTempDir();
      const dim = 12;
      const db = new DB(dbPath, { dim, dtype: DTYPE_INT8, distance: DIST_COSINE });
      const v = randomUnitVector(dim, 55);
      db.put(v, 'q8');
      const hits = db.search(v, 1);
      assert.strictEqual(hits.length, 1);
      assert.strictEqual(db.stats().storage_dtype, DTYPE_INT8);
      db.close();
    });

    it('exports dtype constants', () => {
      assert.strictEqual(DTYPE_FLOAT32, 0);
      assert.strictEqual(DTYPE_FLOAT16, 1);
      assert.strictEqual(DTYPE_INT8, 2);
    });
  });

  describe('sync, stats, compact', () => {
    let dbPath;

    afterEach(() => {
      cleanup(dbPath);
    });

    it('put + sync + stats reflects operations', () => {
      dbPath = createTempDir();
      const db = new DB(dbPath, { dim: 16, distance: DIST_COSINE });
      const vec = randomUnitVector(16, 41);
      db.put(vec, 'one');
      db.sync();
      const s = db.stats();
      assert.strictEqual(s.rows_live, 1);
      assert.ok(s.put_success >= 1);
      assert.ok(s.sync_calls >= 1);
      db.close();
    });

    it('compact drops tombstones', () => {
      dbPath = createTempDir();
      const src = dbPath;
      const dst = path.join(os.tmpdir(), `logosdb-compact-${Date.now()}`);
      const db = new DB(src, { dim: 8, distance: DIST_COSINE });
      const v0 = randomUnitVector(8, 1);
      const id = db.put(v0, 'live');
      db.delete(id);
      db.put(randomUnitVector(8, 2), 'still');
      db.sync();
      db.close();

      compact(src, dst);

      const db2 = new DB(dst, { dim: 8, distance: DIST_COSINE });
      assert.strictEqual(db2.countLive(), 1);
      const st = db2.stats();
      assert.strictEqual(st.tombstones, 0);
      db2.close();
      cleanup(dst);
    });

    it('sync after close throws', () => {
      dbPath = createTempDir();
      const db = new DB(dbPath, { dim: 8 });
      db.close();
      assert.throws(() => db.sync(), /closed/);
    });

    it('compact into non-empty dst throws', () => {
      dbPath = createTempDir();
      const dst = createTempDir('logosdb-compact-dst-');
      fs.writeFileSync(path.join(dst, 'marker.txt'), 'x');
      const db = new DB(dbPath, { dim: 8, distance: DIST_COSINE });
      db.put(randomUnitVector(8, 3));
      db.close();
      assert.throws(() => compact(dbPath, dst), /not empty/i);
      cleanup(dst);
    });
  });

  describe('liveIds, rawVectors, exportNdjson', () => {
    let dbPath;

    afterEach(() => {
      cleanup(dbPath);
    });

    it('liveIds omits deleted ids', () => {
      dbPath = createTempDir();
      const db = new DB(dbPath, { dim: 8, distance: DIST_COSINE });
      db.put(randomUnitVector(8, 1), 'a');
      db.put(randomUnitVector(8, 2), 'b');
      db.put(randomUnitVector(8, 3), 'c');
      db.delete(1);
      const ids = db.liveIds();
      assert.deepStrictEqual(ids, [0, 2]);
      assert.strictEqual(ids.length, db.countLive());
      db.close();
    });

    it('rawVectors decodes float16 storage', () => {
      dbPath = createTempDir();
      const v0 = randomUnitVector(16, 77);
      const db = new DB(dbPath, { dim: 16, dtype: DTYPE_FLOAT16, distance: DIST_COSINE });
      db.put(v0, 'f16');
      const raw = db.rawVectors();
      assert.strictEqual(raw.length, 16);
      for (let i = 0; i < 16; i++) {
        assert.ok(Math.abs(raw[i] - v0[i]) < 0.05);
      }
      db.close();
    });

    it('exportNdjson writes live rows', () => {
      dbPath = createTempDir();
      const db = new DB(dbPath, { dim: 4, distance: DIST_COSINE });
      const v = randomUnitVector(4, 9);
      db.put(v, 'row0');
      db.put(randomUnitVector(4, 10), 'row1');
      db.delete(1);
      const out = path.join(dbPath, 'out.ndjson');
      db.exportNdjson(out);
      const lines = fs.readFileSync(out, 'utf8').trim().split('\n');
      assert.strictEqual(lines.length, 1);
      const row = JSON.parse(lines[0]);
      assert.strictEqual(row.id, 0);
      assert.strictEqual(row.text, 'row0');
      db.close();
    });
  });

  describe('capacity and async (1.8.0)', () => {
    let dbPath;

    afterEach(() => {
      cleanup(dbPath);
    });

    it('grows past maxElements and reopens', () => {
      dbPath = createTempDir();
      const db = new DB(dbPath, { dim: 8, maxElements: 16, efSearch: 64 });
      for (let i = 0; i < 20; i++) {
        db.put(randomUnitVector(8, 500 + i));
      }
      assert.strictEqual(db.indexCapacity(), 32);
      db.close();
      const db2 = new DB(dbPath, { dim: 8, maxElements: 16, efSearch: 64 });
      assert.strictEqual(db2.indexCapacity(), 32);
      db2.close();
    });

    it('rejects insert over maxCapacity', () => {
      dbPath = createTempDir();
      const db = new DB(dbPath, { dim: 8, maxElements: 16, maxCapacity: 20 });
      for (let i = 0; i < 20; i++) {
        db.put(randomUnitVector(8, 600 + i));
      }
      assert.throws(() => db.put(randomUnitVector(8, 699)), /hnsw capacity cap exceeded/);
      assert.strictEqual(db.count(), 20);
      db.close();
    });

    it('accepts Float32Array in and out', () => {
      dbPath = createTempDir();
      const db = new DB(dbPath, { dim: 4, maxElements: 64 });
      const vec = Float32Array.from(randomUnitVector(4, 1));
      const id = db.put(vec, 'f32');
      const row = db.get(id);
      assert.ok(row.embedding instanceof Float32Array);
      assert.strictEqual(row.embedding.length, 4);
      db.close();
    });

    it('searchAsync matches sync search', async () => {
      dbPath = createTempDir();
      const db = new DB(dbPath, { dim: 8, maxElements: 64 });
      const q = randomUnitVector(8, 7);
      db.put(randomUnitVector(8, 8), 'a');
      db.put(q, 'b');
      const sync = db.search(q, 2);
      const asyncHits = await db.searchAsync(q, 2);
      assert.strictEqual(sync.length, asyncHits.length);
      assert.strictEqual(sync[0].id, asyncHits[0].id);
      db.close();
    });

    it('putBatchAsync matches putBatch', async () => {
      dbPath = createTempDir();
      const db = new DB(dbPath, { dim: 4, maxElements: 64 });
      const flat = [];
      for (let i = 0; i < 3; i++) flat.push(...randomUnitVector(4, 20 + i));
      const syncIds = db.putBatch(flat, 3);
      const db2Path = createTempDir();
      const db2 = new DB(db2Path, { dim: 4, maxElements: 64 });
      const asyncIds = await db2.putBatchAsync(Float32Array.from(flat), 3);
      assert.deepStrictEqual(syncIds, asyncIds);
      db.close();
      db2.close();
      cleanup(db2Path);
    });
  });

  describe('Persistence', () => {
    let dbPath;

    afterEach(() => {
      cleanup(dbPath);
    });

    it('should persist data across reopen', () => {
      dbPath = createTempDir();

      // First session: insert
      let db1 = new DB(dbPath, { dim: 32, distance: DIST_COSINE });
      const vec = randomUnitVector(32, 60);
      const id = db1.put(vec, 'persistent');
      db1.close();

      // Second session: verify
      let db2 = new DB(dbPath, { dim: 32, distance: DIST_COSINE });
      assert.strictEqual(db2.countLive(), 1);

      const hits = db2.search(vec, 1);
      assert.strictEqual(hits.length, 1);
      assert.strictEqual(hits[0].text, 'persistent');
      db2.close();
    });
  });
});

// Run tests if executed directly
if (require.main === module) {
  const Mocha = require('mocha');
  const mocha = new Mocha();
  mocha.suite.emit('pre-require', global, 'test.js', mocha);

  // Simple test runner for standalone execution
  console.log('Running LogosDB Node.js tests...\n');
}
