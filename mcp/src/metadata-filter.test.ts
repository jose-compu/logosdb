import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  serializeTags,
  stripTagsSuffix,
  parseTags,
  matchesFilter,
  validateTags,
  validateFilter,
} from './metadata-filter';

// ── serializeTags / stripTagsSuffix / parseTags ────────────────────────────────

test('serializeTags appends JSON suffix', () => {
  const result = serializeTags('hello world', { lang: 'ts', priority: 2 });
  assert.ok(result.startsWith('hello world\n[tags:'));
  assert.ok(result.endsWith(']'));
});

test('parseTags round-trips through serializeTags', () => {
  const tags = { lang: 'typescript', priority: 3, reviewed: true };
  const stored = serializeTags('some text', tags);
  const parsed = parseTags(stored);
  assert.deepEqual(parsed, tags);
});

test('parseTags returns null for text without suffix', () => {
  assert.equal(parseTags('plain text without tags'), null);
  assert.equal(parseTags(''), null);
  assert.equal(parseTags(null), null);
  assert.equal(parseTags(undefined), null);
});

test('parseTags returns null for malformed JSON in suffix', () => {
  const bad = 'some text\n[tags:{not valid json}]';
  assert.equal(parseTags(bad), null);
});

test('stripTagsSuffix removes suffix', () => {
  const original = 'hello world';
  const stored = serializeTags(original, { x: 1 });
  assert.equal(stripTagsSuffix(stored), original);
});

test('stripTagsSuffix is no-op on text without suffix', () => {
  assert.equal(stripTagsSuffix('plain text'), 'plain text');
});

test('serializeTags is idempotent — replaces existing suffix', () => {
  const step1 = serializeTags('text', { a: 1 });
  const step2 = serializeTags(step1, { b: 2 });
  const parsed = parseTags(step2);
  // Only the second tags object should be present
  assert.deepEqual(parsed, { b: 2 });
  assert.ok(!step2.includes('"a":'));
});

test('parseTags handles null tag value', () => {
  const stored = serializeTags('text', { key: null });
  const parsed = parseTags(stored);
  assert.equal(parsed!.key, null);
});

// ── validateTags ──────────────────────────────────────────────────────────────

test('validateTags accepts valid tags', () => {
  const tags = validateTags({ lang: 'ts', priority: 2, active: true, nullable: null });
  assert.equal(tags.lang, 'ts');
  assert.equal(tags.priority, 2);
  assert.equal(tags.active, true);
  assert.equal(tags.nullable, null);
});

test('validateTags rejects non-object', () => {
  assert.throws(() => validateTags('string'), /plain object/);
  assert.throws(() => validateTags([1, 2]), /plain object/);
  assert.throws(() => validateTags(null), /plain object/);
});

test('validateTags rejects value of wrong type', () => {
  assert.throws(() => validateTags({ key: { nested: true } }), /string, number, boolean/);
  assert.throws(() => validateTags({ key: ['a'] }), /string, number, boolean/);
});

test('validateTags rejects too many keys', () => {
  const tooMany = Object.fromEntries(Array.from({ length: 33 }, (_, i) => [`k${i}`, i]));
  assert.throws(() => validateTags(tooMany), /exceed/);
});

test('validateTags rejects long string value', () => {
  assert.throws(() => validateTags({ key: 'x'.repeat(257) }), /exceeds/);
});

// ── matchesFilter — equality ───────────────────────────────────────────────────

test('matchesFilter: plain equality match', () => {
  const tags = { lang: 'typescript', priority: 2 };
  assert.ok(matchesFilter(tags, { lang: 'typescript' }));
  assert.ok(!matchesFilter(tags, { lang: 'python' }));
});

test('matchesFilter: multiple keys ANDed', () => {
  const tags = { lang: 'typescript', priority: 2 };
  assert.ok(matchesFilter(tags, { lang: 'typescript', priority: 2 }));
  assert.ok(!matchesFilter(tags, { lang: 'typescript', priority: 3 }));
});

test('matchesFilter: null tags against non-empty filter → false', () => {
  assert.ok(!matchesFilter(null, { lang: 'typescript' }));
});

test('matchesFilter: empty filter always matches', () => {
  assert.ok(matchesFilter(null, {}));
  assert.ok(matchesFilter({ x: 1 }, {}));
});

// ── matchesFilter — $eq / $ne ─────────────────────────────────────────────────

test('matchesFilter: $eq', () => {
  assert.ok(matchesFilter({ x: 5 }, { x: { $eq: 5 } }));
  assert.ok(!matchesFilter({ x: 5 }, { x: { $eq: 6 } }));
});

test('matchesFilter: $ne', () => {
  assert.ok(matchesFilter({ x: 5 }, { x: { $ne: 6 } }));
  assert.ok(!matchesFilter({ x: 5 }, { x: { $ne: 5 } }));
});

test('matchesFilter: $ne on missing key → false', () => {
  // MongoDB: missing key does not satisfy $ne
  assert.ok(!matchesFilter({ other: 1 }, { x: { $ne: 5 } }));
});

// ── matchesFilter — $in / $nin ────────────────────────────────────────────────

test('matchesFilter: $in', () => {
  assert.ok(matchesFilter({ lang: 'ts' }, { lang: { $in: ['ts', 'js'] } }));
  assert.ok(!matchesFilter({ lang: 'py' }, { lang: { $in: ['ts', 'js'] } }));
});

test('matchesFilter: $nin', () => {
  assert.ok(matchesFilter({ lang: 'py' }, { lang: { $nin: ['ts', 'js'] } }));
  assert.ok(!matchesFilter({ lang: 'ts' }, { lang: { $nin: ['ts', 'js'] } }));
});

test('matchesFilter: $in on missing key → false', () => {
  assert.ok(!matchesFilter(null, { lang: { $in: ['ts'] } }));
});

test('matchesFilter: $nin on missing key → true (undefined not in list)', () => {
  assert.ok(matchesFilter(null, { lang: { $nin: ['ts', 'js'] } }));
});

test('matchesFilter: $nin with empty array always true', () => {
  assert.ok(matchesFilter({ lang: 'ts' }, { lang: { $nin: [] } }));
  assert.ok(matchesFilter(null, { lang: { $nin: [] } }));
});

// ── matchesFilter — range ─────────────────────────────────────────────────────

test('matchesFilter: $gte / $lte numeric range', () => {
  assert.ok(matchesFilter({ n: 5 }, { n: { $gte: 1, $lte: 10 } }));
  assert.ok(!matchesFilter({ n: 11 }, { n: { $gte: 1, $lte: 10 } }));
  assert.ok(!matchesFilter({ n: 0 }, { n: { $gte: 1, $lte: 10 } }));
});

test('matchesFilter: $gt / $lt exclusive bounds', () => {
  assert.ok(matchesFilter({ n: 5 }, { n: { $gt: 4, $lt: 6 } }));
  assert.ok(!matchesFilter({ n: 4 }, { n: { $gt: 4, $lt: 6 } }));
  assert.ok(!matchesFilter({ n: 6 }, { n: { $gt: 4, $lt: 6 } }));
});

test('matchesFilter: range on missing key → false', () => {
  assert.ok(!matchesFilter(null, { n: { $gte: 1 } }));
});

test('matchesFilter: string lexicographic range', () => {
  assert.ok(matchesFilter({ name: 'beta' }, { name: { $gte: 'alpha', $lte: 'gamma' } }));
  assert.ok(!matchesFilter({ name: 'zeta' }, { name: { $gte: 'alpha', $lte: 'gamma' } }));
});

// ── matchesFilter — $exists ───────────────────────────────────────────────────

test('matchesFilter: $exists: true', () => {
  assert.ok(matchesFilter({ lang: 'ts' }, { lang: { $exists: true } }));
  assert.ok(!matchesFilter({ other: 1 }, { lang: { $exists: true } }));
  assert.ok(!matchesFilter(null, { lang: { $exists: true } }));
});

test('matchesFilter: $exists: false', () => {
  assert.ok(matchesFilter({ other: 1 }, { lang: { $exists: false } }));
  assert.ok(!matchesFilter({ lang: 'ts' }, { lang: { $exists: false } }));
  assert.ok(matchesFilter(null, { lang: { $exists: false } }));
});

// ── validateFilter ────────────────────────────────────────────────────────────

test('validateFilter accepts plain object', () => {
  const f = validateFilter({ lang: 'ts', priority: { $gte: 2 } });
  assert.ok(typeof f === 'object');
});

test('validateFilter rejects non-object', () => {
  assert.throws(() => validateFilter('string'), /plain object/);
  assert.throws(() => validateFilter(null), /plain object/);
  assert.throws(() => validateFilter([]), /plain object/);
});
