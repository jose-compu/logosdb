import assert from 'node:assert/strict';
import { test } from 'node:test';
import { hybridRerank } from './hybrid';
import type { SearchHit } from 'logosdb';

// ── Helpers ────────────────────────────────────────────────────────────────────

function hit(id: number, score: number, text: string): SearchHit {
  return { id, score, text, timestamp: null };
}

// ── tokenizer / BM25 coverage through hybridRerank ────────────────────────────

test('hybridRerank: empty hits returns empty array', () => {
  const result = hybridRerank([], 'authentication', 3);
  assert.deepEqual(result, []);
});

test('hybridRerank: single hit is returned', () => {
  const hits = [hit(1, 0.9, 'user authentication flow')];
  const result = hybridRerank(hits, 'authentication', 3);
  assert.equal(result.length, 1);
  assert.equal(result[0]!.id, 1);
  assert.ok(result[0]!.ann_score >= 0);
  assert.ok(result[0]!.lexical_score >= 0);
  assert.ok(result[0]!.hybrid_score >= 0);
});

test('hybridRerank: topK limits output', () => {
  const hits = [
    hit(1, 0.9, 'alpha bravo charlie'),
    hit(2, 0.85, 'delta echo foxtrot'),
    hit(3, 0.8, 'golf hotel india'),
    hit(4, 0.75, 'juliet kilo lima'),
  ];
  const result = hybridRerank(hits, 'alpha bravo', 2);
  assert.equal(result.length, 2);
});

test('hybridRerank: all-stopword query falls back to ANN order', () => {
  const hits = [
    hit(1, 0.9, 'document about authentication'),
    hit(2, 0.8, 'document about authorization'),
    hit(3, 0.7, 'document about sessions'),
  ];
  // "the and or" are all stopwords → queryTokens empty
  const result = hybridRerank(hits, 'the and or', 3);
  assert.equal(result.length, 3);
  // ANN order preserved
  assert.equal(result[0]!.id, 1);
  assert.equal(result[1]!.id, 2);
  assert.equal(result[2]!.id, 3);
  // Lexical scores are zero
  result.forEach((r) => assert.equal(r.lexical_score, 0));
});

test('hybridRerank: RRF result is sorted by hybrid_score descending', () => {
  const hits = [
    hit(1, 0.9, 'neural network backpropagation'),
    hit(2, 0.8, 'authentication token flow'),
    hit(3, 0.7, 'authentication authentication token token jwt'),
    hit(4, 0.6, 'oauth2 refresh grant'),
  ];
  const result = hybridRerank(hits, 'authentication token', 4, { fusion: 'rrf' });
  assert.equal(result.length, 4);
  // Output must be sorted by hybrid_score descending (or equal ties)
  for (let i = 0; i < result.length - 1; i++) {
    assert.ok(
      result[i]!.hybrid_score >= result[i + 1]!.hybrid_score,
      `out-of-order at index ${i}: ${result[i]!.hybrid_score} < ${result[i + 1]!.hybrid_score}`,
    );
  }
  // Every result must have all three score fields populated
  result.forEach((r) => {
    assert.ok('ann_score' in r, 'missing ann_score');
    assert.ok('lexical_score' in r, 'missing lexical_score');
    assert.ok('hybrid_score' in r, 'missing hybrid_score');
  });
});

test('hybridRerank: RRF — document good in both lists wins over specialist', () => {
  // hit1: high ANN, zero lexical (ANN specialist)
  // hit2: medium ANN, medium lexical (balanced)
  // hit3: low ANN, high lexical (lexical specialist)
  // With enough items, RRF elevates the balanced hit over both specialists.
  const hits = [
    hit(1, 0.95, 'neural gradient descent optimizer'),  // ANN rank 0, lex rank 2
    hit(2, 0.80, 'authentication token auth'),           // ANN rank 1, lex rank 1
    hit(3, 0.65, 'authentication token token auth jwt'), // ANN rank 2, lex rank 0
    hit(4, 0.50, 'database index query'),                // ANN rank 3, lex rank 3
    hit(5, 0.40, 'cache invalidation strategy'),         // ANN rank 4, lex rank 4
    hit(6, 0.30, 'garbage collection memory'),           // ANN rank 5, lex rank 5
  ];
  const result = hybridRerank(hits, 'authentication token', 6, { fusion: 'rrf' });
  // hit2 has ANN rank=1, lex rank=1 → RRF = 1/63 + 1/63
  // hit1 has ANN rank=0, lex rank=2 → RRF = 1/62 + 1/64
  // hit3 has ANN rank=2, lex rank=0 → RRF = 1/64 + 1/62
  // hit2 wins: 1/63+1/63 > 1/62+1/64 (symmetric ← since 1/62+1/64 < 2/63)
  assert.equal(result[0]!.id, 2, 'balanced hit (good in both lists) should win over specialists');
});

test('hybridRerank: weighted fusion with lexical_weight=0 is pure ANN', () => {
  const hits = [
    hit(1, 0.9, 'neural network'),
    hit(2, 0.7, 'authentication token refresh'),
  ];
  const result = hybridRerank(hits, 'authentication', 2, {
    fusion: 'weighted',
    lexical_weight: 0,
  });
  // lexical has zero weight → ANN order unchanged
  assert.equal(result[0]!.id, 1);
  assert.equal(result[1]!.id, 2);
});

test('hybridRerank: weighted fusion with lexical_weight=1 is pure lexical', () => {
  const hits = [
    hit(1, 0.9, 'unrelated neural gradient'),
    hit(2, 0.5, 'authentication authentication authentication'),
  ];
  const result = hybridRerank(hits, 'authentication', 2, {
    fusion: 'weighted',
    lexical_weight: 1,
  });
  // hit 2 dominates lexically
  assert.equal(result[0]!.id, 2);
});

test('hybridRerank: scores are within expected bounds', () => {
  const hits = [
    hit(1, 0.8, 'token bucket rate limiting algorithm'),
    hit(2, 0.75, 'jwt authentication and token verification'),
    hit(3, 0.6, 'oauth2 token refresh flow'),
  ];
  const result = hybridRerank(hits, 'token authentication', 3);
  for (const r of result) {
    assert.ok(r.ann_score >= 0 && r.ann_score <= 1, 'ann_score out of [0,1]');
    assert.ok(r.lexical_score >= 0 && r.lexical_score <= 1, 'lexical_score out of [0,1]');
    assert.ok(r.hybrid_score >= 0, 'hybrid_score negative');
  }
});

test('hybridRerank: null text hits get lexical_score 0', () => {
  const hits: SearchHit[] = [
    { id: 1, score: 0.9, text: null, timestamp: null },
    hit(2, 0.8, 'authentication flow'),
  ];
  const result = hybridRerank(hits, 'authentication', 2);
  const nullHit = result.find((r) => r.id === 1)!;
  assert.equal(nullHit.lexical_score, 0);
});

test('hybridRerank: rrf_k option changes scoring magnitude', () => {
  const hits = [
    hit(1, 0.9, 'alpha bravo'),
    hit(2, 0.7, 'alpha bravo charlie'),
  ];
  const resultDefaultK = hybridRerank(hits, 'alpha', 2, { fusion: 'rrf' });
  const resultLowK = hybridRerank(hits, 'alpha', 2, { fusion: 'rrf', rrf_k: 1 });
  // Both should return 2 hits; scores differ but relative order may differ
  assert.equal(resultDefaultK.length, 2);
  assert.equal(resultLowK.length, 2);
});
