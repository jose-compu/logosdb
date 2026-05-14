/**
 * hybrid.ts — Lexical scoring and score-fusion for logosdb hybrid search (#85).
 *
 * Two fusion strategies:
 *   "rrf"      — Reciprocal Rank Fusion  (rank-based, distribution-free)
 *   "weighted" — Linear interpolation    combined = (1-w)*ann + w*lexical
 *
 * Lexical scoring uses a BM25-lite implementation (BM25 without IDF, i.e. purely
 * based on term frequency within each document relative to average document length).
 * This runs entirely in the MCP process over the ANN result set — no core changes.
 */

import type { SearchHit } from 'logosdb';

// ── Types ─────────────────────────────────────────────────────────────────────

export type FusionStrategy = 'rrf' | 'weighted';

export interface HybridOptions {
  /** Fusion strategy (default: "rrf") */
  fusion?: FusionStrategy;
  /**
   * Weight of the lexical score in "weighted" mode (0 = pure ANN, 1 = pure lexical).
   * Default: 0.5
   */
  lexical_weight?: number;
  /**
   * RRF rank constant k (default: 60).
   * Score = 1/(k + rank). Higher k dampens rank differences.
   */
  rrf_k?: number;
}

export interface HybridHit extends SearchHit {
  ann_score: number;
  lexical_score: number;
  hybrid_score: number;
}

// ── Tokenizer ─────────────────────────────────────────────────────────────────

const STOP_WORDS = new Set([
  'a','an','the','and','or','but','in','on','at','to','for','of','with',
  'is','are','was','were','be','been','being','have','has','had','do','does',
  'did','will','would','could','should','may','might','shall','can','this',
  'that','these','those','it','its','i','you','he','she','we','they','my',
  'your','his','her','our','their',
]);

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((t) => t.length > 1 && !STOP_WORDS.has(t));
}

// ── BM25-lite ─────────────────────────────────────────────────────────────────
// No IDF (single-namespace corpus stats not available at query time).
// Score = sum over query terms of: (tf * (k1+1)) / (tf + k1*(1 - b + b*dl/avgdl))
// Normalized to [0, 1] over the result set.

const BM25_K1 = 1.5;
const BM25_B = 0.75;

function bm25Score(
  queryTokens: string[],
  docTokens: string[],
  avgDocLen: number,
): number {
  const dl = docTokens.length;
  const tf = new Map<string, number>();
  for (const t of docTokens) tf.set(t, (tf.get(t) ?? 0) + 1);

  let score = 0;
  for (const qt of queryTokens) {
    const f = tf.get(qt) ?? 0;
    if (f === 0) continue;
    score += (f * (BM25_K1 + 1)) / (f + BM25_K1 * (1 - BM25_B + BM25_B * (dl / avgDocLen)));
  }
  return score;
}

// ── Fusion ─────────────────────────────────────────────────────────────────────

/**
 * Re-rank `hits` (already ordered by ANN score desc) using hybrid scoring.
 * Returns a new array sorted by hybrid_score desc, trimmed to `topK`.
 */
export function hybridRerank(
  hits: SearchHit[],
  query: string,
  topK: number,
  opts: HybridOptions = {},
): HybridHit[] {
  if (hits.length === 0) return [];

  const fusion: FusionStrategy = opts.fusion ?? 'rrf';
  const lexWeight = Math.min(1, Math.max(0, opts.lexical_weight ?? 0.5));
  const rrfK = opts.rrf_k ?? 60;

  const queryTokens = tokenize(query);

  // If all query tokens were stopwords / too short, lexical scoring is meaningless.
  // Return hits in ANN order, trimmed to topK, with zero lexical scores.
  if (queryTokens.length === 0) {
    return hits.slice(0, topK).map((h, i) => ({
      ...h,
      ann_score: Math.round(h.score * 10000) / 10000,
      lexical_score: 0,
      hybrid_score: Math.round(h.score * 10000) / 10000,
    }));
  }

  // Compute lexical scores
  const docTokensList = hits.map((h) => tokenize(h.text ?? ''));
  const avgDocLen =
    docTokensList.reduce((s, d) => s + d.length, 0) / (docTokensList.length || 1);

  const rawLex = docTokensList.map((dt) => bm25Score(queryTokens, dt, avgDocLen));

  // Normalize lexical to [0,1]; if all scores are zero (no term overlap), use 1e-9 floor.
  const maxLex = Math.max(...rawLex, 1e-9);
  const lexScores = rawLex.map((s) => s / maxLex);

  // ANN is already in [0,1] (cosine similarity from logosdb)
  // Build rank maps (0-based, lower = better)
  const annRanks = hits.map((_, i) => i); // already sorted by ANN desc

  const lexOrder = lexScores
    .map((s, i) => [i, s] as [number, number])
    .sort((a, b) => b[1] - a[1]);
  const lexRanks = new Array<number>(hits.length);
  lexOrder.forEach(([origIdx], rank) => { lexRanks[origIdx] = rank; });

  const hybridHits: HybridHit[] = hits.map((h, i) => {
    const annScore = h.score;
    const lexScore = lexScores[i]!;
    let hybridScore: number;

    if (fusion === 'rrf') {
      hybridScore =
        1 / (rrfK + annRanks[i]! + 1) + 1 / (rrfK + lexRanks[i]! + 1);
    } else {
      hybridScore = (1 - lexWeight) * annScore + lexWeight * lexScore;
    }

    return {
      ...h,
      ann_score: Math.round(annScore * 10000) / 10000,
      lexical_score: Math.round(lexScore * 10000) / 10000,
      hybrid_score: Math.round(hybridScore * 10000) / 10000,
    };
  });

  hybridHits.sort((a, b) => b.hybrid_score - a.hybrid_score);
  return hybridHits.slice(0, topK);
}
