/**
 * Situating-context annotations for Contextual Retrieval (#128).
 *
 * `header` prepends a short deterministic prefix to the string that is
 * embedded. The stored row text stays the raw chunk (plus the existing
 * `[file:][chunk:]` display label in the index handler).
 *
 * `off` embeds the raw chunk only (bit-compatible with pre-1.3.0).
 * `llm` is reserved; without a generator it falls back to `header`.
 */

import type { Chunk, ChunkMode } from './chunker.js';

export type ChunkContextMode = 'off' | 'header' | 'llm';

export function resolveChunkContextMode(
  raw?: string | null,
  env: NodeJS.ProcessEnv = process.env,
): ChunkContextMode {
  const v = (raw ?? env.LOGOSDB_CHUNK_CONTEXT ?? 'header').trim().toLowerCase();
  if (v === 'off' || v === '0' || v === 'false' || v === 'no' || v === 'raw') return 'off';
  if (v === 'llm') return 'llm';
  return 'header';
}

/** Effective embed mode: `llm` without a generator is `header`. */
export function effectiveEmbedMode(mode: ChunkContextMode, hasLlm = false): 'off' | 'header' {
  if (mode === 'off') return 'off';
  if (mode === 'llm' && hasLlm) return 'header'; // v2 generator not wired yet
  return 'header';
}

export interface SituatingHeaderInput {
  filePath: string;
  chunkIndex: number;
  chunkCount: number;
  chunk: Chunk;
  resolvedMode: Exclude<ChunkMode, 'auto'>;
  namespace?: string;
  tags?: Record<string, string | number | boolean | null>;
}

/**
 * Deterministic 1–3 line situating header. Never includes the raw chunk body.
 */
export function buildSituatingHeader(input: SituatingHeaderInput): string {
  const { filePath, chunkIndex, chunkCount, chunk, resolvedMode, namespace, tags } = input;
  const loc =
    chunk.lineStart != null && chunk.lineEnd != null
      ? `lines ${chunk.lineStart}-${chunk.lineEnd}`
      : `offset ${chunk.offset}`;
  const lines: string[] = [
    `File: ${filePath}  chunk ${chunkIndex + 1}/${chunkCount}  ${loc}`,
  ];
  if (chunk.heading) {
    const kind = resolvedMode === 'line' ? 'Symbol' : 'Section';
    lines.push(`${kind}: ${chunk.heading}`);
  }
  const extras: string[] = [];
  if (namespace) extras.push(`namespace=${namespace}`);
  if (tags) {
    for (const [k, v] of Object.entries(tags)) {
      if (v === null) continue;
      extras.push(`${k}=${String(v)}`);
    }
  }
  if (extras.length) lines.push(extras.join('  '));
  return lines.join('\n');
}

/** String passed to the embedder. Stored row text must stay `chunk.text`. */
export function embedTextForChunk(
  mode: ChunkContextMode,
  headerInput: SituatingHeaderInput,
  rawText: string,
): string {
  if (effectiveEmbedMode(mode) === 'off') return rawText;
  const header = buildSituatingHeader(headerInput);
  return `${header}\n\n${rawText}`;
}
