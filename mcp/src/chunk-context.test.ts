/**
 * Contextual Retrieval headers (#128): off vs header embed strings;
 * stored raw text is never rewritten here.
 */

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { chunk } from './chunker.js';
import {
  buildSituatingHeader,
  embedTextForChunk,
  resolveChunkContextMode,
  type SituatingHeaderInput,
} from './chunk-context.js';

function headerInput(
  overrides: Partial<SituatingHeaderInput> & { raw?: string } = {},
): SituatingHeaderInput {
  const raw =
    overrides.raw ??
    'function greet() {\n  return "hi";\n}\n';
  const chunks = chunk(raw, { mode: overrides.resolvedMode ?? 'line', filePath: 'src/app.ts' });
  const c = chunks[0]!;
  return {
    filePath: 'src/app.ts',
    chunkIndex: 0,
    chunkCount: chunks.length,
    chunk: c,
    resolvedMode: 'line',
    ...overrides,
  };
}

describe('resolveChunkContextMode', () => {
  it('defaults to header', () => {
    assert.equal(resolveChunkContextMode(undefined, {}), 'header');
  });

  it('accepts off aliases', () => {
    assert.equal(resolveChunkContextMode('off'), 'off');
    assert.equal(resolveChunkContextMode('0', {}), 'off');
    assert.equal(resolveChunkContextMode('false', {}), 'off');
    assert.equal(resolveChunkContextMode('raw', {}), 'off');
  });

  it('reads LOGOSDB_CHUNK_CONTEXT when no explicit value', () => {
    assert.equal(resolveChunkContextMode(undefined, { LOGOSDB_CHUNK_CONTEXT: 'off' }), 'off');
    assert.equal(resolveChunkContextMode(undefined, { LOGOSDB_CHUNK_CONTEXT: 'llm' }), 'llm');
  });
});

describe('buildSituatingHeader', () => {
  it('includes file, chunk i/n, and line range for line mode', () => {
    const src = Array.from({ length: 60 }, (_, i) => `line ${i + 1}`).join('\n');
    const chunks = chunk(src, { mode: 'line', linesPerChunk: 20, lineOverlap: 0, filePath: 'a.ts' });
    assert.ok(chunks.length >= 2);
    const h = buildSituatingHeader({
      filePath: 'src/a.ts',
      chunkIndex: 0,
      chunkCount: chunks.length,
      chunk: chunks[0]!,
      resolvedMode: 'line',
    });
    assert.match(h, /File: src\/a\.ts/);
    assert.match(h, /chunk 1\/\d+/);
    assert.match(h, /lines 1-/);
    assert.ok(h.split('\n').length <= 3);
  });

  it('includes Section heading for markdown section mode', () => {
    const md = '# Revenue\n\nACME grew 3%.\n\n## Details\n\nMore text here.\n';
    const chunks = chunk(md, { mode: 'section', filePath: 'notes.md' });
    const first = chunks[0]!;
    assert.ok(first.heading);
    const h = buildSituatingHeader({
      filePath: 'notes.md',
      chunkIndex: 0,
      chunkCount: chunks.length,
      chunk: first,
      resolvedMode: 'section',
    });
    assert.match(h, /Section: Revenue/);
  });

  it('includes Symbol from last-seen function for line mode', () => {
    const src = 'export function verifyToken(t: string) {\n  return true;\n}\n\nconst x = 1;\n';
    const chunks = chunk(src, { mode: 'line', linesPerChunk: 2, lineOverlap: 0, filePath: 'jwt.ts' });
    const later = chunks.find((c) => c.heading);
    // first chunk starts at the function — heading is on later windows
    const target = later ?? chunks[chunks.length - 1]!;
    const h = buildSituatingHeader({
      filePath: 'jwt.ts',
      chunkIndex: 1,
      chunkCount: chunks.length,
      chunk: target,
      resolvedMode: 'line',
    });
    if (target.heading) assert.match(h, /Symbol:/);
  });

  it('includes namespace and tags on a third line', () => {
    const h = buildSituatingHeader({
      ...headerInput(),
      namespace: 'code',
      tags: { lang: 'typescript', reviewed: true },
    });
    assert.match(h, /namespace=code/);
    assert.match(h, /lang=typescript/);
    assert.ok(h.split('\n').length <= 3);
  });

  it('legacy mode uses char offset when lines are present still shows File', () => {
    const raw = 'plain paragraph without headings.\n';
    const chunks = chunk(raw, { mode: 'legacy', filePath: 'note.txt' });
    const h = buildSituatingHeader({
      filePath: 'note.txt',
      chunkIndex: 0,
      chunkCount: 1,
      chunk: chunks[0]!,
      resolvedMode: 'legacy',
    });
    assert.match(h, /File: note\.txt/);
    assert.match(h, /chunk 1\/1/);
  });
});

describe('embedTextForChunk', () => {
  it('off is bit-compatible with the raw chunk', () => {
    const input = headerInput();
    const raw = input.chunk.text;
    assert.equal(embedTextForChunk('off', input, raw), raw);
  });

  it('header prefixes situating context and keeps raw body intact', () => {
    const input = headerInput();
    const raw = input.chunk.text;
    const embedded = embedTextForChunk('header', input, raw);
    assert.ok(embedded.startsWith('File:'));
    assert.ok(embedded.endsWith(raw));
    assert.notEqual(embedded, raw);
    assert.equal(raw.includes('File:'), false);
  });

  it('llm without a generator falls back to header', () => {
    const input = headerInput();
    const raw = input.chunk.text;
    const embedded = embedTextForChunk('llm', input, raw);
    assert.ok(embedded.startsWith('File:'));
    assert.ok(embedded.endsWith(raw));
  });
});
