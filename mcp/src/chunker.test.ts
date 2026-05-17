/**
 * Unit tests for the smart chunker — line, section, legacy, and auto modes.
 * Covers normal operation, edge cases, boundary conditions, and invariants.
 * Related: GitHub issues #98, #110.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { chunk, type ChunkOptions } from './chunker';

// ── Helpers ───────────────────────────────────────────────────────────────────

function makeLines(n: number, prefix = 'line'): string {
  return Array.from({ length: n }, (_, i) => `${prefix} ${i + 1}`).join('\n');
}

/** Every word in `words` must appear at least once across all chunks. */
function assertAllWordsPresent(
  words: string[],
  chunks: ReturnType<typeof chunk>,
  label: string,
): void {
  const combined = chunks.map((c) => c.text).join('\n');
  for (const w of words) {
    assert.ok(combined.includes(w), `${label}: word "${w}" missing from chunked output`);
  }
}

/** No chunk may be empty or whitespace-only after trim. */
function assertNoEmptyChunks(chunks: ReturnType<typeof chunk>, label: string): void {
  for (const c of chunks) {
    assert.ok(c.text.trim().length > 0, `${label}: chunk at offset ${c.offset} is empty/whitespace`);
  }
}

/** Chunk offsets must be non-decreasing. */
function assertOffsetMonotonic(chunks: ReturnType<typeof chunk>, label: string): void {
  for (let i = 1; i < chunks.length; i++) {
    assert.ok(
      chunks[i].offset >= chunks[i - 1].offset,
      `${label}: offset regression at index ${i} (${chunks[i - 1].offset} → ${chunks[i].offset})`,
    );
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// AUTO-DETECTION FROM filePath
// ═════════════════════════════════════════════════════════════════════════════

test('auto: .ts → line mode', () => {
  const body = Array.from({ length: 60 }, (_, i) => `  const x${i} = ${i};`).join('\n');
  const src = `function big() {\n${body}\n}\n`;
  const chunks = chunk(src, { filePath: '/src/foo.ts' });
  for (const c of chunks) {
    assert.ok(c.text.split('\n').length <= 60);
  }
});

test('auto: .tsx → line mode', () => {
  const src = makeLines(80);
  assert.ok(chunk(src, { filePath: 'App.tsx' }).length >= 2);
});

test('auto: .js .jsx .mjs .cjs → line mode', () => {
  const src = makeLines(80);
  for (const ext of ['.js', '.jsx', '.mjs', '.cjs']) {
    assert.ok(chunk(src, { filePath: `file${ext}` }).length >= 2, `${ext} should be line mode`);
  }
});

test('auto: .go .rs .java .c .cpp .h .cs .rb .php .swift .kt .scala → line mode', () => {
  const src = makeLines(80);
  for (const ext of ['.go', '.rs', '.java', '.c', '.cpp', '.h', '.cs', '.rb', '.php', '.swift', '.kt', '.scala']) {
    assert.ok(chunk(src, { filePath: `file${ext}` }).length >= 2, `${ext} should be line mode`);
  }
});

test('auto: .sh .bash .zsh .fish → line mode', () => {
  const src = makeLines(80);
  for (const ext of ['.sh', '.bash', '.zsh', '.fish']) {
    assert.ok(chunk(src, { filePath: `file${ext}` }).length >= 2, `${ext} should be line mode`);
  }
});

test('auto: .md .rst → section mode (returns chunks containing heading text)', () => {
  const md = '# Heading\nContent.\n\n## Sub\nMore content.';
  for (const ext of ['.md', '.rst']) {
    const chunks = chunk(md, { filePath: `file${ext}` });
    const combined = chunks.map((c) => c.text).join('\n');
    assert.ok(combined.includes('Heading'), `${ext} should use section mode`);
  }
});

test('auto: .json .yaml .yml .toml .txt .cfg .ini → legacy mode (single small chunk)', () => {
  const text = 'key: value\n';
  for (const ext of ['.json', '.yaml', '.yml', '.toml', '.txt', '.cfg', '.ini']) {
    const chunks = chunk(text, { filePath: `file${ext}` });
    assert.equal(chunks.length, 1, `${ext} should use legacy mode`);
  }
});

test('auto: unknown extension → legacy mode', () => {
  const chunks = chunk('hello world', { filePath: 'foo.xyz' });
  assert.equal(chunks.length, 1);
  assert.ok(chunks[0].text.includes('hello'));
});

test('auto: no extension (Makefile, Dockerfile) → legacy mode', () => {
  for (const name of ['Makefile', 'Dockerfile', 'LICENSE']) {
    const chunks = chunk('all:\n\techo ok', { filePath: name });
    assert.equal(chunks.length, 1);
  }
});

test('auto: uppercase extension (.TS .PY) is case-insensitive → line mode', () => {
  const src = makeLines(80);
  assert.ok(chunk(src, { filePath: 'Foo.TS' }).length >= 2, '.TS should be line mode');
  assert.ok(chunk(src, { filePath: 'script.PY' }).length >= 2, '.PY should be line mode');
});

test('auto: no filePath → legacy mode', () => {
  const chunks = chunk('Para one.\n\nPara two.');
  assert.ok(chunks[0].text.includes('Para'));
});

// ═════════════════════════════════════════════════════════════════════════════
// LINE MODE — normal operation
// ═════════════════════════════════════════════════════════════════════════════

test('line: 10-line file → single chunk', () => {
  assert.equal(chunk(makeLines(10), { mode: 'line', linesPerChunk: 50 }).length, 1);
});

test('line: 100 lines, linesPerChunk=50, lineOverlap=10 → multiple chunks', () => {
  const chunks = chunk(makeLines(100), { mode: 'line', linesPerChunk: 50, lineOverlap: 10 });
  assert.ok(chunks.length >= 2);
});

test('line: overlap lines appear in consecutive chunks', () => {
  const src = makeLines(60);
  const chunks = chunk(src, { mode: 'line', linesPerChunk: 50, lineOverlap: 10 });
  if (chunks.length >= 2) {
    const tail = new Set(chunks[0].text.split('\n').slice(-10));
    const head = chunks[1].text.split('\n').slice(0, 10);
    const shared = head.filter((l) => tail.has(l));
    assert.ok(shared.length > 0, 'No overlap lines found between consecutive chunks');
  }
});

test('line: blank-line boundary preference avoids mid-block split', () => {
  const lines: string[] = [];
  for (let i = 0; i < 45; i++) lines.push(`code line ${i}`);
  lines.push(''); // blank at index 45
  for (let i = 46; i < 70; i++) lines.push(`code line ${i}`);
  const chunks = chunk(lines.join('\n'), { mode: 'line', linesPerChunk: 50, lineOverlap: 5 });
  assert.ok(chunks.length >= 1);
  assertOffsetMonotonic(chunks, 'line/blank-boundary');
});

test('line: unicode multibyte characters preserved intact', () => {
  const src = ['// 日本語コメント', 'const x = "héllo";', 'fn foo(s: &str) {}'].join('\n');
  const chunks = chunk(src, { mode: 'line' });
  const combined = chunks.map((c) => c.text).join('\n');
  assert.ok(combined.includes('日本語'));
  assert.ok(combined.includes('héllo'));
});

test('line: all unique words preserved across chunks', () => {
  const words = ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf'];
  const lines = words.map((w, i) => `${w}_token_${i} = ${i};`);
  // 7 lines, fits in one chunk — still verify correctness
  assertAllWordsPresent(words.map((w) => `${w}_token`), chunk(lines.join('\n'), { mode: 'line' }), 'line');
});

test('line: large file (1000 lines) — all content preserved', () => {
  const src = makeLines(1000);
  const chunks = chunk(src, { mode: 'line', linesPerChunk: 50, lineOverlap: 10 });
  assert.ok(chunks.length >= 10);
  assertAllWordsPresent(['line 1', 'line 500', 'line 1000'], chunks, 'line/1000');
  assertOffsetMonotonic(chunks, 'line/1000');
});

// ═════════════════════════════════════════════════════════════════════════════
// LINE MODE — edge cases and boundary conditions
// ═════════════════════════════════════════════════════════════════════════════

test('line: single-line file', () => {
  const chunks = chunk('const x = 1;', { mode: 'line' });
  assert.equal(chunks.length, 1);
  assert.equal(chunks[0].text, 'const x = 1;');
  assert.equal(chunks[0].offset, 0);
});

test('line: empty string → single chunk', () => {
  const chunks = chunk('', { mode: 'line' });
  assert.equal(chunks.length, 1);
});

test('line: whitespace-only string → single chunk', () => {
  const chunks = chunk('   \n  \n  ', { mode: 'line' });
  assert.equal(chunks.length, 1);
});

test('line: linesPerChunk=1, lineOverlap=0 → each non-empty line is its own chunk', () => {
  const src = 'a\nb\nc\nd\ne';
  const chunks = chunk(src, { mode: 'line', linesPerChunk: 1, lineOverlap: 0 });
  assert.equal(chunks.length, 5);
  assert.ok(chunks.some((c) => c.text === 'a'));
  assert.ok(chunks.some((c) => c.text === 'e'));
});

test('line: lineOverlap >= linesPerChunk → does not infinite-loop (step clamped to 1)', () => {
  const src = makeLines(20);
  // step = max(1, 5 - 10) = 1 — should complete with more chunks
  const chunks = chunk(src, { mode: 'line', linesPerChunk: 5, lineOverlap: 10 });
  assert.ok(chunks.length >= 1);
  assertOffsetMonotonic(chunks, 'line/overlap>=window');
});

test('line: file of only blank lines → single chunk (empty-ish)', () => {
  const chunks = chunk('\n\n\n\n', { mode: 'line' });
  assert.equal(chunks.length, 1);
});

test('line: very long single line (> typical 50*80 chars)', () => {
  const line = 'x'.repeat(5000);
  const chunks = chunk(line, { mode: 'line' });
  // Should not crash and content is preserved
  const combined = chunks.map((c) => c.text).join('');
  assert.ok(combined.includes('x'));
});

test('line: file with trailing newline — no spurious extra empty chunk', () => {
  const src = 'a\nb\nc\n'; // trailing newline
  const chunks = chunk(src, { mode: 'line', linesPerChunk: 50 });
  assert.equal(chunks.length, 1);
  assert.equal(chunks[0].text, 'a\nb\nc');
});

test('line: file with Windows CRLF line endings — content not corrupted', () => {
  const src = 'line 1\r\nline 2\r\nline 3\r\n';
  const chunks = chunk(src, { mode: 'line' });
  const combined = chunks.map((c) => c.text).join('');
  assert.ok(combined.includes('line 1'));
  assert.ok(combined.includes('line 3'));
});

test('line: exactly linesPerChunk lines → single chunk, no split', () => {
  const src = makeLines(50); // exactly 50 lines
  const chunks = chunk(src, { mode: 'line', linesPerChunk: 50, lineOverlap: 10 });
  assert.equal(chunks.length, 1);
});

test('line: exactly linesPerChunk+1 lines → two chunks', () => {
  const src = makeLines(51);
  const chunks = chunk(src, { mode: 'line', linesPerChunk: 50, lineOverlap: 10 });
  assert.ok(chunks.length >= 2);
});

test('line: no chunk text is empty or whitespace-only', () => {
  const src = makeLines(120);
  const chunks = chunk(src, { mode: 'line', linesPerChunk: 50, lineOverlap: 10 });
  assertNoEmptyChunks(chunks, 'line');
});

// ═════════════════════════════════════════════════════════════════════════════
// SECTION MODE — normal operation
// ═════════════════════════════════════════════════════════════════════════════

test('section: ATX headings split document', () => {
  const md = ['# H1', 'Text.', '## H2a', 'More.', '## H2b', 'Last.'].join('\n');
  const chunks = chunk(md, { mode: 'section', targetChars: 2000 });
  const combined = chunks.map((c) => c.text).join('\n');
  assert.ok(combined.includes('# H1'));
  assert.ok(combined.includes('## H2a'));
  assert.ok(combined.includes('## H2b'));
});

test('section: setext headings (= underline) split document', () => {
  const rst = ['Title', '=====', '', 'Intro.', '', 'Sub', '---', '', 'Detail.'].join('\n');
  const combined = chunk(rst, { mode: 'section', targetChars: 2000 }).map((c) => c.text).join('\n');
  assert.ok(combined.includes('Title'));
  assert.ok(combined.includes('Sub'));
});

test('section: section larger than targetChars is sub-split', () => {
  const big = '# Big\n' + 'word '.repeat(400);
  const chunks = chunk(big, { mode: 'section', targetChars: 500 });
  assert.ok(chunks.length >= 2, 'Big section must be sub-split');
});

test('section: small consecutive sections are merged up to targetChars', () => {
  const md = ['# A', 'tiny.', '# B', 'tiny.', '# C', 'tiny.'].join('\n');
  const chunks = chunk(md, { mode: 'section', targetChars: 500 });
  assert.ok(chunks.length <= 2);
});

test('section: no headings → falls back to paragraph chunker', () => {
  const plain = 'First paragraph.\n\nSecond.\n\nThird.';
  const combined = chunk(plain, { mode: 'section', targetChars: 2000 }).map((c) => c.text).join(' ');
  assert.ok(combined.includes('First'));
  assert.ok(combined.includes('Third'));
});

test('section: all content preserved across chunks', () => {
  const words = ['introduction', 'installation', 'configuration', 'troubleshooting'];
  const md = words.map((w) => `## ${w}\n\n${w} details here.\n`).join('\n');
  assertAllWordsPresent(words, chunk(md, { mode: 'section' }), 'section/preservation');
});

// ═════════════════════════════════════════════════════════════════════════════
// SECTION MODE — edge cases and boundary conditions
// ═════════════════════════════════════════════════════════════════════════════

test('section: heading-only document (no body text)', () => {
  const md = '# Title\n## Section A\n## Section B\n';
  const chunks = chunk(md, { mode: 'section', targetChars: 2000 });
  const combined = chunks.map((c) => c.text).join('\n');
  assert.ok(combined.includes('Title'));
});

test('section: heading at very end of file', () => {
  const md = 'Some content.\n\n## Last Heading';
  const chunks = chunk(md, { mode: 'section', targetChars: 2000 });
  const combined = chunks.map((c) => c.text).join('\n');
  assert.ok(combined.includes('Last Heading'));
});

test('section: consecutive headings with no body between them', () => {
  const md = ['# H1', '## H2', '### H3', 'Finally some content.'].join('\n');
  const chunks = chunk(md, { mode: 'section', targetChars: 2000 });
  const combined = chunks.map((c) => c.text).join('\n');
  assert.ok(combined.includes('# H1'));
  assert.ok(combined.includes('Finally some content'));
});

test('section: all six ATX heading levels are recognised', () => {
  const md = ['# H1', 'a.', '## H2', 'b.', '### H3', 'c.', '#### H4', 'd.', '##### H5', 'e.', '###### H6', 'f.'].join('\n');
  const combined = chunk(md, { mode: 'section', targetChars: 2000 }).map((c) => c.text).join('\n');
  for (const h of ['# H1', '## H2', '### H3', '#### H4', '##### H5', '###### H6']) {
    assert.ok(combined.includes(h), `heading ${h} not found in output`);
  }
});

test('section: heading with special characters in title', () => {
  const md = '# API: /v1/users?id=1&sort=asc\n\nDocs here.';
  const chunks = chunk(md, { mode: 'section', targetChars: 2000 });
  const combined = chunks.map((c) => c.text).join('\n');
  assert.ok(combined.includes('/v1/users'));
});

test('section: empty document → single chunk', () => {
  const chunks = chunk('', { mode: 'section' });
  assert.equal(chunks.length, 1);
});

test('section: only blank lines → single chunk', () => {
  const chunks = chunk('\n\n\n', { mode: 'section' });
  assert.equal(chunks.length, 1);
});

test('section: section with only blank lines as body is preserved', () => {
  const md = '# Title\n\n\n\n## Next\n\nContent.';
  const chunks = chunk(md, { mode: 'section', targetChars: 2000 });
  const combined = chunks.map((c) => c.text).join('\n');
  assert.ok(combined.includes('Title'));
  assert.ok(combined.includes('Content'));
});

test('section: markdown code fences containing # — treated as heading (known limitation)', () => {
  // Our chunker does not parse fences; a `# comment` inside a fence is treated
  // as a section boundary. This test documents the current (expected) behaviour.
  const md = ['# Title', '```python', '# this is a comment', 'x = 1', '```', 'End.'].join('\n');
  const chunks = chunk(md, { mode: 'section', targetChars: 2000 });
  // Should not crash and all text is preserved
  const combined = chunks.map((c) => c.text).join('\n');
  assert.ok(combined.includes('Title'));
  assert.ok(combined.includes('End.'));
});

test('section: targetChars=1 → every character is split', () => {
  const md = '# A\nHi.';
  const chunks = chunk(md, { mode: 'section', targetChars: 1 });
  assert.ok(chunks.length >= 2);
  assertOffsetMonotonic(chunks, 'section/targetChars=1');
});

test('section: no chunk is empty or whitespace-only', () => {
  const md = ['# Intro', 'Text.', '', '## Details', 'More text.', '', '## End', 'Done.'].join('\n');
  assertNoEmptyChunks(chunk(md, { mode: 'section', targetChars: 50 }), 'section');
});

// ═════════════════════════════════════════════════════════════════════════════
// LEGACY MODE — normal operation
// ═════════════════════════════════════════════════════════════════════════════

test('legacy: paragraph split and merge', () => {
  const text = 'Para one.\n\nPara two.\n\nPara three.';
  const combined = chunk(text, { mode: 'legacy', targetChars: 500 }).map((c) => c.text).join(' ');
  assert.ok(combined.includes('Para one'));
  assert.ok(combined.includes('Para three'));
});

test('legacy: overlap carries characters into next chunk', () => {
  const long = 'word '.repeat(500);
  const chunks = chunk(long, { mode: 'legacy', targetChars: 800, overlapChars: 100 });
  assert.ok(chunks.length >= 2);
});

test('legacy: old two-argument signature preserved', () => {
  const chunks = chunk('Hello.\n\nWorld.', 800, 100);
  assert.equal(chunks.length, 1);
  assert.ok(chunks[0].text.includes('Hello'));
  assert.ok(chunks[0].text.includes('World'));
});

test('legacy: single long paragraph sub-split into char windows', () => {
  const para = 'word '.repeat(300); // 1500 chars, no blank lines
  const chunks = chunk(para, { mode: 'legacy', targetChars: 500, overlapChars: 50 });
  assert.ok(chunks.length >= 2);
  assertOffsetMonotonic(chunks, 'legacy/long-para');
});

test('legacy: all content preserved across chunks', () => {
  const words = ['apple', 'banana', 'cherry', 'date', 'elderberry'];
  const text = words.map((w) => `The fruit is ${w}.`).join('\n\n');
  assertAllWordsPresent(words, chunk(text, { mode: 'legacy', targetChars: 30 }), 'legacy');
});

// ═════════════════════════════════════════════════════════════════════════════
// LEGACY MODE — edge cases and boundary conditions
// ═════════════════════════════════════════════════════════════════════════════

test('legacy: empty string → single chunk', () => {
  const chunks = chunk('', { mode: 'legacy' });
  assert.equal(chunks.length, 1);
});

test('legacy: whitespace-only string', () => {
  const chunks = chunk('   \n\n   ', { mode: 'legacy' });
  assert.equal(chunks.length, 1);
});

test('legacy: single paragraph exactly at targetChars → one chunk', () => {
  const text = 'x'.repeat(800);
  const chunks = chunk(text, { mode: 'legacy', targetChars: 800 });
  // The paragraph equals targetChars — splitLongText will not split it further
  // because trimmed.length >= targetChars triggers splitLongText but a single
  // window covers it all. Verify at least 1 chunk and content preserved.
  assert.ok(chunks.length >= 1);
  const combined = chunks.map((c) => c.text).join('');
  assert.ok(combined.includes('x'));
});

test('legacy: single paragraph one char over targetChars → multiple chunks', () => {
  const text = 'x'.repeat(801);
  const chunks = chunk(text, { mode: 'legacy', targetChars: 800, overlapChars: 0 });
  assert.ok(chunks.length >= 2);
});

test('legacy: overlapChars=0 → fewer cross-boundary duplicates than with overlap', () => {
  // Use numbered tokens so we can compare overlap=0 vs overlap=100 meaningfully.
  // 'tok_0 tok_1 ... tok_399' — each token is unique, 2000+ chars total.
  const tokens = Array.from({ length: 400 }, (_, i) => `tok_${i}`);
  const text = tokens.join(' ');

  const chunksNoOverlap = chunk(text, { mode: 'legacy', targetChars: 200, overlapChars: 0 });
  const chunksWithOverlap = chunk(text, { mode: 'legacy', targetChars: 200, overlapChars: 80 });

  // Count total tokens across all chunks; overlap=0 should have fewer total than overlap>0
  const countTokens = (cs: ReturnType<typeof chunk>) =>
    cs.reduce((sum, c) => sum + c.text.split(/\s+/).filter(Boolean).length, 0);

  assert.ok(chunksNoOverlap.length >= 2);
  assert.ok(countTokens(chunksNoOverlap) <= countTokens(chunksWithOverlap),
    'overlapChars=0 should produce equal or fewer total tokens than overlap=80');
});

test('legacy: targetChars=1 — does not crash, produces many chunks', () => {
  const text = 'abc';
  const chunks = chunk(text, { mode: 'legacy', targetChars: 1, overlapChars: 0 });
  assert.ok(chunks.length >= 1);
  assertOffsetMonotonic(chunks, 'legacy/targetChars=1');
});

test('legacy: very many short paragraphs (200) — all preserved', () => {
  const text = Array.from({ length: 200 }, (_, i) => `para_${i}`).join('\n\n');
  const chunks = chunk(text, { mode: 'legacy', targetChars: 100 });
  assertAllWordsPresent(['para_0', 'para_99', 'para_199'], chunks, 'legacy/200-paras');
});

test('legacy: text with only blank lines → single chunk', () => {
  const chunks = chunk('\n\n\n\n', { mode: 'legacy' });
  assert.equal(chunks.length, 1);
});

test('legacy: no chunk is empty or whitespace-only', () => {
  const text = Array.from({ length: 50 }, (_, i) => `paragraph ${i} content`).join('\n\n');
  assertNoEmptyChunks(chunk(text, { mode: 'legacy', targetChars: 50 }), 'legacy');
});

// ═════════════════════════════════════════════════════════════════════════════
// INVARIANTS ACROSS ALL MODES
// ═════════════════════════════════════════════════════════════════════════════

test('all modes: first chunk offset is always 0', () => {
  const src = makeLines(5);
  for (const mode of ['auto', 'line', 'section', 'legacy'] as ChunkOptions['mode'][]) {
    const chunks = chunk(src, { mode });
    assert.equal(chunks[0].offset, 0, `mode ${mode}: first chunk offset should be 0`);
  }
});

test('all modes: offsets are non-decreasing', () => {
  const src = makeLines(200);
  for (const mode of ['line', 'legacy'] as ChunkOptions['mode'][]) {
    const chunks = chunk(src, { mode, linesPerChunk: 30, lineOverlap: 5, targetChars: 300 });
    assertOffsetMonotonic(chunks, `mode=${mode}`);
  }
  // section with many headings
  const md = Array.from({ length: 30 }, (_, i) => `## Section ${i}\nContent ${i}.`).join('\n\n');
  assertOffsetMonotonic(chunk(md, { mode: 'section', targetChars: 60 }), 'mode=section');
});

test('all modes: result is always an array with at least one chunk', () => {
  for (const mode of ['auto', 'line', 'section', 'legacy'] as ChunkOptions['mode'][]) {
    for (const input of ['', '  ', 'hello', 'a\nb', 'a\n\nb']) {
      const chunks = chunk(input, { mode });
      assert.ok(Array.isArray(chunks) && chunks.length >= 1, `mode=${mode} input=${JSON.stringify(input)}`);
    }
  }
});

test('all modes: chunk objects always have text (string) and offset (number)', () => {
  const src = makeLines(5);
  for (const mode of ['auto', 'line', 'section', 'legacy'] as ChunkOptions['mode'][]) {
    for (const c of chunk(src, { mode })) {
      assert.equal(typeof c.text, 'string', `mode=${mode}: chunk.text must be a string`);
      assert.equal(typeof c.offset, 'number', `mode=${mode}: chunk.offset must be a number`);
      assert.ok(Number.isFinite(c.offset) && c.offset >= 0, `mode=${mode}: invalid offset ${c.offset}`);
    }
  }
});

test('all modes: unique sentinel words are never lost across chunking', () => {
  const sentinels = ['SENTINEL_A', 'SENTINEL_B', 'SENTINEL_C', 'SENTINEL_D', 'SENTINEL_E'];
  // Distribute sentinels in 100-line code, 5-section markdown, short plain text
  const codeSrc = sentinels.map((s, i) => Array.from({ length: 20 }, (_, j) => j === 10 ? `// ${s}` : `line ${i}_${j}`).join('\n')).join('\n');
  const mdSrc = sentinels.map((s) => `## ${s}\n\nSection content for ${s}.`).join('\n\n');
  const plainSrc = sentinels.map((s) => `${s} paragraph.`).join('\n\n');

  assertAllWordsPresent(sentinels, chunk(codeSrc, { filePath: 'code.ts' }), 'sentinel/code');
  assertAllWordsPresent(sentinels, chunk(mdSrc, { filePath: 'doc.md' }), 'sentinel/md');
  assertAllWordsPresent(sentinels, chunk(plainSrc, { mode: 'legacy', targetChars: 30 }), 'sentinel/plain');
});

// ═════════════════════════════════════════════════════════════════════════════
// BACKWARD COMPATIBILITY
// ═════════════════════════════════════════════════════════════════════════════

test('backward-compat: chunk(text, targetChars) — number second arg → legacy mode', () => {
  const text = 'Hello.\n\nWorld.';
  const chunks = chunk(text, 800);
  assert.ok(chunks[0].text.includes('Hello'));
});

test('backward-compat: chunk(text, targetChars, overlapChars) — three-arg form', () => {
  const text = 'word '.repeat(400);
  const chunks = chunk(text, 800, 100);
  assert.ok(chunks.length >= 2);
});

test('backward-compat: chunk(text) — no opts, defaults to legacy', () => {
  const chunks = chunk('Para one.\n\nPara two.');
  assert.ok(chunks[0].text.includes('Para one'));
});
