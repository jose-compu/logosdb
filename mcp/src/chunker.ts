/**
 * Smart text chunker for logosdb-mcp-server.
 *
 * Three strategies, selected by `mode` (or auto-detected from `filePath`):
 *
 *   "auto"    — (default) picks the best strategy per file extension:
 *                 · code files (.ts .py .go .rs .java .c .cpp …) → "line"
 *                 · docs (.md .rst)                               → "section"
 *                 · config/data (.json .yaml .toml .txt …)        → "legacy"
 *
 *   "line"    — sliding line-window (code files).
 *               Target ~50 lines per chunk, ~10-line overlap.
 *               Prefers blank-line boundaries at window edges to avoid
 *               splitting mid-function. Best for retrieval precision on code.
 *
 *   "section" — heading-aware (Markdown / RST).
 *               Splits on ATX headings (# … ###### …) and Setext underlines
 *               (=== / ---). Small consecutive sections are merged up to
 *               `targetChars`. Large single sections are sub-split with the
 *               paragraph chunker. Best for docs and README-style files.
 *
 *   "legacy"  — original paragraph / character merge (any file type).
 *               Splits on blank lines, merges short paragraphs up to
 *               `targetChars` (default 800), carries `overlapChars` (default
 *               100) into the next chunk. Long single paragraphs are sub-split
 *               into character-window slices. Use this mode when you want the
 *               old behaviour for A/B testing, compatibility with an existing
 *               namespace, or for dense config/data files where paragraph merge
 *               is appropriate. Enable globally with `LOGOSDB_CHUNK_MODE=legacy`
 *               or per-call with `chunking: "legacy"` on `logosdb_index_file`.
 *
 * Related: GitHub issue #98.
 */

import * as path from 'path';

export interface Chunk {
  text: string;
  /** approximate character offset in original text */
  offset: number;
}

export type ChunkMode = 'auto' | 'line' | 'section' | 'legacy';

export interface ChunkOptions {
  /** Target character count per chunk (legacy / section modes). Default: 800. */
  targetChars?: number;
  /** Character overlap carried into the next chunk (legacy mode). Default: 100. */
  overlapChars?: number;
  /** Chunking strategy. Default: "auto". */
  mode?: ChunkMode;
  /**
   * Absolute or relative file path — used by "auto" to pick the right strategy.
   * Ignored unless mode is "auto".
   */
  filePath?: string;
  /** Lines per chunk (line mode). Default: 50. */
  linesPerChunk?: number;
  /** Lines of overlap between consecutive line-window chunks. Default: 10. */
  lineOverlap?: number;
}

// ── Extension sets ────────────────────────────────────────────────────────────

/** Source-code file extensions → line-window chunking. */
const CODE_EXTS = new Set([
  '.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs',
  '.py', '.go', '.rs', '.java',
  '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp',
  '.cs', '.rb', '.php', '.swift', '.kt', '.scala',
  '.sh', '.bash', '.zsh', '.fish',
  '.sql',
]);

/** Documentation file extensions → section-aware chunking. */
const SECTION_EXTS = new Set(['.md', '.rst']);

function resolveMode(opts: ChunkOptions): ChunkMode {
  const m = opts.mode ?? 'auto';
  if (m !== 'auto') return m;
  if (!opts.filePath) return 'legacy';
  const ext = path.extname(opts.filePath).toLowerCase();
  if (CODE_EXTS.has(ext)) return 'line';
  if (SECTION_EXTS.has(ext)) return 'section';
  return 'legacy';
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Chunk `text` according to `opts`.
 * The old two-argument signature `chunk(text, targetChars, overlapChars)` is
 * preserved for callers that have not yet migrated.
 */
export function chunk(text: string, opts?: ChunkOptions | number, _legacyOverlap?: number): Chunk[] {
  if (!text.trim()) return [{ text: '', offset: 0 }];

  let resolvedOpts: ChunkOptions;
  if (typeof opts === 'number') {
    resolvedOpts = {
      targetChars: opts,
      overlapChars: _legacyOverlap ?? 100,
      mode: 'legacy',
    };
  } else {
    resolvedOpts = opts ?? {};
  }

  const mode = resolveMode(resolvedOpts);

  switch (mode) {
    case 'line':
      return chunkByLines(
        text,
        resolvedOpts.linesPerChunk ?? 50,
        resolvedOpts.lineOverlap ?? 10,
      );
    case 'section':
      return chunkBySection(
        text,
        resolvedOpts.targetChars ?? 800,
        resolvedOpts.overlapChars ?? 100,
      );
    default:
      return chunkByParagraph(
        text,
        resolvedOpts.targetChars ?? 800,
        resolvedOpts.overlapChars ?? 100,
      );
  }
}

// ── Strategy: line-window (code files) ───────────────────────────────────────

/**
 * Sliding line-window chunker.
 * Each chunk is `linesPerChunk` lines; consecutive chunks overlap by
 * `lineOverlap` lines. Splits prefer blank-line boundaries within the last
 * 10 lines of a window, avoiding mid-statement cuts where possible.
 */
function chunkByLines(text: string, linesPerChunk: number, lineOverlap: number): Chunk[] {
  const lines = text.split('\n');
  if (lines.length === 0) return [{ text: text.trim(), offset: 0 }];

  // Build per-line byte offsets into the original text.
  const lineOffsets: number[] = new Array(lines.length);
  let off = 0;
  for (let i = 0; i < lines.length; i++) {
    lineOffsets[i] = off;
    off += lines[i].length + 1; // +1 for the '\n' we split on
  }

  const step = Math.max(1, linesPerChunk - lineOverlap);
  const chunks: Chunk[] = [];
  let start = 0;

  while (start < lines.length) {
    const rawEnd = Math.min(start + linesPerChunk, lines.length);

    // Prefer a blank-line split within the last 10 lines of the window.
    let splitAt = rawEnd;
    if (rawEnd < lines.length) {
      for (let j = rawEnd - 1; j >= Math.max(rawEnd - 10, start + 1); j--) {
        if (lines[j].trim() === '') {
          splitAt = j + 1;
          break;
        }
      }
    }

    const chunkText = lines.slice(start, splitAt).join('\n').trim();
    if (chunkText) {
      chunks.push({ text: chunkText, offset: lineOffsets[start] });
    }

    if (splitAt >= lines.length) break;
    start = Math.max(start + 1, splitAt - lineOverlap);
  }

  return chunks.length ? chunks : [{ text: text.trim(), offset: 0 }];
}

// ── Strategy: section-aware (Markdown / RST) ─────────────────────────────────

/**
 * ATX headings (`# … ` through `###### …`) and Setext underlines (`===`, `---`).
 * Matches lines that *begin* a new section boundary.
 */
const ATX_HEADING_RE = /^#{1,6}\s/;
const SETEXT_UNDERLINE_RE = /^[=\-]{3,}\s*$/;

/**
 * Section-aware chunker for Markdown and RST.
 *
 * Algorithm:
 *   1. Walk lines and detect heading boundaries.
 *   2. Accumulate content into sections.
 *   3. Merge small consecutive sections up to `targetChars`.
 *   4. Large individual sections are sub-split with the paragraph chunker.
 */
function chunkBySection(text: string, targetChars: number, overlapChars: number): Chunk[] {
  const lines = text.split('\n');
  if (lines.length === 0) return chunkByParagraph(text, targetChars, overlapChars);

  // Build per-line offsets.
  const lineOffsets: number[] = new Array(lines.length);
  let off = 0;
  for (let i = 0; i < lines.length; i++) {
    lineOffsets[i] = off;
    off += lines[i].length + 1;
  }

  // Identify section start indices.
  const sectionStarts: number[] = [0];
  for (let i = 0; i < lines.length; i++) {
    if (i === 0) continue;
    const line = lines[i];
    const prev = lines[i - 1];
    const isAtx = ATX_HEADING_RE.test(line);
    // Setext: prev line is text, current line is all = or -
    const isSetext =
      SETEXT_UNDERLINE_RE.test(line) &&
      prev.trim().length > 0 &&
      !SETEXT_UNDERLINE_RE.test(prev) &&
      !ATX_HEADING_RE.test(prev);

    if (isAtx || isSetext) {
      // For setext, the heading starts on the *previous* line.
      const start = isSetext ? i - 1 : i;
      // Avoid duplicate starts (e.g. setext was already counted via prev iter).
      if (start > sectionStarts[sectionStarts.length - 1]) {
        sectionStarts.push(start);
      }
    }
  }

  // No headings found — fall back to paragraph chunker.
  if (sectionStarts.length <= 1 && !ATX_HEADING_RE.test(lines[0])) {
    return chunkByParagraph(text, targetChars, overlapChars);
  }

  // Build raw sections: [{text, offset}].
  const rawSections: { text: string; offset: number }[] = [];
  for (let s = 0; s < sectionStarts.length; s++) {
    const from = sectionStarts[s];
    const to = s + 1 < sectionStarts.length ? sectionStarts[s + 1] : lines.length;
    const secText = lines.slice(from, to).join('\n').trim();
    if (secText) rawSections.push({ text: secText, offset: lineOffsets[from] });
  }

  if (rawSections.length === 0) return [{ text: text.trim(), offset: 0 }];

  // Merge small sections; sub-split large ones.
  const result: Chunk[] = [];
  let buffer = '';
  let bufferOffset = 0;

  const flushBuffer = () => {
    if (buffer.trim()) result.push({ text: buffer.trim(), offset: bufferOffset });
    buffer = '';
  };

  for (const sec of rawSections) {
    if (sec.text.length > targetChars) {
      flushBuffer();
      const sub = chunkByParagraph(sec.text, targetChars, overlapChars);
      for (const c of sub) result.push({ text: c.text, offset: sec.offset + c.offset });
    } else if (buffer && buffer.length + sec.text.length + 2 > targetChars) {
      flushBuffer();
      buffer = sec.text;
      bufferOffset = sec.offset;
    } else {
      if (!buffer) bufferOffset = sec.offset;
      buffer = buffer ? buffer + '\n\n' + sec.text : sec.text;
    }
  }
  flushBuffer();

  return result.length ? result : [{ text: text.trim(), offset: 0 }];
}

// ── Strategy: legacy paragraph/char (everything else) ────────────────────────

/**
 * Split a long string that has no internal blank-line boundaries into
 * character-window chunks with overlap. Used as a fallback when a single
 * paragraph exceeds `targetChars`.
 */
function splitLongText(
  text: string,
  targetChars: number,
  overlapChars: number,
  baseOffset: number,
): Chunk[] {
  const chunks: Chunk[] = [];
  const step = Math.max(1, targetChars - overlapChars);
  let pos = 0;
  while (pos < text.length) {
    const end = Math.min(pos + targetChars, text.length);
    const slice = text.slice(pos, end).trim();
    if (slice) chunks.push({ text: slice, offset: baseOffset + pos });
    if (end >= text.length) break;
    pos += step;
  }
  return chunks.length ? chunks : [{ text: text.trim(), offset: baseOffset }];
}

/**
 * Original paragraph-aware character chunker.
 * Splits on blank lines, merges short paragraphs up to `targetChars`,
 * carries `overlapChars` into the next chunk.
 * Paragraphs longer than `targetChars` are sub-split by character window.
 */
function chunkByParagraph(text: string, targetChars: number, overlapChars: number): Chunk[] {
  const paragraphs = text.split(/\n{2,}/);
  const chunks: Chunk[] = [];
  let buffer = '';
  let bufferOffset = 0;
  let currentOffset = 0;

  const flushBuffer = () => {
    const t = buffer.trim();
    if (!t) return;
    if (t.length > targetChars) {
      // Long buffer with no blank-line breaks — split by char window.
      chunks.push(...splitLongText(t, targetChars, overlapChars, bufferOffset));
    } else {
      chunks.push({ text: t, offset: bufferOffset });
    }
    buffer = '';
  };

  for (const para of paragraphs) {
    const trimmed = para.trim();
    if (!trimmed) {
      currentOffset += para.length + 2;
      continue;
    }

    if (trimmed.length >= targetChars) {
      // This paragraph alone exceeds the target — flush buffer, then split it.
      flushBuffer();
      chunks.push(...splitLongText(trimmed, targetChars, overlapChars, currentOffset));
      bufferOffset = currentOffset + trimmed.length;
    } else if (buffer && buffer.length + trimmed.length + 1 > targetChars) {
      flushBuffer();
      const overlap = trimmed.slice(0, overlapChars);
      buffer = overlap + '\n' + trimmed;
      bufferOffset = currentOffset - overlap.length;
    } else {
      if (!buffer) bufferOffset = currentOffset;
      buffer = buffer ? buffer + '\n\n' + trimmed : trimmed;
    }

    currentOffset += para.length + 2;
  }

  flushBuffer();

  return chunks.length ? chunks : [{ text: text.trim(), offset: 0 }];
}
