/**
 * Unit tests for MCP input validation (security.ts).
 * Covers clamp helpers, text/metadata validation, NFC normalisation, bidi
 * rejection, path confinement, and file-walk safety.
 * Related: GitHub issues #74, #108, #109, #110.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

import {
  assertNoDisallowedControls,
  CHUNK_SIZE_MAX,
  CHUNK_SIZE_MIN,
  clampChunkSize,
  clampTopK,
  collectFilesSafe,
  MAX_METADATA_CHARS,
  MAX_TEXT_CHARS,
  normalizeAndValidateUserText,
  resolveIndexablePath,
  TOP_K_MAX,
  TOP_K_MIN,
  validateMetadata,
  validateUserText,
} from './security';

// ═════════════════════════════════════════════════════════════════════════════
// clampChunkSize
// ═════════════════════════════════════════════════════════════════════════════

test('clampChunkSize: below MIN is clamped to MIN', () => {
  assert.equal(clampChunkSize(50, 800), CHUNK_SIZE_MIN);
  assert.equal(clampChunkSize(1, 800), CHUNK_SIZE_MIN);
});

test('clampChunkSize: above MAX is clamped to MAX', () => {
  assert.equal(clampChunkSize(200_000, 800), CHUNK_SIZE_MAX);
  // Infinity is not finite → base = fallback (800), then clamped to [MIN, MAX] → 800
  assert.equal(clampChunkSize(Infinity, 800), 800);
  // But a very large finite fallback itself gets clamped to MAX
  assert.equal(clampChunkSize(NaN, 999_999), CHUNK_SIZE_MAX);
});

test('clampChunkSize: exactly at MIN passes', () => {
  assert.equal(clampChunkSize(CHUNK_SIZE_MIN, 800), CHUNK_SIZE_MIN);
});

test('clampChunkSize: exactly at MAX passes', () => {
  assert.equal(clampChunkSize(CHUNK_SIZE_MAX, 800), CHUNK_SIZE_MAX);
});

test('clampChunkSize: NaN → uses fallback', () => {
  assert.equal(clampChunkSize(NaN, 900), 900);
});

test('clampChunkSize: 0 → uses fallback (0 is not > 0)', () => {
  assert.equal(clampChunkSize(0, 800), 800);
});

test('clampChunkSize: negative → uses fallback', () => {
  assert.equal(clampChunkSize(-100, 800), 800);
});

test('clampChunkSize: float → truncated, then clamped', () => {
  assert.equal(clampChunkSize(500.9, 800), 500);
});

test('clampChunkSize: fallback itself is clamped to [MIN, MAX]', () => {
  // fallback below MIN → MIN
  assert.equal(clampChunkSize(NaN, 10), CHUNK_SIZE_MIN);
  // fallback above MAX → MAX
  assert.equal(clampChunkSize(NaN, 999_999), CHUNK_SIZE_MAX);
});

// ═════════════════════════════════════════════════════════════════════════════
// clampTopK
// ═════════════════════════════════════════════════════════════════════════════

test('clampTopK: valid range [1, 500] passes', () => {
  assert.equal(clampTopK(1), TOP_K_MIN);
  assert.equal(clampTopK(3), 3);
  assert.equal(clampTopK(TOP_K_MAX), TOP_K_MAX);
});

test('clampTopK: exactly at boundaries', () => {
  assert.equal(clampTopK(TOP_K_MIN), TOP_K_MIN);
  assert.equal(clampTopK(TOP_K_MAX), TOP_K_MAX);
});

test('clampTopK: 0 → throws', () => {
  assert.throws(() => clampTopK(0), /top_k must be between/);
});

test('clampTopK: negative → throws', () => {
  assert.throws(() => clampTopK(-1), /top_k must be between/);
});

test('clampTopK: above MAX → throws', () => {
  assert.throws(() => clampTopK(TOP_K_MAX + 1), /top_k must be between/);
});

test('clampTopK: float is truncated — 0.9 → trunc → 0 → throws', () => {
  assert.throws(() => clampTopK(0.9), /top_k must be between/);
});

test('clampTopK: 500.9 → trunc → 500 → passes', () => {
  assert.equal(clampTopK(500.9), TOP_K_MAX);
});

test('clampTopK: NaN → treated as TOP_K_MIN (1) → passes', () => {
  assert.equal(clampTopK(NaN), TOP_K_MIN);
});

test('clampTopK: Infinity → not finite → treated as TOP_K_MIN → passes', () => {
  assert.equal(clampTopK(Infinity), TOP_K_MIN);
});

test('clampTopK: -Infinity → treated as TOP_K_MIN → passes', () => {
  assert.equal(clampTopK(-Infinity), TOP_K_MIN);
});

// ═════════════════════════════════════════════════════════════════════════════
// assertNoDisallowedControls
// ═════════════════════════════════════════════════════════════════════════════

test('control: NUL (0x00) → throws', () => {
  assert.throws(() => assertNoDisallowedControls('\u0000', 't'), /NUL/);
});

test('control: all C0 controls except HT (9) LF (10) CR (13) → throw', () => {
  for (let code = 1; code < 32; code++) {
    if (code === 9 || code === 10 || code === 13) continue;
    assert.throws(
      () => assertNoDisallowedControls(String.fromCharCode(code), 't'),
      /disallowed control/,
      `C0 code ${code} (0x${code.toString(16)}) should be rejected`,
    );
  }
});

test('control: HT (9), LF (10), CR (13) → allowed', () => {
  assert.doesNotThrow(() => assertNoDisallowedControls('\t', 't'));
  assert.doesNotThrow(() => assertNoDisallowedControls('\n', 't'));
  assert.doesNotThrow(() => assertNoDisallowedControls('\r', 't'));
});

test('control: empty string → allowed', () => {
  assert.doesNotThrow(() => assertNoDisallowedControls('', 't'));
});

test('control: DEL (0x7F) → allowed (not in current C0 range check)', () => {
  // DEL is 127 — above the < 32 check. Documenting current behaviour.
  assert.doesNotThrow(() => assertNoDisallowedControls('\x7f', 't'));
});

test('control: C1 controls (0x80–0x9F) → allowed (not checked)', () => {
  // C1 controls are above 32 — current policy does not reject them.
  for (let code = 0x80; code <= 0x9f; code++) {
    assert.doesNotThrow(
      () => assertNoDisallowedControls(String.fromCharCode(code), 't'),
      `C1 code 0x${code.toString(16)} should be allowed`,
    );
  }
});

test('control: mixed valid/invalid → throws on first invalid', () => {
  assert.throws(() => assertNoDisallowedControls('valid\x01invalid', 't'), /disallowed control/);
});

test('control: string ending with NUL → throws', () => {
  assert.throws(() => assertNoDisallowedControls('text\u0000', 't'), /NUL/);
});

// ═════════════════════════════════════════════════════════════════════════════
// normalizeAndValidateUserText — NFC normalisation
// ═════════════════════════════════════════════════════════════════════════════

test('NFC: NFD input (e + combining acute) → normalised to NFC é', () => {
  const result = normalizeAndValidateUserText('\u0065\u0301', 'text');
  assert.equal(result, '\u00e9');
  assert.equal(result.length, 1);
});

test('NFC: NFC input passes through unchanged', () => {
  const nfc = 'Héllo wörld';
  assert.equal(normalizeAndValidateUserText(nfc, 'text'), nfc);
});

test('NFC: Korean Hangul NFD decomposed → composed NFC', () => {
  // 각 in NFD = ᄀ + ᅡ + ᆨ (U+1100 U+1161 U+11A8) → NFC = U+AC01
  const nfd = '\u1100\u1161\u11A8';
  const result = normalizeAndValidateUserText(nfd, 'text');
  assert.equal(result, '\uAC01');
});

test('NFC: plain ASCII passes unchanged', () => {
  const s = 'Hello, world!\t123\n';
  assert.doesNotThrow(() => normalizeAndValidateUserText(s, 'text'));
  assert.equal(normalizeAndValidateUserText(s, 'text'), s);
});

test('NFC: empty string passes', () => {
  assert.equal(normalizeAndValidateUserText('', 'text'), '');
});

test('NFC: multilingual clinical text (accents, CJK, Arabic) passes', () => {
  const s = 'Patient: Müller, José. Diagnosis: 正常 (normal). ملاحظات: لا يوجد.';
  assert.doesNotThrow(() => normalizeAndValidateUserText(s, 'text'));
});

test('NFC: emoji and supplementary planes pass', () => {
  assert.doesNotThrow(() => normalizeAndValidateUserText('😀 🎉 🔐 text', 'text'));
});

// ═════════════════════════════════════════════════════════════════════════════
// normalizeAndValidateUserText — bidi and invisible char rejection
// ═════════════════════════════════════════════════════════════════════════════

test('bidi: LTR mark U+200E rejected', () => {
  assert.throws(() => normalizeAndValidateUserText('hello\u200eworld', 'text'), /bidi/i);
});

test('bidi: RTL mark U+200F rejected', () => {
  assert.throws(() => normalizeAndValidateUserText('hello\u200fworld', 'text'), /bidi/i);
});

test('bidi: LTR embedding U+202A rejected', () => {
  assert.throws(() => normalizeAndValidateUserText('\u202a', 'text'), /bidi/i);
});

test('bidi: RTL embedding U+202B rejected', () => {
  assert.throws(() => normalizeAndValidateUserText('\u202b', 'text'), /bidi/i);
});

test('bidi: pop directional formatting U+202C rejected', () => {
  assert.throws(() => normalizeAndValidateUserText('\u202c', 'text'), /bidi/i);
});

test('bidi: LTR override U+202D rejected', () => {
  assert.throws(() => normalizeAndValidateUserText('\u202d', 'text'), /bidi/i);
});

test('bidi: RTL override U+202E rejected', () => {
  assert.throws(() => normalizeAndValidateUserText('ab\u202ecd', 'text'), /bidi/i);
});

test('bidi: LRI isolate U+2066 rejected', () => {
  assert.throws(() => normalizeAndValidateUserText('\u2066text\u2069', 'text'), /bidi/i);
});

test('bidi: RLI isolate U+2067 rejected', () => {
  assert.throws(() => normalizeAndValidateUserText('\u2067', 'text'), /bidi/i);
});

test('bidi: FSI isolate U+2068 rejected', () => {
  assert.throws(() => normalizeAndValidateUserText('\u2068', 'text'), /bidi/i);
});

test('bidi: PDI (pop directional isolate) U+2069 rejected', () => {
  assert.throws(() => normalizeAndValidateUserText('\u2069', 'text'), /bidi/i);
});

test('bidi: zero-width space U+200B rejected', () => {
  assert.throws(() => normalizeAndValidateUserText('zero\u200bwidth', 'text'), /bidi/i);
});

test('bidi: zero-width non-joiner U+200C rejected', () => {
  assert.throws(() => normalizeAndValidateUserText('a\u200cb', 'text'), /bidi/i);
});

test('bidi: zero-width joiner U+200D rejected', () => {
  assert.throws(() => normalizeAndValidateUserText('a\u200db', 'text'), /bidi/i);
});

test('bidi: word joiner U+2060 rejected', () => {
  assert.throws(() => normalizeAndValidateUserText('a\u2060b', 'text'), /bidi/i);
});

test('bidi: BOM U+FEFF at position 0 allowed', () => {
  assert.doesNotThrow(() => normalizeAndValidateUserText('\ufeffok', 'text'));
});

test('bidi: BOM U+FEFF at position > 0 rejected', () => {
  assert.throws(() => normalizeAndValidateUserText('abc\ufeffdef', 'text'), /bidi/i);
});

test('bidi: language tag U+E0000 rejected', () => {
  assert.throws(() => normalizeAndValidateUserText('a\u{E0000}b', 'text'), /bidi/i);
});

test('bidi: language tag U+E007F rejected', () => {
  assert.throws(() => normalizeAndValidateUserText('a\u{E007F}b', 'text'), /bidi/i);
});

test('bidi: NUL still rejected after NFC normalisation', () => {
  assert.throws(() => normalizeAndValidateUserText('a\u0000b', 'text'));
});

test('bidi: bidi char at end of string → still caught', () => {
  assert.throws(() => normalizeAndValidateUserText('text ends with bidi\u200e', 'text'), /bidi/i);
});

// ═════════════════════════════════════════════════════════════════════════════
// validateUserText
// ═════════════════════════════════════════════════════════════════════════════

test('validateUserText: empty string passes', () => {
  assert.doesNotThrow(() => validateUserText('', 'text'));
});

test('validateUserText: exactly at MAX_TEXT_CHARS passes', () => {
  assert.doesNotThrow(() => validateUserText('x'.repeat(MAX_TEXT_CHARS), 'text'));
});

test('validateUserText: one over MAX_TEXT_CHARS → throws', () => {
  assert.throws(() => validateUserText('x'.repeat(MAX_TEXT_CHARS + 1), 'text'), /exceeds maximum length/);
});

test('validateUserText: NFD input is normalised to NFC in return value', () => {
  const result = validateUserText('caf\u0065\u0301', 'text');
  assert.equal(result, 'café');
});

test('validateUserText: bidi char rejected even inside long string', () => {
  const s = 'a'.repeat(100) + '\u200e' + 'b'.repeat(100);
  assert.throws(() => validateUserText(s, 'text'), /bidi/i);
});

test('validateUserText: C0 control rejected', () => {
  assert.throws(() => validateUserText('hello\x02world', 'text'));
});

// ═════════════════════════════════════════════════════════════════════════════
// validateMetadata
// ═════════════════════════════════════════════════════════════════════════════

test('validateMetadata: undefined passes, returns undefined', () => {
  assert.equal(validateMetadata(undefined), undefined);
});

test('validateMetadata: empty string passes', () => {
  assert.doesNotThrow(() => validateMetadata(''));
});

test('validateMetadata: normal value passes', () => {
  assert.equal(validateMetadata('ok'), 'ok');
});

test('validateMetadata: exactly at MAX_METADATA_CHARS passes', () => {
  assert.doesNotThrow(() => validateMetadata('y'.repeat(MAX_METADATA_CHARS)));
});

test('validateMetadata: one over MAX_METADATA_CHARS → throws', () => {
  assert.throws(() => validateMetadata('y'.repeat(MAX_METADATA_CHARS + 1)), /exceeds maximum length/);
});

test('validateMetadata: bidi char rejected', () => {
  assert.throws(() => validateMetadata('meta\u202aval'), /bidi/i);
});

test('validateMetadata: C0 control rejected', () => {
  assert.throws(() => validateMetadata('label\x01value'));
});

test('validateMetadata: NFC normalisation applied', () => {
  const result = validateMetadata('\u0065\u0301'); // NFD é
  assert.equal(result, '\u00e9');
});

// ═════════════════════════════════════════════════════════════════════════════
// resolveIndexablePath — path confinement
// ═════════════════════════════════════════════════════════════════════════════

test('resolveIndexablePath: non-existent path → throws "does not exist"', () => {
  assert.throws(
    () => resolveIndexablePath('/this/path/absolutely/does/not/exist/9999'),
    /does not exist/,
  );
});

test('resolveIndexablePath: rejects null bytes in path string', () => {
  assert.throws(() => resolveIndexablePath('/tmp/foo\u0000bar'));
});

test('resolveIndexablePath: rejects ../../ traversal escaping cwd', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'logosdb-trav-'));
  const prev = process.cwd();
  process.chdir(tmp);
  try {
    assert.throws(() => resolveIndexablePath('../../etc/passwd'));
    assert.throws(() => resolveIndexablePath('../../../usr'));
  } finally {
    process.chdir(prev);
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test('resolveIndexablePath: rejects escape via symlink to outside dir', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'logosdb-sym-'));
  const inside = path.join(tmp, 'in');
  const outside = path.join(tmp, 'out');
  fs.mkdirSync(inside);
  fs.mkdirSync(outside);
  fs.writeFileSync(path.join(outside, 'secret.txt'), 'secret', 'utf8');
  fs.symlinkSync(outside, path.join(inside, 'bad'), 'dir');

  const prev = process.cwd();
  process.chdir(inside);
  try {
    assert.throws(() => resolveIndexablePath('bad/secret.txt'));
  } finally {
    process.chdir(prev);
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test('resolveIndexablePath: rejects double symlink escape', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'logosdb-sym2-'));
  const inside = path.join(tmp, 'in');
  const middle = path.join(tmp, 'middle');
  const outside = path.join(tmp, 'out');
  fs.mkdirSync(inside);
  fs.mkdirSync(middle);
  fs.mkdirSync(outside);
  fs.writeFileSync(path.join(outside, 'target.txt'), 'secret', 'utf8');
  // inside/hop → middle, middle/hop2 → outside
  fs.symlinkSync(middle, path.join(inside, 'hop'), 'dir');
  fs.symlinkSync(outside, path.join(middle, 'hop2'), 'dir');

  const prev = process.cwd();
  process.chdir(inside);
  try {
    assert.throws(() => resolveIndexablePath('hop/hop2/target.txt'));
  } finally {
    process.chdir(prev);
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test('resolveIndexablePath: file inside cwd passes', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'logosdb-ok-'));
  const file = path.join(tmp, 'good.txt');
  fs.writeFileSync(file, 'ok', 'utf8');

  const prev = process.cwd();
  process.chdir(tmp);
  try {
    const resolved = resolveIndexablePath('good.txt');
    assert.ok(typeof resolved === 'string');
    assert.ok(resolved.endsWith('good.txt'));
  } finally {
    process.chdir(prev);
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test('resolveIndexablePath: directory inside cwd passes', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'logosdb-dir-'));
  const sub = path.join(tmp, 'subdir');
  fs.mkdirSync(sub);

  const prev = process.cwd();
  process.chdir(tmp);
  try {
    const resolved = resolveIndexablePath('subdir');
    assert.ok(resolved.endsWith('subdir'));
  } finally {
    process.chdir(prev);
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

// ═════════════════════════════════════════════════════════════════════════════
// collectFilesSafe — recursive file walking
// ═════════════════════════════════════════════════════════════════════════════

test('collectFilesSafe: returns .ts file within root', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'logosdb-walk-'));
  fs.writeFileSync(path.join(tmp, 'a.ts'), '// hi', 'utf8');
  const files = collectFilesSafe(tmp);
  assert.ok(files.some((f) => f.endsWith('a.ts')));
  fs.rmSync(tmp, { recursive: true, force: true });
});

test('collectFilesSafe: empty directory → []', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'logosdb-empty-'));
  const files = collectFilesSafe(tmp);
  assert.deepEqual(files, []);
  fs.rmSync(tmp, { recursive: true, force: true });
});

test('collectFilesSafe: non-indexable extension (.exe .bin .png) → not included', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'logosdb-ext-'));
  for (const name of ['binary.exe', 'image.png', 'data.bin', 'archive.zip']) {
    fs.writeFileSync(path.join(tmp, name), 'bytes', 'utf8');
  }
  fs.writeFileSync(path.join(tmp, 'ok.ts'), '// ok', 'utf8');
  const files = collectFilesSafe(tmp);
  assert.ok(files.some((f) => f.endsWith('ok.ts')));
  for (const ext of ['.exe', '.png', '.bin', '.zip']) {
    assert.ok(!files.some((f) => f.endsWith(ext)), `${ext} should be excluded`);
  }
  fs.rmSync(tmp, { recursive: true, force: true });
});

test('collectFilesSafe: all SKIP_DIRS are excluded', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'logosdb-skip-'));
  const skipDirs = ['node_modules', '.git', '.venv', '__pycache__', '.next', 'dist', 'build', 'out', 'coverage', '.turbo'];
  for (const dir of skipDirs) {
    fs.mkdirSync(path.join(tmp, dir));
    fs.writeFileSync(path.join(tmp, dir, 'evil.ts'), 'bad', 'utf8');
  }
  fs.writeFileSync(path.join(tmp, 'ok.ts'), '// ok', 'utf8');
  const files = collectFilesSafe(tmp);
  for (const dir of skipDirs) {
    assert.ok(!files.some((f) => f.includes(dir)), `${dir} should be skipped`);
  }
  assert.ok(files.some((f) => f.endsWith('ok.ts')));
  fs.rmSync(tmp, { recursive: true, force: true });
});

test('collectFilesSafe: hidden directories (dot-prefix) are skipped', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'logosdb-hidden-'));
  fs.mkdirSync(path.join(tmp, '.hidden'));
  fs.writeFileSync(path.join(tmp, '.hidden', 'file.ts'), 'bad', 'utf8');
  fs.writeFileSync(path.join(tmp, 'visible.ts'), '// ok', 'utf8');
  const files = collectFilesSafe(tmp);
  assert.ok(!files.some((f) => f.includes('.hidden')));
  assert.ok(files.some((f) => f.endsWith('visible.ts')));
  fs.rmSync(tmp, { recursive: true, force: true });
});

test('collectFilesSafe: deeply nested directories are walked', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'logosdb-deep-'));
  const deep = path.join(tmp, 'a', 'b', 'c', 'd', 'e');
  fs.mkdirSync(deep, { recursive: true });
  fs.writeFileSync(path.join(deep, 'deep.ts'), '// deep', 'utf8');
  const files = collectFilesSafe(tmp);
  assert.ok(files.some((f) => f.endsWith('deep.ts')));
  fs.rmSync(tmp, { recursive: true, force: true });
});

test('collectFilesSafe: multiple indexable extensions in one tree are all collected', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'logosdb-multi-'));
  const exts = ['.ts', '.py', '.go', '.md', '.rs', '.json', '.yaml'];
  for (const ext of exts) {
    fs.writeFileSync(path.join(tmp, `file${ext}`), 'content', 'utf8');
  }
  const files = collectFilesSafe(tmp);
  for (const ext of exts) {
    assert.ok(files.some((f) => f.endsWith(ext)), `${ext} should be collected`);
  }
  fs.rmSync(tmp, { recursive: true, force: true });
});

test('collectFilesSafe: symlink to file inside root is collected', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'logosdb-slink-'));
  const real = path.join(tmp, 'real.ts');
  const link = path.join(tmp, 'link.ts');
  fs.writeFileSync(real, '// real', 'utf8');
  fs.symlinkSync(real, link, 'file');
  const files = collectFilesSafe(tmp);
  // Either the real file or the symlink (or both) should appear
  assert.ok(files.some((f) => f.endsWith('real.ts') || f.endsWith('link.ts')));
  fs.rmSync(tmp, { recursive: true, force: true });
});

test('collectFilesSafe: symlink to directory outside root is not followed', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'logosdb-slinkout-'));
  const inside = path.join(tmp, 'in');
  const outside = path.join(tmp, 'out');
  fs.mkdirSync(inside);
  fs.mkdirSync(outside);
  fs.writeFileSync(path.join(outside, 'secret.ts'), 'secret', 'utf8');
  fs.symlinkSync(outside, path.join(inside, 'escape'), 'dir');
  const files = collectFilesSafe(inside);
  assert.ok(!files.some((f) => f.includes('secret.ts')), 'symlink escape should be blocked');
  fs.rmSync(tmp, { recursive: true, force: true });
});

test('collectFilesSafe: all returned paths are absolute', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'logosdb-abspaths-'));
  fs.writeFileSync(path.join(tmp, 'check.ts'), '// x', 'utf8');
  const files = collectFilesSafe(tmp);
  for (const f of files) {
    assert.ok(path.isAbsolute(f), `Path is not absolute: ${f}`);
  }
  fs.rmSync(tmp, { recursive: true, force: true });
});

test('collectFilesSafe: all returned paths exist on disk', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'logosdb-exist-'));
  fs.writeFileSync(path.join(tmp, 'a.ts'), '// a', 'utf8');
  fs.writeFileSync(path.join(tmp, 'b.py'), '# b', 'utf8');
  const files = collectFilesSafe(tmp);
  for (const f of files) {
    assert.ok(fs.existsSync(f), `Returned path does not exist: ${f}`);
  }
  fs.rmSync(tmp, { recursive: true, force: true });
});
