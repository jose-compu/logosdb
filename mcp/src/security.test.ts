import assert from 'node:assert/strict';
import { test } from 'node:test';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

import {
  assertNoDisallowedControls,
  clampChunkSize,
  clampTopK,
  collectFilesSafe,
  resolveIndexablePath,
  validateMetadata,
  validateUserText,
} from './security';

test('clampChunkSize bounds', () => {
  assert.equal(clampChunkSize(50, 800), 128);
  assert.equal(clampChunkSize(200000, 800), 64 * 1024);
  assert.equal(clampChunkSize(NaN, 900), 900);
});

test('clampTopK', () => {
  assert.equal(clampTopK(3), 3);
  assert.throws(() => clampTopK(0));
  assert.throws(() => clampTopK(501));
});

test('control characters rejected', () => {
  assert.throws(() => assertNoDisallowedControls('a\u0000b', 't'));
  assert.throws(() => assertNoDisallowedControls('a\u0001b', 't'));
  assert.doesNotThrow(() => assertNoDisallowedControls('a\tb\nc', 't'));
});

test('validateUserText length', () => {
  const big = 'x'.repeat(512 * 1024 + 1);
  assert.throws(() => validateUserText(big, 'text'));
});

test('validateMetadata optional', () => {
  assert.equal(validateMetadata(undefined), undefined);
  assert.equal(validateMetadata('ok'), 'ok');
});

test('resolveIndexablePath rejects escape via symlink', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'logosdb-mcp-sec-'));
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

test('collectFilesSafe stays under root', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'logosdb-mcp-walk-'));
  fs.writeFileSync(path.join(tmp, 'a.ts'), '// hi', 'utf8');
  const files = collectFilesSafe(tmp);
  assert.ok(files.some((f) => f.endsWith('a.ts')));
  fs.rmSync(tmp, { recursive: true, force: true });
});
