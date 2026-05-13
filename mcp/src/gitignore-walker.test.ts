/**
 * Tests for `.gitignore`-aware directory walking (issue #101).
 * Each case builds a synthetic tree under a fresh tmpdir, optionally with a
 * `.git/` marker so the walker treats it as a working tree.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

import { collectFilesSafe } from './security';
import {
  buildGitignoreContext,
  findGitRoot,
  isIgnoredByStack,
  maybePushLocalGitignore,
  respectGitignoreDefault,
} from './gitignore-walker';

function mktmp(prefix = 'logosdb-gitignore-'): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

function write(p: string, body: string): void {
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.writeFileSync(p, body, 'utf8');
}

function markGit(root: string): void {
  fs.mkdirSync(path.join(root, '.git'), { recursive: true });
}

function cdGuard<T>(dir: string, fn: () => T): T {
  const prev = process.cwd();
  process.chdir(dir);
  try {
    return fn();
  } finally {
    process.chdir(prev);
  }
}

function basenames(files: string[]): Set<string> {
  return new Set(files.map((f) => path.basename(f)));
}

// ── findGitRoot ──────────────────────────────────────────────────────────────

test('findGitRoot returns enclosing .git dir', () => {
  const tmp = mktmp();
  try {
    markGit(tmp);
    write(path.join(tmp, 'src', 'a.ts'), '// hi');
    const got = findGitRoot(path.join(tmp, 'src'));
    assert.equal(got !== null && fs.realpathSync.native(got), fs.realpathSync.native(tmp));
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test('findGitRoot returns null when no .git is found', () => {
  const tmp = mktmp();
  try {
    write(path.join(tmp, 'a.ts'), '// hi');
    // Bound the search so it does not climb above tmp into a host repo.
    assert.equal(findGitRoot(tmp, [tmp]), null);
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test('findGitRoot stops at a configured boundary', () => {
  const tmp = mktmp();
  try {
    markGit(tmp); // .git lives at tmp
    const sub = path.join(tmp, 'sub');
    fs.mkdirSync(sub, { recursive: true });
    // Boundary = sub  -> walker must not see tmp/.git
    assert.equal(findGitRoot(sub, [sub]), null);
    // Without boundary, it finds tmp.
    const got = findGitRoot(sub);
    assert.equal(got !== null && fs.realpathSync.native(got), fs.realpathSync.native(tmp));
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

// ── Stack composition primitives ─────────────────────────────────────────────

test('isIgnoredByStack honours pattern semantics from a single .gitignore', () => {
  const tmp = mktmp();
  try {
    write(path.join(tmp, '.gitignore'), 'build/\n*.log\n!keep.log\n');
    const stack = maybePushLocalGitignore([], tmp);
    assert.equal(stack.length, 1);
    assert.equal(isIgnoredByStack(stack, path.join(tmp, 'build'), true), true);
    assert.equal(isIgnoredByStack(stack, path.join(tmp, 'app.log'), false), true);
    assert.equal(isIgnoredByStack(stack, path.join(tmp, 'keep.log'), false), false);
    assert.equal(isIgnoredByStack(stack, path.join(tmp, 'src', 'a.ts'), false), false);
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test('isIgnoredByStack composes nested .gitignore frames', () => {
  const tmp = mktmp();
  try {
    write(path.join(tmp, '.gitignore'), '*.tmp\n');
    write(path.join(tmp, 'sub', '.gitignore'), 'secret.txt\n');
    const root = maybePushLocalGitignore([], tmp);
    const nested = maybePushLocalGitignore(root, path.join(tmp, 'sub'));

    assert.equal(isIgnoredByStack(nested, path.join(tmp, 'sub', 'secret.txt'), false), true);
    assert.equal(isIgnoredByStack(nested, path.join(tmp, 'sub', 'foo.tmp'), false), true);
    assert.equal(isIgnoredByStack(nested, path.join(tmp, 'sub', 'foo.ts'), false), false);
    // Parent frame must not see the nested rule.
    assert.equal(isIgnoredByStack(root, path.join(tmp, 'sub', 'secret.txt'), false), false);
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

// ── End-to-end via collectFilesSafe ──────────────────────────────────────────

test('collectFilesSafe skips files matched by repo .gitignore', () => {
  const tmp = fs.realpathSync.native(mktmp());
  try {
    markGit(tmp);
    write(path.join(tmp, '.gitignore'), 'generated/\n*.log\n');
    write(path.join(tmp, 'src', 'a.ts'), '// hi');
    write(path.join(tmp, 'src', 'b.ts'), '// hi');
    write(path.join(tmp, 'generated', 'big.ts'), '// noise');
    write(path.join(tmp, 'app.log'), 'noise');
    write(path.join(tmp, 'README.md'), 'ok');

    cdGuard(tmp, () => {
      const got = basenames(collectFilesSafe(tmp, { respectGitignore: true }));
      assert.ok(got.has('a.ts'));
      assert.ok(got.has('b.ts'));
      assert.ok(got.has('README.md'));
      assert.ok(!got.has('big.ts'), 'generated/big.ts must be ignored');
      assert.ok(!got.has('app.log'), '*.log must be ignored');
    });
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test('collectFilesSafe honours nested .gitignore', () => {
  const tmp = fs.realpathSync.native(mktmp());
  try {
    markGit(tmp);
    write(path.join(tmp, '.gitignore'), '*.tmp\n');
    write(path.join(tmp, 'pkg', '.gitignore'), 'private/\n');
    write(path.join(tmp, 'pkg', 'a.ts'), '// keep');
    write(path.join(tmp, 'pkg', 'private', 'secret.ts'), '// hidden');
    write(path.join(tmp, 'pkg', 'note.tmp'), 'noise');

    cdGuard(tmp, () => {
      const got = basenames(collectFilesSafe(tmp, { respectGitignore: true }));
      assert.ok(got.has('a.ts'));
      assert.ok(!got.has('secret.ts'), 'nested private/ rule must apply');
      assert.ok(!got.has('note.tmp'), 'root *.tmp must apply to nested dir');
    });
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test('collectFilesSafe honours .git/info/exclude', () => {
  const tmp = fs.realpathSync.native(mktmp());
  try {
    markGit(tmp);
    write(path.join(tmp, '.git', 'info', 'exclude'), 'scratch.ts\n');
    write(path.join(tmp, 'keep.ts'), '// keep');
    write(path.join(tmp, 'scratch.ts'), '// drop');

    cdGuard(tmp, () => {
      const got = basenames(collectFilesSafe(tmp, { respectGitignore: true }));
      assert.ok(got.has('keep.ts'));
      assert.ok(!got.has('scratch.ts'), '.git/info/exclude must be honoured');
    });
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test('collectFilesSafe handles negation patterns (!keep)', () => {
  const tmp = fs.realpathSync.native(mktmp());
  try {
    markGit(tmp);
    write(path.join(tmp, '.gitignore'), '*.log\n!important.log\n');
    write(path.join(tmp, 'a.log'), 'drop'); // .log is not indexable, but
    write(path.join(tmp, 'a.ts'), '// keep');
    // Build a file that *is* indexable and that demonstrates negation effect:
    // *.md is allow, but if we ignore *.md and re-include keep.md, we should
    // see keep.md back.
    write(path.join(tmp, '.gitignore'), '*.md\n!keep.md\n');
    write(path.join(tmp, 'drop.md'), '# drop');
    write(path.join(tmp, 'keep.md'), '# keep');

    cdGuard(tmp, () => {
      const got = basenames(collectFilesSafe(tmp, { respectGitignore: true }));
      assert.ok(got.has('a.ts'));
      assert.ok(got.has('keep.md'), 'negation should re-include keep.md');
      assert.ok(!got.has('drop.md'), '*.md should drop drop.md');
    });
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test('collectFilesSafe with respectGitignore=true but no .git falls back', () => {
  const tmp = fs.realpathSync.native(mktmp());
  try {
    // No .git marker. .gitignore should be IGNORED (legacy behaviour preserved).
    write(path.join(tmp, '.gitignore'), '*.ts\n');
    write(path.join(tmp, 'a.ts'), '// kept under legacy filter');
    write(path.join(tmp, 'b.md'), 'kept');

    cdGuard(tmp, () => {
      const got = basenames(collectFilesSafe(tmp, { respectGitignore: true }));
      // Without a .git, the .gitignore is moot; .ts is indexable.
      assert.ok(got.has('a.ts'));
      assert.ok(got.has('b.md'));
    });
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test('collectFilesSafe legacy default (no opts) ignores .gitignore', () => {
  const tmp = fs.realpathSync.native(mktmp());
  try {
    markGit(tmp);
    write(path.join(tmp, '.gitignore'), '*.ts\n');
    write(path.join(tmp, 'a.ts'), '// would be dropped if gitignore were on');

    cdGuard(tmp, () => {
      // No opts -> legacy filter only. a.ts must be present.
      const got = basenames(collectFilesSafe(tmp));
      assert.ok(got.has('a.ts'), 'opt-in flag is required');
    });
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

// ── env override ─────────────────────────────────────────────────────────────

test('respectGitignoreDefault env override', () => {
  const prev = process.env.LOGOSDB_RESPECT_GITIGNORE;
  try {
    delete process.env.LOGOSDB_RESPECT_GITIGNORE;
    assert.equal(respectGitignoreDefault(), true);

    process.env.LOGOSDB_RESPECT_GITIGNORE = '';
    assert.equal(respectGitignoreDefault(), true);

    process.env.LOGOSDB_RESPECT_GITIGNORE = '1';
    assert.equal(respectGitignoreDefault(), true);

    for (const off of ['0', 'false', 'FALSE', 'no', 'off']) {
      process.env.LOGOSDB_RESPECT_GITIGNORE = off;
      assert.equal(respectGitignoreDefault(), false, `expected ${off} -> false`);
    }
  } finally {
    if (prev === undefined) delete process.env.LOGOSDB_RESPECT_GITIGNORE;
    else process.env.LOGOSDB_RESPECT_GITIGNORE = prev;
  }
});

// ── buildGitignoreContext is robust when files are absent ────────────────────

test('buildGitignoreContext with empty git root', () => {
  const tmp = fs.realpathSync.native(mktmp());
  try {
    markGit(tmp);
    const ctx = buildGitignoreContext(tmp);
    assert.equal(ctx.gitRoot, tmp);
    // No .gitignore and no info/exclude means no frames.
    assert.equal(ctx.rootStack.length, 0);
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});
