/**
 * `.gitignore`-aware directory walking for `logosdb_index_file` (issue #101).
 *
 * Composes ignore rules the way Git does:
 *   - the root `.gitignore` of the enclosing working tree,
 *   - every nested `.gitignore` discovered during the walk (per-directory scope),
 *   - `.git/info/exclude`,
 *   - the global excludes file (`$XDG_CONFIG_HOME/git/ignore`, falling back to
 *     `~/.config/git/ignore`).
 *
 * Pattern semantics are delegated to the `ignore` npm package (Git-compatible:
 * leading `/`, trailing `/`, `**`, negations with `!`, etc.).
 *
 * This module never reaches outside the resolved Git working tree.
 */

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

import ignoreFactory, { type Ignore } from 'ignore';

export interface IgnoreFrame {
  /** Absolute directory the patterns are scoped to (i.e. the dir holding the `.gitignore`). */
  dir: string;
  ig: Ignore;
}

export interface GitignoreContext {
  /** Absolute path to the enclosing Git working tree. */
  gitRoot: string;
  /** Stack of always-active frames (git root + global + info/exclude). */
  rootStack: IgnoreFrame[];
}

/** Walk parents looking for a `.git` dir/file. Stops at any `stopAt` boundary
 *  (use these to keep discovery inside `process.cwd()` / `LOGOSDB_INDEX_ROOT`). */
export function findGitRoot(startDir: string, stopAt: readonly string[] = []): string | null {
  let cur = path.resolve(startDir);
  const stops = stopAt.map((s) => path.resolve(s));
  while (true) {
    if (fs.existsSync(path.join(cur, '.git'))) {
      return cur;
    }
    if (stops.some((s) => path.relative(s, cur) === '')) {
      return null; // hit a configured boundary
    }
    const parent = path.dirname(cur);
    if (parent === cur) return null; // filesystem root
    cur = parent;
  }
}

function safeReadText(p: string): string | null {
  try {
    return fs.readFileSync(p, 'utf8');
  } catch {
    return null;
  }
}

function globalIgnoreFile(): string | null {
  // Prefer XDG; fall back to ~/.config/git/ignore. We do not parse ~/.gitconfig
  // for `core.excludesFile` (would require a child process); a documented
  // limitation noted in mcp/README.md.
  const xdg = process.env.XDG_CONFIG_HOME;
  if (xdg) {
    const p = path.join(xdg, 'git', 'ignore');
    if (fs.existsSync(p)) return p;
  }
  const home = os.homedir();
  if (home) {
    const p = path.join(home, '.config', 'git', 'ignore');
    if (fs.existsSync(p)) return p;
  }
  return null;
}

/** Build the always-active ignore stack for a given Git root.
 *  - Global excludes contribute a frame at `gitRoot` (Git treats them as
 *    repo-wide rules; testing them with a path relative to gitRoot is fine).
 *  - Repo `.gitignore` and `.git/info/exclude` likewise scope at gitRoot. */
export function buildGitignoreContext(gitRoot: string): GitignoreContext {
  const stack: IgnoreFrame[] = [];

  const ig = ignoreFactory();
  let added = false;

  const globalPath = globalIgnoreFile();
  if (globalPath) {
    const txt = safeReadText(globalPath);
    if (txt) {
      ig.add(txt);
      added = true;
    }
  }

  const repoGitignore = safeReadText(path.join(gitRoot, '.gitignore'));
  if (repoGitignore) {
    ig.add(repoGitignore);
    added = true;
  }

  const infoExclude = safeReadText(path.join(gitRoot, '.git', 'info', 'exclude'));
  if (infoExclude) {
    ig.add(infoExclude);
    added = true;
  }

  if (added) {
    stack.push({ dir: gitRoot, ig });
  }

  return { gitRoot, rootStack: stack };
}

/** Push a `.gitignore` from `dir` (if any) onto the stack. Returns the same
 *  stack reference when nothing was added, so callers can use identity to
 *  decide whether to allocate a child copy. */
export function maybePushLocalGitignore(stack: IgnoreFrame[], dir: string): IgnoreFrame[] {
  const txt = safeReadText(path.join(dir, '.gitignore'));
  if (txt === null) return stack;
  const ig = ignoreFactory();
  ig.add(txt);
  return [...stack, { dir, ig }];
}

/** Test a path (absolute) against every frame in the stack.
 *  Convention: each frame's `ig` receives the path **relative to that frame**,
 *  with a trailing `/` when it is a directory (Git semantics for `pattern/`).
 *  Cross-frame negations are not honoured (mirrors how globby / `ignore`'s
 *  composed usage already approximates Git for nested files). */
export function isIgnoredByStack(
  stack: readonly IgnoreFrame[],
  absPath: string,
  isDir: boolean,
): boolean {
  for (const frame of stack) {
    const rel = path.relative(frame.dir, absPath);
    if (!rel || rel.startsWith('..') || path.isAbsolute(rel)) continue;
    // ignore@^7 always wants POSIX separators.
    const norm = rel.split(path.sep).join('/') + (isDir ? '/' : '');
    if (frame.ig.ignores(norm)) return true;
  }
  return false;
}

/** Should `logosdb_index_file` apply gitignore filtering by default?
 *  Returns `true` if `LOGOSDB_RESPECT_GITIGNORE` is unset or truthy. */
export function respectGitignoreDefault(): boolean {
  const raw = process.env.LOGOSDB_RESPECT_GITIGNORE;
  if (raw === undefined || raw === '') return true;
  const v = raw.toLowerCase();
  return !(v === '0' || v === 'false' || v === 'no' || v === 'off');
}
