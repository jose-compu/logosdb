/**
 * Input validation for MCP tool boundaries (path confinement, size limits, control characters).
 * See SECURITY.md and GitHub issue #74.
 */

import * as fs from 'fs';
import * as path from 'path';

import {
  buildGitignoreContext,
  findGitRoot,
  isIgnoredByStack,
  maybePushLocalGitignore,
  type GitignoreContext,
  type IgnoreFrame,
} from './gitignore-walker';

/** Max UTF-16 code units for indexed text or search query. */
export const MAX_TEXT_CHARS = 512 * 1024;
export const MAX_METADATA_CHARS = 32 * 1024;
export const MAX_FILE_BYTES = 25 * 1024 * 1024;
export const CHUNK_SIZE_MIN = 128;
export const CHUNK_SIZE_MAX = 64 * 1024;
export const TOP_K_MIN = 1;
export const TOP_K_MAX = 500;

/** Same extension / directory policy as the indexer (single source). */
export const INDEXABLE_EXTENSIONS = new Set([
  '.ts',
  '.tsx',
  '.js',
  '.jsx',
  '.mjs',
  '.cjs',
  '.py',
  '.go',
  '.rs',
  '.java',
  '.c',
  '.cpp',
  '.h',
  '.hpp',
  '.cs',
  '.rb',
  '.php',
  '.swift',
  '.kt',
  '.scala',
  '.sh',
  '.bash',
  '.zsh',
  '.fish',
  '.md',
  '.rst',
  '.txt',
  '.toml',
  '.yaml',
  '.yml',
  '.json',
  '.sql',
  '.graphql',
  '.proto',
  '.env.example',
  '.cfg',
  '.ini',
]);

export const SKIP_DIRS = new Set([
  'node_modules',
  '.git',
  '.venv',
  '__pycache__',
  '.next',
  'dist',
  'build',
  'out',
  'coverage',
  '.turbo',
]);

function indexRoots(): string[] {
  const roots: string[] = [];
  const custom = process.env.LOGOSDB_INDEX_ROOT?.trim();
  if (custom) {
    roots.push(path.resolve(custom));
  }
  roots.push(path.resolve(process.cwd()));
  return roots;
}

function isInsideRoot(candidateReal: string, rootReal: string): boolean {
  const rel = path.relative(rootReal, candidateReal);
  if (rel === '') return true;
  return !rel.startsWith('..') && !path.isAbsolute(rel);
}

/**
 * Resolve a user-supplied path to a real path that must stay inside
 * `process.cwd()` or `LOGOSDB_INDEX_ROOT` (if set), after symlink resolution.
 */
export function resolveIndexablePath(userPath: string): string {
  const resolvedInput = path.resolve(userPath);
  if (!fs.existsSync(resolvedInput)) {
    throw new Error(`Path does not exist: ${userPath}`);
  }

  let candidateReal: string;
  try {
    candidateReal = fs.realpathSync.native(resolvedInput);
  } catch {
    throw new Error(`Cannot resolve path: ${userPath}`);
  }

  for (const root of indexRoots()) {
    let rootReal: string;
    try {
      rootReal = fs.realpathSync.native(root);
    } catch {
      continue;
    }
    if (isInsideRoot(candidateReal, rootReal)) {
      return candidateReal;
    }
  }

  throw new Error(
    'Path must be inside process.cwd() or LOGOSDB_INDEX_ROOT (if set). ' +
      'Paths that escape via symlinks are rejected.',
  );
}

export interface CollectFilesOptions {
  /**
   * When true, honour `.gitignore` (root + nested + `.git/info/exclude` + global
   * excludes) **in addition to** the static `SKIP_DIRS` / hidden / extension
   * filters. If no enclosing Git working tree is found (walking up to
   * `process.cwd()` / `LOGOSDB_INDEX_ROOT`), this option is a no-op.
   * Default: `false` (callers in `index.ts` derive a default per-call).
   */
  respectGitignore?: boolean;
}

export function collectFilesSafe(rootDirReal: string, options: CollectFilesOptions = {}): string[] {
  const rootReal = fs.realpathSync.native(rootDirReal);
  const results: string[] = [];

  let gitignore: GitignoreContext | null = null;
  if (options.respectGitignore) {
    const gitRoot = findGitRoot(rootReal, indexRoots());
    if (gitRoot !== null) {
      gitignore = buildGitignoreContext(gitRoot);
    }
  }

  function walk(dir: string, stack: IgnoreFrame[]): void {
    const localStack = gitignore ? maybePushLocalGitignore(stack, dir) : stack;

    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      if (entry.name.startsWith('.') && entry.isDirectory()) continue;
      if (SKIP_DIRS.has(entry.name)) continue;
      const full = path.join(dir, entry.name);

      if (gitignore && localStack.length > 0) {
        if (isIgnoredByStack(localStack, full, entry.isDirectory())) continue;
      }

      let entryReal: string;
      try {
        entryReal = fs.realpathSync.native(full);
      } catch {
        continue;
      }
      if (!isInsideRoot(entryReal, rootReal)) continue;

      if (entry.isDirectory()) {
        walk(full, localStack);
      } else if (entry.isFile() && INDEXABLE_EXTENSIONS.has(path.extname(entry.name))) {
        results.push(entryReal);
      }
    }
  }

  walk(rootDirReal, gitignore ? gitignore.rootStack : []);
  return results;
}

export function assertNoDisallowedControls(s: string, label: string): void {
  for (let i = 0; i < s.length; i++) {
    const code = s.charCodeAt(i);
    if (code === 0) {
      throw new Error(`${label} must not contain NUL`);
    }
    if (code < 32 && code !== 9 && code !== 10 && code !== 13) {
      throw new Error(`${label} contains disallowed control character`);
    }
  }
}

/**
 * Bidi override and invisible format code points that can make stored text
 * render differently than what the model sees.
 *
 * Rejected:
 *   U+200B ZERO WIDTH SPACE
 *   U+200C ZERO WIDTH NON-JOINER
 *   U+200D ZERO WIDTH JOINER
 *   U+200E LEFT-TO-RIGHT MARK
 *   U+200F RIGHT-TO-LEFT MARK
 *   U+202A–U+202E bidi embedding/override controls
 *   U+2060 WORD JOINER
 *   U+2066–U+2069 bidi isolate controls
 *   U+FEFF BOM / ZERO WIDTH NO-BREAK SPACE (when not at position 0)
 *   U+E0000–U+E007F language tag characters
 */
function assertNoBidiOrInvisible(s: string, label: string): void {
  for (let i = 0; i < s.length; i++) {
    const code = s.codePointAt(i) ?? 0;
    if (
      code === 0x200b || code === 0x200c || code === 0x200d ||
      code === 0x200e || code === 0x200f ||
      (code >= 0x202a && code <= 0x202e) ||
      code === 0x2060 ||
      (code >= 0x2066 && code <= 0x2069) ||
      (code === 0xfeff && i !== 0) ||
      (code >= 0xe0000 && code <= 0xe007f)
    ) {
      throw new Error(
        `${label} contains disallowed Unicode format/bidi character (U+${code.toString(16).toUpperCase().padStart(4, '0')})`,
      );
    }
    // Skip surrogate pair second code unit
    if (code > 0xffff) i++;
  }
}

/**
 * Normalize `s` to NFC and validate that it contains no disallowed control or
 * bidi/invisible Unicode code points.
 *
 * NFC normalization ensures that semantically identical strings (e.g. NFC vs
 * NFD decompositions of the same character) produce the same embedding input,
 * preventing duplicate "same" content from accumulating in the DB.
 */
export function normalizeAndValidateUserText(s: string, label: string): string {
  const normalized = s.normalize('NFC');
  assertNoDisallowedControls(normalized, label);
  assertNoBidiOrInvisible(normalized, label);
  return normalized;
}

export function validateUserText(s: string, label: string): string {
  if (s.length > MAX_TEXT_CHARS) {
    throw new Error(`${label} exceeds maximum length (${MAX_TEXT_CHARS} characters)`);
  }
  return normalizeAndValidateUserText(s, label);
}

export function validateMetadata(metadata: string | undefined): string | undefined {
  if (metadata === undefined) return undefined;
  if (metadata.length > MAX_METADATA_CHARS) {
    throw new Error(`metadata exceeds maximum length (${MAX_METADATA_CHARS} characters)`);
  }
  return normalizeAndValidateUserText(metadata, 'metadata');
}

export function clampChunkSize(raw: number, fallback: number): number {
  const base = Number.isFinite(raw) && raw > 0 ? Math.trunc(raw) : fallback;
  return Math.min(CHUNK_SIZE_MAX, Math.max(CHUNK_SIZE_MIN, base));
}

export function clampTopK(raw: number): number {
  const v = Number.isFinite(raw) ? Math.trunc(raw) : TOP_K_MIN;
  if (v < TOP_K_MIN || v > TOP_K_MAX) {
    throw new Error(`top_k must be between ${TOP_K_MIN} and ${TOP_K_MAX}`);
  }
  return v;
}

export function readFileBoundedUtf8(filePath: string): string {
  const st = fs.statSync(filePath);
  if (!st.isFile()) {
    throw new Error(`Not a regular file: ${filePath}`);
  }
  if (st.size > MAX_FILE_BYTES) {
    throw new Error(
      `File exceeds maximum size for indexing (${MAX_FILE_BYTES} bytes): ${filePath}`,
    );
  }
  return fs.readFileSync(filePath, 'utf-8');
}
