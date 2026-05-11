/**
 * Input validation for MCP tool boundaries (path confinement, size limits, control characters).
 * See SECURITY.md and GitHub issue #74.
 */

import * as fs from 'fs';
import * as path from 'path';

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

export function collectFilesSafe(rootDirReal: string): string[] {
  const rootReal = fs.realpathSync.native(rootDirReal);
  const results: string[] = [];

  function walk(dir: string): void {
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
      let entryReal: string;
      try {
        entryReal = fs.realpathSync.native(full);
      } catch {
        continue;
      }
      if (!isInsideRoot(entryReal, rootReal)) continue;

      if (entry.isDirectory()) {
        walk(full);
      } else if (entry.isFile() && INDEXABLE_EXTENSIONS.has(path.extname(entry.name))) {
        results.push(entryReal);
      }
    }
  }

  walk(rootDirReal);
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

export function validateUserText(s: string, label: string): string {
  if (s.length > MAX_TEXT_CHARS) {
    throw new Error(`${label} exceeds maximum length (${MAX_TEXT_CHARS} characters)`);
  }
  assertNoDisallowedControls(s, label);
  return s;
}

export function validateMetadata(metadata: string | undefined): string | undefined {
  if (metadata === undefined) return undefined;
  if (metadata.length > MAX_METADATA_CHARS) {
    throw new Error(`metadata exceeds maximum length (${MAX_METADATA_CHARS} characters)`);
  }
  assertNoDisallowedControls(metadata, 'metadata');
  return metadata;
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
