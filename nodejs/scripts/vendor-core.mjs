#!/usr/bin/env node
/**
 * Copy C++ core + headers into nodejs/deps/core so the npm tarball is self-contained
 * (binding.gyp must not reference parent ../ paths outside the published package).
 *
 * Run from repo root: node nodejs/scripts/vendor-core.mjs
 * Or from nodejs/: npm run vendor-core
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const NODEJS_ROOT = path.join(__dirname, '..');
const REPO_ROOT = path.join(NODEJS_ROOT, '..');
const DEST = path.join(NODEJS_ROOT, 'deps', 'core');

function rmrf(p) {
  fs.rmSync(p, { recursive: true, force: true });
}

function cp(src, dst) {
  fs.mkdirSync(path.dirname(dst), { recursive: true });
  fs.copyFileSync(src, dst);
}

function cpDir(srcDir, dstDir, filter) {
  fs.mkdirSync(dstDir, { recursive: true });
  for (const ent of fs.readdirSync(srcDir, { withFileTypes: true })) {
    const s = path.join(srcDir, ent.name);
    const d = path.join(dstDir, ent.name);
    if (filter && !filter(s, ent)) continue;
    if (ent.isDirectory()) cpDir(s, d, filter);
    else if (ent.isFile()) cp(s, d);
  }
}

function main() {
  const includeSrc = path.join(REPO_ROOT, 'include');
  const srcDir = path.join(REPO_ROOT, 'src');
  const nlohmann = path.join(REPO_ROOT, 'third_party', 'nlohmann', 'json.hpp');
  const hnswDir = path.join(REPO_ROOT, 'third_party', 'hnswlib', 'hnswlib');

  if (!fs.existsSync(includeSrc) || !fs.existsSync(srcDir)) {
    console.error(
      '[vendor-core] Expected monorepo layout (include/, src/) next to nodejs/. Skip if publishing from incomplete tree.',
    );
    process.exit(1);
  }

  rmrf(DEST);
  cpDir(includeSrc, path.join(DEST, 'include'));
  fs.mkdirSync(path.join(DEST, 'src'), { recursive: true });
  for (const f of fs.readdirSync(srcDir)) {
    if (f.endsWith('.cpp') || f.endsWith('.h')) {
      cp(path.join(srcDir, f), path.join(DEST, 'src', f));
    }
  }
  const nlohmannDst = path.join(DEST, 'third_party', 'nlohmann', 'json.hpp');
  fs.mkdirSync(path.dirname(nlohmannDst), { recursive: true });
  fs.copyFileSync(nlohmann, nlohmannDst);
  cpDir(hnswDir, path.join(DEST, 'third_party', 'hnswlib', 'hnswlib'));
  console.log('[vendor-core] Wrote', DEST);
}

main();
