#!/usr/bin/env bash
# Wire the repo-tracked hooks under .githooks/ into the local clone.
# One-shot, idempotent.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

chmod +x .githooks/pre-commit
git config core.hooksPath .githooks

echo "[githooks] core.hooksPath -> .githooks (active hooks: $(ls .githooks | grep -v -E '^(install\.sh|README\.md)$' | xargs))"
echo "[githooks] skip once: LOGOSDB_SKIP_HOOK=1 git commit ...   (or: --no-verify)"
