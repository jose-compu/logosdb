# Local git hooks

Repo-tracked hooks live here so every contributor gets the same `pre-commit` autofix without sharing `.git/hooks/`.

## Install (once per clone)

```bash
./.githooks/install.sh
```

That sets `core.hooksPath = .githooks` for **this clone only** (it never touches your global git config).

## What the hook does

`pre-commit` runs **`clang-format` 18.x** on staged `*.cpp` / `*.h` / `*.hpp` under `src/`, `include/`, `tests/`, `tools/`, **auto-fixes in place**, and **re-stages** any file it rewrote so the commit goes out already-formatted (matching CI: `.github/workflows/ci.yml`).

The hook prefers, in order:
1. `clang-format-18` on `$PATH`
2. `/opt/homebrew/opt/llvm@18/bin/clang-format` (macOS arm64, `brew install llvm@18`)
3. `/usr/local/opt/llvm@18/bin/clang-format` (macOS Intel)
4. Whatever `clang-format` is on `$PATH` (warns if it is not `18.x`)

## Skip a commit (rare)

```bash
LOGOSDB_SKIP_HOOK=1 git commit ...
# or:
git commit --no-verify ...
```

Use sparingly — CI will still fail if the resulting tree is not 18.x-clean.

## Why not the global `~/.git/hooks/`?

`core.hooksPath` is a local config knob; it does not affect any other repo on your machine.
