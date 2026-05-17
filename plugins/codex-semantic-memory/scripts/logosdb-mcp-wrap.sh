#!/usr/bin/env bash
# logosdb-mcp-wrap.sh — launch logosdb-mcp-server with the correct LOGOSDB_PATH.
#
# Memory scope is controlled by LOGOS_MEMORY_MODE (default: global).
#
#   global  (default) — all sessions share ~/.codex/.logosdb
#                        Turn records and searches are visible across ALL projects.
#   project            — storage is isolated to <project-root>/.logosdb
#                        Each repository has its own independent memory.
#
# Set in shell profile:
#   export LOGOS_MEMORY_MODE=project     # per-repo isolation
#   (omit / set to "global" for the default cross-project shared memory)

set -euo pipefail

MODE="${LOGOS_MEMORY_MODE:-global}"

if [ "$MODE" = "project" ]; then
  # Use the nearest git root, falling back to cwd
  PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
  export LOGOSDB_PATH="${PROJECT_ROOT}/.logosdb"
else
  # Global: shared across all projects — default
  export LOGOSDB_PATH="${HOME}/.codex/.logosdb"
fi

# Locate logosdb-mcp-server: installed globally, or fall back to npx
if command -v logosdb-mcp-server >/dev/null 2>&1; then
  exec logosdb-mcp-server
else
  exec npx --yes logosdb-mcp-server
fi
