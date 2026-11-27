#!/usr/bin/env bash
# Launch logosdb-mcp-server 1.4.7+ with LOGOSDB_PATH and LOGOSDB_INDEX_ROOT.
#
# LOGOS_MEMORY_MODE (default: global)
#   global   — database at ~/.codex/.logosdb
#   project  — database at <project>/.logosdb
#
# Indexing is confined to LOGOSDB_INDEX_ROOT. When that variable is unset,
# the wrapper sets it to the git root of the current directory, or to the
# current directory when there is no git root. Codex should start this
# process with the session directory as cwd (do not set plugin cwd).

set -euo pipefail

PIN="1.4.7"

if [ -z "${LOGOSDB_INDEX_ROOT:-}" ]; then
  if git rev-parse --show-toplevel >/dev/null 2>&1; then
    LOGOSDB_INDEX_ROOT="$(git rev-parse --show-toplevel)"
  else
    LOGOSDB_INDEX_ROOT="${PWD}"
  fi
  export LOGOSDB_INDEX_ROOT
fi

if [ -z "${LOGOSDB_PATH:-}" ]; then
  MODE="${LOGOS_MEMORY_MODE:-global}"
  if [ "$MODE" = "project" ]; then
    export LOGOSDB_PATH="${LOGOSDB_INDEX_ROOT}/.logosdb"
  else
    export LOGOSDB_PATH="${HOME}/.codex/.logosdb"
  fi
fi

version_at_least() {
  # $1 = have, $2 = need. sort -V puts the older version first.
  [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n 1)" = "$2" ]
}

if [ -n "${LOGOSDB_MCP_BIN:-}" ]; then
  exec "$LOGOSDB_MCP_BIN"
fi

if command -v logosdb-mcp-server >/dev/null 2>&1; then
  got="$(logosdb-mcp-server --version 2>/dev/null | head -n 1 | tr -d '[:space:]' || true)"
  if [ -n "$got" ] && version_at_least "$got" "$PIN"; then
    exec logosdb-mcp-server
  fi
fi

exec npx --yes "logosdb-mcp-server@${PIN}"
