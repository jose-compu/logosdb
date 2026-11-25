#!/usr/bin/env bash
# Exit 0 when every version field matches the core (pyproject) and MCP versions.
# Usage: ./scripts/check_versions.sh   (run from repo root)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

X="$(grep -m1 '^version' pyproject.toml | cut -d'"' -f2)"
A="$(node -p "require('./mcp/package.json').version")"
fail() { echo "MISMATCH: $1"; exit 1; }

grep -q "project(logosdb VERSION $X " CMakeLists.txt || fail CMakeLists.txt
grep -q "LOGOSDB_VERSION_STRING \"$X\"" include/logosdb/logosdb.h || fail logosdb.h
grep -q "LOGOSDB_VERSION_STRING \"$X\"" nodejs/deps/core/include/logosdb/logosdb.h || fail "deps/core (run vendor-core)"
[ "$(node -p "require('./nodejs/package.json').version")" = "$X" ] \
  || fail nodejs/package.json
node -e "const p=require('./mcp/package.json');const want='^'+process.argv[1].replace(/\\.\\d+\$/,'.0');process.exit(p.dependencies.logosdb===want?0:1)" "$X" \
  || fail "mcp logosdb range"
[ "$(jq -r .version mcp/server.json)" = "$A" ] \
  && [ "$(jq -r '.packages[0].version' mcp/server.json)" = "$A" ] \
  || fail mcp/server.json
[ "$(jq -r '.packages.nodejs.version' package-lock.json)" = "$X" ] \
  && [ "$(jq -r '.packages.mcp.version' package-lock.json)" = "$A" ] \
  || fail "package-lock (run npm install)"

exit 0