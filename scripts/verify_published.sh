#!/usr/bin/env bash
# Verify a release is live on registries and the GitHub release asset (no GitHub REST API).
# Usage: ./scripts/verify_published.sh vX.Y.Z MCP_VERSION
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 vX.Y.Z MCP_VERSION" >&2
  exit 2
fi

TAG="$1"
MCP="$2"
CORE="${TAG#v}"
REPO="jose-compu/logosdb"
ATOM="https://github.com/${REPO}/releases.atom"
BANNER_URL="https://github.com/${REPO}/releases/download/${TAG}/linkedin-v${CORE}.png"
SITE="https://jose-compu.github.io/logosdb/"

failures=0
ok() { echo "OK: $*"; }
bad() { echo "FAIL: $*"; failures=$((failures + 1)); }

echo "=== verify_published ${TAG} (core ${CORE}, MCP ${MCP}) ==="

# PyPI
pypi_ver="$(curl -sf "https://pypi.org/pypi/logosdb/${CORE}/json" | jq -r .info.version 2>/dev/null || true)"
if [[ "$pypi_ver" == "$CORE" ]]; then
  nfiles="$(curl -sf "https://pypi.org/pypi/logosdb/${CORE}/json" | jq -r '.urls | length')"
  ok "PyPI logosdb ${CORE} (${nfiles} files)"
else
  bad "PyPI logosdb expected ${CORE}, got ${pypi_ver:-<missing>}"
fi

# npm core
npm_core="$(npm view "logosdb@${CORE}" version 2>/dev/null || true)"
if [[ "$npm_core" == "$CORE" ]]; then
  ok "npm logosdb@${CORE}"
else
  bad "npm logosdb expected ${CORE}, got ${npm_core:-<missing>}"
fi

# npm MCP
npm_mcp="$(npm view "logosdb-mcp-server@${MCP}" version 2>/dev/null || true)"
mcp_dep="$(npm view "logosdb-mcp-server@${MCP}" dependencies.logosdb 2>/dev/null || true)"
if [[ "$npm_mcp" == "$MCP" ]]; then
  ok "npm logosdb-mcp-server@${MCP} (dep ${mcp_dep})"
else
  bad "npm MCP expected ${MCP}, got ${npm_mcp:-<missing>}"
fi

# Release in Atom feed (title should mention tag)
atom_title="$(curl -sf "$ATOM" | sed -n 's:.*<title>\([^<]*\)</title>.*:\1:p' | sed -n '2p' || true)"
if echo "$atom_title" | grep -qE "(v)?${CORE}\b"; then
  ok "releases.atom lists ${CORE} (${atom_title})"
else
  bad "releases.atom missing release for ${CORE} (newest title: ${atom_title:-<none>})"
fi

# Banner asset (HTTP status only)
banner_status="$(curl -sI "$BANNER_URL" | head -1 | tr -d '\r')"
if echo "$banner_status" | grep -qE '200|302'; then
  ok "banner asset ${BANNER_URL}"
else
  bad "banner asset ${BANNER_URL}: ${banner_status:-no response}"
fi

# Live site version badge (cache-bust)
badge="$(curl -sf "${SITE}?v=$(date +%s)" | grep -oP '(?<=<p class="version-badge">)[^<]+' | head -1 || true)"
if [[ "$badge" == "$CORE" ]]; then
  ok "website badge ${CORE}"
else
  bad "website badge expected ${CORE}, got ${badge:-<missing>}"
fi

echo "=== done (${failures} failure(s)) ==="
[[ "$failures" -eq 0 ]]