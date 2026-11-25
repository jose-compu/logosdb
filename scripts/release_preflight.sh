#!/usr/bin/env bash
# Release go/no-go before starting version bumps (no GitHub REST API).
# Usage: ./scripts/release_preflight.sh [logosdb]
#        ./scripts/release_preflight.sh --force logosdb
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
  shift
fi

PROJECT="${1:-logosdb}"
if [[ "$PROJECT" != "logosdb" ]]; then
  echo "Only logosdb is supported." >&2
  exit 2
fi

REPO="jose-compu/logosdb"
ATOM="https://github.com/${REPO}/releases.atom"
MIN_DAYS=6

reasons=()
pass() { echo "  + $*"; }
warn() { reasons+=("$*"); echo "  - $*"; }

echo "=== release_preflight (${PROJECT}) ==="

# Current versions on main (expected live before bump)
V="$(grep -m1 '^version' pyproject.toml | cut -d'"' -f2)"
M="$(node -p "require('./mcp/package.json').version")"
echo "Main declares core ${V}, MCP ${M}"

# 6-day rule from Atom <updated> of newest entry
atom_xml="$(curl -sf "$ATOM" || true)"
if [[ -z "$atom_xml" ]]; then
  warn "Could not fetch releases.atom"
else
  latest_updated="$(echo "$atom_xml" | grep -m1 '<updated>' | sed 's:.*<updated>\([^<]*\)</updated>.*:\1:' || true)"
  latest_title="$(echo "$atom_xml" | sed -n 's:.*<title>\([^<]*\)</title>.*:\1:p' | sed -n '2p' || true)"
  if [[ -n "$latest_updated" ]]; then
    now_epoch="$(date -u +%s)"
    rel_epoch="$(date -u -d "$latest_updated" +%s 2>/dev/null || date -u -j -f "%Y-%m-%dT%H:%M:%SZ" "$latest_updated" +%s 2>/dev/null || echo 0)"
    if [[ "$rel_epoch" -gt 0 ]]; then
      age_days=$(( (now_epoch - rel_epoch) / 86400 ))
      echo "Latest release feed: ${latest_title} (${latest_updated}, ${age_days}d ago)"
      if [[ "$age_days" -lt "$MIN_DAYS" && "$FORCE" -eq 0 ]]; then
        warn "Last release is only ${age_days}d old (minimum ${MIN_DAYS}d; use --force to override)"
      else
        pass "6-day rule satisfied (${age_days}d >= ${MIN_DAYS} or --force)"
      fi
    else
      warn "Could not parse release date: ${latest_updated}"
    fi
  fi
fi

# In-flight release/* on origin: tip not merged to main (open release work)
git fetch origin main --quiet 2>/dev/null || true
inflight_count=0
while read -r _sha ref; do
  [[ -z "${ref:-}" ]] && continue
  br="${ref#refs/heads/}"
  if git merge-base --is-ancestor "origin/${br}" origin/main 2>/dev/null; then
    continue
  fi
  warn "Unmerged release branch on origin: ${br}"
  inflight_count=$((inflight_count + 1))
done < <(git ls-remote --heads origin 'refs/heads/release/*' 2>/dev/null || true)
if [[ "$inflight_count" -eq 0 ]]; then
  pass "No unmerged release/* branches on origin"
fi

# Previous version fully published
echo "Checking registries for main versions (must match live)..."
if ! "$ROOT/scripts/verify_published.sh" "v${V}" "${M}"; then
  warn "Published check failed for v${V} / MCP ${M} (fix before a new release)"
fi

echo ""
if [[ ${#reasons[@]} -eq 0 ]]; then
  echo "GO"
  exit 0
fi
echo "NO-GO"
for r in "${reasons[@]}"; do
  echo "  * $r"
done
exit 1