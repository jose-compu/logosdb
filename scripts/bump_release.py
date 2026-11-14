#!/usr/bin/env python3
"""
Bump the static GitHub Pages site (docs/index.html on the website branch).

Usage:
  python3 scripts/bump_release.py --config scripts/release.example.json
  python3 scripts/bump_release.py --dry-run --config my-release.json

The JSON config drives hero version, headline block (between HTML markers),
footer line, and MCP integration version. Edit flyer/feature cards manually
when the release theme changes materially (same as past 1.2.0 / 1.3.0 updates).

Markers in docs/index.html:
  - <p class="version-badge">…</p>
  - <!-- RELEASE_HEADLINE_START --> … <!-- RELEASE_HEADLINE_END -->
  - footer .footer-copy
  - integrations MCP @version in #integrations
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "index.html"


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def bump_index(html: str, cfg: dict) -> str:
    core = cfg["core_version"]
    mcp = cfg["mcp_version"]
    n8n = cfg.get("n8n_version", "1.0.3")

    html = re.sub(
        r'(<p class="version-badge">)[^<]+(</p>)',
        rf"\g<1>{core}\g<2>",
        html,
        count=1,
    )

    headline = cfg["headline_html"].strip()
    html = re.sub(
        r"<!-- RELEASE_HEADLINE_START -->.*?<!-- RELEASE_HEADLINE_END -->",
        f"<!-- RELEASE_HEADLINE_START -->\n{headline}\n        <!-- RELEASE_HEADLINE_END -->",
        html,
        flags=re.DOTALL,
    )

    html = re.sub(
        r'(<p class="footer-copy">)LogosDB [^·]+ · MCP [^·]+(?: · n8n [^·]+)? · Jose',
        rf"\g<1>LogosDB {core} · MCP {mcp} · n8n {n8n} · Jose",
        html,
        count=1,
    )

    html = re.sub(
        r"logosdb-mcp-server@\d+\.\d+\.\d+",
        f"logosdb-mcp-server@{mcp}",
        html,
    )

    return html


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True, help="Release JSON config")
    ap.add_argument(
        "--index",
        type=Path,
        default=INDEX,
        help=f"Path to index.html (default: {INDEX})",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print diff summary only")
    args = ap.parse_args()

    if not args.index.is_file():
        print(f"Missing {args.index}", file=sys.stderr)
        return 1

    cfg = load_config(args.config)
    for key in ("core_version", "mcp_version", "headline_html"):
        if key not in cfg:
            print(f"Config missing required key: {key}", file=sys.stderr)
            return 1

    original = args.index.read_text(encoding="utf-8")
    updated = bump_index(original, cfg)

    if updated == original:
        print("No changes applied.", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"Would update {args.index} for core {cfg['core_version']}")
        return 0

    args.index.write_text(updated, encoding="utf-8")
    print(f"Updated {args.index} for {cfg['core_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())