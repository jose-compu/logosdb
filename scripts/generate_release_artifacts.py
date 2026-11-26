#!/usr/bin/env python3
"""
Generate release copy from a single JSON file (releases/vX.Y.Z.json).

Usage:
  python3 scripts/generate_release_artifacts.py releases/v1.5.0.json --target all
  python3 scripts/generate_release_artifacts.py releases/v1.5.0.json --target changelog
  python3 scripts/generate_release_artifacts.py releases/v1.5.0.json --target readme-table
  python3 scripts/generate_release_artifacts.py releases/v1.5.0.json --target release-body
  python3 scripts/generate_release_artifacts.py releases/v1.5.0.json --target banner-yaml-snippet
  python3 scripts/generate_release_artifacts.py releases/v1.5.0.json --target website-json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from textwrap import dedent

TARGETS = (
    "changelog",
    "readme-table",
    "release-body",
    "banner-yaml-snippet",
    "website-json",
    "all",
)


def load_cfg(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    for key in (
        "core_version",
        "mcp_version",
        "date",
        "title",
        "prev_tag",
        "change_summary",
    ):
        if key not in cfg:
            raise SystemExit(f"Missing required field: {key}")
    cfg.setdefault("n8n_version", "1.0.3")
    cfg.setdefault("release_kind", "Minor")
    cfg.setdefault("issue", None)
    cfg.setdefault("issue_title", cfg["title"])
    cfg.setdefault("bindings_note", "")
    cfg.setdefault("tagline", "Embedded vector DB for AI agents and RAG")
    cfg.setdefault("release_tagline", "")
    cfg.setdefault("highlights", [])
    cfg.setdefault("install", "pip install logosdb   ·   npm i logosdb")
    cfg.setdefault("github", "github.com/jose-compu/logosdb")
    return cfg


def mcp_range(core: str) -> str:
    parts = core.split(".")
    return f"^{parts[0]}.{parts[1]}.0"


def changelog(cfg: dict) -> str:
    c, m, d = cfg["core_version"], cfg["mcp_version"], cfg["date"]
    issue = cfg.get("issue")
    title = cfg.get("issue_title", cfg["title"])
    prefix = f"**#{issue} — {title}.**" if issue else f"**{title}.**"
    body = cfg["change_summary"].strip()
    bindings = cfg.get("bindings_note", "").strip()
    lines = [
        f"{c}  ({d}) — core / PyPI / npm `logosdb`",
        "-" * len(f"{c}  ({d}) — core / PyPI / npm `logosdb`"),
        "",
        f"  * {prefix} {body}",
    ]
    if bindings:
        lines.append(f"    {bindings}")
    lines.append(f"  * Ships with `logosdb-mcp-server` **{m}** (depends on `logosdb` {mcp_range(c)}).")
    lines.append("")
    lines.append(f"{m}  ({d}) — `logosdb-mcp-server` only")
    lines.append("-" * 47)
    lines.append("")
    lines.append(
        f"  * Depend on `logosdb` **{mcp_range(c)}** "
        "(npm publish workflow in `.github/workflows/publish.yml`)."
    )
    lines.append("")
    return "\n".join(lines)


def readme_table(cfg: dict) -> str:
    c, m, n = cfg["core_version"], cfg["mcp_version"], cfg["n8n_version"]
    return dedent(
        f"""\
        | Package | Version | Install |
        |---------|---------|---------|
        | [logosdb](https://pypi.org/project/logosdb/) (Python / PyPI) | {c} | `pip install logosdb=={c}` |
        | [logosdb](https://www.npmjs.com/package/logosdb) (Node.js) | {c} | `npm install logosdb@{c}` |
        | [logosdb-mcp-server](https://www.npmjs.com/package/logosdb-mcp-server) | {m} | `npm install logosdb-mcp-server@{m}` |
        | [n8n-nodes-logosdb](https://www.npmjs.com/package/n8n-nodes-logosdb) | {n} | `npm install n8n-nodes-logosdb@{n}` |
        """
    ).strip() + "\n"


def release_body(cfg: dict) -> str:
    c, m = cfg["core_version"], cfg["mcp_version"]
    prev = cfg["prev_tag"].lstrip("v")
    kind = cfg.get("release_kind", "Minor")
    issue = cfg.get("issue")
    title = cfg.get("issue_title", cfg["title"])
    bullet = (
        f"- **#{issue} — {title}.** {cfg['change_summary'].strip()}"
        if issue
        else f"- **{title}.** {cfg['change_summary'].strip()}"
    )
    bindings = cfg.get("bindings_note", "").strip()
    if bindings:
        bullet += f" {bindings}"
    n8n = cfg["n8n_version"]
    return dedent(
        f"""\
        ## LogosDB {c}

        {kind} release: {cfg.get('release_tagline') or cfg['change_summary'].strip()}

        ### Packages

        | Package | Version |
        |---|---|
        | `logosdb` (PyPI) | **{c}** |
        | `logosdb` (npm) | **{c}** |
        | `logosdb-mcp-server` (npm) | **{m}** (dep range `{mcp_range(c)}`) |
        | `n8n-nodes-logosdb` (npm) | {n8n} (unchanged, dep range `^…`) |

        ### Changes

        {bullet}

        **Full Changelog**: https://github.com/jose-compu/logosdb/compare/v{prev}...v{c}
        """
    ).strip() + "\n"


def banner_yaml_snippet(cfg: dict) -> str:
    c, m = cfg["core_version"], cfg["mcp_version"]
    lines = [
        f'version: "{c}"',
        f'title: "{cfg["title"]}"',
        f'date: "{cfg["date"]}"',
        f'tagline: "{cfg["tagline"]}"',
    ]
    if cfg.get("release_tagline"):
        lines.append(f'release_tagline: "{cfg["release_tagline"]}"')
    if cfg.get("highlights"):
        lines.append("highlights:")
        for h in cfg["highlights"]:
            if isinstance(h, str):
                lines.append(f'  - "{h}"')
            else:
                lines.append(f'  - name: "{h["name"]}"')
                lines.append(f'    desc: "{h["desc"]}"')
    lines.append("packages:")
    lines.append(f'  - {{ label: "PyPI logosdb",           version: "{c}" }}')
    lines.append(f'  - {{ label: "npm logosdb",            version: "{c}" }}')
    lines.append(f'  - {{ label: "npm logosdb-mcp-server", version: "{m}" }}')
    lines.append(f'install: "{cfg.get("install", "pip install logosdb   ·   npm i logosdb")}"')
    lines.append(f'github: "{cfg.get("github", "github.com/jose-compu/logosdb")}"')
    return "\n".join(lines) + "\n"


def website_json(cfg: dict) -> str:
    c, m, n = cfg["core_version"], cfg["mcp_version"], cfg["n8n_version"]
    summary = cfg["change_summary"].strip()
    headline = cfg.get("headline_html")
    if not headline:
        headline = dedent(
            f"""\
            <h2>{c} — {cfg['title']}</h2>
                    <p class="section-intro">
                      {summary}
                    </p>

                    <table class="pkg-table" aria-label="Published package versions">
                      <thead>
                        <tr><th>Package</th><th>Version</th><th>Registry</th></tr>
                      </thead>
                      <tbody>
                        <tr><td><code>logosdb</code></td><td><strong>{c}</strong></td><td>PyPI, npm</td></tr>
                        <tr><td><code>logosdb-mcp-server</code></td><td><strong>{m}</strong></td><td>npm (<code>{mcp_range(c)}</code>)</td></tr>
                        <tr><td><code>n8n-nodes-logosdb</code></td><td>{n}</td><td>npm (unchanged)</td></tr>
                      </tbody>
                    </table>

                    <a class="release-banner" href="https://github.com/jose-compu/logosdb/releases/tag/v{c}" target="_blank" rel="noopener">
                      <img src="https://github.com/jose-compu/logosdb/releases/download/v{c}/linkedin-v{c}.png" alt="LogosDB {c} release" width="1200" height="675" loading="lazy" />
                    </a>"""
        ).strip()
    payload = {
        "core_version": c,
        "mcp_version": m,
        "n8n_version": n,
        "headline_html": headline,
    }
    return json.dumps(payload, indent=2) + "\n"


def emit(target: str, cfg: dict) -> None:
    if target == "changelog":
        print(changelog(cfg), end="")
    elif target == "readme-table":
        print(readme_table(cfg), end="")
    elif target == "release-body":
        print(release_body(cfg), end="")
    elif target == "banner-yaml-snippet":
        print(banner_yaml_snippet(cfg), end="")
    elif target == "website-json":
        print(website_json(cfg), end="")
    else:
        raise SystemExit(f"Unknown target: {target}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path, help="releases/vX.Y.Z.json")
    ap.add_argument(
        "--target",
        choices=TARGETS,
        default="all",
        help="Which artifact to print (default: all)",
    )
    args = ap.parse_args()
    if not args.config.is_file():
        print(f"Missing {args.config}", file=sys.stderr)
        return 1
    cfg = load_cfg(args.config)
    if args.target == "all":
        for t in TARGETS:
            if t == "all":
                continue
            print(f"### {t}\n")
            emit(t, cfg)
            print()
    else:
        emit(args.target, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())