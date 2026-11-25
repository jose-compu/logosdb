#!/usr/bin/env python3
"""
Bump coordinated version fields, then vendor-core and root npm install.

Usage:
  python3 scripts/bump_version.py X.Y.Z A.B.C
  python3 scripts/bump_version.py --dry-run X.Y.Z A.B.C

Core X.Y.Z must be semver. MCP A.B.C sets mcp/package.json, server.json (x2),
and logosdb dependency range ^X.Y.0 (patch of core forced to .0).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_core(ver: str) -> tuple[int, int, int]:
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", ver)
    if not m:
        raise SystemExit(f"Invalid core version: {ver}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def logosdb_range(core: str) -> str:
    major, minor, _patch = parse_core(core)
    return f"^{major}.{minor}.0"


def bump_pyproject(path: Path, ver: str) -> str:
    text = path.read_text(encoding="utf-8")
    new, n = re.subn(
        r'^(version\s*=\s*)"[^"]+"',
        rf'\1"{ver}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise SystemExit("Could not update pyproject.toml version")
    return new


def bump_cmake(path: Path, ver: str) -> str:
    text = path.read_text(encoding="utf-8")
    new, n = re.subn(
        r"project\(logosdb VERSION \d+\.\d+\.\d+",
        f"project(logosdb VERSION {ver}",
        text,
        count=1,
    )
    if n != 1:
        raise SystemExit("Could not update CMakeLists.txt version")
    return new


def bump_logosdb_h(path: Path, ver: str) -> str:
    major, minor, patch = parse_core(ver)
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"#define LOGOSDB_VERSION_MAJOR \d+",
        f"#define LOGOSDB_VERSION_MAJOR {major}",
        text,
        count=1,
    )
    text = re.sub(
        r"#define LOGOSDB_VERSION_MINOR \d+",
        f"#define LOGOSDB_VERSION_MINOR {minor}",
        text,
        count=1,
    )
    text = re.sub(
        r"#define LOGOSDB_VERSION_PATCH \d+",
        f"#define LOGOSDB_VERSION_PATCH {patch}",
        text,
        count=1,
    )
    text = re.sub(
        r'#define LOGOSDB_VERSION_STRING "[^"]+"',
        f'#define LOGOSDB_VERSION_STRING "{ver}"',
        text,
        count=1,
    )
    return text


def bump_node_package(path: Path, ver: str) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["version"] = ver
    return json.dumps(data, indent=2) + "\n"


def bump_mcp_package(path: Path, core: str, mcp: str) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["version"] = mcp
    deps = data.setdefault("dependencies", {})
    deps["logosdb"] = logosdb_range(core)
    return json.dumps(data, indent=2) + "\n"


def bump_server_json(path: Path, mcp: str) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["version"] = mcp
    if data.get("packages") and len(data["packages"]) > 0:
        data["packages"][0]["version"] = mcp
    return json.dumps(data, indent=2) + "\n"


def write_or_preview(path: Path, new: str, dry_run: bool, label: str) -> None:
    old = path.read_text(encoding="utf-8") if path.is_file() else ""
    if old == new:
        print(f"  {label}: unchanged")
        return
    if dry_run:
        print(f"  {label}: would update")
        return
    path.write_text(new, encoding="utf-8")
    print(f"  {label}: updated")


def run_cmd(cmd: list[str], dry_run: bool) -> None:
    print(f"  $ {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("core_version", help="Core / PyPI / npm logosdb version X.Y.Z")
    ap.add_argument("mcp_version", help="logosdb-mcp-server version A.B.C")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned edits and commands only",
    )
    ap.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip vendor-core and npm install (also implied by --dry-run)",
    )
    args = ap.parse_args()

    core = args.core_version
    mcp = args.mcp_version
    dry = args.dry_run

    print(f"Bump core {core}, MCP {mcp}" + (" (dry-run)" if dry else ""))

    write_or_preview(
        ROOT / "pyproject.toml",
        bump_pyproject(ROOT / "pyproject.toml", core),
        dry,
        "pyproject.toml",
    )
    write_or_preview(
        ROOT / "CMakeLists.txt",
        bump_cmake(ROOT / "CMakeLists.txt", core),
        dry,
        "CMakeLists.txt",
    )
    write_or_preview(
        ROOT / "include/logosdb/logosdb.h",
        bump_logosdb_h(ROOT / "include/logosdb/logosdb.h", core),
        dry,
        "include/logosdb/logosdb.h",
    )
    write_or_preview(
        ROOT / "nodejs/package.json",
        bump_node_package(ROOT / "nodejs/package.json", core),
        dry,
        "nodejs/package.json",
    )
    write_or_preview(
        ROOT / "mcp/package.json",
        bump_mcp_package(ROOT / "mcp/package.json", core, mcp),
        dry,
        "mcp/package.json",
    )
    write_or_preview(
        ROOT / "mcp/server.json",
        bump_server_json(ROOT / "mcp/server.json", mcp),
        dry,
        "mcp/server.json",
    )

    if args.skip_install or dry:
        print("Skipping vendor-core and npm install")
        return 0

    run_cmd(["npm", "run", "vendor-core", "-w", "logosdb"], dry)
    run_cmd(["npm", "install"], dry)
    print("Run ./scripts/check_versions.sh to confirm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())