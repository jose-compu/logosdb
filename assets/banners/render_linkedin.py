#!/usr/bin/env python3
"""Render LinkedIn release banner (1200x627) using bundled fonts and release metadata."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FONTS = HERE / "fonts"
DEFAULT_LOGO = ROOT / "logo.png"
CANVAS_W, CANVAS_H = 1200, 627

BG = (10, 10, 11)
PANEL = (20, 20, 22)
PANEL_BORDER = (42, 42, 46)
CHIP_BG = (24, 24, 27)
CHIP_BORDER = (63, 63, 70)
TITLE = (244, 244, 245)
BODY = (212, 212, 216)
MUTED = (161, 161, 170)
TEAL = (45, 212, 191)
TEAL_DIM = (20, 184, 166)
RULE = (39, 39, 42)
ACCENT_BAR = (45, 212, 191)

TAGLINE_DEFAULT = "The SQLite for AI agents and RAG"
GITHUB_DEFAULT = "github.com/jose-compu/logosdb"


@dataclass
class BannerData:
    version: str
    title: str
    date: str
    tagline: str = TAGLINE_DEFAULT
    highlights: list[str] = field(default_factory=list)
    packages: list[tuple[str, str]] = field(default_factory=list)
    install: str = ""
    github: str = GITHUB_DEFAULT


class FontSet:
    def __init__(self) -> None:
        self.display_xl = _load("SpaceGrotesk-Bold.woff", 64)
        self.display_lg = _load("SpaceGrotesk-Bold.woff", 52)
        self.heading = _load("SpaceGrotesk-Medium.woff", 28)
        self.body = _load("Inter-Regular.ttf", 22)
        self.body_sm = _load("Inter-Regular.ttf", 18)
        self.label = _load("Inter-SemiBold.ttf", 16)
        self.footer = _load("Inter-Regular.ttf", 17)
        self.mono = _load("JetBrainsMono-Medium.ttf", 19)
        self.mono_sm = _load("JetBrainsMono-Regular.ttf", 16)
        self.chip = _load("Inter-SemiBold.ttf", 15)


def _font_path(name: str) -> Path:
    path = FONTS / name
    if not path.is_file():
        raise FileNotFoundError(f"bundled font missing: {path}")
    return path


def _load(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(_font_path(name)), size=size)


def _text_w(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def _text_h(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return box[3] - box[1]


def _draw_chip(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    label: str,
    version: str,
    fonts: FontSet,
) -> int:
    text = f"{label}  {version}"
    pad_x, pad_y = 14, 8
    tw = _text_w(draw, text, fonts.chip)
    th = _text_h(draw, text, fonts.chip)
    w, h = tw + pad_x * 2, th + pad_y * 2
    draw.rounded_rectangle((x, y, x + w, y + h), radius=10, fill=CHIP_BG, outline=CHIP_BORDER, width=1)
    draw.text((x + pad_x, y + pad_y - 1), text, fill=BODY, font=fonts.chip)
    return w


def _parse_package(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"--package expects label=version, got {raw!r}")
    label, ver = raw.split("=", 1)
    label, ver = label.strip(), ver.strip()
    if not label or not ver:
        raise argparse.ArgumentTypeError(f"--package expects label=version, got {raw!r}")
    return label, ver


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("PyYAML required for YAML config (pip install pyyaml)") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("config root must be an object")
    return data


def _packages_from_config(raw: Any) -> list[tuple[str, str]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("packages must be a list")
    out: list[tuple[str, str]] = []
    for item in raw:
        if isinstance(item, dict):
            label = str(item.get("label", "")).strip()
            ver = str(item.get("version", "")).strip()
        elif isinstance(item, str) and "=" in item:
            label, ver = _parse_package(item)
        else:
            raise ValueError("each package needs label and version")
        if label and ver:
            out.append((label, ver))
    return out


def build_banner_data(args: argparse.Namespace) -> BannerData:
    base: dict[str, Any] = {}
    if args.config:
        base = _load_config(args.config)

    version = args.version or base.get("version")
    title = args.title or base.get("title")
    date = args.date or base.get("date")
    if not version or not title or not date:
        raise SystemExit("--version, --title, and --date are required (or set in --config)")

    highlights = list(base.get("highlights") or [])
    highlights.extend(args.highlight or [])

    packages = _packages_from_config(base.get("packages"))
    packages.extend(args.package or [])

    install = args.install or base.get("install") or ""
    tagline = args.tagline or base.get("tagline") or TAGLINE_DEFAULT
    github = args.github or base.get("github") or GITHUB_DEFAULT

    return BannerData(
        version=str(version).lstrip("v"),
        title=str(title),
        date=str(date),
        tagline=str(tagline),
        highlights=[str(h) for h in highlights],
        packages=packages,
        install=str(install),
        github=str(github),
    )


def render(data: BannerData, logo_path: Path, out_path: Path) -> None:
    fonts = FontSet()
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(img)

    draw.rectangle((0, 0, 6, CANVAS_H), fill=ACCENT_BAR)

    panel = (72, 72, 252, 272)
    draw.rounded_rectangle(panel, radius=24, fill=PANEL, outline=PANEL_BORDER, width=2)
    logo = Image.open(logo_path).convert("RGBA")
    logo.thumbnail((152, 152), Image.Resampling.LANCZOS)
    lx = 72 + (180 - logo.width) // 2
    ly = 72 + (200 - logo.height) // 2
    img.paste(logo, (lx, ly), logo)

    x0 = 292
    draw.text((x0, 76), "LogosDB", fill=TITLE, font=fonts.display_lg)
    draw.text((x0, 134), data.tagline, fill=MUTED, font=fonts.body_sm)
    draw.line([(x0, 168), (CANVAS_W - 72, 168)], fill=RULE, width=2)

    ver_label = f"v{data.version}"
    draw.text((x0, 192), ver_label, fill=TEAL, font=fonts.display_xl)
    ver_w = _text_w(draw, ver_label, fonts.display_xl)
    title_x = x0 + ver_w + 28
    title_max = CANVAS_W - 72 - title_x
    title_font = fonts.heading
    title_lines: list[str] = []
    words = data.title.split()
    line = ""
    for word in words:
        trial = f"{line} {word}".strip()
        if _text_w(draw, trial, title_font) <= title_max:
            line = trial
        else:
            if line:
                title_lines.append(line)
            line = word
    if line:
        title_lines.append(line)
    title_y = 208
    for ln in title_lines[:2]:
        draw.text((title_x, title_y), ln, fill=TITLE, font=title_font)
        title_y += 34

    highlight_y = 292
    bullet_r = 5
    for line in data.highlights[:3]:
        cy = highlight_y + 12
        draw.ellipse((x0, cy - bullet_r, x0 + bullet_r * 2, cy + bullet_r), fill=TEAL_DIM)
        draw.text((x0 + 22, highlight_y), line, fill=BODY, font=fonts.body)
        highlight_y += 40

    pkg_y = 410
    pkg_x = x0
    gap = 12
    for label, ver in data.packages[:4]:
        w = _draw_chip(draw, pkg_x, pkg_y, label, ver, fonts)
        pkg_x += w + gap
        if pkg_x > CANVAS_W - 200:
            break

    if data.install:
        inst_y = 468
        pad = 16
        mono = fonts.mono
        tw = _text_w(draw, data.install, mono)
        box_w = min(tw + pad * 2, CANVAS_W - 72 - x0)
        box_h = _text_h(draw, data.install, mono) + pad * 2
        draw.rounded_rectangle(
            (x0, inst_y, x0 + box_w, inst_y + box_h),
            radius=12,
            fill=PANEL,
            outline=CHIP_BORDER,
            width=1,
        )
        draw.text((x0 + pad, inst_y + pad - 2), data.install, fill=TEAL, font=mono)

    footer_y = CANVAS_H - 44
    draw.text((x0, footer_y), data.github, fill=MUTED, font=fonts.footer)
    date_w = _text_w(draw, data.date, fonts.footer)
    draw.text((CANVAS_W - 72 - date_w, footer_y), data.date, fill=MUTED, font=fonts.footer)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG", optimize=True)
    with Image.open(out_path) as check:
        w, h = check.size
        assert (w, h) == (CANVAS_W, CANVAS_H), f"expected 1200x627, got {w}x{h}"


def main() -> int:
    p = argparse.ArgumentParser(description="Render LogosDB LinkedIn release banner (1200×627 PNG)")
    p.add_argument("--config", type=Path, help="JSON or YAML file with banner fields")
    p.add_argument("--version", help="Release version (e.g. 1.3.0)")
    p.add_argument("--title", help="Release title headline")
    p.add_argument("--date", help="Release date (YYYY-MM-DD)")
    p.add_argument("--tagline", help=f"Product tagline (default: {TAGLINE_DEFAULT!r})")
    p.add_argument("--highlight", action="append", default=[], help="Highlight line (repeatable)")
    p.add_argument("--package", type=_parse_package, action="append", default=[], metavar="LABEL=VER")
    p.add_argument("--install", help="Install commands shown in monospace")
    p.add_argument("--github", help=f"Footer repo URL (default: {GITHUB_DEFAULT})")
    p.add_argument("--logo", type=Path, default=DEFAULT_LOGO)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if not args.logo.is_file():
        print(f"logo not found: {args.logo}", file=sys.stderr)
        return 1

    try:
        data = build_banner_data(args)
    except (ValueError, SystemExit) as exc:
        print(exc, file=sys.stderr)
        return 1

    if not data.highlights:
        print("warning: no highlights; pass --highlight or set highlights in config", file=sys.stderr)
    if not data.packages:
        print("warning: no packages; pass --package or set packages in config", file=sys.stderr)

    render(data, args.logo, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())