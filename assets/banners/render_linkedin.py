#!/usr/bin/env python3
"""Render LinkedIn release banner (1200x627) from template + official logo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOGO = ROOT / "logo.png"
CANVAS_W, CANVAS_H = 1200, 627
BG = (10, 10, 11)
PANEL = (20, 20, 22)
PANEL_BORDER = (42, 42, 46)
TITLE_COLOR = (244, 244, 245)
SUBTITLE_COLOR = (45, 212, 191)
VERSION_COLOR = (161, 161, 170)
MUTED = (136, 136, 136)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            ]
        )
    for path in candidates:
        p = Path(path)
        if p.is_file():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def render(version: str, title: str, logo_path: Path, out_path: Path) -> None:
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(img)

    # Logo panel
    panel = (80, 80, 260, 260)
    draw.rounded_rectangle(panel, radius=24, fill=PANEL, outline=PANEL_BORDER, width=2)

    logo = Image.open(logo_path).convert("RGBA")
    max_side = 160
    logo.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    lx = 80 + (180 - logo.width) // 2
    ly = 80 + (180 - logo.height) // 2
    img.paste(logo, (lx, ly), logo)

    draw.text((300, 95), "LogosDB", fill=TITLE_COLOR, font=_font(72, bold=True))
    draw.text((300, 175), title, fill=SUBTITLE_COLOR, font=_font(30))
    draw.line([(300, 250), (1120, 250)], fill=(39, 39, 42), width=2)
    draw.text((300, 360), f"Release {version}", fill=VERSION_COLOR, font=_font(36))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG", optimize=True)
    with Image.open(out_path) as check:
        w, h = check.size
        assert (w, h) == (CANVAS_W, CANVAS_H), f"expected 1200x627, got {w}x{h}"


def main() -> int:
    p = argparse.ArgumentParser(description="Render LogosDB LinkedIn banner PNG")
    p.add_argument("--version", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--logo", type=Path, default=DEFAULT_LOGO)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if not args.logo.is_file():
        print(f"logo not found: {args.logo}", file=sys.stderr)
        return 1
    render(args.version, args.title, args.logo, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())