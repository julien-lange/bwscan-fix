"""Build labeled, tiled "grid" PNGs for visual comparison.

Each photo's variants are rendered as thumbnails and stitched into one
contact-sheet-style image with a filename header and per-tile labels, then
saved under ``review/`` (or ``output/`` for the batch verify sheet).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

TILE_W = 500
TILE_H = 360

_TTF_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
)


def to_pil_rgb(arr: np.ndarray, width: int) -> Image.Image:
    """Convert a 16-bit array to an RGB thumbnail, preserving aspect."""
    arr = np.clip(arr, 0, 65535).astype(np.uint16)
    if arr.ndim == 2:
        gray = Image.fromarray((arr >> 8).astype(np.uint8)).convert("RGB")
        img = gray
    else:
        img = Image.fromarray((arr >> 8).astype(np.uint8))
    img.thumbnail((width, width * 10))
    return img


def make_grid(
    thumbnails: Iterable[Tuple[str, Image.Image]],
    cols: int = 3,
    header: Optional[str] = None,
) -> Image.Image:
    """Lay out labeled tiles into a single RGB image."""
    tiles: List[Tuple[str, Image.Image]] = list(thumbnails)
    rows = (len(tiles) + cols - 1) // cols
    pad, label_h, header_h = 12, 26, (34 if header else 4)
    width = cols * TILE_W + (cols + 1) * pad
    height = header_h + rows * (TILE_H + label_h) + (rows + 1) * pad

    canvas = Image.new("RGB", (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    font = _font(15)
    small = _font(13)

    if header:
        draw.text((pad, 8), header, fill=(240, 240, 240), font=font)

    for i, (label, thumb) in enumerate(tiles):
        r, c = divmod(i, cols)
        x = pad + c * (TILE_W + pad)
        y = header_h + pad + r * (TILE_H + label_h)
        frame = thumb.copy()
        frame.thumbnail((TILE_W, TILE_H))
        canvas.paste(frame, (x + (TILE_W - frame.width) // 2, y))
        draw.text(
            (x, y + TILE_H),
            label,
            fill=(210, 210, 210),
            font=small,
        )
    return canvas


def _font(size: int) -> ImageFont.ImageFont:
    for path in _TTF_CANDIDATES:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def write_grid(
    img: Image.Image,
    out_dir: str | Path,
    filename: str,
    parent: str = "review",
) -> Path:
    base = Path(out_dir) / parent
    base.mkdir(parents=True, exist_ok=True)
    path = base / filename
    img.save(path, "PNG")
    return path


def open_image(path: str | Path) -> None:
    """Open a file in the default macOS viewer (Preview)."""
    for opener in (["open", str(path)],):
        try:
            subprocess.run(opener, check=True, capture_output=True)
        except (OSError, subprocess.CalledProcessError):
            continue
        return


def close_image(path: str | Path) -> None:
    """Close the matching Preview window(s).

    Opens just the grid PNG named after the photo, so we can target that one
    window (by its document name) and close it once a pick has been made —
    otherwise Preview piles up one window per photo across the roll.
    """
    name = Path(path).name
    script = (
        "tell application \"Preview\"\n"
        f"  close (every window whose name contains \"{name}\")\n"
        "end tell"
    )
    try:
        subprocess.run(
            ["osascript", "-e", script], check=True, capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass