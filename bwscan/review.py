"""Terminal review loop: view a grid per photo, pick the best variant.

Flow per scan, one at a time:

1. analyze (cached) + render the default grid to ``review/<stem>_grid.png``,
   then open it in Preview;
2. prompt ``Pick 1-6, c (customize), s (skip), q (quit):``;
3. ``c`` asks for black%/white% clips, re-renders just that variant as an
   extra tile and reopens the grid;
4. selections (and skips) are recorded in the profile store; after the batch
   a learned default is recomputed if enough picks exist.

Run ``bwscan apply`` afterwards (or pass ``--apply`` here) to write the
full-resolution outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from . import apply as apply_mod
from . import grid as grid_mod
from . import io_tiff, profile_store, stats, variants

PREVIEW_W = 640


def downsample(arr: np.ndarray, width: int = PREVIEW_W) -> np.ndarray:
    """Shrink a 16-bit array for fast preview rendering."""
    h, w = arr.shape
    if w <= width:
        return arr
    scale = w / width
    new_h = max(1, int(round(h / scale)))
    xs = np.linspace(0, w - 1, width).astype(int)
    ys = np.linspace(0, h - 1, new_h).astype(int)
    return arr[np.ix_(ys, xs)]


def render_row(
    small: np.ndarray,
    stati: stats.ImageStats,
    extra: Tuple[Tuple[str, variants.Variant], ...] = (),
) -> List[Tuple[str, object]]:
    """One labeled thumbnail per variant, for the contact-sheet grid."""
    out: List[Tuple[str, object]] = []
    for i, v in enumerate(variants.DEFAULT_GRID):
        black_val, white_val = variants.clip_values(stati, v)
        rendered = variants.render_variant(small, v, black_val, white_val)
        curve_note = " [s]" if v.curve == "s" else ""
        out.append(
            (
                f"{i+1}. {v.label}  b={v.black_pct}% w={v.white_pct}%{curve_note}",
                grid_mod.to_pil_rgb(rendered, PREVIEW_W),
            )
        )
    for tile_no, v in extra:
        black_val, white_val = variants.clip_values(stati, v)
        rendered = variants.render_variant(small, v, black_val, white_val)
        out.append(
            (f"{tile_no}. {v.label}  b={v.black_pct}% w={v.white_pct}%",
             grid_mod.to_pil_rgb(rendered, PREVIEW_W))
        )
    return out


def _prompt(text: str, inp, out) -> str:
    print(text, end="", flush=True, file=out)
    try:
        line = inp.readline()
    except AttributeError:
        line = input()
    if line == "":
        return "q"
    return line.strip().lower()


def _parse_clip(raw: str, default: float, out) -> float:
    raw = raw.strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"  (ignoring '{raw}', using {default})", file=out)
        return default


def review_batch(
    folder: str | Path,
    no_open: bool = False,
    apply_after: bool = False,
    output_dir: str | Path = "output",
    min_picks: int = profile_store.MIN_PICKS_DEFAULT,
    profile_dir: Optional[str | Path] = None,
    inp=None,
    out=None,
) -> int:
    """Run the interactive loop. Returns number of photos reviewed."""
    out = out or sys.stdout
    inp = inp or sys.stdin
    folder = Path(folder)
    tiffs = [t for t in apply_mod.find_tiffs(folder) if "_grid" not in t.name]
    if not tiffs:
        print(f"No TIFFs found in {folder}", file=out)
        return 0

    profile = profile_store.ProfileStore(
        roll=folder, profile_dir=profile_dir, min_picks=min_picks
    )
    reviewed = 0
    for tiff in tiffs:
        try:
            stati = stats.analyze_tiff(tiff)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! could not analyze {tiff.name}: {exc}", file=out)
            continue

        small = downsample(io_tiff.read_tiff(tiff))
        extra: List[Tuple[int, variants.Variant]] = []
        while True:
            tiles = render_row(small, stati, tuple(extra))
            grid_img = grid_mod.make_grid(tiles, header=f"{tiff.name}  ({tiff.parent})")
            path = grid_mod.write_grid(grid_img, folder, f"{tiff.stem}_grid.png")
            if no_open:
                print(f"  [grid] {path}", file=out)
            else:
                grid_mod.close_image(path)
                grid_mod.open_image(path)

            n = len(tiles)
            print("", file=out)
            print(f"  {tiff.name}", file=out)
            for i, v in enumerate(variants.DEFAULT_GRID, start=1):
                curve_note = " s-curve" if v.curve == "s" else ""
                print(
                    f"    {i}. {v.label:9s} b={v.black_pct}% w={v.white_pct}%"
                    f" g={v.gamma}{curve_note}",
                    file=out,
                )
            for tile_no, v in extra:
                print(
                    f"    {tile_no}. {v.label:9s} b={v.black_pct}% w={v.white_pct}%",
                    file=out,
                )
            raw = _prompt(f"  Pick [1-{n}] or c/s/q: ", inp, out)

            if raw in ("", "h", "?"):
                continue
            if raw == "q":
                if not no_open:
                    grid_mod.close_image(path)
                if apply_after:
                    _apply_now(folder, output_dir, out, profile)
                return reviewed
            if raw == "s":
                print(f"  skipped {tiff.name}", file=out)
                if not no_open:
                    grid_mod.close_image(path)
                reviewed += 1
                break
            if raw == "c":
                b = _parse_clip(_prompt(f"  black clip % [{0.1}]: ", inp, out), 0.1, out)
                w = _parse_clip(_prompt(f"  white clip % [{99.9}]: ", inp, out), 99.9, out)
                if b >= w:
                    print("  black must be < white; ignoring", file=out)
                    continue
                custom = variants.Variant("custom", b, w, "linear")
                extra = [t for t in extra if t[1].label != "custom"]
                extra.append((len(variants.DEFAULT_GRID) + len(extra) + 1, custom))
                continue
            try:
                tile = int(raw)
            except ValueError:
                print(f"  unrecognised input '{raw}'", file=out)
                continue
            if tile < 1 or tile > n:
                print(f"  pick must be 1..{n}", file=out)
                continue

            if tile - 1 < len(variants.DEFAULT_GRID):
                v = variants.DEFAULT_GRID[tile - 1]
            else:
                v = extra[tile - 1 - len(variants.DEFAULT_GRID)][1]
            custom = v.label == "custom"
            profile.record_pick(
                filename=tiff.name,
                black_pct=v.black_pct,
                white_pct=v.white_pct,
                curve=v.curve,
                tile=tile,
                variant_label="custom" if custom else v.label,
                custom=custom,
                gamma=v.gamma,
                s_weight=v.s_weight if v.curve == "s" else 0.0,
            )
            print(
                f"  -> tile {tile} ({v.label}) b={v.black_pct} w={v.white_pct}",
                file=out,
            )
            if not no_open:
                grid_mod.close_image(path)
            reviewed += 1
            break

    learned = profile.learned
    if learned:
        print(
            f"\n  learned default so far: b={learned['black_pct']}% "
            f"w={learned['white_pct']}% g={learned.get('gamma', 1.0)} "
            f"({learned['curve']}, n={learned['n']})",
            file=out,
        )
    if apply_after:
        _apply_now(folder, output_dir, out, profile)
    return reviewed


def _apply_now(folder: Path, output_dir: Path, out, profile) -> None:
    print(f"\n  applying selections to {output_dir}/ ...", file=out)
    for tiff, status in apply_mod.apply_batch(
        folder, output_dir, profile=profile
    ):
        print(f"    {status:9s} {tiff.name}", file=out)