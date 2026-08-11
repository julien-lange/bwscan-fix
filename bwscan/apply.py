"""Full-resolution application of chosen (or learned) tonal remaps.

For each scan: the exact black/white clip pixel values are recomputed from
the *loaded* 16-bit array (np.percentile), the chosen LUT is applied, and
the result is written as a 16-bit RGB TIFF to ``output/``. Scans without a
pick (and without a usable learned default) are passed through unmodified.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from . import io_tiff, profile_store, variants


def find_tiffs(folder: str | Path, recursive: bool = False) -> List[Path]:
    folder = Path(folder)
    pattern = "**/*.tif" if recursive else "*.tif"
    found = sorted(folder.glob(pattern))
    found += sorted(folder.glob("**/*.tiff" if recursive else "*.tiff"))
    return sorted(set(found))


def resolve_settings(
    profile: profile_store.ProfileStore,
    roll: str,
    filename: str,
    use_default: bool,
) -> Optional[dict]:
    """Decide the tonal settings for one scan, or None to copy untouched.

    Precedence: an explicit pick for this file, then the learned default
    (when ``use_default``/``--default`` is requested).
    """
    pick = profile.pick_for(roll, filename)
    if pick is not None:
        return pick
    if use_default and profile.learned is not None:
        return {
            "black_pct": profile.learned["black_pct"],
            "white_pct": profile.learned["white_pct"],
            "curve": profile.learned["curve"],
            "gamma": profile.learned.get("gamma", 1.0),
            "s_weight": profile.learned.get("s_weight", 0.0),
            "tile": 0,
            "variant_label": "learned",
            "custom": False,
        }
    return None


def process_one(
    tiff: Path,
    settings: Optional[dict],
    output_dir: Path,
) -> str:
    """Apply one file. Returns a status string: 'applied', 'unchanged'."""
    if settings is None:
        io_tiff.copy_unchanged(tiff, output_dir / tiff.name)
        return "unchanged"
    arr = io_tiff.read_tiff(tiff)
    black_val = int(np.percentile(arr, float(settings["black_pct"])))
    white_val = int(np.percentile(arr, float(settings["white_pct"])))
    curve = settings.get("curve", "linear")
    gamma = settings.get("gamma", 1.0)
    s_weight = settings.get("s_weight", 0.0)
    lut = variants.build_lut(black_val, white_val, curve, gamma, s_weight)
    out = lut[np.clip(arr, 0, 65535).astype(np.int64)]
    io_tiff.write_tiff(output_dir / tiff.name, out)
    return "applied"


def apply_batch(
    folder: str | Path,
    output_dir: str | Path,
    use_default: bool = False,
    jpeg: bool = False,
    recursive: bool = False,
    profile: Optional[profile_store.ProfileStore] = None,
) -> List[Tuple[Path, str]]:
    """Apply all scans in a folder; returns (file, status) pairs."""
    folder = Path(folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    roll = folder.name
    if profile is None:
        profile = profile_store.ProfileStore(roll)
    results: List[Tuple[Path, str]] = []
    for tiff in find_tiffs(folder, recursive=recursive):
        settings = resolve_settings(profile, roll, tiff.name, use_default)
        status = process_one(tiff, settings, output_dir)
        if jpeg and status == "applied":
            export_jpeg(output_dir / tiff.name)
        results.append((tiff, status))
    return results


def export_jpeg(tiff_path: Path) -> Path:
    """8-bit JPEG quick-share from the (already processed) TIFF."""
    from PIL import Image

    jpg = tiff_path.with_suffix(".jpg")
    Image.open(tiff_path).convert("RGB").save(jpg, "JPEG", quality=92)
    return jpg


def elapsed(t0: float) -> str:
    return f"{time.time() - t0:.1f}s"