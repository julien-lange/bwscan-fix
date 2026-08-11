"""Per-image histogram + percentile analysis, with a JSON cache.

For each TIFF we record min/max/mean/stddev, a set of tail percentiles and a
fixed-resolution histogram. The cache (`<folder>/.bwscan_cache/<stem>.json`)
keys on file size + mtime so re-runs skip recomputation until a scan changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import io_tiff

N_BINS = 4096
PERCENTILES = (0.05, 0.1, 0.5, 1.0, 50.0, 99.0, 99.5, 99.9, 99.95)
FULL_RANGE = 65536
CACHE_FORMAT = 1


@dataclass
class ImageStats:
    """Analysis of one scan."""

    width: int
    height: int
    min: int
    max: int
    mean: float
    stddev: float
    percentiles: Dict[str, int] = field(default_factory=dict)
    hist: List[int] = field(default_factory=list)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "ImageStats":
        flat = arr.astype(np.float64).ravel()
        percentiles = {
            str(p): int(round(float(np.percentile(flat, p))))
            for p in PERCENTILES
        }
        hist, _ = np.histogram(arr, bins=N_BINS, range=(0, FULL_RANGE))
        return cls(
            width=arr.shape[1],
            height=arr.shape[0],
            min=int(arr.min()),
            max=int(arr.max()),
            mean=float(flat.mean()),
            stddev=float(flat.std()),
            percentiles=percentiles,
            hist=[int(v) for v in hist],
        )

    def to_dict(self) -> dict:
        return {
            "format": CACHE_FORMAT,
            "width": self.width,
            "height": self.height,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "stddev": self.stddev,
            "percentiles": self.percentiles,
            "hist": self.hist,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ImageStats":
        return cls(
            width=d["width"],
            height=d["height"],
            min=d["min"],
            max=d["max"],
            mean=d["mean"],
            stddev=d["stddev"],
            percentiles={k: int(v) for k, v in d["percentiles"].items()},
            hist=[int(v) for v in d["hist"]],
        )

    def value_at(self, pct: float) -> int:
        """Interpolated pixel value at percentile ``pct`` (0-100)."""
        if pct <= 0.0:
            return self.min
        if pct >= 100.0:
            return self.max
        total = float(sum(self.hist))
        if total <= 0:
            return 0
        target = total * (pct / 100.0)
        cdf = np.cumsum(self.hist)
        idx = int(np.searchsorted(cdf, target))
        prev = float(cdf[idx - 1]) if idx > 0 else 0.0
        cur = float(cdf[idx])
        frac = (target - prev) / (cur - prev) if cur > prev else 0.0
        width = FULL_RANGE / N_BINS
        return int(round((idx + frac) * width))


def cache_dir_for(folder: str | Path) -> Path:
    return Path(folder) / ".bwscan_cache"


def cache_path_for(tiff: str | Path) -> Path:
    tiff = Path(tiff)
    return cache_dir_for(tiff.parent) / f"{tiff.stem}.json"


def _cache_meta(tiff: Path) -> dict:
    return {
        "source": tiff.name,
        "size": tiff.stat().st_size,
        "mtime": tiff.stat().st_mtime,
    }


def load_cached(tiff: str | Path) -> Optional[dict]:
    """Return cached stats dict if valid, else None."""
    tiff = Path(tiff)
    cache = cache_path_for(tiff)
    if not cache.exists():
        return None
    try:
        st = json.loads(cache.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if st.get("format") != CACHE_FORMAT:
        return None
    if _cache_meta(tiff) != st.get("_meta"):
        return None
    return st


def analyze_tiff(tiff: str | Path, use_cache: bool = True) -> ImageStats:
    """Analyze a scan, reusing the on-disk cache when it is still valid."""
    tiff = Path(tiff)
    if use_cache:
        cached = load_cached(tiff)
        if cached is not None:
            return ImageStats.from_dict(cached)
    arr = io_tiff.read_tiff(tiff)
    stats = ImageStats.from_array(arr)
    _store(tiff, stats)
    return stats


def analyze_array(arr: np.ndarray) -> ImageStats:
    return ImageStats.from_array(arr)


def _store(tiff: Path, stats: ImageStats) -> Path:
    cache_dir = cache_dir_for(tiff.parent)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path_for(tiff)
    d = stats.to_dict()
    d["_meta"] = _cache_meta(tiff)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d), encoding="utf-8")
    tmp.replace(path)
    return path


def ascii_histogram(stats: ImageStats, width: int = 56) -> List[str]:
    """Compact log-scaled text histogram for the terminal.

    Renders ~24 evenly spaced buckets; the left/right edge labels mark the
    historical 0.1%/99.9% percentile values so the shape reads at a glance.
    """
    hist = np.asarray(stats.hist, dtype=np.float64) + 1.0
    n = min(24, N_BINS)
    block = n and (N_BINS // n) or 1
    hist = np.add.reduceat(hist, np.arange(0, N_BINS, block))[:n]
    hist = np.log10(hist)
    hist = hist - hist.min()
    hist = hist / (hist.max() or 1.0)
    edge_lo = stats.percentiles.get("0.1", 0)
    edge_hi = stats.percentiles.get("99.9", 65535)
    lines = [f"p0.1={edge_lo}  p99.9={edge_hi}"]
    step = FULL_RANGE / n
    for i, h in enumerate(hist):
        bar_len = int(round(h * width))
        lines.append(f"{int(round(i*step)):>6} |{'#'*bar_len}")
    return lines