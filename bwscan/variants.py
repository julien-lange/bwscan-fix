"""Candidate tonal remaps and their application.

Every photo gets the same small, consistent grid of variants to start with:
a family of looks that differ clearly in both clip (black/white percentile
tails) and midtone shape (gamma lift/darken plus an S-curve strength). Each
variant is a pure per-pixel remap expressed as a 65536-entry lookup table,
so the same code path drives both cheap preview thumbnails and full-
resolution output (the LUT is O(1) per pixel either way).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from . import stats as stats_mod

FULL_RANGE = 65536

S_CURVE_WEIGHT = 0.4


@dataclass(frozen=True)
class Variant:
    """A named tonal remap.

    ``gamma`` (applied as ``y ** gamma`` after the clip) lifts midtones when
    < 1 and darkens/deepens them when > 1. ``s_weight`` is the S-curve blend:
    0 = linear, 1 = full smoothstep, steepening midtones and shaving the
    blacks/whites — i.e. photographic "contrast".
    """

    label: str
    black_pct: float
    white_pct: float
    curve: str = "linear"
    gamma: float = 1.0
    s_weight: float = S_CURVE_WEIGHT


DEFAULT_GRID: Tuple[Variant, ...] = (
    Variant("flat", 0.1, 99.9, gamma=0.65, s_weight=0.0),
    Variant("soft", 0.1, 99.5, gamma=0.85, s_weight=0.15),
    Variant("std", 0.1, 99.0, gamma=1.0, s_weight=0.3),
    Variant("contrast", 0.2, 98.5, gamma=1.1, s_weight=0.55),
    Variant("hard", 0.5, 98.0, gamma=1.2, s_weight=0.75),
    Variant("s-curve", 0.2, 98.8, curve="s", s_weight=1.0),
)


def render_variant(
    arr: np.ndarray,
    variant: Variant,
    black_val: int,
    white_val: int,
) -> np.ndarray:
    """Apply a variant to a 16-bit array, returning a uint16 array."""
    lut = build_lut(black_val, white_val, variant.curve, variant.gamma,
                    variant.s_weight)
    return lut[np.clip(arr, 0, FULL_RANGE - 1).astype(np.int64)]


def build_lut(
    black_val: int,
    white_val: int,
    curve: str = "linear",
    gamma: float = 1.0,
    s_weight: float = S_CURVE_WEIGHT,
) -> np.ndarray:
    """Lookup table mapping input value -> output value."""
    lo = float(black_val)
    hi = float(white_val)
    if hi - lo < 1.0:
        lo, hi = 0.0, float(FULL_RANGE - 1)
    v = np.arange(FULL_RANGE, dtype=np.float64)
    y = (v - lo) / (hi - lo)
    np.clip(y, 0.0, 1.0, out=y)
    if s_weight > 0.0 and curve == "s":
        smooth = y * y * (3.0 - 2.0 * y)
        y = (1.0 - s_weight) * y + s_weight * smooth
    if curve not in ("linear", "s"):
        raise ValueError(f"Unknown curve {curve!r}")
    if gamma != 1.0:
        y = np.power(y, gamma)
    return (y * (FULL_RANGE - 1) + 0.5).astype(np.uint16)


def to_params(variant: Variant) -> dict:
    """Variant parameters flattened for persistence (profile/apply)."""
    return {
        "curve": variant.curve,
        "gamma": variant.gamma,
        "s_weight": variant.s_weight,
    }


def clip_values(stati: stats_mod.ImageStats, variant: Variant) -> Tuple[int, int]:
    """Exact clip pixel values for a variant from a stats object."""
    return stati.value_at(variant.black_pct), stati.value_at(variant.white_pct)