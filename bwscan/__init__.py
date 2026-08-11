"""B&W negative scan optimizer.

Package sized for one fixed pipeline (Kentmere 400 / Pentax Spotmatic ES2 /
Plustek 7200 via VueScan). It analyzes 16-bit TIFF scans, proposes a small
grid of tonal remaps (black/white percentile clips, optional mild S-curve),
lets you pick the best one per photo in the terminal, and records the picks
so a default can be learned for future rolls.
"""

__version__ = "0.1.0"

from . import (  # noqa: F401
    apply,
    grid,
    io_tiff,
    profile_store,
    review,
    stats,
    variants,
)

__all__ = [
    "apply",
    "grid",
    "io_tiff",
    "profile_store",
    "review",
    "stats",
    "variants",
]