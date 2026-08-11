"""Load and save 16-bit TIFFs for the Plustek/VueScan pipeline.

VueScan writes 16-bit *RGB* TIFFs for a B&W negative even though the data
is effectively grayscale (the three channels usually differ slightly). We
collapse to a single 16-bit channel for analysis, then write output back
as 3-channel RGB so results stay visually comparable with VueScan output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

RGB_MODES = ("luma", "green", "max")

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def read_tiff(path: str | Path) -> np.ndarray:
    """Read any single-page TIFF and return a 2-D uint16 array."""
    arr = tifffile.imread(path)
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[2] in (3, 4):
            arr = _rgb_to_gray(arr[..., :3])
        else:
            raise ValueError(f"Unsupported TIFF shape {arr.shape} for {path}")
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected a single 2-D image in {path}, got shape {arr.shape}"
        )
    return _to_uint16(arr)


def _rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    r, g, b = (rgb[..., c].astype(np.float64) for c in range(3))
    if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(
        rgb[..., 1], rgb[..., 2]
    ):
        return r
    return _LUMA[0] * r + _LUMA[1] * g + _LUMA[2] * b


def _to_uint16(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint16:
        return np.asarray(arr, dtype=np.uint16)
    if arr.dtype == np.uint8:
        return (np.asarray(arr, dtype=np.uint16) * 257).astype(np.uint16)
    arr = np.asarray(arr, dtype=np.float64)
    if np.issubdtype(arr.dtype, np.floating):
        if arr.min() >= 0.0 and arr.max() <= 1.0:
            return np.clip(arr * 65535.0, 0, 65535).astype(np.uint16)
    return np.clip(arr, 0, 65535).astype(np.uint16)


def make_rgb(arr: np.ndarray) -> np.ndarray:
    """Replicate a 2-D grayscale array across 3 channels."""
    arr = np.clip(arr, 0, 65535).astype(np.uint16)
    if arr.ndim == 3:
        return arr
    return np.repeat(arr[:, :, None], 3, axis=2)


def write_tiff(path: str | Path, arr: np.ndarray) -> Path:
    """Write a 16-bit 3-channel TIFF (grayscale replicated to RGB)."""
    path = Path(path)
    tifffile.imwrite(path, make_rgb(arr), photometric="rgb")
    return path


def copy_unchanged(src: str | Path, dst: str | Path) -> None:
    """Byte-for-byte copy so unprocessed scans are passed through untouched."""
    import shutil

    shutil.copyfile(src, dst)