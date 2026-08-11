import numpy as np
import pytest
from PIL import Image

from bwscan import grid

from common import make_tiff


def test_grid_roundtrip(tmp_path):
    arr = np.full((40, 60), 3000, dtype=np.uint16)
    tiles = [
        ("1. soft  b=0.1% w=99.9%", grid.to_pil_rgb(arr, 160)),
        ("2. hard  b=1.0% w=99.0%", grid.to_pil_rgb(arr + 4000, 160)),
        ("3. s-curve [s]", grid.to_pil_rgb(arr + 8000, 160)),
    ]
    img = grid.make_grid(tiles, cols=2, header="neg00.tif (folder)")
    assert img.mode == "RGB"
    assert img.size[0] > 2 * 160
    path = grid.write_grid(img, tmp_path, "test_grid.png")
    assert path.exists()
    loaded = Image.open(path)
    assert loaded.size == img.size


def test_to_pil_aspect(tmp_path):
    tiff = make_tiff(tmp_path / "x.tif", size=(50, 100))
    from bwscan import io_tiff

    arr = io_tiff.read_tiff(tiff)
    pil = grid.to_pil_rgb(arr, 80)
    w, h = pil.size
    assert w <= 80 and h <= 800
    assert h / w == pytest.approx(50 / 100)