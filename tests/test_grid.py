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


def test_open_image_uses_background_flag(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        raise OSError  # simulate 'open' missing so fallback path is exercised

    monkeypatch.setattr("subprocess.run", fake_run)
    grid.open_image(tmp_path / "x.png")
    assert calls, "expected at least one open attempt"
    assert calls[0][0] == "open" and "-g" in calls[0]


def test_close_image_targets_filename(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        class R:
            returncode = 0
        return R()

    monkeypatch.setattr("subprocess.run", fake_run)
    grid.close_image(tmp_path / "neg00_grid.png")
    joined = " ".join(calls[0])
    assert "Preview" in joined and "neg00_grid.png" in joined


def test_to_pil_aspect(tmp_path):
    tiff = make_tiff(tmp_path / "x.tif", size=(50, 100))
    from bwscan import io_tiff

    arr = io_tiff.read_tiff(tiff)
    pil = grid.to_pil_rgb(arr, 80)
    w, h = pil.size
    assert w <= 80 and h <= 800
    assert h / w == pytest.approx(50 / 100)