import numpy as np
import tifffile

from bwscan import io_tiff


def test_write_read_roundtrip(tmp_path):
    arr = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    tmp_path.mkdir(exist_ok=True)
    out = io_tiff.write_tiff(tmp_path / "out.tif", arr)
    img = tifffile.imread(out)
    assert img.ndim == 3 and img.shape[2] == 3
    back = io_tiff.read_tiff(out)
    assert back.shape == arr.shape
    assert back.dtype == np.uint16
    assert np.array_equal(back, arr)


def test_rgb_to_gray(tmp_path):
    path = (tmp_path / "x.tif")
    rgb = np.zeros((8, 8, 3), dtype=np.uint16)
    rgb[..., 0] = 1000
    rgb[..., 1] = 2000
    rgb[..., 2] = 3000
    tifffile.imwrite(str(path), rgb)
    gray = io_tiff.read_tiff(path)
    expected = 1000 * 0.2126 + 2000 * 0.7152 + 3000 * 0.0722
    assert np.allclose(gray, expected, atol=1)


def test_uint8_scaled_to_16bit(tmp_path):
    path = tmp_path / "x.tif"
    u8 = np.full((8, 8), 128, dtype=np.uint8)
    tifffile.imwrite(str(path), u8)
    arr = io_tiff.read_tiff(path)
    assert arr.dtype == np.uint16
    assert np.all(arr == 128 * 257)


def test_make_rgb_roundtrip():
    arr = np.arange(0, 65536, 16, dtype=np.uint16).reshape(64, 64)
    rgb = io_tiff.make_rgb(arr)
    assert rgb.shape == (64, 64, 3)
    assert np.array_equal(rgb[..., 0], rgb[..., 1])


def test_copy_unchanged(tmp_path):
    src = tmp_path / "src.tif"
    tifffile.imwrite(str(src), np.zeros((4, 4, 3), dtype=np.uint16))
    dst = tmp_path / "dst.tif"
    io_tiff.copy_unchanged(src, dst)
    assert dst.read_bytes() == src.read_bytes()