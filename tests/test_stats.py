import json

import numpy as np

from bwscan import stats


def test_from_array_basics():
    arr = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    s = stats.ImageStats.from_array(arr)
    assert s.min == 0
    assert s.max == 65535
    assert s.width == 256 and s.height == 256
    assert s.percentiles["50.0"] == 32768
    assert len(s.hist) == stats.N_BINS
    assert sum(s.hist) == 65536


def test_value_at_matches_percentiles():
    rng = np.random.default_rng(3)
    arr = np.clip(rng.normal(loc=3000, scale=2000, size=(200, 200)), 0,
                  65535).astype(np.uint16)
    s = stats.ImageStats.from_array(arr)
    for pct, exact in [("0.1", None), ("50.0", None)]:
        v = s.value_at(float(pct))
        assert 0 <= v <= 65535
    # interpolation should be near the true percentile within a bin width
    exact = float(np.percentile(arr, 5.0))
    width = 65536 / stats.N_BINS
    assert abs(s.value_at(5.0) - exact) <= width + 1


def test_cache_roundtrip_and_invalidation(tmp_path):
    from common import make_tiff

    tiff = make_tiff(tmp_path / "a.tif", size=(32, 32))
    s1 = stats.analyze_tiff(tiff)
    cpath = stats.cache_path_for(tiff)
    assert cpath.exists()
    d = json.loads(cpath.read_text())
    assert d["percentiles"] == s1.percentiles

    s2 = stats.analyze_tiff(tiff)
    assert s2.percentiles == s1.percentiles
    assert stats.load_cached(tiff) is not None

    tiff.write_bytes(tiff.read_bytes() + b"\x00")
    assert stats.load_cached(tiff) is None
    s3 = stats.analyze_tiff(tiff)
    assert s3.percentiles == s1.percentiles
    assert stats.load_cached(tiff) is not None


def test_ascii_histogram_renders(tmp_path):
    from common import make_tiff

    tiff = make_tiff(tmp_path / "b.tif", size=(32, 32))
    s = stats.analyze_tiff(tiff)
    lines = stats.ascii_histogram(s)
    assert len(lines) > 10
    assert "p0.1=" in lines[0]