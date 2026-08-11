import numpy as np

from bwscan import stats, variants


def test_build_lut_endpoints():
    lut = variants.build_lut(100, 60000, curve="linear", gamma=1.0, s_weight=0.0)
    assert lut[0] == 0
    assert lut[65535] == 65535
    assert lut[100] == 0
    assert lut[60000] == 65535
    assert (np.diff(lut.astype(np.int64)) >= 0).all()


def test_all_grid_luts_monotonic_and_endpoints():
    # each tile must span the full range and never invert tones
    rng = np.random.default_rng(0)
    arr = (rng.uniform(0, 65536, size=(64, 64))).astype(np.uint16)
    stati = stats.ImageStats.from_array(arr)
    for v in variants.DEFAULT_GRID:
        b, w = variants.clip_values(stati, v)
        lut = variants.build_lut(b, w, v.curve, v.gamma, v.s_weight)
        assert (np.diff(lut.astype(np.int64)) >= 0).all()
        assert lut.min() == 0 and lut.max() == 65535


def test_grid_variants_are_visibly_different():
    # the whole point: tiles must not look identical. Harder variants should
    # clearly deepen shadows (more pixels pushed below quarter-tone) and show
    # measurable mean differences against the "std" tile.
    rng = np.random.default_rng(7)
    arr = np.clip(rng.normal(loc=10000, scale=7000, size=(300, 300)),
                  0, 65535).astype(np.uint16)
    stati = stats.ImageStats.from_array(arr)
    outs = {}
    for v in variants.DEFAULT_GRID:
        b, w = variants.clip_values(stati, v)
        outs[v.label] = variants.render_variant(arr, v, b, w)

    std = outs["std"]
    for label in ("flat", "soft", "hard", "s-curve"):
        diff = np.abs(outs[label].astype(int) - std.astype(int)).mean() / 257
        assert diff > 2.0, f"{label} too similar to std ({diff:.1f} levels)"

    dark = {l: (o < 16384).mean() for l, o in outs.items()}
    assert dark["hard"] > dark["std"] > dark["flat"]
    std_std = outs["std"].std()
    assert outs["hard"].std() > std_std > outs["flat"].std()


def test_render_variant_bounds():
    rng = np.random.default_rng(0)
    arr = (rng.uniform(0, 65536, size=(16, 16))).astype(np.uint16)
    v = variants.Variant("test", 1.0, 99.0)
    stati = stats.ImageStats.from_array(arr)
    black_val, white_val = variants.clip_values(stati, v)
    out = variants.render_variant(arr, v, black_val, white_val)
    assert out.dtype == np.uint16
    assert out.min() == 0
    assert out.max() == 65535


def test_default_grid_is_complete():
    labels = [v.label for v in variants.DEFAULT_GRID]
    assert "s-curve" in labels
    assert len(variants.DEFAULT_GRID) >= 5
    for v in variants.DEFAULT_GRID:
        assert v.black_pct < v.white_pct


def test_clip_values_ordering():
    rng = np.random.default_rng(1)
    arr = (rng.uniform(0, 65536, size=(64, 64))).astype(np.uint16)
    stati = stats.ImageStats.from_array(arr)
    for v in variants.DEFAULT_GRID:
        b, w = variants.clip_values(stati, v)
        assert b <= w