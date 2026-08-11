import io

import numpy as np

from bwscan import apply, io_tiff, profile_store, review, variants


def test_apply_applied_and_full_range(roll, tmp_path):
    out = tmp_path / "out"
    results = apply.apply_batch(roll, out, use_default=False)
    assert all(status == "unchanged" for _, status in results)

    store = profile_store.ProfileStore(roll, profile_dir=tmp_path / "prof",
                                       min_picks=5)
    for f in sorted(p.name for p in roll.glob("*.tif")):
        store.record_pick(f, 0.1, 99.5, "linear", 1, "std")
    results = apply.apply_batch(roll, out, profile=store)
    assert all(status == "applied" for _, status in results)

    first = io_tiff.read_tiff(out / sorted(p.name for p in roll.glob('*.tif'))[0])
    assert first.min() == 0 and first.max() == 65535


def test_resolve_precedence(roll, tmp_path):
    profdir = tmp_path / "prof"
    store = profile_store.ProfileStore("roll01", profile_dir=profdir, min_picks=5)
    for f in sorted(p.name for p in roll.glob("*.tif")):
        store.record_pick(f, 0.1, 99.5, "linear", 1, "std")
    store.record_pick("neg00.tif", 0.5, 99.0, "linear", 5, "hard")

    s = apply.resolve_settings(store, "roll01", "neg00.tif", use_default=True)
    assert s["variant_label"] == "hard"
    s2 = apply.resolve_settings(store, "roll01", "neg01.tif", use_default=True)
    assert s2["variant_label"] == "std"  # explicit pick wins over learned
    s3 = apply.resolve_settings(store, "roll01", "neg02.tif", use_default=False)
    assert s3["variant_label"] == "std"


def test_apply_uses_recorded_tone_curve(roll, tmp_path):
    profdir = tmp_path / "prof"
    store = profile_store.ProfileStore("roll01", profile_dir=profdir, min_picks=1)
    first = sorted(p.name for p in roll.glob("*.tif"))[0]
    store.record_pick(first, 0.1, 99.0, "s", 6, "s-curve",
                      gamma=1.0, s_weight=1.0)

    out = tmp_path / "out"
    statuses = dict(apply.apply_batch(roll, out, profile=store))
    assert statuses[roll / first] == "applied"
    for other in sorted((p for p in roll.glob("*.tif") if p.name != first)):
        assert statuses[other] == "unchanged"

    # full smoothstep S-curve must steepen midtones vs a plain linear clip
    src = io_tiff.read_tiff(roll / first)
    src_flat = src.astype(np.float64).ravel()
    black_val = int(np.percentile(src_flat, 0.1))
    white_val = int(np.percentile(src_flat, 99.0))
    linear_out = variants.build_lut(black_val, white_val, "linear")
    linear_std = linear_out[np.clip(src, 0, 65535).astype(np.int64)].std()
    out_std = io_tiff.read_tiff(out / first).std()
    assert out_std > linear_std


def test_review_pick_and_skip(roll, tmp_path):
    out = io.StringIO()
    lines = io.StringIO("2\ns\n")
    review.review_batch(roll, no_open=True, inp=lines, out=out,
                        min_picks=5, profile_dir=str(tmp_path / "prof"))
    txt = out.getvalue()
    assert "-> tile 2" in txt
    assert "skipped" in txt
    store = profile_store.ProfileStore(roll, profile_dir=tmp_path / "prof",
                                       min_picks=5)
    first = sorted(f.name for f in roll.glob("*.tif"))[0]
    rec = store.pick_for(roll.name, first)
    assert rec is not None and rec["tile"] == 2
    second = sorted(f.name for f in roll.glob("*.tif"))[1]
    assert store.pick_for(roll.name, second) is None


def test_review_custom_tile(roll, tmp_path):
    out = io.StringIO()
    lines = io.StringIO("c\n0.2\n99.8\n7\n")
    review.review_batch(roll, no_open=True, inp=lines, out=out,
                        min_picks=5, profile_dir=str(tmp_path / "prof"))
    txt = out.getvalue()
    assert "Pick [1-7] or c/s/q" in txt
    assert "7. custom" in txt
    assert "-> tile 7" in txt
    store = profile_store.ProfileStore(roll, profile_dir=tmp_path / "prof",
                                       min_picks=5)
    rec = store.pick_for(roll.name, sorted(f.name for f in roll.glob("*.tif"))[0])
    assert rec["custom"] is True
    assert rec["black_pct"] == 0.2
    assert rec["white_pct"] == 99.8