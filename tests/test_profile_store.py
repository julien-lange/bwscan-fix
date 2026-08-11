import pytest

from bwscan import profile_store


def test_record_and_find_by_roll(tmp_path):
    store = profile_store.ProfileStore("roll01", profile_dir=tmp_path,
                                       min_picks=1)
    store.record_pick("a.tif", 0.1, 99.9, "linear", 1, "soft")
    store.record_pick("a.tif", 0.5, 99.0, "linear", 2, "hard", custom=True)

    pick = store.pick_for("roll01", "a.tif")
    assert pick["variant_label"] == "hard"
    assert store.pick_for("roll02", "a.tif") is None

    # reload from disk
    again = profile_store.ProfileStore("roll01", profile_dir=tmp_path,
                                       min_picks=1)
    assert again.pick_for("roll01", "a.tif")["tile"] == 2


def test_learned_default_needs_min_picks(tmp_path):
    store = profile_store.ProfileStore("roll01", profile_dir=tmp_path,
                                       min_picks=5)
    assert store.learned is None
    for i in range(4):
        store.record_pick(f"f{i}.tif", 0.1, 99.5, "linear", 1, "std")
    assert store.learned is None
    store.record_pick("f4.tif", 0.1, 99.5, "linear", 1, "std")
    assert store.learned is not None
    assert store.learned["black_pct"] == 0.1
    assert store.learned["white_pct"] == 99.5


def test_custom_picks_excluded_from_learning(tmp_path):
    store = profile_store.ProfileStore("roll01", profile_dir=tmp_path,
                                       min_picks=2)
    store.record_pick("a.tif", 0.1, 99.5, "linear", 1, "std")
    store.record_pick("b.tif", 0.9, 98.0, "linear", 7, "custom", custom=True)
    assert store.learned is None
    store.record_pick("c.tif", 0.1, 99.5, "linear", 1, "std")
    assert store.learned is not None
    assert store.learned["black_pct"] == 0.1


def test_learnt_uses_median_not_mode(tmp_path):
    store = profile_store.ProfileStore("roll01", profile_dir=tmp_path,
                                       min_picks=3)
    for i, (b, w) in enumerate([(0.05, 99.95), (0.1, 99.9), (1.0, 99.0)]):
        store.record_pick(f"f{i}.tif", b, w, "linear", 1, "v")
    assert store.learned["black_pct"] == 0.1
    assert store.learned["white_pct"] == 99.9


def test_learned_carries_tone_and_pick_records_gamma(tmp_path):
    store = profile_store.ProfileStore("roll01", profile_dir=tmp_path,
                                       min_picks=2)
    store.record_pick("a.tif", 0.1, 99.0, "linear", 3, "std",
                      gamma=1.0, s_weight=0.3)
    store.record_pick("b.tif", 0.2, 98.8, "linear", 5, "hard",
                      gamma=1.2, s_weight=0.75)
    store.record_pick("c.tif", 0.5, 98.0, "linear", 5, "hard",
                      gamma=1.2, s_weight=0.75)
    pick = store.pick_for("roll01", "b.tif")
    assert pick["gamma"] == 1.2
    assert pick["s_weight"] == pytest.approx(0.75)
    assert store.learned["gamma"] == pytest.approx(1.2)
    assert store.learned["s_weight"] == pytest.approx(0.75)