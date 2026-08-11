import pytest

from common import make_tiff


@pytest.fixture
def roll(tmp_path):
    folder = tmp_path / "roll01"
    folder.mkdir()
    for i in range(5):
        make_tiff(folder / f"neg{i:02d}.tif", seed=i)
    return folder


@pytest.fixture
def profile_dir(tmp_path):
    return tmp_path / "profiles"