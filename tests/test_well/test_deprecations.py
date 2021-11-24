import pytest
from packaging import version

from xtgeo import Well
from xtgeo import version as xtgeo_version


@pytest.fixture
def any_well():
    return Well()


def test_default_well_warns():
    with pytest.warns(DeprecationWarning, match="default"):
        Well()


def test_from_file_warns(any_well, tmp_path):
    if version.parse(xtgeo_version) < version.parse("2.16"):
        pytest.skip()
    any_well.to_file(tmp_path / "mywell.w")

    with pytest.warns(DeprecationWarning, match="from_file is deprecated"):
        any_well.from_file(tmp_path / "mywell.w")


def test_init_from_file_warns(any_well, tmp_path):
    any_well.to_file(tmp_path / "mywell.w")

    with pytest.warns(DeprecationWarning, match="from file"):
        Well(tmp_path / "mywell.w")
