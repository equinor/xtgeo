import pytest
import xtgeo
from packaging import version
from xtgeo import Well
from xtgeo import version as xtgeo_version


@pytest.fixture(name="any_well")
def fixture_any_well():
    with pytest.warns(DeprecationWarning, match="empty"):
        return Well()


@pytest.fixture(name="any_well_file")
def fixture_any_well_file(any_well, tmp_path):
    any_well.to_file(tmp_path / "mywell.w")
    return tmp_path / "mywell.w"


def test_default_well_warns():
    with pytest.warns(DeprecationWarning, match="default"):
        Well()


def test_from_file_warns(any_well, tmp_path):
    if version.parse(xtgeo_version) < version.parse("2.16"):
        pytest.skip()
    any_well.to_file(tmp_path / "mywell.w")

    with pytest.warns(DeprecationWarning, match="from_file is deprecated"):
        any_well.from_file(tmp_path / "mywell.w")


def test_init_from_file_warns(any_well_file):
    with pytest.warns(DeprecationWarning, match="from file"):
        Well(any_well_file)


def test_blockedwell_init_from_file_warns(any_well_file):
    with pytest.warns(DeprecationWarning, match="from file"):
        xtgeo.BlockedWell(any_well_file)


def test_wells_init_warns(any_well_file):

    with pytest.warns(DeprecationWarning, match="directly from file"):
        xtgeo.Wells([any_well_file])


def test_blockedwells_from_files_warns(any_well_file):
    if version.parse(xtgeo_version) < version.parse("2.16"):
        pytest.skip()
    with pytest.warns(DeprecationWarning, match="from_files is deprecated"):
        xtgeo.BlockedWells().from_files([any_well_file])
