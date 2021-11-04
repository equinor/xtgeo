import pytest
from packaging import version

from xtgeo import Well, Wells
from xtgeo import version as xtgeo_version


@pytest.fixture
def any_well():
    return Well()


@pytest.fixture
def any_wells(any_well):
    ws = Wells()
    ws.wells = [any_well]
    return ws


def test_wells_init_warns():
    if version.parse(xtgeo_version) < version.parse("2.16"):
        pytest.skip()
    with pytest.warns(DeprecationWarning):
        Wells()


def test_wells_methods_warns(any_wells):
    if version.parse(xtgeo_version) < version.parse("2.16"):
        pytest.skip()

    with pytest.warns(DeprecationWarning, match="from_files is deprecated"):
        any_wells.from_files([])

    with pytest.warns(DeprecationWarning, match="get_well is deprecated"):
        any_wells.get_well("")
