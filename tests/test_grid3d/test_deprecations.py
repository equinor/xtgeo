import pytest
from packaging import version

from xtgeo import Grid, GridProperties
from xtgeo import version as xtgeo_version


@pytest.fixture
def any_grid():
    grd = Grid()
    grd.create_box(dimension=(5, 5, 5))
    return grd


@pytest.fixture
def any_gridprop(any_grid):
    return any_grid.get_dz()


@pytest.fixture
def any_gridproperties(any_grid, any_gridprop):
    gps = GridProperties(*any_grid.dimensions)
    gps.append_props([any_gridprop])
    return gps


def test_grid_numpify_warns(any_grid):
    if version.parse(xtgeo_version) < version.parse("2.7"):
        pytest.skip()

    with pytest.warns(DeprecationWarning, match="numpify_carrays is deprecated"):
        any_grid.numpify_carrays()

    with pytest.warns(DeprecationWarning, match="get_cactnum is deprecated"):
        any_grid.get_cactnum()


def test_grid_get_indices_warns(any_grid):
    if version.parse(xtgeo_version) < version.parse("1.16"):
        pytest.skip()

    with pytest.warns(DeprecationWarning, match="get_indices is deprecated"):
        any_grid.get_indices()


def test_grid_mask_warns(any_grid):
    if version.parse(xtgeo_version) >= version.parse("3.0"):
        pytest.fail(reason="mask option should be removed")
    with pytest.warns(DeprecationWarning, match="mask"):
        any_grid.get_actnum(mask=True)

    with pytest.warns(DeprecationWarning, match="mask"):
        any_grid.get_ijk(mask=True)

    with pytest.warns(DeprecationWarning, match="mask"):
        any_grid.get_xyz(mask=True)


def test_gridprop_mask_warns(any_gridprop):
    if version.parse(xtgeo_version) >= version.parse("3.0"):
        pytest.fail(reason="mask option should be removed")
    with pytest.warns(DeprecationWarning, match="mask"):
        any_gridprop.get_actnum(mask=True)


def test_gridprops_mask_warns(any_gridproperties):
    if version.parse(xtgeo_version) >= version.parse("3.0"):
        pytest.fail(reason="mask option should be removed")
    with pytest.warns(DeprecationWarning, match="mask"):
        any_gridproperties.get_actnum(mask=True)
