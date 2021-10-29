import pathlib

import pytest
from packaging import version

import xtgeo
from xtgeo import GridProperties, GridProperty
from xtgeo import version as xtgeo_version


@pytest.fixture
def any_grid():
    grd = xtgeo.create_box_grid((5, 5, 5))
    return grd


@pytest.fixture
def any_gridprop(any_grid):
    return any_grid.get_dz()


@pytest.fixture
def any_gridproperties(any_grid, any_gridprop):
    gps = GridProperties(*any_grid.dimensions)
    gps.append_props([any_gridprop])
    return gps


def test_grid_from_file_warns(any_grid):
    if version.parse(xtgeo_version) < version.parse("2.16"):
        pytest.skip()

    any_grid.to_file("grid.roff", fformat="roff")

    with pytest.warns(DeprecationWarning, match="from_file is deprecated"):
        any_grid.from_file("grid.roff", fformat="roff")


def test_grid_from_hdf_warns(any_grid):
    if version.parse(xtgeo_version) < version.parse("2.16"):
        pytest.skip()

    any_grid.to_hdf("grid.hdf")

    with pytest.warns(DeprecationWarning, match="from_hdf is deprecated"):
        any_grid.from_hdf("grid.hdf")


def test_grid_from_xtgf_warns(any_grid):
    if version.parse(xtgeo_version) < version.parse("2.16"):
        pytest.skip()

    any_grid.to_xtgf("grid.xtg")

    with pytest.warns(DeprecationWarning, match="from_xtgf is deprecated"):
        any_grid.from_xtgf("grid.xtg")


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


def test_gridproperty_deprecated_init(testpath):
    with pytest.warns(DeprecationWarning, match="Default initialization"):
        gp = GridProperty()
        assert gp.ncol == 4
        assert gp.nrow == 3
        assert gp.nlay == 5

    with pytest.warns(DeprecationWarning, match="from file name"):
        GridProperty(pathlib.Path(testpath) / "3dgrids/bri/b_poro.roff", fformat="roff")


def test_gridproperties_deprecated_init(testpath):
    with pytest.warns(
        DeprecationWarning, match="The GridProperties class is deprecated"
    ):
        GridProperties()


def test_gridprops_non_list_methods_warns(any_gridproperties, testpath):
    if version.parse(xtgeo_version) < version.parse("2.16"):
        pytest.skip()

    testpath = pathlib.Path(testpath)
    with pytest.warns(DeprecationWarning, match="__getitem__ is deprecated"):
        any_gridproperties["dZ"]

    with pytest.warns(DeprecationWarning, match="names is deprecated"):
        any_gridproperties.names

    with pytest.warns(DeprecationWarning, match="props is deprecated"):
        any_gridproperties.props

    with pytest.warns(DeprecationWarning, match="dates is deprecated"):
        any_gridproperties.dates

    with pytest.warns(DeprecationWarning, match="describe is deprecated"):
        any_gridproperties.describe()

    with pytest.warns(DeprecationWarning, match="generate_hash is deprecated"):
        any_gridproperties.generate_hash()

    with pytest.warns(DeprecationWarning, match="get_ijk is deprecated"):
        any_gridproperties.get_ijk()

    with pytest.warns(DeprecationWarning, match="get_actnum is deprecated"):
        any_gridproperties.get_actnum()

    with pytest.warns(DeprecationWarning, match="get_prop_by_name is deprecated"):
        any_gridproperties.get_prop_by_name("dZ")

    with pytest.warns(DeprecationWarning, match="get_dataframe is deprecated"):
        any_gridproperties.get_dataframe()

    with pytest.warns(DeprecationWarning, match="dataframe is deprecated"):
        any_gridproperties.dataframe()

    with pytest.warns(DeprecationWarning, match="scan_keywords is deprecated"):
        any_gridproperties.scan_keywords(testpath / "3dgrids/reek/REEK.UNRST")

    with pytest.warns(DeprecationWarning, match="scan_dates is deprecated"):
        any_gridproperties.scan_dates(testpath / "3dgrids/reek/REEK.UNRST")

    with pytest.warns(DeprecationWarning, match="scan_keywords is deprecated"):
        any_gridproperties.scan_keywords(testpath / "3dgrids/reek/REEK.UNRST")

    with pytest.warns(DeprecationWarning, match="from_file is deprecated"):
        any_gridproperties.from_file(
            testpath / "3dgrids/reek/REEK.UNRST",
            names=["PRESSURE"],
            dates="all",
            grid=xtgeo.grid_from_file(testpath / "3dgrids/reek/REEK.EGRID"),
        )
    with pytest.warns(DeprecationWarning, match="append_props is deprecated"):
        any_gridproperties.append_props(list(any_gridproperties))
