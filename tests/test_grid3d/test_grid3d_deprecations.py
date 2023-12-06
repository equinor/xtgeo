import io
import pathlib

import numpy as np
import pytest
import xtgeo
from hypothesis import given
from packaging import version
from xtgeo import GridProperties, GridProperty
from xtgeo.common.version import __version__ as xtgeo_version

from .gridprop_generator import grid_properties


@pytest.fixture(name="any_grid")
def fixture_any_grid():
    return xtgeo.create_box_grid((5, 5, 5))


@pytest.fixture(name="any_gridprop")
def fixture_any_gridprop(any_grid):
    return any_grid.get_dz()


@pytest.fixture(name="any_gridproperties")
def fixture_any_gridproperties(any_gridprop):
    return GridProperties(props=[any_gridprop])


def test_gridproperties_init_deprecations(any_gridprop):
    with pytest.warns(DeprecationWarning):
        GridProperties(ncol=10)
    with pytest.warns(DeprecationWarning):
        GridProperties(nrow=10)
    with pytest.warns(DeprecationWarning):
        GridProperties(nlay=10)

    with pytest.raises(ValueError, match="Giving both ncol/nrow/nlay and props"):
        with pytest.warns(DeprecationWarning):
            GridProperties(nlay=10, props=[any_gridprop])


def test_grid_from_file_warns(tmp_path, any_grid):
    if version.parse(xtgeo_version) < version.parse("2.16"):
        pytest.skip()

    any_grid.to_file(tmp_path / "grid.roff", fformat="roff")

    with pytest.warns(DeprecationWarning, match="from_file is deprecated"):
        any_grid.from_file(tmp_path / "grid.roff", fformat="roff")


def test_grid_from_hdf_warns(tmp_path, any_grid):
    if version.parse(xtgeo_version) < version.parse("2.16"):
        pytest.skip()

    any_grid.to_hdf(tmp_path / "grid.hdf")

    with pytest.warns(DeprecationWarning, match="from_hdf is deprecated"):
        any_grid.from_hdf(tmp_path / "grid.hdf")


def test_grid_from_xtgf_warns(tmp_path, any_grid):
    if version.parse(xtgeo_version) < version.parse("2.16"):
        pytest.skip()

    any_grid.to_xtgf(tmp_path / "grid.xtg")

    with pytest.warns(DeprecationWarning, match="from_xtgf is deprecated"):
        any_grid.from_xtgf(tmp_path / "grid.xtg")


def test_grid_mask_warns(any_grid):
    if version.parse(xtgeo_version) >= version.parse("4.0"):
        pytest.fail(reason="mask option should be removed")
    with pytest.warns(DeprecationWarning, match="mask"):
        any_grid.get_actnum(mask=True)

    with pytest.warns(DeprecationWarning, match="mask"):
        any_grid.get_ijk(mask=True)

    with pytest.warns(DeprecationWarning, match="mask"):
        any_grid.get_xyz(mask=True)


@given(grid_properties())
def test_unknown_name_deprecate(gridprop):
    buf = io.BytesIO()
    gridprop.to_file(buf, fformat="roff")

    buf.seek(0)

    gridprop2 = xtgeo.gridproperty_from_file(buf, fformat="roff")

    assert gridprop2.name == gridprop.name
    np.testing.assert_allclose(gridprop2.values, gridprop.values)

    # deprecated name="unknown"
    buf.seek(0)

    with pytest.warns(DeprecationWarning, match="name='unknown'"):
        gridprop3 = xtgeo.gridproperty_from_file(buf, name="unknown", fformat="roff")

    assert gridprop3.name == gridprop.name


def test_gridprop_mask_warns(any_gridprop):
    if version.parse(xtgeo_version) >= version.parse("4.0"):
        pytest.fail(reason="mask option should be removed")
    with pytest.warns(DeprecationWarning, match="mask"):
        any_gridprop.get_actnum(mask=True)


def test_gridprops_mask_warns(any_gridproperties):
    if version.parse(xtgeo_version) >= version.parse("4.0"):
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
