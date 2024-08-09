import pathlib

import pytest
import xtgeo
from packaging import version
from xtgeo import GridProperties
from xtgeo.common.version import __version__ as xtgeo_version


@pytest.fixture(name="any_grid")
def fixture_any_grid():
    return xtgeo.create_box_grid((5, 5, 5))


@pytest.fixture(name="any_gridprop")
def fixture_any_gridprop(any_grid):
    return any_grid.get_dz()


@pytest.fixture(name="any_gridproperties")
def fixture_any_gridproperties(any_gridprop):
    return GridProperties(props=[any_gridprop])


IFILE1 = pathlib.Path("3dgrids/reek/REEK.INIT")
GFILE1 = pathlib.Path("3dgrids/reek/REEK.EGRID")


def test_gridproperties_init_deprecations(any_gridprop):
    with pytest.warns(DeprecationWarning):
        GridProperties(ncol=10)
    with pytest.warns(DeprecationWarning):
        GridProperties(nrow=10)
    with pytest.warns(DeprecationWarning):
        GridProperties(nlay=10)

    with pytest.raises(
        ValueError,
        match="Giving both ncol/nrow/nlay and props",
    ), pytest.warns(
        DeprecationWarning,
    ):
        GridProperties(nlay=10, props=[any_gridprop])


def test_gridprops_from_file(testdata_path):
    g = xtgeo.grid_from_file(testdata_path / GFILE1, fformat="egrid")
    v1 = xtgeo.gridproperties_from_file(
        testdata_path / IFILE1, fformat="init", names=["PORO", "PORV"], grid=g
    )

    v2 = xtgeo.GridProperties()
    with pytest.warns(DeprecationWarning, match="from_file is deprecated"):
        v2.from_file(
            testdata_path / IFILE1, fformat="init", names=["PORO", "PORV"], grid=g
        )

    assert v1.generate_hash() == v2.generate_hash()


def test_gridprops_mask_warns(any_gridproperties):
    if version.parse(xtgeo_version) >= version.parse("4.0"):
        pytest.fail(reason="mask option should be removed")
    with pytest.warns(DeprecationWarning, match="mask"):
        any_gridproperties.get_actnum(mask=True)
