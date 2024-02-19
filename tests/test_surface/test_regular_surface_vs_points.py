import pathlib

import pytest
import xtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.surface import RegularSurface
from xtgeo.xyz import Points

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

FTOP1 = pathlib.Path("surfaces/reek/1/reek_stooip_map.gri")


@pytest.fixture(name="reek_map")
def fixture_reek_map(testdata_path):
    logger.info("Loading surface")
    return xtgeo.surface_from_file(testdata_path / FTOP1)


def test_list_xy_points_as_numpies(reek_map):
    """Get the list of the coordinates"""

    # logger.info('Loading surface')
    # xs = RegularSurface(ftop1)
    xs = reek_map
    assert xs.ncol == 99

    # get coordinates as numpys
    xc, _ = xs.get_xy_values()

    assert xc[55, 55] == 462219.75


def test_map_to_points(tmp_path, reek_map, testdata_path):
    """Get the list of the coordinates"""

    px = Points()

    surf = xtgeo.surface_from_file(testdata_path / FTOP1)

    assert isinstance(surf, RegularSurface)

    assert surf.values.mean() == pytest.approx(0.5755830099, abs=0.001)

    px = xtgeo.points_from_surface(surf)

    # convert to a Points instance
    px = xtgeo.points_from_surface(reek_map)

    outf = tmp_path / "points_from_surf_reek.poi"
    px.to_file(outf)

    assert px.get_dataframe()["X_UTME"].min() == 456719.75
    assert px.get_dataframe()["Z_TVDSS"].mean() == pytest.approx(0.57558, abs=0.001)

    # read the output for comparison
    pxx = xtgeo.points_from_file(outf)

    assert px.get_dataframe()["Z_TVDSS"].mean() == pytest.approx(
        pxx.get_dataframe()["Z_TVDSS"].mean(), abs=0.00001
    )
