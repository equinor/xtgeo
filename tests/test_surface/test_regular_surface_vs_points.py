from os.path import join

import pytest

import xtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.surface import RegularSurface
from xtgeo.xyz import Points

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TPATH = xtg.testpathobj

# =============================================================================
# Do tests
# =============================================================================
ftop1 = TPATH / "surfaces/reek/1/reek_stooip_map.gri"


@pytest.fixture()
def reek_map():
    logger.info("Loading surface")
    return xtgeo.surface_from_file(ftop1)


def test_list_xy_points_as_numpies(reek_map):
    """Get the list of the coordinates"""

    # logger.info('Loading surface')
    # xs = RegularSurface(ftop1)
    xs = reek_map
    assert xs.ncol == 99

    # get coordinates as numpys
    xc, yc = xs.get_xy_values()

    assert xc[55, 55] == 462219.75


def test_map_to_points(tmpdir, reek_map):
    """Get the list of the coordinates"""

    px = Points()

    surf = xtgeo.surface_from_file(ftop1)

    assert isinstance(surf, RegularSurface)

    assert surf.values.mean() == pytest.approx(0.5755830099, abs=0.001)

    px.from_surface(surf)

    # convert to a Points instance
    px = Points(reek_map)
    # or
    # px.from_surface(...)

    outf = join(tmpdir, "points_from_surf_reek.poi")
    px.to_file(outf)

    assert px.dataframe["X_UTME"].min() == 456719.75
    assert px.dataframe["Z_TVDSS"].mean() == pytest.approx(0.57558, abs=0.001)

    # read the output for comparison
    pxx = Points(outf)

    assert px.dataframe["Z_TVDSS"].mean() == pytest.approx(
        pxx.dataframe["Z_TVDSS"].mean(), abs=0.00001
    )
