import os
import pytest

from xtgeo.surface import RegularSurface
from xtgeo.common import XTGeoDialog
from xtgeo.xyz import Points
import tests.test_common.test_xtg as tsetup

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

td = xtg.tmpdir

# =============================================================================
# Do tests
# =============================================================================
ftop1 = "../xtgeo-testdata/surfaces/reek/1/reek_stooip_map.gri"


@pytest.fixture()
def reek_map():
    logger.info("Loading surface")
    return RegularSurface(ftop1)


def test_list_xy_points_as_numpies(reek_map):
    """Get the list of the coordinates"""

    # logger.info('Loading surface')
    # xs = RegularSurface(ftop1)
    xs = reek_map
    assert xs.ncol == 99

    # get coordinates as numpys
    xc, yc = xs.get_xy_values()

    assert xc[55, 55] == 462219.75


def test_map_to_points(reek_map):
    """Get the list of the coordinates"""

    px = Points()

    surf = RegularSurface(ftop1)

    assert isinstance(surf, RegularSurface)

    tsetup.assert_almostequal(surf.values.mean(), 0.5755830099, 0.001)

    px.from_surface(surf)

    # convert to a Points instance
    px = Points(reek_map)
    # or
    # px.from_surface(...)

    outf = os.path.join(td, "points_from_surf_reek.poi")
    px.to_file(outf)

    assert px.dataframe["X_UTME"].min() == 456719.75
    tsetup.assert_almostequal(px.dataframe["Z_TVDSS"].mean(), 0.57558, 0.001)

    # read the output for comparison
    pxx = Points(outf)

    tsetup.assert_almostequal(
        px.dataframe["Z_TVDSS"].mean(), pxx.dataframe["Z_TVDSS"].mean(), 0.00001
    )
