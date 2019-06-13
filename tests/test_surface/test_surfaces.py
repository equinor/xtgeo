# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function

from os.path import join

import xtgeo
import test_common.test_xtg as tsetup

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TMPD = xtg.tmpdir
TESTPATH = xtg.testpath


# =============================================================================
# Do tests
# =============================================================================

TESTSET1A = "../xtgeo-testdata/surfaces/reek/1/topreek_rota.gri"
TESTSET1B = "../xtgeo-testdata/surfaces/reek/1/basereek_rota.gri"
TESTSETG1 = "../xtgeo-testdata/3dgrids/reek/reek_geo_grid.roff"


def test_create():
    """Create simple Surfaces instance"""

    logger.info("Simple case...")

    top = xtgeo.RegularSurface(TESTSET1A)
    base = xtgeo.RegularSurface(TESTSET1B)
    surfs = xtgeo.Surfaces()
    surfs.surfaces = [top, base]
    surfs.describe()

    assert isinstance(surfs.surfaces[0], xtgeo.RegularSurface)
    assert isinstance(surfs, xtgeo.Surfaces)


def test_get_surfaces_from_3dgrid():
    """Create surfaces from a 3D grid"""

    mygrid = xtgeo.Grid(TESTSETG1)
    surfs = xtgeo.Surfaces()
    surfs.from_grid3d(mygrid, rfactor=2)
    surfs.describe()

    tsetup.assert_almostequal(surfs.surfaces[-1].values.mean(), 1742.28, 0.02)
    tsetup.assert_almostequal(surfs.surfaces[-1].values.min(), 1589.62, 0.02)
    tsetup.assert_almostequal(surfs.surfaces[-1].values.max(), 1977.20, 0.02)
    tsetup.assert_almostequal(surfs.surfaces[0].values.mean(), 1697.02, 0.02)

    for srf in surfs.surfaces:
        srf.to_file(join(TMPD, srf.name + ".gri"))
