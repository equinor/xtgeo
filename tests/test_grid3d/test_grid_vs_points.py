# -*- coding: utf-8 -*-

import xtgeo

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.basiclogger(__name__)

TDMP = xtg.tmpdir
TESTPATH = xtg.testpath

# =============================================================================
# Do tests
# =============================================================================

reekgrid = '../xtgeo-testdata/3dgrids/reek/REEK.EGRID'


def test_get_ijk_from_points():
    """Testing getting IJK cooridnates from points"""
    g1 = xtgeo.grid3d.Grid(reekgrid)

    pointset = [(462000, 594200, 1750), (462200, 594250, 1760)]

    po = xtgeo.Points(pointset)

    ijk = g1.get_ijk_from_points(po)
