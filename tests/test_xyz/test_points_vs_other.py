# -*- coding: utf-8 -*-
from os.path import join

import xtgeo
from xtgeo.common import XTGeoDialog
import tests.test_common.test_xtg as tsetup

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TMPD = xtg.tmpdir
TSTPATH = xtg.testpath

# =========================================================================
# Do tests
# =========================================================================

SFILE1A = join(TSTPATH, "surfaces/reek/1/topupperreek.gri")
SFILE2A = join(TSTPATH, "surfaces/reek/2/01_topreek_rota.gri")
PFILE3 = join(TSTPATH, "points/reek/1/pointset3.poi")


def test_snap_to_surface():
    """Import XYZ points from file."""

    mypoints = xtgeo.Points(PFILE3)
    assert mypoints.nrow == 20

    surf1 = xtgeo.RegularSurface(SFILE1A)

    mypoints.snap_surface(surf1)
    assert mypoints.nrow == 11

    tsetup.assert_almostequal(mypoints.dataframe["Z_TVDSS"].mean(), 1661.45, 0.01)

    # repeat,using surface whithg rotaion and partial masks

    mypoints = xtgeo.Points(PFILE3)
    surf2 = xtgeo.RegularSurface(SFILE2A)

    mypoints.snap_surface(surf2)
    assert mypoints.nrow == 12
    tsetup.assert_almostequal(mypoints.dataframe["Z_TVDSS"].mean(), 1687.45, 0.01)

    # alternative; keep values as is using activeobnly=False
    mypoints = xtgeo.Points(PFILE3)
    mypoints.snap_surface(surf2, activeonly=False)
    assert mypoints.nrow == 20
    tsetup.assert_almostequal(mypoints.dataframe["Z_TVDSS"].mean(), 1012.47, 0.01)
    mypoints.to_file(join(TMPD, "snapped_point.poi"))
