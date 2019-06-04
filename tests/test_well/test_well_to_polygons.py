# -*- coding: utf-8 -*-

import os

import xtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TMPD = xtg.tmpdir
TESTPATH = xtg.testpath

# =========================================================================
# Do tests
# =========================================================================

WFILE = os.path.join(TESTPATH, "wells/reek/1/OP_1.w")


def test_well_to_polygons():
    """Import well from file and amke a Polygons object"""

    mywell = xtgeo.Well(WFILE)

    poly = mywell.get_polygon()

    assert isinstance(poly, xtgeo.xyz.Polygons)
    print(poly.dataframe)

    assert mywell.dataframe["X_UTME"].mean() == poly.dataframe["X_UTME"].mean()
