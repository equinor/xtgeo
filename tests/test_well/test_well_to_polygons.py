# -*- coding: utf-8 -*-

import os

import xtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TPATH = xtg.testpathobj

# =========================================================================
# Do tests
# =========================================================================

WFILE = os.path.join(TPATH, "wells/reek/1/OP_1.w")


def test_well_to_polygons():
    """Import well from file and amke a Polygons object"""

    mywell = xtgeo.well_from_file(WFILE)

    poly = mywell.get_polygons()

    assert isinstance(poly, xtgeo.xyz.Polygons)
    print(poly.dataframe)

    assert mywell.dataframe["X_UTME"].mean() == poly.dataframe["X_UTME"].mean()
