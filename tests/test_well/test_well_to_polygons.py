import os

import xtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)


def test_well_to_polygons(testdata_path):
    """Import well from file and amke a Polygons object"""

    WFILE = os.path.join(testdata_path, "wells/reek/1/OP_1.w")
    mywell = xtgeo.well_from_file(WFILE)

    poly = mywell.get_polygons()

    assert isinstance(poly, xtgeo.xyz.Polygons)

    assert (
        mywell.get_dataframe()["X_UTME"].mean() == poly.get_dataframe()["X_UTME"].mean()
    )
