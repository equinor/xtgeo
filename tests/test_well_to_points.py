import os
import sys
import logging
from xtgeo.well import Well
from xtgeo.common import XTGeoDialog


path = 'TMP'
try:
    os.makedirs(path)
except OSError:
    if not os.path.isdir(path):
        raise

xtg = XTGeoDialog()
format = xtg.loggingformat

logging.basicConfig(format=format, stream=sys.stdout)
logging.getLogger().setLevel(xtg.logginglevel)  # root logger!

logger = logging.getLogger(__name__)

# =========================================================================
# Do tests
# =========================================================================


def test_wellzone_to_points():
    """Import well from file and put zone boundaries to a Pandas object."""

    wfile = "../xtgeo-testdata/wells/tro/1/31_2-E-1_H.w"

    mywell = Well(wfile, zonelogname="ZONELOG")
    logger.info("Imported {}".format(wfile))

    # get the zpoints which is a Pandas
    zpoints = mywell.get_zonation_points()

    print(zpoints)
