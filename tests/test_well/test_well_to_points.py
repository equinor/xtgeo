from xtgeo.well import Well
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

TMPD = xtg.tmpdir

logger = xtg.basiclogger(__name__)

# =========================================================================
# Do tests
# =========================================================================


def test_wellzone_to_points():
    """Import well from file and put zone boundaries to a Pandas object."""

    wfile = "../xtgeo-testdata/wells/reek/1/OP_1.w"

    mywell = Well(wfile, zonelogname="Zonelog")
    logger.info("Imported %s", wfile)

    # get the zpoints which is a Pandas
    zpoints = mywell.get_zonation_points()

    print(zpoints)
