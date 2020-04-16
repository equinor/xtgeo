from xtgeo.well import Well
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

TMPD = xtg.tmpdir

logger = xtg.basiclogger(__name__)


def test_wellzone_to_points():
    """Import well from file and put zone boundaries to a Pandas object."""

    wfile = "../xtgeo-testdata/wells/etc/otest.rmswell"

    mywell = Well(wfile, zonelogname="Zone_model", mdlogname="M_DEPTH")
    logger.info("Imported %s", wfile)

    # get the zpoints which is a Pandas
    zpoints = mywell.get_zonation_points(use_undef=False)

    print(zpoints.to_string())
