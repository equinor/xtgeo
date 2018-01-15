import os
import os.path
import sys

from xtgeo.surface import RegularSurface
from xtgeo.common import XTGeoDialog
import tests.test_setup as tsetup

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    sys.exit(-9)

td = xtg.tmpdir
testpath = xtg.testpath

# =============================================================================
# Do tests
# =============================================================================


@tsetup.skipunlessroxar  # disable=F405 # noqa:<Error No>
def test_getsurface():
    """Get a surface from a RMS project."""

    logger.info('Simple case...')

    project = "/private/jriv/tmp/fossekall/fossekall.rms10.1.1"

    x = RegularSurface()
    x.from_roxar(project, name='TopIle', category="DepthSurface")

    x.to_file("TMP/topile.gri")

    tsetup.assert_equal(x.ncol, 273, "NCOL of top Ile from RMS")

    tsetup.assert_almostequal(x.values.mean(), 2771.82236, 0.001)
