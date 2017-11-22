import os
import os.path
import sys

from xtgeo.surface import RegularSurface
from xtgeo.common import XTGeoDialog
from .test_xtg import assert_equal, assert_almostequal

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg._testsetup():
    sys.exit(-9)

td = xtg.tmpdir
testpath = xtg.testpath

try:
    roxenv = int(os.environ['ROXENV'])
except Exception:
    roxenv = 0

print(roxenv)

if roxenv != 1:
    print("Do not run ROXENV tests")
else:
    print("Will run ROXENV tests")

# =============================================================================
# Do tests
# =============================================================================


def test_getsurface():
    """Get a surface from a RMS project."""

    if roxenv == 1:

        logger.info('Simple case...')

        project = "/private/jriv/tmp/fossekall.rms10.0.0"

        x = RegularSurface()
        x.from_roxar(project, name='TopIle', category="DepthSurface")

        x.to_file("TMP/topile.gri")

        assert_equal(x.nx, 273, "NX of top Ile from RMS")

        assert_almostequal(x.values.mean(), 2771.82236, 0.001)
    else:
        pass
