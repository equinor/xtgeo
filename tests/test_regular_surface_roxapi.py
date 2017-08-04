import unittest
import numpy as np
import os
import os.path
import sys
import logging

from xtgeo.surface import RegularSurface
from xtgeo.common import XTGeoDialog


path = 'TMP'
try:
    os.makedirs(path)
except OSError:
    if not os.path.isdir(path):
        raise

xtg = XTGeoDialog()

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


class TestSurfaceRoxapi(unittest.TestCase):
    """Testing suite for surfaces"""

    def getlogger(self, name):

        # if isinstance(self.logger):
        #     return

        format = xtg.loggingformat

        logging.basicConfig(format=format, stream=sys.stdout)
        logging.getLogger().setLevel(xtg.logginglevel)  # root logger!

        self.logger = logging.getLogger(name)

    def test_getsurface(self):
        """
        get a surface from a RMS project.
        """

        if roxenv == 1:
            self.getlogger('test_getsurface')

            self.logger.info('Simple case...')

            project = "/private/jriv/tmp/fossekall.rms10.0.0"

            x = RegularSurface()
            x.from_roxar(project, name='TopIle', category="DepthSurface")

            x.to_file("TMP/topile.gri")

            self.assertEqual(x.nx, 273, "NX of top Ile from RMS")

            self.assertAlmostEqual(x.values.mean(), 2771.82236, places=3)
        else:
            pass


if __name__ == '__main__':

    logging.basicConfig(stream=sys.stderr)
    logging.getLogger('').setLevel(logging.DEBUG)

    print()
    unittest.main()
