import unittest
import os
import os.path
import sys
import logging
import warnings

from xtgeo.surface import RegularSurface
from xtgeo.cube import Cube
from xtgeo.common import XTGeoDialog


path = 'TMP'
try:
    os.makedirs(path)
except OSError:
    if not os.path.isdir(path):
        raise

xtg = XTGeoDialog()

try:
    bigtest = int(os.environ['BIGTEST'])
except Exception:
    bigtest = 0


def custom_formatwarning(msg, *a):
    # ignore everything except the message
    return str(msg) + '\n'

warnings.formatwarning = custom_formatwarning

# =============================================================================
# Do tests
# =============================================================================


class Test(unittest.TestCase):
    """Testing suite for surfaces"""

    def getlogger(self, name):

        # if isinstance(self.logger):
        #     return

        format = xtg.loggingformat

        logging.basicConfig(format=format, stream=sys.stdout)
        logging.getLogger().setLevel(xtg.logginglevel)  # root logger!

        self.logger = logging.getLogger(name)

    def test_slice_nearest(self):
        """
        Slice a cube with a surface, nearest node
        """

        self.getlogger('test_slice_nearest')

        self.logger.info("Loading surface")
        x = RegularSurface()
        x.from_file("../../testdata/Surface/G/gullfaks_top.irapbin")

        y = x.copy()

        x.to_file("TMP/surf_slice_cube_initial.gri")

        self.logger.info("Loading cube")
        cc = Cube()
        cc.from_file("../../testdata/Cube/GF/gf_depth_1985_10_01.segy")
        self.logger.info("Loading cube, done")

        # Now slice
        self.logger.info("Slicing...")
        x.slice_cube(cc)
        self.logger.info("Slicing...done")

        x.to_file("TMP/surf_slice_cube.gri")
        self.assertAlmostEqual(x.values.mean(), -0.0755, places=2)
        self.logger.info("Avg X is {}".format(x.values.mean()))

        # try same ting with swapaxes active ==================================
        y = RegularSurface()
        y.from_file("../../testdata/Surface/G/gullfaks_top.irapbin")

        cc.swapaxes()
        # Now slice
        self.logger.info("Slicing... (now with swapaxes)")
        y.slice_cube(cc)
        self.logger.info("Slicing...done")

        y.to_file("TMP/surf_slice_cube_y.gri")
        self.assertAlmostEqual(y.values.mean(), -0.0755, places=2)
        self.logger.info("Avg Y is {}".format(y.values.mean()))

    def test_slice_interpol(self):
        """
        Slice a cube with a surface, using trilinear interpol.
        """

        self.getlogger('test_slice_interpol')

        self.logger.info("Loading surface")
        x = RegularSurface()
        x.from_file("../../testdata/Surface/G/gullfaks_top.irapbin")

        self.logger.info("Loading cube")
        cc = Cube()
        cc.from_file("../../testdata/Cube/GF/gf_depth_1985_10_01.segy")
        self.logger.info("Loading cube, done")

        if cc.yflip == -1:
            self.logger.info("Swap axes...")
            cc.swapaxes()

        # Now slice
        self.logger.info("Slicing...")
        x.slice_cube(cc, sampling=1)
        self.logger.info("Slicing...done")
        x.to_file("TMP/surf_slice_cube_interpol.gri")

        self.assertAlmostEqual(x.values.mean(), -0.07363, places=5)

    def test_slice2(self):
        """
        Slice a larger cube with a surface
        """

        self.getlogger('test_slice2')

        if bigtest == 0:
            warnings.warn("TEST SKIPPED, enable with env BIGTEST = 1  .... ")
            return

        cfile = "/project/gullfaks/resmod/gfmain_brent/2015a/users/eza/"\
                + "r004/r004_20170303/sim2seis/input/"\
                + "4D_Res_PSTM_LM_ST8511D11_Full_rmsexport.segy"

        print(cfile)

        if os.path.isfile(cfile):

            self.logger.info("Loading surface")
            x = RegularSurface()
            x.from_file("../../testdata/Surface/G/gullfaks_top.irapbin")

            zd = RegularSurface()
            zd.from_file("../../testdata/Surface/G/gullfaks_top.irapbin")

            self.logger.info("Loading cube")
            cc = Cube()
            cc.from_file(cfile)
            self.logger.info("Loading cube, done")

            # Now slice
            self.logger.info("Slicing a large cube...")
            x.slice_cube(cc, zsurf=zd)
            self.logger.info("Slicing a large cube ... done")
            x.to_file("TMP/surf_slice2_cube.gri")
        else:
            self.logger.warning("No big file; skip test")


if __name__ == '__main__':

    logging.basicConfig(stream=sys.stderr)
    logging.getLogger('').setLevel(logging.DEBUG)

    print()
    unittest.main()

    print("OK")
