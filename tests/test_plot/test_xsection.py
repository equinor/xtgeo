import unittest
import os
import sys
import glob
import logging
import matplotlib.pyplot as plt

from xtgeo.plot import XSection
from xtgeo.well import Well
from xtgeo.xyz import Polygons
from xtgeo.surface import RegularSurface
from xtgeo.common import XTGeoDialog


path = 'TMP'
try:
    os.makedirs(path)
except OSError:
    if not os.path.isdir(path):
        raise

xtg = XTGeoDialog()

# =========================================================================
# Do tests
# =========================================================================


class Test(unittest.TestCase):
    """Testing suite for wells"""

    def getlogger(self, name):

        format = xtg.loggingformat

        logging.basicConfig(format=format, stream=sys.stdout)
        logging.getLogger().setLevel(xtg.logginglevel)  # root logger!

        self.logger = logging.getLogger(name)

    def test_very_basic(self):
        """Just test that matplotlib works."""
        plt.title('Hello world')
        plt.show()

    def test_simple_plot(self):
        """
        Test as simple XSECT plot
        """
        self.getlogger('test_simple_plot')

        mywell = Well()
        mywell.from_file("../xtgeo-testdata/wells/gfb/1/34_10-A-42.w")

        mysurfaces = []
        mysurf = RegularSurface()
        mysurf.from_file('../xtgeo-testdata/surfaces/gfb/1/'
                         'gullfaks_top.irapbin')
        for i in range(10):
            xsurf = mysurf.copy()
            xsurf.values = xsurf.values + i * 20
            mysurfaces.append(xsurf)

        myplot = XSection(zmin=1700, zmax=2500, well=mywell,
                          surfaces=mysurfaces)

        # set the color table, from file
        clist = [0, 1, 222, 3, 5, 7, 3, 12, 11, 10, 9, 8]
        cfil1 = 'xtgeo'
        cfil2 = '../xtgeo-testdata/etc/colortables/colfacies.txt'
        myplot.set_colortable(cfil1, colorlist=None)

        myplot.canvas(title="Manamana", subtitle="My Dear Well")

        myplot.plot_surfaces(fill=True)

        myplot.plot_well()

        myplot.plot_map()

        myplot.show()

    def test_troll1(self):
        """
        Test XSect for a Troll well.
        """
        self.getlogger('test_troll1')

        myfield = Polygons()
        myfield.from_file('../xtgeo-testdata/polygons/tro/1/troll.xyz')

        mywells = []
        wnames = glob.glob("../xtgeo-testdata/wells/tro/2/31_2-G-4_H.w")
        wnames.sort()
        for wname in wnames:
            mywell = Well()
            mywell.from_file(wname)
            mywells.append(mywell)

        self.logger.info("Wells are read...")

        mysurfaces = []
        surfnames = glob.glob("../xtgeo-testdata/surfaces/tro/2/*.gri")
        surfnames.sort()
        for fname in surfnames:
            self.logger.info("Read {}".format(fname))
            mysurf = RegularSurface()
            mysurf.from_file(fname)
            mysurfaces.append(mysurf)

        # Troll lobes
        mylobes = []
        surfnames = glob.glob("../xtgeo-testdata/surfaces/tro/3/*.gri")
        surfnames.sort()
        for fname in surfnames:
            self.logger.info("Read {}".format(fname))
            mysurf = RegularSurface()
            mysurf.from_file(fname)
            mylobes.append(mysurf)

        for wo in mywells:
            myplot = XSection(zmin=1300, zmax=2000, well=wo,
                              surfaces=mysurfaces, zonelogshift=-1,
                              outline=myfield)

            myplot.canvas(title=wo.truewellname,
                          subtitle="Before my corrections",
                          infotext="Heisan sveisan", figscaling=1.2)

            # myplot.colortable('xtgeo')

            myplot.plot_surfaces(fill=True)

            myplot.plot_surfaces(surfaces=mylobes, fill=False, linewidth=4,
                                 legendtitle="Lobes", fancyline=True)

            myplot.plot_well()

            myplot.plot_wellmap()

            myplot.plot_map()

            if not myplot.show():
                print("Could not plot for some reason")

            myplot.savefig('x.png')


if __name__ == '__main__':

    unittest.main()
