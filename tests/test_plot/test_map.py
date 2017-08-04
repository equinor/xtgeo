import unittest
import os
import sys
import logging
import matplotlib.pyplot as plt

from xtgeo.plot import Map
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


class TestMap(unittest.TestCase):
    """Testing suite for map plots."""

    def getlogger(self, name):

        format = xtg.loggingformat

        logging.basicConfig(format=format, stream=sys.stdout)
        logging.getLogger().setLevel(xtg.logginglevel)  # root logger!

        self.logger = logging.getLogger(name)

    def test_very_basic(self):
        """Just test that matplotlib works."""
        plt.title('Hello world')
        plt.show()

    def test_very_basic_to_file(self):
        """Just test that matplotlib works, to a file."""
        plt.title('Hello world')
        plt.savefig('TMP/verysimple.png')

    def test_simple_plot(self):
        """Test as simple map plot only making an instance++ and plot."""
        self.getlogger('test_simple_plot')

        mysurf = RegularSurface()
        mysurf.from_file('../xtgeo-testdata/surfaces/gfb/1/'
                         'gullfaks_top.irapbin')

        # just make the instance, with a lot of defaults behind the scene
        myplot = Map()
        myplot.canvas(title='My o my')
        myplot.set_colortable('gist_ncar')
        myplot.plot_surface(mysurf)

        myplot.show()

    def test_more_features_plot(self):
        """Map with some more features added, such as label rotation."""
        self.getlogger('test_more_features_plot')

        mysurf = RegularSurface()
        mysurf.from_file('../xtgeo-testdata/surfaces/gfb/1/'
                         'gullfaks_top.irapbin')

        # just make the instance, with a lot of defaults behind the scene
        myplot = Map()
        myplot.canvas(title='Label rotation')
        myplot.set_colortable('gist_rainbow_r')
        myplot.plot_surface(mysurf, minvalue=1250, maxvalue=2200,
                            xlabelrotation=45)

        myplot.show()


if __name__ == '__main__':

    unittest.main()
