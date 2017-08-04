import unittest
import numpy as np
import os
import logging
import sys

from xtgeo.grid3d import Grid
from xtgeo.grid3d import GridProperty
from xtgeo.surface import RegularSurface
from xtgeo.common import XTGeoDialog

path = 'TMP'
try:
    os.makedirs(path)
except OSError:
    if not os.path.isdir(path):
        raise

# set default level
xtg = XTGeoDialog()


class TestEtcMakeHcpvMaps(unittest.TestCase):
    """Testing suite making HC thickness maps"""

    def getlogger(self, name):

        # if isinstance(self.logger):
        #     return

        fm = '%(msecs)6.2f Line: %(lineno)4d %(name)44s [%(funcName)40s()]'\
             + '%(levelname)8s:'\
             + '\t%(message)s'

        logging.basicConfig(format=fm, stream=sys.stdout)
        logging.getLogger().setLevel(logging.DEBUG)  # root logger!

        self.logger = logging.getLogger(name)

    def test_hcpvfz1(self):

        self.getlogger(sys._getframe(1).f_code.co_name)

        self.logger.info('Name is {}'.format(__name__))
        g = Grid()
        self.logger.info("Import roff...")
        g.from_file('../../testdata/Zone/emerald_hetero_grid.roff',
                    fformat="roff")

        # get the hcpv
        st = GridProperty()
        to = GridProperty()

        st.from_file('../../testdata/props/em/em1/emerald_hetero.roff',
                     name='Oil_HCPV')

        to.from_file('../../testdata/props/em/em1/emerald_hetero.roff',
                     name='Oil_bulk')

        # get the dz and the coordinates
        dz = g.get_dz()
        xc, yc, zc = g.get_xyz()

        hcpfz = np.array(st.values)
        hcpfz = dz.values * st.values / to.values

        hcpfz = np.reshape(hcpfz, (g.nx, g.ny, g.nz), order='F')

        # make a map... estimate from xc and yc
        xmin = xc.values.min()
        xmax = xc.values.max()
        ymin = yc.values.min()
        ymax = yc.values.max()
        xinc = (xmax - xmin) / 50
        yinc = (ymax - ymin) / 50

        self.logger.debug("xmin xmax ymin ymax, xinc, yinc: {} {} {} {} {} {} "
                          .format(xmin, xmax, ymin, ymax, xinc, yinc))

        hcmap = RegularSurface(nx=50, ny=50, xinc=xinc, yinc=yinc,
                               xori=xmin, yori=ymin, values=np.zeros((50, 50)))

        zp = np.ones((g.nx, g.ny, g.nz))
        # now make hcpf map

        hcmap.hc_thickness_from_3dprops(xprop=xc.values3d, yprop=yc.values3d,
                                        hcpfzprop=hcpfz, zoneprop=zp)

        hcmap.quickplot(filename='TMP/tull.png')
        #        os.system("eog /tmp/tull.png")


if __name__ == '__main__':

    unittest.main()
