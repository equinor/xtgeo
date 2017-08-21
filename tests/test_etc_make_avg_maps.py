import unittest
import numpy as np
import numpy.ma as ma
import os
import sys
import logging

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

# =============================================================================
# This tests a combination of methods, in order to produce maps of HC thickness
# =============================================================================
gfile1 = '../xtgeo-testdata/3dgrids/bri/B.GRID'
ifile1 = '../xtgeo-testdata/3dgrids/bri/B.INIT'

gfile2 = '../xtgeo-testdata/3dgrids/gfb/GULLFAKS.EGRID'
ifile2 = '../xtgeo-testdata/3dgrids/gfb/GULLFAKS.INIT'
rfile2 = '../xtgeo-testdata/3dgrids/gfb/GULLFAKS.UNRST'


class TestEtcMakeAvgMaps(unittest.TestCase):
    """Testing suite making avg maps"""

    def getlogger(self, name):

        # if isinstance(self.logger):
        #     return

        format = xtg.loggingformat

        logging.basicConfig(format=format, stream=sys.stdout)
        logging.getLogger().setLevel(xtg.logginglevel)  # root logger!

        self.logger = logging.getLogger(name)

    def test_avg01(self):
        """
        Make average map from very simple Eclipse.
        """
        self.getlogger('test_avg01')

        g = Grid()
        g.from_file(gfile1, fformat="grid")

        # get the poro
        po = GridProperty()
        po.from_file(ifile1, fformat='init', name='PORO', grid=g)

        # get the dz and the coordinates
        dz = g.get_dz(mask=False)
        xc, yc, zc = g.get_xyz(mask=False)

        # get actnum
        actnum = g.get_actnum()

        # convert from masked numpy to ordinary
        xcuse = np.copy(xc.values3d)
        ycuse = np.copy(yc.values3d)
        dzuse = np.copy(dz.values3d)
        pouse = np.copy(po.values3d)

        # dz must be zero for undef cells
        dzuse[actnum.values3d < 0.5] = 0.0
        pouse[actnum.values3d < 0.5] = 0.0

        # make a map... estimate from xc and yc

        avgmap = RegularSurface(nx=55, ny=50, xinc=400, yinc=375,
                                xori=-100, yori=0, values=np.zeros((55, 50)))

        avgmap.avg_from_3dprop(xprop=xcuse, yprop=ycuse,
                               mprop=pouse, dzprop=dzuse,
                               layer_minmax=(1, 9), truncate_le=0.001)

        avgmap.quickplot(filename='TMP/tmp_poro.png')
        avgmap.to_file('TMP/tmp.poro.gri')

        self.assertAlmostEqual(avgmap.values.mean(), 0.264, places=3)

    def test_avg02(self):
        """
        Make average map from Gullfaks Eclipse.
        """
        self.getlogger('test_avg02')

        g = Grid()
        g.from_file(gfile2, fformat="egrid")

        # get the poro
        po = GridProperty()
        po.from_file(ifile2, fformat='init', name='PORO', grid=g)

        # get the dz and the coordinates
        dz = g.get_dz(mask=False)
        xc, yc, zc = g.get_xyz(mask=False)

        # get actnum
        actnum = g.get_actnum()

        # convert from masked numpy to ordinary
        xcuse = np.copy(xc.values3d)
        ycuse = np.copy(yc.values3d)
        dzuse = np.copy(dz.values3d)
        pouse = np.copy(po.values3d)

        # dz must be zero for undef cells
        dzuse[actnum.values3d < 0.5] = 0.0
        pouse[actnum.values3d < 0.5] = 0.0

        # make a map... estimate from xc and yc

        avgmap = RegularSurface(nx=220, ny=260, xinc=50, yinc=50,
                                xori=451100, yori=6779700,
                                values=np.zeros((220, 260)))

        avgmap.avg_from_3dprop(xprop=xcuse, yprop=ycuse,
                               mprop=pouse, dzprop=dzuse,
                               layer_minmax=(5, 6),
                               truncate_le=None)

        avgmap.quickplot(filename='TMP/tmp_poro2.png', xlabelrotation=30)
        avgmap.to_file('TMP/tmp.poro.gri', fformat='irap_ascii')

        self.logger.info(avgmap.values.mean())
        self.assertAlmostEqual(avgmap.values.mean(), 0.158, places=3)

    def test_avg03(self):
        """
        Make average depth map where Sw is between 0.3 and 0.33
        """

        self.getlogger('test_avg03')

        self.logger.info("Reading Grid file")
        g = Grid()
        g.from_file(gfile2, fformat="egrid")

        # get the sw
        self.logger.info("Reading GridProperty file")
        sw = GridProperty()
        sw.from_file(rfile2, fformat='unrst', name='SWAT', date=19851001,
                     grid=g)

        # # # get the sw2
        # # sw2=GridProperty()
        # # sw2.from_file('../../testdata/Zone/GULLFAKS.UNRST', fformat='unrst',
        # #               name='SWAT', date=20150101, grid=g)

        # self.logger.info("Compute...")
        # # get the dz and the coordinates
        # dz = g.get_dz(mask=False)
        # xc, yc, zc = g.get_xyz(mask=False)

        # # get actnum
        # actnum = g.get_actnum()

        # # convert from masked numpy to ordinary
        # xcuse = np.copy(xc.values3d)
        # ycuse = np.copy(yc.values3d)
        # zcuse = np.copy(zc.values3d)
        # # zc2use = np.copy(zc.values3d) #
        # dzuse = np.copy(dz.values3d)

        # swuse = np.copy(sw.values3d)
        # # sw2use = np.copy(sw2.values3d)

        # # dz must be zero for undef cells
        # dzuse[actnum.values3d < 0.5] = 0.0
        # swuse[actnum.values3d < 0.5] = 0.0
        # # sw2use[actnum.values3d<0.5] = 0.0

        # zcuse[swuse < 0.3] = 0.0
        # zcuse[swuse > 0.33] = 0.0

        # # zc2use[sw2use<0.3]=0.0
        # # zc2use[sw2use>0.33]=0.0

        # zcuse = ma.array(zcuse)
        # zcuse = ma.masked_less(zcuse, 999)  # e.g. depth 999

        # # zc2use = ma.array(zc2use)
        # # zc2use = ma.masked_less(zc2use, 999) # e.g. depth 999

        # self.logger.info(zcuse.min())
        # self.logger.info(zcuse.max())

        # dzuse[zcuse < zcuse.min()] = 0.0
        # dzuse[zcuse > zcuse.max()] = 0.0

        # # dz2use[zc2use<zc2use.min()] = 0.0
        # # dz2use[zc2use>zc2use.max()] = 0.0

        # zcuse = zcuse.filled(0.0)

        # # make a map... estimate from xc and yc

        # avgmap = RegularSurface(nx=220, ny=260, xinc=50, yinc=50,
        #                         xori=451100, yori=6779700,
        #                         values=np.zeros((220, 260)))

        # avgmap.avg_from_3dprop(xprop=xcuse, yprop=ycuse,
        #                        mprop=zcuse, dzprop=dzuse,
        #                        layer_minmax=(1, 47),
        #                        truncate_le=10)

        # avgmap.quickplot(filename='TMP/tmp_depth.png')
        # avgmap.to_file('TMP/tmp_depth.gri', fformat='irap_binary')

        # # avgmap2 = RegularSurface(nx=220, ny=260, xinc=50, yinc=50,
        # #                          xori=451100, yori=6779700,
        # #                          values=np.zeros((220,260)))

        # # avgmap2.avg_from_3dprop(xprop=xcuse, yprop=ycuse,
        # #                         mprop=zc2use, dz2prop=dzuse,
        # #                         lay_minmax=(1,47),
        # #                         truncate_le=10)

        # # avgmap2.quickplot(filename='TMP/tmp2_depth.png')
        # # avgmap2.to_file('TMP/tmp2_depth.gri', fformat='irap_binary')

        # self.logger.info("See output on TMP ...")


if __name__ == '__main__':

    unittest.main()
