import pytest
import numpy as np
import os
import sys
import logging

from xtgeo.grid3d import Grid
from xtgeo.grid3d import GridProperty
from xtgeo.surface import RegularSurface
from xtgeo.xyz import Polygons
from xtgeo.common import XTGeoDialog

import test_common.test_xtg as tsetup

path = 'TMP'
try:
    os.makedirs(path)
except OSError:
    if not os.path.isdir(path):
        raise

# set default level
xtg = XTGeoDialog()

logging.basicConfig(format=xtg.loggingformat, stream=sys.stdout)
logging.getLogger().setLevel(xtg.logginglevel)

logger = logging.getLogger(__name__)

# =============================================================================
# This tests a combination of methods, in order to produce maps of HC thickness
# =============================================================================
gfile1 = '../xtgeo-testdata/3dgrids/bri/B.GRID'
ifile1 = '../xtgeo-testdata/3dgrids/bri/B.INIT'

gfile2 = '../xtgeo-testdata/3dgrids/reek/REEK.EGRID'
ifile2 = '../xtgeo-testdata/3dgrids/reek/REEK.INIT'
rfile2 = '../xtgeo-testdata/3dgrids/reek/REEK.UNRST'

ffile1 = '../xtgeo-testdata/polygons/reek/1/top_upper_reek_faultpoly.zmap'


@tsetup.skipifroxar
def test_avg01():
    """Make average map from very simple Eclipse."""

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

    assert avgmap.values.mean() == pytest.approx(0.264, abs=0.001)


@tsetup.skipifroxar
def test_avg02():
    """Make average map from Gullfaks Eclipse."""
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

    avgmap = RegularSurface(nx=200, ny=250, xinc=50, yinc=50,
                            xori=457000, yori=5927000,
                            values=np.zeros((200, 250)))

    avgmap.avg_from_3dprop(xprop=xcuse, yprop=ycuse,
                           mprop=pouse, dzprop=dzuse,
                           layer_minmax=(5, 6),
                           truncate_le=None)

    # add the faults in plot
    fau = Polygons(ffile1, fformat='zmap')
    fspec = {'faults': fau}

    avgmap.quickplot(filename='TMP/tmp_poro2.png', xlabelrotation=30,
                     faults=fspec)
    avgmap.to_file('TMP/tmp.poro.gri', fformat='irap_ascii')

    logger.info(avgmap.values.mean())
    assert avgmap.values.mean() == pytest.approx(0.0828, abs=0.001)


@tsetup.skipifroxar
def test_avg03():
    """Make average depth map where Sw is between 0.3 and 0.33"""

    logger.info("Reading Grid file")
    g = Grid()
    g.from_file(gfile2, fformat="egrid")

    # get the sw
    logger.info("Reading GridProperty file")
    sw = GridProperty()
    sw.from_file(rfile2, fformat='unrst', name='SWAT', date=19991201,
                 grid=g)

    # # # get the sw2
    # # sw2=GridProperty()
    # # sw2.from_file('../../testdata/Zone/GULLFAKS.UNRST', fformat='unrst',
    # #               name='SWAT', date=20150101, grid=g)

    # logger.info("Compute...")
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

    # logger.info(zcuse.min())
    # logger.info(zcuse.max())

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

    # logger.info("See output on TMP ...")
