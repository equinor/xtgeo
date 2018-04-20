import os
import sys
import logging

import pytest
import numpy as np
import numpy.ma as ma

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
    po = ma.filled(po.values3d, fill_value=po.values.mean())

    # get the dz and the coordinates
    dz = g.get_dz(mask=False)
    dz = ma.filled(dz.values3d)
    xc, yc, zc = g.get_xyz(mask=False)
    xc = ma.filled(xc.values3d)
    yc = ma.filled(yc.values3d)
    zc = ma.filled(zc.values3d)


    # get actnum
    actnum = g.get_actnum()
    actnum = ma.filled(actnum.values3d)

    # dz must be zero for undef cells
    dz[actnum < 0.5] = 0.0

    zoneprop = np.ones((actnum.shape))

    avgmap = RegularSurface(nx=55, ny=50, xinc=400, yinc=375,
                            xori=-100, yori=0, values=np.zeros((55, 50)))


    avgmap.avg_from_3dprop(xprop=xc, yprop=yc, zoneprop=zoneprop,
                           mprop=po, dzprop=dz, truncate_le=0.001,
                           zone_minmax=(1, 1))

    avgmap.quickplot(filename='TMP/tmp_poro.png')
    avgmap.to_file('TMP/tmp.poro.gri')

    assert avgmap.values.mean() == pytest.approx(0.264, abs=0.001)


@tsetup.skipifroxar
def test_avg02():
    """Make average map from Reek Eclipse."""
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
    zuse = np.ones((xcuse.shape))

    avgmap = RegularSurface(nx=200, ny=250, xinc=50, yinc=50,
                            xori=457000, yori=5927000,
                            values=np.zeros((200, 250)))

    avgmap.avg_from_3dprop(xprop=xcuse, yprop=ycuse, zoneprop=zuse,
                           zone_minmax=(1, 1),
                           mprop=pouse, dzprop=dzuse,
                           truncate_le=None)

    # add the faults in plot
    fau = Polygons(ffile1, fformat='zmap')
    fspec = {'faults': fau}

    avgmap.quickplot(filename='TMP/tmp_poro2.png', xlabelrotation=30,
                     faults=fspec)
    avgmap.to_file('TMP/tmp.poro.gri', fformat='irap_ascii')

    logger.info(avgmap.values.mean())
    assert avgmap.values.mean() == pytest.approx(0.1653, abs=0.01)


@tsetup.skipifroxar
def test_avg03():
    """Make average map from Reek Eclipse, speed up by zone_avg."""
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
    actnum = actnum.get_npvalues3d()

    # convert from masked numpy to ordinary
    xcuse = xc.get_npvalues3d()
    ycuse = yc.get_npvalues3d()
    dzuse = dz.get_npvalues3d(fill_value=0.0)
    pouse = po.get_npvalues3d(fill_value=0.0)

    # dz must be zero for undef cells
    dzuse[actnum < 0.5] = 0.0
    pouse[actnum < 0.5] = 0.0

    # make a map... estimate from xc and yc
    zuse = np.ones((xcuse.shape))

    avgmap = RegularSurface(nx=200, ny=250, xinc=50, yinc=50,
                            xori=457000, yori=5927000,
                            values=np.zeros((200, 250)))

    avgmap.avg_from_3dprop(xprop=xcuse, yprop=ycuse, zoneprop=zuse,
                           zone_minmax=(1, 1),
                           mprop=pouse, dzprop=dzuse,
                           truncate_le=None, zone_avg=True)

    # add the faults in plot
    fau = Polygons(ffile1, fformat='zmap')
    fspec = {'faults': fau}

    avgmap.quickplot(filename='TMP/tmp_poro3.png', xlabelrotation=30,
                     faults=fspec)
    avgmap.to_file('TMP/tmp.poro3.gri', fformat='irap_ascii')

    logger.info(avgmap.values.mean())
    assert avgmap.values.mean() == pytest.approx(0.1653, abs=0.01)
