import os
import logging
import sys

import pytest
import numpy as np
import numpy.ma as ma

from xtgeo.grid3d import Grid
from xtgeo.grid3d import GridProperty
from xtgeo.surface import RegularSurface
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

roff1_grid = '../xtgeo-testdata/3dgrids/eme/1/emerald_hetero_grid.roff'
roff1_props = '../xtgeo-testdata/3dgrids/eme/1/emerald_hetero.roff'


@tsetup.skipifroxar
def test_hcpvfz1():
    """HCPV thickness map."""

    # It is important that inpyut are pure numpies, not masked

    logger.info('Name is {}'.format(__name__))
    g = Grid()
    logger.info("Import roff...")
    g.from_file(roff1_grid, fformat="roff")

    # get the hcpv
    st = GridProperty()
    to = GridProperty()

    st.from_file(roff1_props, name='Oil_HCPV')

    to.from_file(roff1_props, name='Oil_bulk')

    # get the dz and the coordinates, with no mask (ie get value for outside)
    dz = g.get_dz(mask=False)
    xc, yc, zc = g.get_xyz(mask=False)

    xcv = ma.filled(xc.values3d)
    ycv = ma.filled(yc.values3d)
    dzv = ma.filled(dz.values3d)

    hcpfz = ma.filled(st.values3d, fill_value=0.0)
    tov = ma.filled(to.values3d, fill_value=10)
    tov[tov < 1.0e-32] = 1.0e-32
    hcpfz = hcpfz * dzv / tov

    # make a map... estimate from xc and yc
    xmin = xcv.min()
    xmax = xcv.max()
    ymin = ycv.min()
    ymax = ycv.max()
    xinc = (xmax - xmin) / 50
    yinc = (ymax - ymin) / 50

    logger.debug("xmin xmax ymin ymax, xinc, yinc: {} {} {} {} {} {} "
                 .format(xmin, xmax, ymin, ymax, xinc, yinc))

    hcmap = RegularSurface(nx=50, ny=50, xinc=xinc, yinc=yinc,
                           xori=xmin, yori=ymin, values=np.zeros((50, 50)))

    hcmap2 = RegularSurface(nx=50, ny=50, xinc=xinc, yinc=yinc,
                            xori=xmin, yori=ymin, values=np.zeros((50, 50)))

    zp = np.ones((g.ncol, g.nrow, g.nlay))
    # now make hcpf map

    t1 = xtg.timer()
    hcmap.hc_thickness_from_3dprops(xprop=xcv, yprop=ycv, dzprop=dzv,
                                    hcpfzprop=hcpfz, zoneprop=zp,
                                    zone_minmax=(1, 1))

    assert hcmap.values.mean() == pytest.approx(1.447, abs=0.1)

    t2 = xtg.timer(t1)

    logger.info('Speed basic is {}'.format(t2))

    t1 = xtg.timer()
    hcmap2.hc_thickness_from_3dprops(xprop=xcv, yprop=ycv, dzprop=dzv,
                                     hcpfzprop=hcpfz, zoneprop=zp, coarsen=2,
                                     zone_avg=True, zone_minmax=(1, 1),
                                     mask_outside=True)
    t2 = xtg.timer(t1)

    logger.info('Speed zoneavg coarsen 2 is {}'.format(t2))

    hcmap.quickplot(filename='TMP/quickplot_hcpv.png')
    hcmap2.quickplot(filename='TMP/quickplot_hcpv_zavg_coarsen.png')
    logger.debug(hcmap.values.mean())
