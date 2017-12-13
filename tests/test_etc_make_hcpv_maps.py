import pytest
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

logging.basicConfig(format=xtg.loggingformat, stream=sys.stdout)
logging.getLogger().setLevel(xtg.logginglevel)

logger = logging.getLogger(__name__)

roff1_grid = '../xtgeo-testdata/3dgrids/eme/1/emerald_hetero_grid.roff'
roff1_props = '../xtgeo-testdata/3dgrids/eme/1/emerald_hetero.roff'


def test_hcpvfz1():

    logger.info('Name is {}'.format(__name__))
    g = Grid()
    logger.info("Import roff...")
    g.from_file(roff1_grid, fformat="roff")

    # get the hcpv
    st = GridProperty()
    to = GridProperty()

    st.from_file(roff1_props, name='Oil_HCPV')

    to.from_file(roff1_props, name='Oil_bulk')

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

    logger.debug("xmin xmax ymin ymax, xinc, yinc: {} {} {} {} {} {} "
                 .format(xmin, xmax, ymin, ymax, xinc, yinc))

    hcmap = RegularSurface(nx=50, ny=50, xinc=xinc, yinc=yinc,
                           xori=xmin, yori=ymin, values=np.zeros((50, 50)))

    zp = np.ones((g.nx, g.ny, g.nz))
    # now make hcpf map

    hcmap.hc_thickness_from_3dprops(xprop=xc.values3d, yprop=yc.values3d,
                                    hcpfzprop=hcpfz, zoneprop=zp)

    hcmap.quickplot(filename='TMP/quickplot_hcpv.png')
    logger.debug(hcmap.values.mean())

    assert hcmap.values.mean() == pytest.approx(3, abs=0.1)
