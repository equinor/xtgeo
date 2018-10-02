# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import os
import sys
import logging
import matplotlib.pyplot as plt
import pytest

from xtgeo.plot import Grid3DSlice
from xtgeo.grid3d import Grid
from xtgeo.grid3d import GridProperty
from xtgeo.common import XTGeoDialog
import test_common.test_xtg as tsetup

path = 'TMP'
try:
    os.makedirs(path)
except OSError:
    if not os.path.isdir(path):
        raise

xtg = XTGeoDialog()

format = xtg.loggingformat

logging.basicConfig(format=format, stream=sys.stdout)
logging.getLogger().setLevel(xtg.logginglevel)  # root logger!

logger = logging.getLogger(__name__)

if 'XTG_SHOW' in os.environ:
    xtgshow = True
else:
    xtgshow = False

logger.info('Use env variable XTG_SHOW to show interactive plots to screen')

# =========================================================================
# Do tests
# =========================================================================

usefile1 = '../xtgeo-testdata/3dgrids/reek/reek_sim_grid.roff'
usefile2 = '../xtgeo-testdata/3dgrids/reek/reek_sim_poro.roff'
usefile3 = '../xtgeo-testdata/etc/colortables/rainbow_reverse.rmscolor'


@tsetup.skipifroxar
def test_very_basic():
    """Just test that matplotlib works."""
    assert 'matplotlib' in str(plt)

    plt.title('Hello world')
    plt.savefig('TMP/helloworld1.png')
    plt.savefig('TMP/helloworld1.svg')
    if xtgshow:
        plt.show()
    logger.info('Very basic plotting')
    plt.close()


@tsetup.skipifroxar
def test_slice_simple():
    """Trigger XSection class, and do some simple things basically."""
    layslice = Grid3DSlice()

    mygrid = Grid(usefile1)
    myprop = GridProperty(usefile2, grid=mygrid, name='PORO')

    assert myprop.values.mean() == pytest.approx(0.1677, abs=0.001)

    layslice.canvas(title='My Grid plot')
    wnd = (454000, 455000, 6782000, 6783000)
    layslice.plot_gridslice(mygrid, myprop, window=wnd, colormap=usefile3)

    if xtgshow:
        layslice.show()
    else:
        print('Output to screen disabled (will plotto screen); '
              'use XTG_SHOW env variable')
        layslice.savefig('TMP/layerslice.png')


@tsetup.skipifroxar
def test_slice_plot_many_grid_layers():
    """Loop over layers and produce both SVG and PNG files to file"""

    mygrid = Grid(usefile1)
    myprop = GridProperty(usefile2, grid=mygrid, name='PORO')

    nlayers = mygrid.nz + 1

    layslice2 = Grid3DSlice()

    for k in range(1, nlayers, 4):
        print('Layer {} ...'.format(k))
        layslice2.canvas(title='Porosity for layer ' + str(k))
        layslice2.plot_gridslice(mygrid, myprop, colormap=usefile3, index=k,
                                 minvalue=0.18, maxvalue=0.36)
        layslice2.savefig('TMP/layerslice2_' + str(k) + '.svg', fformat='svg',
                          last=False)
        layslice2.savefig('TMP/layerslice2_' + str(k) + '.png')
