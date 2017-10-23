#!/usr/bin/env python -u
import os
import sys
import logging
import pytest

from xtgeo.grid3d import Grid
from xtgeo.grid3d import GridProperty
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
# Do tests
# =============================================================================
emegfile = '../xtgeo-testdata/3dgrids/eme/1/emerald_hetero_grid.roff'
emerfile = '../xtgeo-testdata/3dgrids/eme/1/emerald_hetero_region.roff'


def getlogger(name):

    format = xtg.loggingformat

    logging.basicConfig(format=format, stream=sys.stdout)
    logging.getLogger().setLevel(xtg.logginglevel)  # root logger!

    logger = logging.getLogger(name)
    return logger


def test_hybridgrid1():
    """Making a hybridgrid for Emerald case (ROFF and GRDECL"""

    logger = getlogger(__name__)

    logger.info('Read grid...')
    grd = Grid(emegfile)
    logger.info('Read grid... done, NZ is {}'.format(grd.nlay))
    grd.to_file('TMP/test_hybridgrid1_asis.grdecl', fformat='grdecl')

    logger.info('Convert...')
    nhdiv = 40
    newnlay = grd.nlay * 2 + nhdiv

    grd.convert_to_hybrid(nhdiv=nhdiv, toplevel=1700, bottomlevel=1740)

    logger.info('Hybrid grid... done, NLAY is now {}'.format(grd.nlay))

    assert grd.nlay == newnlay, "New NLAY number"

    dz = grd.get_dz()

    assert dz.values3d.mean() == pytest.approx(1.395, abs=0.01)

    logger.info('Export...')
    grd.to_file('TMP/test_hybridgrid1.roff')

    logger.info('Read grid 2...')
    grd2 = Grid('TMP/test_hybridgrid1_asis.grdecl')
    logger.info('Read grid... done, NLAY is {}'.format(grd2.nlay))

    logger.info('Convert...')
    nhdiv = 40
    newnz = grd2.nz * 2 + nhdiv

    grd2.convert_to_hybrid(nhdiv=nhdiv, toplevel=1700, bottomlevel=1740)

    logger.info('Hybrid grid... done, NZ is now {}'.format(grd2.nz))

    assert grd2.nlay == newnz, "New NLAY number"

    dz = grd2.get_dz()

    assert dz.values3d.mean() == pytest.approx(1.395, abs=0.01)


def test_hybridgrid2():
    """Making a hybridgrid for Emerald case in region"""

    logger = getlogger('test_hybridgrid2')

    logger.info('Read grid...')
    grd = Grid(emegfile)
    logger.info('Read grid... done, NLAY is {}'.format(grd.nlay))

    reg = GridProperty()
    reg.from_file(emerfile, name='REGION')

    nhdiv = 40

    grd.convert_to_hybrid(nhdiv=nhdiv, toplevel=1650, bottomlevel=1690,
                          region=reg, region_number=1)

    grd.to_file('TMP/test_hybridgrid2.roff')


def test_inactivate_thin_cells():
    """Make hybridgrid for Emerald case in region, and inactive thin cells"""

    logger = getlogger('test_hybridgrid2')

    logger.info('Read grid...')
    grd = Grid(emegfile)
    logger.info('Read grid... done, NLAY is {}'.format(grd.nlay))

    reg = GridProperty()
    reg.from_file(emerfile, name='REGION')

    nhdiv = 40

    grd.convert_to_hybrid(nhdiv=nhdiv, toplevel=1650, bottomlevel=1690,
                          region=reg, region_number=1)

    grd.inactivate_by_dz(0.001)

    grd.to_file('TMP/test_hybridgrid2_inact_thin.roff')
