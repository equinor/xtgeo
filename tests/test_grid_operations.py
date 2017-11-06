#!/usr/bin/env python -u
import os
import sys
import logging
import pytest

from xtgeo.grid3d import Grid
from xtgeo.grid3d import GridProperty
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg._testsetup():
    sys.exit(-9)

skiplargetest = pytest.mark.skipif(xtg.bigtest is False,
                                   reason="Big tests skip")

td = xtg.tmpdir
testpath = xtg.testpath

skiplargetest = pytest.mark.skipif(xtg.bigtest is False,
                                   reason="Big tests skip")


# =============================================================================
# Do tests
# =============================================================================
emegfile = '../xtgeo-testdata/3dgrids/eme/1/emerald_hetero_grid.roff'
emerfile = '../xtgeo-testdata/3dgrids/eme/1/emerald_hetero_region.roff'

maugfile = '../xtgeo-testdata/3dgrids/mau/mau.roff'

def test_hybridgrid1():
    """Making a hybridgrid for Emerald case (ROFF and GRDECL"""

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


def test_refine_vertically():
    """Do a grid refinement vertically."""

    logger.info('Read grid...')

    grd = Grid(emegfile)
    logger.info('Read grid... done, NLAY is {}'.format(grd.nlay))

    avg_dz1 = grd.get_dz().values3d.mean()

    # idea; either a scalar (all cells), or a dictionary for zone wise
    grd.refine_vertically(3)

    avg_dz2 = grd.get_dz().values3d.mean()

    assert avg_dz1 == pytest.approx(3 * avg_dz2, abs=0.0001)

    grd.inactivate_by_dz(0.001)

    grd.to_file('TMP/test_refined_by_3.roff')


@skiplargetest
def test_refine_vertically_mau():
    """Do a grid refinement vertically, Maureen case."""

    logger.info('Read grid...')

    grd = Grid(maugfile)
    logger.info('Read grid... done, NLAY is {}'.format(grd.nlay))
    logger.info('Read grid... done, NCOL is {}'.format(grd.ncol))
    logger.info('Read grid... done, NROW is {}'.format(grd.nrow))

    avg_dz1 = grd.get_dz().values3d.mean()

    logger.info('AVG dZ prior is {}'.format(avg_dz1))

    grd.refine_vertically(2)

    avg_dz2 = grd.get_dz().values3d.mean()

    logger.info('AVG dZ post refine is {}'.format(avg_dz2))

    assert avg_dz1 == pytest.approx(2 * avg_dz2, abs=0.0001)

    grd.inactivate_by_dz(0.00001)

    grd.to_file('TMP/test_refined_by_2_mau.roff')
