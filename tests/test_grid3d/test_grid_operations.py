#!/usr/bin/env python -u
import sys

import pytest
import numpy as np

from xtgeo.grid3d import Grid
from xtgeo.grid3d import GridProperty
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    sys.exit(-9)

td = xtg.tmpdir
testpath = xtg.testpath

# =============================================================================
# Do tests
# =============================================================================
emegfile = '../xtgeo-testdata/3dgrids/eme/1/emerald_hetero_grid.roff'
emerfile = '../xtgeo-testdata/3dgrids/eme/1/emerald_hetero_region.roff'

emegfile2 = '../xtgeo-testdata/3dgrids/eme/2/emerald_hetero_grid.roff'
emezfile2 = '../xtgeo-testdata/3dgrids/eme/2/emerald_hetero.roff'


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



def test_refine_vertically_per_zone():
    """Do a grid refinement vertically, via a dict per zone."""

    logger.info('Read grid...')

    grd = Grid(emegfile2)
    logger.info('Read grid... done, NLAY is {}'.format(grd.nlay))
    grd.to_file('TMP/test_refined_by_dict_initial.roff')
    dz1 = grd.get_dz().values

    zone = GridProperty(emezfile2, grid=grd, name='Zone')
    logger.info('Zone values min max: %s %s', zone.values.min(),
                zone.values.max())

    logger.info('Subgrids list: %s', grd.subgrids)

    refinement = {1: 4, 2: 2}
    grd.refine_vertically(refinement, zoneprop=zone)

    grd.to_file('TMP/test_refined_by_dict.roff')


def test_crop_grid():
    """Crop a grid."""

    logger.info('Read grid...')

    grd = Grid(emegfile)
    logger.info('Read grid... done, NLAY is {}'.format(grd.nlay))
    logger.info('Read grid...NCOL, NROW, NLAY is {} {} {}'
                .format(grd.ncol, grd.nrow, grd.nlay))

    grd.do_cropping(((30, 60), (20, 40), (1, 46)))

    grd.to_file('TMP/grid_cropped.roff')
