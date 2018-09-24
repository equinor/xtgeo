# -*- coding: utf-8 -*-
from __future__ import division, absolute_import
from __future__ import print_function

from os.path import join
from filecmp import cmp

import pytest

from xtgeo.grid3d import Grid
from xtgeo.grid3d import GridProperty
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TDP = xtg.tmpdir
TESTPATH = xtg.testpath

# =============================================================================
# Do tests
# =============================================================================
EMEGFILE = '../xtgeo-testdata/3dgrids/eme/1/emerald_hetero_grid.roff'
EMERFILE = '../xtgeo-testdata/3dgrids/eme/1/emerald_hetero_region.roff'

EMEGFILE2 = '../xtgeo-testdata/3dgrids/eme/2/emerald_hetero_grid.roff'
EMEZFILE2 = '../xtgeo-testdata/3dgrids/eme/2/emerald_hetero.roff'


def test_hybridgrid1():
    """Making a hybridgrid for Emerald case (ROFF and GRDECL"""

    logger.info('Read grid...')
    grd = Grid(EMEGFILE)
    logger.info('Read grid... done, NZ is {}'.format(grd.nlay))
    grd.to_file(join(TDP, 'test_hybridgrid1_asis.grdecl'), fformat='grdecl')

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
    grd = Grid(EMEGFILE)
    logger.info('Read grid... done, NLAY is {}'.format(grd.nlay))

    reg = GridProperty()
    reg.from_file(EMERFILE, name='REGION')

    nhdiv = 40

    grd.convert_to_hybrid(nhdiv=nhdiv, toplevel=1650, bottomlevel=1690,
                          region=reg, region_number=1)

    grd.to_file('TMP/test_hybridgrid2.roff')


def test_inactivate_thin_cells():
    """Make hybridgrid for Emerald case in region, and inactive thin cells"""

    logger.info('Read grid...')
    grd = Grid(EMEGFILE)
    logger.info('Read grid... done, NLAY is {}'.format(grd.nlay))

    reg = GridProperty()
    reg.from_file(EMERFILE, name='REGION')

    nhdiv = 40

    grd.convert_to_hybrid(nhdiv=nhdiv, toplevel=1650, bottomlevel=1690,
                          region=reg, region_number=1)

    grd.inactivate_by_dz(0.001)

    grd.to_file('TMP/test_hybridgrid2_inact_thin.roff')


def test_refine_vertically():
    """Do a grid refinement vertically."""

    logger.info('Read grid...')

    grd = Grid(EMEGFILE)
    logger.info('Read grid... done, NLAY is {}'.format(grd.nlay))
    logger.info('Subgrids before: %s', grd.get_subgrids())

    avg_dz1 = grd.get_dz().values3d.mean()

    # idea; either a scalar (all cells), or a dictionary for zone wise
    grd.refine_vertically(3)

    avg_dz2 = grd.get_dz().values3d.mean()

    assert avg_dz1 == pytest.approx(3 * avg_dz2, abs=0.0001)

    logger.info('Subgrids after: %s', grd.get_subgrids())
    grd.inactivate_by_dz(0.001)

    grd.to_file('TMP/test_refined_by_3.roff')


def test_refine_vertically_per_zone():
    """Do a grid refinement vertically, via a dict per zone."""

    logger.info('Read grid...')

    grd_orig = Grid(EMEGFILE2)
    grd = grd_orig.copy()

    logger.info('Read grid... done, NLAY is {}'.format(grd.nlay))
    grd.to_file('TMP/test_refined_by_dict_initial.roff')

    logger.info('Subgrids before: %s', grd.get_subgrids())

    zone = GridProperty(EMEZFILE2, grid=grd, name='Zone')
    logger.info('Zone values min max: %s %s', zone.values.min(),
                zone.values.max())

    logger.info('Subgrids list: %s', grd.subgrids)

    refinement = {1: 4, 2: 2}
    grd.refine_vertically(refinement, zoneprop=zone)

    grd1s = grd.get_subgrids()
    logger.info('Subgrids after: %s', grd1s)

    grd.to_file('TMP/test_refined_by_dict.roff')

    grd = grd_orig.copy()
    grd.refine_vertically(refinement)  # no zoneprop
    grd2s = grd.get_subgrids()
    logger.info('Subgrids after: %s', grd2s)
    assert list(grd1s.values()) == list(grd2s.values())

    grd = grd_orig.copy()
    with pytest.raises(RuntimeError):
        grd.refine_vertically({1: 200}, zoneprop=zone)


def test_copy_grid():
    """Crop a grid."""

    grd = Grid(EMEGFILE2)
    grd2 = grd.copy()

    grd.to_file(join('TMP', 'gcp1.roff'))
    grd2.to_file(join('TMP', 'gcp2.roff'))
    assert cmp(join('TMP', 'gcp1.roff'), join('TMP', 'gcp2.roff')) is True


def test_crop_grid():
    """Crop a grid."""

    logger.info('Read grid...')

    grd = Grid(EMEGFILE2)
    zprop = GridProperty(EMEZFILE2, name='Zone', grid=grd)

    logger.info('Read grid... done, NLAY is {}'.format(grd.nlay))
    logger.info('Read grid...NCOL, NROW, NLAY is {} {} {}'
                .format(grd.ncol, grd.nrow, grd.nlay))

    grd.crop((30, 60), (20, 40), (1, 46), props=[zprop])

    grd.to_file(join('TMP', 'grid_cropped.roff'))

    grd2 = Grid(join('TMP', 'grid_cropped.roff'))

    assert grd2.ncol == 31


def test_crop_grid_after_copy():
    """Copy a grid, then crop and check number of active cells."""

    logger.info('Read grid...')

    grd = Grid(EMEGFILE2)
    grd.describe()
    zprop = GridProperty(EMEZFILE2, name='Zone', grid=grd)
    grd.describe(details=True)

    logger.info(grd.dimensions)

    grd2 = grd.copy()
    grd2.describe(details=True)

    logger.info('GRD2 props: %s', grd2.props)
    assert grd.propnames == grd2.propnames

    logger.info('GRD2 number of active cells: %s', grd2.nactive)
    act = grd.get_actnum()
    logger.info(act.values.shape)
    logger.info('ZPROP: %s', zprop._values.shape)

    grd2.crop((1, 30), (40, 80), (23, 46))

    grd2.describe(details=True)
