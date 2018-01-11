# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function

import os
import pytest

from xtgeo.grid3d import Grid
from xtgeo.grid3d import GridProperty
from xtgeo.common import XTGeoDialog

from ..test_common.test_xtg import assert_almostequal

# set default level
xtg = XTGeoDialog()

logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

td = xtg.tmpdir
testpath = xtg.testpath

# =============================================================================
# Do tests
# =============================================================================

testfile1 = '../xtgeo-testdata/3dgrids/reek/reek_sim_poro.roff'
testfile2 = '../xtgeo-testdata/3dgrids/eme/1/emerald_hetero.roff'
testfile3 = '../xtgeo-testdata/3dgrids/bri/B.GRID'
testfile4 = '../xtgeo-testdata/3dgrids/bri/B.INIT'
testfile5 = '../xtgeo-testdata/3dgrids/reek/REEK.EGRID'
testfile6 = '../xtgeo-testdata/3dgrids/reek/REEK.INIT'
testfile7 = '../xtgeo-testdata/3dgrids/reek/REEK.UNRST'
testfile8 = '../xtgeo-testdata/3dgrids/reek/reek_sim_zone.roff'
testfile8a = '../xtgeo-testdata/3dgrids/reek/reek_sim_grid.roff'
testfile9 = testfile1
testfile10 = '../xtgeo-testdata/3dgrids/bri/b_grid.roff'
testfile11 = '../xtgeo-testdata/3dgrids/bri/b_poro.roff'


def test_create():
    x = GridProperty()
    assert x.ncol == 5, 'NCOL'
    assert x.nrow == 12, 'NROW'

    m = GridProperty(discrete=True)
    (repr(m.values))


def test_roffbin_import1():

    logger.info('Name is {}'.format(__name__))

    x = GridProperty()
    logger.info("Import roff...")
    x.from_file(testfile1, fformat="roff", name='PORO')

    logger.info(repr(x.values))
    logger.info(x.values.dtype)
    logger.info("Mean porosity is {}".format(x.values.mean()))
    assert x.values.mean() == pytest.approx(0.1677, abs=0.001)


def test_roffbin_import2():
    """Import roffbin, with several props in one file."""

    logger.info('Name is {}'.format(__name__))
    dz = GridProperty()
    logger.info("Import roff...")
    dz.from_file(testfile2, fformat="roff", name='Z_increment')

    logger.info(repr(dz.values))
    logger.info(dz.values.dtype)
    logger.info("Mean DZ is {}".format(dz.values.mean()))

    hc = GridProperty()
    logger.info("Import roff...")
    hc.from_file(testfile2, fformat="roff", name='Oil_HCPV')

    logger.info(repr(hc.values))
    logger.info(hc.values.dtype)
    logger.info(hc.values3d.shape)
    ncol, nrow, nlay = hc.values3d.shape

    assert nrow == 100, 'NROW from shape (Emerald)'

    logger.info("Mean HCPV is {}".format(hc.values.mean()))


def test_eclinit_import():
    """Property import from Eclipse. Needs a grid object first. Eclipse GRID"""

    logger.info('Name is {}'.format(__name__))
    gg = Grid(testfile3, fformat="grid")
    po = GridProperty()
    logger.info("Import INIT...")
    po.from_file(testfile4, fformat="init", name='PORO', grid=gg,
                 apiversion=2)

    assert po.ncol == 20, 'NX from B.INIT'

    logger.debug(po.values[0:400])
    assert float(po.values3d[1:2, 13:14, 0:1]) == \
        pytest.approx(0.17146, abs=0.001), 'PORO in cell 2 14 1'

    # discrete prop
    eq = GridProperty(testfile4, fformat="init", name='EQLNUM', grid=gg)
    logger.info(eq.values[0:400])
    assert eq.values3d[12:13, 13:14, 0:1] == 3, 'EQLNUM in cell 13 14 1'


def test_eclinit_import_reek():
    """Property import from Eclipse. Reek"""

    # let me guess the format (shall be egrid)
    gg = Grid(testfile5, fformat='egrid')
    assert gg.ncol == 40, "Reek NX"

    logger.info("Import INIT...")
    po = GridProperty(testfile6, name='PORO', grid=gg)

    logger.info(po.values.mean())
    assert po.values.mean() == pytest.approx(0.1677, abs=0.0001)

    pv = GridProperty(testfile6, name='PORV', grid=gg)
    logger.info(pv.values.mean())


def test_eclunrst_import_reek():
    """Property UNRST import from Eclipse. Reek"""

    gg = Grid(testfile5, fformat='egrid')

    logger.info("Import RESTART (UNIFIED) ...")
    press = GridProperty(testfile7, name='PRESSURE', fformat='unrst',
                         date=19991201, grid=gg)

    assert_almostequal(press.values.mean(), 334.5232, 0.0001)

def test_export_roff():
    """Property import from Eclipse. Then export to roff."""

    gg = Grid()
    gg.from_file(testfile3, fformat="grid")
    po = GridProperty()
    logger.info("Import INIT...")
    po.from_file(testfile4, fformat="init", name='PORO', grid=gg, apiversion=2)

    po.to_file(os.path.join(td, 'bdata.roff'), name='PORO')
    pox = GridProperty(os.path.join(td, 'bdata.roff'), name='PORO')

    print(po.values.mean())

    assert po.values.mean() == pytest.approx(pox.values.mean(), abs=0.0001)


def test_io_roff_discrete():
    """Import ROFF discrete property; then TODO! export to ROFF int."""

    logger.info('Name is {}'.format(__name__))
    po = GridProperty()
    po.from_file(testfile8, fformat="roff", name='Zone')

    logger.info("\nCodes ({})\n{}".format(po.ncodes, po.codes))

    # tests:
    assert po.ncodes == 3
    logger.debug(po.codes[3])
    assert po.codes[3] == 'Below_Low_reek'

    # export discrete to ROFF ...TODO!
    # po.to_file(os.join.path(td, zone_export.roff'))


def test_get_all_corners():
    """Get X Y Z for all corners as XTGeo GridProperty objects"""

    grid = Grid(testfile8a)
    allc = grid.get_xyz_corners()

    x0 = allc[0]
    y0 = allc[1]
    z0 = allc[2]
    x1 = allc[3]
    y1 = allc[4]
    z1 = allc[5]

    # top of cell layer 2 in cell 5 5 (if 1 index start as RMS)
    assert x0.values3d[4, 4, 1] == pytest.approx(457387.718, abs=0.01)
    assert y0.values3d[4, 4, 1] == pytest.approx(5935461.29790, abs=0.01)
    assert z0.values3d[4, 4, 1] == pytest.approx(1728.9429, abs=0.01)

    assert x1.values3d[4, 4, 1] == pytest.approx(457526.55367, abs=0.01)
    assert y1.values3d[4, 4, 1] == pytest.approx(5935542.02467, abs=0.01)
    assert z1.values3d[4, 4, 1] == pytest.approx(1728.57898, abs=0.01)


def test_get_cell_corners():
    """Get X Y Z for one cell as tuple"""

    grid = Grid(testfile8a)
    clist = grid.get_xyz_cell_corners(ijk=(4, 4, 1))
    logger.debug(clist)

    assert_almostequal(clist[0], 457168.358886, 0.001)


def test_get_xy_values_for_webportal():
    """Get lists on webportal format"""

    grid = Grid(testfile8a)
    prop = GridProperty(testfile9, grid=grid, name='PORO')

    start = xtg.timer()
    coord, valuelist = prop.get_xy_value_lists(grid=grid)
    elapsed = xtg.timer(start)
    logger.info('Elapsed {}'.format(elapsed))

    grid = Grid(testfile10)
    prop = GridProperty(testfile11, grid=grid, name='PORO')

    coord, valuelist = prop.get_xy_value_lists(grid=grid, mask=False)

    logger.info('Cell 1 1 1 coords\n{}.'.format(coord[0][0]))
    assert coord[0][0][0] == (454.875, 318.5)
    assert valuelist[0][0] == -999.0


def test_get_xy_values_for_webportal_ecl():
    """Get lists on webportal format (Eclipse input)"""

    grid = Grid(testfile5)
    prop = GridProperty(testfile6, grid=grid, name='PORO')

    coord, valuelist = prop.get_xy_value_lists(grid=grid)
    assert_almostequal(coord[0][0][0][1], 5935688.22412, 0.001)
