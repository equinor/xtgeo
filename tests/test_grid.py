#!/usr/bin/env python -u

import sys
import pytest

from xtgeo.grid3d import Grid
from xtgeo.grid3d import GridProperty
from xtgeo.common import XTGeoDialog
from .test_xtg import assert_equal, assert_almostequal

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
reekfile = '../xtgeo-testdata/3dgrids/reek/REEK.EGRID'


def test_import_wrong():
    """Importing wrong fformat, etc"""

    with pytest.raises(ValueError) as e_info:
        logger.warning(e_info)
        g = Grid().from_file(emegfile, fformat='roffdum')
        assert_equal(g.ncol, 70)


def test_import_guess():
    """Import with guessing fformat"""

    g = Grid().from_file(emegfile)

    assert_equal(g.ncol, 70)


def test_roffbin_import0():
    """Import ROFF on the form Grid().from_file and Grid(..)"""

    g = Grid().from_file(emegfile, fformat="roff")

    assert isinstance(g, Grid)

    g = Grid(emegfile, fformat="roff")

    assert isinstance(g, Grid)


def test_roffbin_import1():
    """Test roff binary import case 1"""

    g = Grid()
    logger.info("Import roff...")
    g.from_file(emegfile, fformat="roff")

    assert_equal(g.ncol, 70, txt='Grid NCOL Emerald')
    assert_equal(g.nlay, 46, txt='Grid NLAY Emerald')

    # extract ACTNUM parameter as a property instance (a GridProperty)
    act = g.get_actnum()

    logger.info('ACTNUM is {}'.format(act))
    logger.debug('ACTNUM values are \n{}'.format(act.values[888:999]))

    # get dZ...
    dz = g.get_dz()

    logger.info('DZ is {}'.format(act))
    logger.info('DZ values are \n{}'.format(dz.values[888:999]))

    dzval = dz.values3d
    # get the value is cell 32 73 1 shall be 2.761
    mydz = float(dzval[31:32, 72:73, 0:1])
    assert_almostequal(mydz, 2.761, 0.001, txt='Grid DZ Emerald')

    # get dX dY
    logger.info('Get dX dY')
    dx, dy = g.get_dxdy()

    mydx = float(dx.values3d[31:32, 72:73, 0:1])
    mydy = float(dy.values3d[31:32, 72:73, 0:1])

    assert_almostequal(mydx, 118.51, 0.01, txt='Grid DX Emerald')
    assert_almostequal(mydy, 141.21, 0.01, txt='Grid DY Emerald')

    # get X Y Z coordinates (as GridProperty objects) in one go
    logger.info('Get X Y Z...')
    x, y, z = g.get_xyz(names=['xxx', 'yyy', 'zzz'])

    logger.info('X is {}'.format(act))
    logger.debug('X values are \n{}'.format(x.values[888:999]))

    assert_equal(x.name, 'xxx', txt='Name of X coord')
    x.name = 'Xerxes'

    logger.info('X name is now {}'.format(x.name))

    logger.info('Y is {}'.format(act))
    logger.debug('Y values are \n{}'.format(y.values[888:999]))

    # attach some properties to grid
    g.props = [x, y]

    logger.info(g.props)
    g.props = [z]

    logger.info(g.props)

    g.props.append(x)
    logger.info(g.propnames)

    # get the property of name Xerxes
    myx = g.get_prop_by_name('Xerxes')
    if myx is None:
        logger.info(myx)
    else:
        logger.info("Got nothing!")


def test_eclgrid_import1():
    """Eclipse GRID import."""

    g = Grid()
    logger.info("Import Eclipse GRID...")
    g.from_file('../xtgeo-testdata/3dgrids/gfb/G1.GRID',
                fformat="grid")

    assert_equal(g.ncol, 20, txt='Grid NCOL from Eclipse')
    assert_equal(g.nrow, 20, txt='Grid NROW from Eclipse')


def test_eclgrid_import2():
    """Eclipse EGRID import."""
    g = Grid()
    logger.info("Import Eclipse GRID...")
    g.from_file('../xtgeo-testdata/3dgrids/gfb/GULLFAKS.EGRID',
                fformat="egrid")

    assert_equal(g.ncol, 99, txt='EGrid NX from Eclipse')
    assert_equal(g.nrow, 120, txt='EGrid NY from Eclipse')
    assert_equal(g.nactive, 368004, txt='EGrid NTOTAL from Eclipse')
    assert_equal(g.ntotal, 558360, txt='EGrid NACTIVE from Eclipse')


def test_eclgrid_import3():
    """Eclipse GRDECL import and translate"""

    g = Grid()
    logger.info("Import Eclipse GRDECL...")
    g.from_file('../xtgeo-testdata/3dgrids/gfb/g1_comments.grdecl',
                fformat="grdecl")

    mylist = g.get_geometrics()

    xori1 = mylist[0]

    # translate the coordinates
    g.translate_coordinates(translate=(100, 100, 10), flip=(1, 1, 1))

    mylist = g.get_geometrics()

    xori2 = mylist[0]

    # check if origin is translated 100m in X
    assert_equal(xori1 + 100, xori2, txt='Translate X distance')

    g.to_file('TMP/g1_translate.roff', fformat="roff_binary")


def test_geometrics_reek():
    """Import Reek and test geometrics"""

    g = Grid(reekfile, fformat='egrid')

    geom = g.get_geometrics(return_dict=True, cellcenter=False)

    for key, val in geom.items():
        logger.info('{} is {}'.format(key, val))

    # compared with RMS info:
    assert_almostequal(geom['xmin'], 456510.6, 0.1, 'Xmin')
    assert_almostequal(geom['ymax'], 5938935.5, 0.1, 'Ymax')

    # cellcenter True:
    geom = g.get_geometrics(return_dict=True, cellcenter=True)
    assert_almostequal(geom['xmin'], 456620, 1, 'Xmin cell center')


def test_simple_io():
    """Test various import and export formats"""

    gg = Grid('../xtgeo-testdata/3dgrids/gfb/GULLFAKS.EGRID',
              fformat="egrid")

    gg.to_file("TMP/gullfaks_test.roff")


def test_ecl_run():
    """Test import an eclrun with dates and export to roff after a diff"""

    eclroot = '../xtgeo-testdata/3dgrids/gfb/GULLFAKS'
    dates = [19851001, 20150101]
    rprops = ['PRESSURE', 'SWAT']

    gg = Grid(eclroot, fformat='eclipserun', restartdates=dates,
              restartprops=rprops)

    # get the property object:
    pres1 = gg.get_prop_by_name('PRESSURE_20150101')
    assert_almostequal(pres1.values.mean(), 239.505447, 0.001)

    pres1.to_file("TMP/pres1.roff")

    pres2 = gg.get_prop_by_name('PRESSURE_19851001')

    if isinstance(pres2, GridProperty):
        pass

    logger.debug(pres1.values)
    logger.debug(pres2.values)
    logger.debug(pres1)

    pres1.values = pres1.values - pres2.values
    logger.debug(pres1.values)
    logger.debug(pres1)
    avg = pres1.values.mean()
    # ok checked in RMS:
    assert_almostequal(avg, -93.046011, 0.001)

    pres1.to_file("TMP/pressurediff.roff", name="PRESSUREDIFF")
