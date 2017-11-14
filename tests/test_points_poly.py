# -*- coding: utf-8 -*-
import sys
import pytest
from xtgeo.xyz import XYZ
from xtgeo.xyz import Points
from xtgeo.xyz import Polygons

from xtgeo.common import XTGeoDialog
from .test_grid import assert_almostequal

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg._testsetup():
    sys.exit(-9)

td = xtg.tmpdir
testpath = xtg.testpath

skiplargetest = pytest.mark.skipif(xtg.bigtest is False,
                                   reason="Big tests skip")

# =========================================================================
# Do tests
# =========================================================================


def test_xyz():
    """Import XYZ module from file, should not be possible as it is abc."""

    ok = False
    try:
        myxyz = XYZ()
    except TypeError as tt:
        ok = True
        logger.info(tt)
        assert 'abstract' in str(tt)
    else:
        logger.info(myxyz)

    assert ok is True


def test_import():
    """Import XYZ points from file."""

    pfile = "../xtgeo-testdata/points/eme/1/emerald_10_random.poi"

    mypoints = Points(pfile)

    logger.debug(mypoints.dataframe)

    x0 = mypoints.dataframe['X'].values[0]
    logger.debug(x0)
    assert_almostequal(x0, 460842.434326, 0.001)


def test_import_zmap():
    """Import XYZ polygons on ZMAP format from file"""

    pfile = "../xtgeo-testdata/polygons/gfb/faults_zone10.zmap"

    mypol = Polygons()

    mypol.from_file(pfile, fformat='zmap')

    nn = mypol.nrows
    assert nn == 16666
    x0 = mypol.dataframe['X'].values[0]
    y1 = mypol.dataframe['Y'].values[nn - 1]

    assert_almostequal(x0, 457357.78125, 0.001)
    assert_almostequal(y1, 6790785.5, 0.01)


def test_import_export_polygons():
    """Import XYZ polygons from file. Modify, and export."""

    pfile = "../xtgeo-testdata/points/eme/1/emerald_10_random.poi"

    mypoly = Polygons()

    mypoly.from_file(pfile)

    z0 = mypoly.dataframe['Z'].values[0]

    assert_almostequal(z0, 2266.996338, 0.001)

    logger.debug(mypoly.dataframe)

    mypoly.dataframe['Z'] += 100

    mypoly.to_file(td + '/polygon_export.xyz', fformat='xyz')

    # reimport and check
    mypoly2 = Polygons(td + '/polygon_export.xyz')

    assert_almostequal(z0 + 100, mypoly2.dataframe['Z'].values[0], 0.001)
