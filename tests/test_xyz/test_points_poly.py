# -*- coding: utf-8 -*-
import sys
import pytest
import numpy as np
from xtgeo.xyz import XYZ
from xtgeo.xyz import Points
from xtgeo.xyz import Polygons

from xtgeo.common import XTGeoDialog
import test_common.test_xtg as tsetup

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    sys.exit(-9)

td = xtg.tmpdir
testpath = xtg.testpath

skiplargetest = pytest.mark.skipif(xtg.bigtest is False,
                                   reason='Big tests skip')

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

    pfile = '../xtgeo-testdata/points/eme/1/emerald_10_random.poi'

    mypoints = Points(pfile)  # should guess based on extesion

    logger.debug(mypoints.dataframe)

    x0 = mypoints.dataframe['X'].values[0]
    logger.debug(x0)
    tsetup.assert_almostequal(x0, 460842.434326, 0.001)


def test_import_zmap_and_xyz():
    """Import XYZ polygons on ZMAP and XYZ format from file"""

    pfile1a = '../xtgeo-testdata/polygons/reek/1/top_upper_reek_faultpoly.zmap'
    pfile1b = '../xtgeo-testdata/polygons/reek/1/top_upper_reek_faultpoly.xyz'
    pfile1c = '../xtgeo-testdata/polygons/reek/1/top_upper_reek_faultpoly.pol'

    mypol2a = Polygons()
    mypol2b = Polygons()
    mypol2c = Polygons()

    mypol2a.from_file(pfile1a, fformat='zmap')
    mypol2b.from_file(pfile1b)
    mypol2c.from_file(pfile1c)

    assert mypol2a.nrow == mypol2b.nrow
    assert mypol2b.nrow == mypol2c.nrow

    logger.info(mypol2a.nrow, mypol2b.nrow)

    logger.info(mypol2a.dataframe)
    logger.info(mypol2b.dataframe)

    for col in ['X', 'Y', 'Z', 'ID']:
        status = np.allclose(mypol2a.dataframe[col].values,
                             mypol2b.dataframe[col].values)

        assert status is True


def test_import_export_polygons():
    """Import XYZ polygons from file. Modify, and export."""

    pfile = '../xtgeo-testdata/points/eme/1/emerald_10_random.poi'

    mypoly = Polygons()

    mypoly.from_file(pfile, fformat='xyz')

    z0 = mypoly.dataframe['Z'].values[0]

    tsetup.assert_almostequal(z0, 2266.996338, 0.001)

    logger.debug(mypoly.dataframe)

    mypoly.dataframe['Z'] += 100

    mypoly.to_file(td + '/polygon_export.xyz', fformat='xyz')

    # reimport and check
    mypoly2 = Polygons(td + '/polygon_export.xyz')

    tsetup.assert_almostequal(z0 + 100, mypoly2.dataframe['Z'].values[0], 0.001)
