# -*- coding: utf-8 -*-
from os.path import join
import numpy as np
from xtgeo.xyz import XYZ
from xtgeo.xyz import Points
from xtgeo.xyz import Polygons

from xtgeo.common import XTGeoDialog
import test_common.test_xtg as tsetup

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TMPD = xtg.tmpdir
TSTPATH = xtg.testpath

# =========================================================================
# Do tests
# =========================================================================

PFILE1A = join(TSTPATH, 'polygons/reek/1/top_upper_reek_faultpoly.zmap')
PFILE1B = join(TSTPATH, 'polygons/reek/1/top_upper_reek_faultpoly.xyz')
PFILE1C = join(TSTPATH, 'polygons/reek/1/top_upper_reek_faultpoly.pol')
PFILE = join(TSTPATH, 'points/eme/1/emerald_10_random.poi')
POLSET2 = join(TSTPATH, 'polygons/reek/1/polset2.pol')
POLSET3 = join(TSTPATH, 'polygons/etc/outline.pol')
POINTSET2 = join(TSTPATH, 'points/reek/1/pointset2.poi')
POINTSET3 = join(TSTPATH, 'points/battle/1/many.rmsattr')


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

    mypoints = Points(PFILE)  # should guess based on extesion

    logger.debug(mypoints.dataframe)

    x0 = mypoints.dataframe['X_UTME'].values[0]
    logger.debug(x0)
    tsetup.assert_almostequal(x0, 460842.434326, 0.001)


def test_import_zmap_and_xyz():
    """Import XYZ polygons on ZMAP and XYZ format from file"""

    mypol2a = Polygons()
    mypol2b = Polygons()
    mypol2c = Polygons()

    mypol2a.from_file(PFILE1A, fformat='zmap')
    mypol2b.from_file(PFILE1B)
    mypol2c.from_file(PFILE1C)

    assert mypol2a.nrow == mypol2b.nrow
    assert mypol2b.nrow == mypol2c.nrow

    logger.info(mypol2a.nrow, mypol2b.nrow)

    logger.info(mypol2a.dataframe)
    logger.info(mypol2b.dataframe)

    for col in ['X_UTME', 'Y_UTMN', 'Z_TVDSS', 'POLY_ID']:
        status = np.allclose(mypol2a.dataframe[col].values,
                             mypol2b.dataframe[col].values)

        assert status is True


def test_import_rmsattr_format():
    """Import points with attributes from RMS attr format"""

    mypoi = Points()

    mypoi.from_file(POINTSET3, fformat='rms_attr')

    logger.info(id(mypoi))
    logger.info(mypoi._df.head())
    # print(mypoi.dataframe.columns[3:])
    print(mypoi.dataframe['VerticalSep'].dtype)
    mypoi.to_file('TMP/attrs.rmsattr', fformat='rms_attr')


def test_import_export_polygons():
    """Import XYZ polygons from file. Modify, and export."""

    mypoly = Polygons()

    mypoly.from_file(PFILE, fformat='xyz')

    z0 = mypoly.dataframe['Z_TVDSS'].values[0]

    tsetup.assert_almostequal(z0, 2266.996338, 0.001)

    logger.debug(mypoly.dataframe)

    mypoly.dataframe['Z_TVDSS'] += 100

    mypoly.to_file(TMPD + '/polygon_export.xyz', fformat='xyz')

    # reimport and check
    mypoly2 = Polygons(TMPD + '/polygon_export.xyz')

    tsetup.assert_almostequal(z0 + 100,
                              mypoly2.dataframe['Z_TVDSS'].values[0], 0.001)


def test_polygon_boundary():
    """Import XYZ polygons from file and test boundary function."""

    mypoly = Polygons()

    mypoly.from_file(PFILE, fformat='xyz')

    boundary = mypoly.get_boundary()

    tsetup.assert_almostequal(boundary[0], 460595.6036, 0.0001)
    tsetup.assert_almostequal(boundary[4], 2025.952637, 0.0001)
    tsetup.assert_almostequal(boundary[5], 2266.996338, 0.0001)


def test_polygon_filter_byid():
    """Filter a Polygon by a list of ID's"""

    pol = Polygons(POLSET3)

    assert pol.dataframe["POLY_ID"].iloc[0] == 0
    assert pol.dataframe["POLY_ID"].iloc[-1] == 3

    pol.filter_byid()
    assert pol.dataframe["POLY_ID"].iloc[-1] == 0

    pol = Polygons(POLSET3)
    pol.filter_byid([1, 3])

    assert pol.dataframe["POLY_ID"].iloc[0] == 1
    assert pol.dataframe["POLY_ID"].iloc[-1] == 3

    pol = Polygons(POLSET3)
    pol.filter_byid(2)

    assert pol.dataframe["POLY_ID"].iloc[0] == 2
    assert pol.dataframe["POLY_ID"].iloc[-1] == 2

    pol = Polygons(POLSET3)
    pol.filter_byid(99)  # not present; should remove all rows
    assert pol.nrow == 0


def test_polygon_hlen():
    """Test the hlen operation"""

    pol = Polygons(POLSET3)
    pol.hlen()
    print(pol.dataframe)

    pol.filter_byid()
    hlen = pol.get_shapely_objects()[0].length
    assert (abs(pol.dataframe[pol.hname].iloc[-1] - hlen)) < 0.001
    assert (abs(pol.dataframe[pol.dhname].iloc[0] - 1761.148)) < 0.01


def test_points_in_polygon():
    """Import XYZ points and do operations if inside or outside"""

    poi = Points(POINTSET2)
    pol = Polygons(POLSET2)
    assert poi.nrow == 30
    logger.info(poi.dataframe)

    # remove points in polygon
    poi.operation_polygons(pol, 0, opname='eli',
                           where=True)

    logger.info(poi.dataframe)
    assert poi.nrow == 19
    poi.to_file(join(TMPD, 'poi_test.poi'))

    poi = Points(POINTSET2)
    # remove points outside polygon
    poi.operation_polygons(pol, 0, opname='eli', inside=False,
                           where=True)
    logger.info(poi.dataframe)
    assert poi.nrow == 1


def test_rescale_polygon():
    """Take a polygons set and rescale/resample"""

    pol = Polygons(POLSET2)

    df = pol.dataframe[0:3]

    df.at[0, 'X_UTME'] = 0.0
    df.at[1, 'X_UTME'] = 100.0
    df.at[2, 'X_UTME'] = 100.0

    df.at[0, 'Y_UTMN'] = 20.0
    df.at[1, 'Y_UTMN'] = 20.0
    df.at[2, 'Y_UTMN'] = 100.0

    df.at[0, 'Z_TVDSS'] = 0.0
    df.at[1, 'Z_TVDSS'] = 1000.0
    df.at[2, 'Z_TVDSS'] = 2000.0

    pol.dataframe = df

    pol.rescale(20)
    logger.info(pol.dataframe)

    assert pol.dataframe.at[1, 'X_UTME'] == 20.0

    pol = Polygons(POLSET2)
    pol.rescale(100)

    logger.info(pol.dataframe)
    assert abs(pol.dataframe.at[369, 'X_UTME'] - 462708.614654) < 0.0001


def test_fence_from_polygon():
    """Test polygons get_fence method"""

    pol = Polygons(POLSET2)

    df = pol.dataframe[0:3]

    df.at[0, 'X_UTME'] = 0.0
    df.at[1, 'X_UTME'] = 100.0
    df.at[2, 'X_UTME'] = 100.0

    df.at[0, 'Y_UTMN'] = 20.0
    df.at[1, 'Y_UTMN'] = 20.0
    df.at[2, 'Y_UTMN'] = 100.0

    df.at[0, 'Z_TVDSS'] = 0.0
    df.at[1, 'Z_TVDSS'] = 1000.0
    df.at[2, 'Z_TVDSS'] = 2000.0

    pol.dataframe = df

    fence = pol.get_fence(distance=100, extend=4, name="SOMENAME", asnumpy=False,
                          atleast=10)
    logger.info(fence.dataframe)

    assert fence.dataframe.at[13, 'X_UTME'] == 100.0


def test_polygons_quickplot():

    pol = Polygons(POLSET2)
    pol.quickplot()
