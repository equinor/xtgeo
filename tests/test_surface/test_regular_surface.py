# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function

import os
import os.path

import six

import pytest
import numpy as np

import xtgeo
from xtgeo.common import XTGeoDialog
import test_common.test_xtg as tsetup

if six.PY3:
    from pathlib import Path


xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TMPD = xtg.tmpdir
TPATH = xtg.testpath

XTGSHOW = False
if "XTG_SHOW" in os.environ:
    XTGSHOW = True

# =============================================================================
# Do tests
# =============================================================================

TESTSET1 = "../xtgeo-testdata/surfaces/reek/1/topreek_rota.gri"
TESTSET2 = "../xtgeo-testdata/surfaces/reek/1/topupperreek.gri"
TESTSET3 = "../xtgeo-testdata/surfaces/reek/1/topupperreek.fgr"
TESTSET4A = "../xtgeo-testdata/surfaces/etc/ib_test-horizon.map"  # IJXYZ table
TESTSET4B = "../xtgeo-testdata/surfaces/etc/ijxyz1.map"  # IJXYZ table
TESTSET4D = "../xtgeo-testdata/surfaces/etc/ijxyz1.dat"  # IJXYZ table OW
TESTSET4C = "../xtgeo-testdata/surfaces/etc/testx_1500_edit1.map"
TESTSET5 = "../xtgeo-testdata/surfaces/reek/2/02_midreek_rota.gri"
TESTSET6A = "../xtgeo-testdata/surfaces/etc/seabed_p.pmd"
TESTSET6B = "../xtgeo-testdata/surfaces/etc/seabed_p.gri"
TESTSET6C = "../xtgeo-testdata/surfaces/etc/seabed_p_v2.pmd"

FENCE1 = "../xtgeo-testdata/polygons/reek/1/fence.pol"


def test_create():
    """Create default surface"""

    logger.info("Simple case...")

    x = xtgeo.RegularSurface()
    tsetup.assert_equal(x.ncol, 5, "NX")
    tsetup.assert_equal(x.nrow, 3, "NY")
    val = x.values
    xdim, _ydim = val.shape
    tsetup.assert_equal(xdim, 5, "NX from DIM")
    x.describe()


# @tsetup.skipifwindows
def test_ijxyz_import1():
    """Import some IJ XYZ format, typical seismic."""
    logger.info("Import and export...")

    xsurf = xtgeo.RegularSurface()
    xsurf.from_file(TESTSET4A, fformat="ijxyz")
    xsurf.describe()
    tsetup.assert_almostequal(xsurf.xori, 600413.048444, 0.0001)
    tsetup.assert_almostequal(xsurf.xinc, 25.0, 0.0001)
    assert xsurf.ncol == 280
    assert xsurf.nrow == 1341
    xsurf.to_file(os.path.join(TMPD, "ijxyz_set4a.gri"))


# @tsetup.skipifwindows
def test_ijxyz_import2():
    """Import some IJ XYZ small set with YFLIP-1"""
    logger.info("Import and export...")

    xsurf = xtgeo.RegularSurface()
    xsurf.from_file(TESTSET4B, fformat="ijxyz")
    xsurf.describe()
    tsetup.assert_almostequal(xsurf.values.mean(), 5037.5840, 0.001)
    assert xsurf.ncol == 51
    assert xsurf.yflip == -1
    assert xsurf.nactive == 2578
    xsurf.to_file(os.path.join(TMPD, "ijxyz_set4b.gri"))


# @tsetup.skipifwindows
def test_ijxyz_import4_ow_messy_dat():
    """Import some IJ XYZ small set with YFLIP -1 from OW messy dat format."""
    logger.info("Import and export...")

    xsurf = xtgeo.RegularSurface()
    xsurf.from_file(TESTSET4D, fformat="ijxyz")
    xsurf.describe()
    tsetup.assert_almostequal(xsurf.values.mean(), 5037.5840, 0.001)
    assert xsurf.ncol == 51
    assert xsurf.yflip == -1
    assert xsurf.nactive == 2578
    xsurf.to_file(os.path.join(TMPD, "ijxyz_set4d.gri"))


# @tsetup.skipifwindows
def test_ijxyz_import3():
    """Import some IJ XYZ small set yet again"""
    logger.info("Import and export...")

    xsurf = xtgeo.RegularSurface()
    xsurf.from_file(TESTSET4C, fformat="ijxyz")
    xsurf.describe()
    xsurf.to_file(os.path.join(TMPD, "ijxyz_set4c.gri"))


def test_irapbin_import1():
    """Import Reek Irap binary."""
    logger.info("Import and export...")

    xsurf = xtgeo.RegularSurface(TESTSET2)
    assert xsurf.ncol == 1264
    assert xsurf.nrow == 2010
    tsetup.assert_almostequal(xsurf.values[11, 0], 1678.89733887, 0.00001)
    tsetup.assert_almostequal(xsurf.values[1263, 2009], 1893.75, 0.01)
    xsurf.describe()


@tsetup.skipifpython2
def test_irapbin_import_use_pathib():
    """Import Reek Irap binary."""
    logger.info("Import and export...")

    pobj = Path(TESTSET2)

    xsurf = xtgeo.RegularSurface(pobj)
    assert xsurf.ncol == 1264
    assert xsurf.nrow == 2010


def test_irapbin_import_quickplot():
    """Import Reek Irap binary and do quickplot."""
    logger.info("Import and export...")

    xsurf = xtgeo.RegularSurface(TESTSET2)
    cmap = "gist_earth"

    if XTGSHOW:
        xsurf.quickplot(colormap=cmap)
    else:
        xsurf.quickplot(os.path.join(TMPD, "qplot.jpg"), colormap=cmap)


def test_irapbin_import_metadatafirst():
    """Import Reek Irap binary, first with metadata only, then values."""
    logger.info("Import and export...")

    nsurf = 10
    sur = []
    t1 = xtg.timer()
    for ix in range(nsurf):
        sur.append(xtgeo.RegularSurface(TESTSET2, values=False))
    t2 = xtg.timer(t1)
    logger.info("Loading %s surfaces lazy took %s secs.", nsurf, t2)
    assert sur[nsurf - 1].ncol == 1264

    t1 = xtg.timer()
    for ix in range(nsurf):
        sur[ix].load_values()
    t2 = xtg.timer(t1)
    logger.info("Loading %s surfaces actual values took %s secs.", nsurf, t2)

    assert sur[nsurf - 1].ncol == 1264
    assert sur[nsurf - 1].nrow == 2010
    tsetup.assert_almostequal(sur[nsurf - 1].values[11, 0], 1678.89733887, 0.00001)


def test_petromodbin_import_export():
    """Import Petromod PDM binary example."""
    logger.info("Import and export...")

    petromod = xtgeo.RegularSurface(TESTSET6A)
    irapbin = xtgeo.RegularSurface(TESTSET6B)
    assert petromod.ncol == irapbin.ncol
    assert petromod.nrow == irapbin.nrow
    assert petromod.values1d[200000] == irapbin.values1d[200000]

    testfile = os.path.join(TMPD, "petromod.pmd")
    petromod.to_file(testfile, fformat="petromod")
    petromod_again = xtgeo.RegularSurface(testfile)
    assert petromod_again.values1d[200000] == irapbin.values1d[200000]

    # test with roation 0 and rotation origins 0
    petromod = xtgeo.RegularSurface(TESTSET6C)
    assert petromod.ncol == irapbin.ncol
    assert petromod.nrow == irapbin.nrow
    assert petromod.values1d[200000] == irapbin.values1d[200000]

    testfile = os.path.join(TMPD, "petromod_other_units.pmd")
    petromod.to_file(testfile, fformat="petromod", pmd_dataunits=(16, 300))
    petromod.from_file(testfile, fformat="petromod")

    with pytest.raises(ValueError):
        petromod.to_file("TMP/null", fformat="petromod", pmd_dataunits=(-2, 999))


def test_swapaxes():
    """Import Reek Irap binary and swap axes."""

    xsurf = xtgeo.RegularSurface(TESTSET5)
    xsurf.describe()
    logger.info(xsurf.yflip)
    xsurf.to_file("TMP/notswapped.gri")
    val1 = xsurf.values.copy()
    xsurf.swapaxes()
    xsurf.describe()
    logger.info(xsurf.yflip)
    xsurf.to_file("TMP/swapped.gri")
    xsurf.swapaxes()
    val2 = xsurf.values.copy()
    xsurf.to_file("TMP/swapped_reswapped.gri")
    valdiff = val2 - val1
    tsetup.assert_almostequal(valdiff.mean(), 0.0, 0.00001)
    tsetup.assert_almostequal(valdiff.std(), 0.0, 0.00001)


def test_irapasc_import1():
    """Import Reek Irap ascii."""
    logger.info("Import and export...")

    xsurf = xtgeo.RegularSurface(TESTSET3, fformat="irap_ascii")
    assert xsurf.ncol == 1264
    assert xsurf.nrow == 2010
    tsetup.assert_almostequal(xsurf.values[11, 0], 1678.89733887, 0.00001)
    tsetup.assert_almostequal(xsurf.values[1263, 2009], 1893.75, 0.01)


def test_irapasc_export():
    """Export Irap ASCII (1)."""

    logger.info("Export to Irap Classic")
    x = xtgeo.RegularSurface()
    x.to_file("TMP/irap.fgr", fformat="irap_ascii")

    fstatus = False
    if os.path.isfile("TMP/irap.fgr") is True:
        fstatus = True

    assert fstatus is True


def test_zmapasc_irapasc_export():
    """Export Irap ASCII and then ZMAP ASCII format, last w auto derotation."""

    x = xtgeo.RegularSurface(TESTSET1)
    x.to_file("TMP/reek.fgr", fformat="irap_ascii")
    x.to_file("TMP/reek.zmap", fformat="zmap_ascii")

    fstatus = False
    if os.path.isfile("TMP/reek.zmap") is True:
        fstatus = True

    assert fstatus is True


def test_irapasc_export_and_import():
    """Export Irap ASCII and binary and import again."""

    logger.info("Export to Irap Classic and Binary")

    x = xtgeo.RegularSurface(
        ncol=120,
        nrow=100,
        xori=1000,
        yori=5000,
        xinc=40,
        yinc=20,
        values=np.random.rand(120, 100),
    )
    tsetup.assert_equal(x.ncol, 120)

    mean1 = x.values.mean()

    x.to_file("TMP/irap2_a.fgr", fformat="irap_ascii")
    x.to_file("TMP/irap2_b.gri", fformat="irap_binary")

    fsize = os.path.getsize("TMP/irap2_b.gri")
    logger.info(fsize)
    tsetup.assert_equal(fsize, 48900)

    # import irap ascii
    y = xtgeo.RegularSurface()
    y.from_file("TMP/irap2_a.fgr", fformat="irap_ascii")

    mean2 = y.values.mean()

    tsetup.assert_almostequal(mean1, mean2, 0.0001)


def test_minmax_rotated_map():
    """Min and max of rotated map"""
    logger.info("Import and export...")

    x = xtgeo.RegularSurface()
    x.from_file(TESTSET1, fformat="irap_binary")

    tsetup.assert_almostequal(x.xmin, 454637.6, 0.1)
    tsetup.assert_almostequal(x.xmax, 468895.1, 0.1)
    tsetup.assert_almostequal(x.ymin, 5925995.0, 0.1)
    tsetup.assert_almostequal(x.ymax, 5939998.7, 0.1)


@tsetup.bigtest
def test_irapbin_io():
    """Import and export Irap binary."""
    logger.info("Import and export...")

    x = xtgeo.RegularSurface()
    x.from_file(TESTSET1, fformat="irap_binary")

    x.to_file("TMP/reek1_test.fgr", fformat="irap_ascii")

    logger.debug("NX is %s", x.ncol)

    tsetup.assert_equal(x.ncol, 554)

    # get the 1D numpy
    v1d = x.get_zval()

    logger.info("Mean VALUES are: %s", np.nanmean(v1d))

    zval = x.values

    # add value via numpy
    zval = zval + 300
    # update
    x.values = zval

    tsetup.assert_almostequal(x.values.mean(), 1998.648, 0.01)

    x.to_file("TMP/reek1_plus_300_a.fgr", fformat="irap_ascii")
    x.to_file("TMP/reek1_plus_300_b.gri", fformat="irap_binary")

    mfile = TESTSET1

    # direct import
    y = xtgeo.RegularSurface(mfile)
    tsetup.assert_equal(y.ncol, 554)

    # semidirect import
    cc = xtgeo.RegularSurface().from_file(mfile)
    tsetup.assert_equal(cc.ncol, 554)


def test_get_values1d():
    """Get the 1D array, different variants as masked, notmasked, order, etc"""

    xmap = xtgeo.RegularSurface()
    print(xmap.values)

    v1d = xmap.get_values1d(order="C", asmasked=False, fill_value=-999)

    assert v1d.tolist() == [
        1.0,
        6.0,
        11.0,
        2.0,
        7.0,
        12.0,
        3.0,
        8.0,
        -999.0,
        4.0,
        9.0,
        14.0,
        5.0,
        10.0,
        15.0,
    ]

    v1d = xmap.get_values1d(order="C", asmasked=True, fill_value=-999)
    print(v1d)

    assert v1d.tolist() == [
        1.0,
        6.0,
        11.0,
        2.0,
        7.0,
        12.0,
        3.0,
        8.0,
        None,
        4.0,
        9.0,
        14.0,
        5.0,
        10.0,
        15.0,
    ]

    v1d = xmap.get_values1d(order="F", asmasked=False, fill_value=-999)

    assert v1d.tolist() == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        -999.0,
        14.0,
        15.0,
    ]

    v1d = xmap.get_values1d(order="F", asmasked=True)

    assert v1d.tolist() == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        None,
        14.0,
        15.0,
    ]


def test_ij_map_indices():
    """Get the IJ MAP indices"""

    xmap = xtgeo.RegularSurface()
    print(xmap.values)

    ixc, jyc = xmap.get_ij_values1d(activeonly=True, order="C")

    assert ixc.tolist() == [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5]
    assert jyc.tolist() == [1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3]

    ixc, jyc = xmap.get_ij_values1d(activeonly=False, order="C")

    assert ixc.tolist() == [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
    assert jyc.tolist() == [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]

    ixc, jyc = xmap.get_ij_values1d(activeonly=False, order="F")

    assert ixc.tolist() == [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    assert jyc.tolist() == [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]


def test_get_xy_values():
    """Get the XY coordinate values as 2D arrays"""

    xmap = xtgeo.RegularSurface()

    xcv, _ycv = xmap.get_xy_values(order="C")

    xxv = xcv.ravel(order="K")
    tsetup.assert_almostequal(xxv[1], 0.0, 0.001)

    xcv, _ycv = xmap.get_xy_values(order="F")
    xxv = xcv.ravel(order="K")
    tsetup.assert_almostequal(xxv[1], 25.0, 0.001)

    xcv, _ycv = xmap.get_xy_values(order="C", asmasked=True)

    xxv = xcv.ravel(order="K")
    tsetup.assert_almostequal(xxv[1], 0.0, 0.001)

    xcv, _ycv = xmap.get_xy_values(order="F", asmasked=True)

    xxv = xcv.ravel(order="K")
    tsetup.assert_almostequal(xxv[1], 25.0, 0.001)


def test_get_xy_values1d():
    """Get the XY coordinate values"""

    xmap = xtgeo.RegularSurface()

    xcv, _ycv = xmap.get_xy_values1d(activeonly=False, order="C")

    tsetup.assert_almostequal(xcv[1], 0.0, 0.001)

    xcv, _ycv = xmap.get_xy_values1d(activeonly=False, order="F")

    tsetup.assert_almostequal(xcv[1], 25.0, 0.001)

    xcv, _ycv = xmap.get_xy_values1d(activeonly=True, order="C")

    tsetup.assert_almostequal(xcv[1], 0.0, 0.001)

    xcv, _ycv = xmap.get_xy_values1d(activeonly=True, order="F")

    tsetup.assert_almostequal(xcv[1], 25.0, 0.001)


def test_dataframe_simple():
    """Get a pandas Dataframe object"""

    xmap = xtgeo.RegularSurface(TESTSET1)

    dfrc = xmap.dataframe(ijcolumns=True, order="C", activeonly=True)

    tsetup.assert_almostequal(dfrc["X_UTME"][2], 465956.274, 0.01)


@tsetup.bigtest
def test_dataframe_more():
    """Get a pandas Dataframe object, more detailed testing"""

    xmap = xtgeo.RegularSurface(TESTSET1)

    xmap.describe()

    dfrc = xmap.dataframe(ijcolumns=True, order="C", activeonly=True)
    dfrf = xmap.dataframe(ijcolumns=True, order="F", activeonly=True)

    dfrc.to_csv(os.path.join(TMPD, "regsurf_df_c.csv"))
    dfrf.to_csv(os.path.join(TMPD, "regsurf_df_f.csv"))
    xmap.to_file(os.path.join(TMPD, "regsurf_df.ijxyz"), fformat="ijxyz")

    tsetup.assert_almostequal(dfrc["X_UTME"][2], 465956.274, 0.01)
    tsetup.assert_almostequal(dfrf["X_UTME"][2], 462679.773, 0.01)

    dfrcx = xmap.dataframe(ijcolumns=False, order="C", activeonly=True)
    dfrcx.to_csv(os.path.join(TMPD, "regsurf_df_noij_c.csv"))
    dfrcy = xmap.dataframe(
        ijcolumns=False, order="C", activeonly=False, fill_value=None
    )
    dfrcy.to_csv(os.path.join(TMPD, "regsurf_df_noij_c_all.csv"))


def test_get_xy_value_lists_small():
    """Get the xy list and value list from small test case"""

    x = xtgeo.RegularSurface()  # default instance

    xylist, valuelist = x.get_xy_value_lists(valuefmt="8.3f", xyfmt="12.2f")

    logger.info(xylist[2])
    logger.info(valuelist[2])

    tsetup.assert_equal(valuelist[2], 3.0)


@tsetup.bigtest
def test_get_xy_value_lists_reek():
    """Get the xy list and value list"""

    x = xtgeo.RegularSurface()
    x.from_file(TESTSET1, fformat="irap_binary")

    xylist, valuelist = x.get_xy_value_lists(valuefmt="8.3f", xyfmt="12.2f")

    logger.info(xylist[2])
    logger.info(valuelist[2])

    tsetup.assert_equal(valuelist[2], 1910.445)


def test_topology():
    """Testing topology between two surfaces."""

    logger.info("Test if surfaces are similar...")

    mfile = TESTSET1

    x = xtgeo.RegularSurface(mfile)
    y = xtgeo.RegularSurface(mfile)

    status = x.compare_topology(y)
    assert status is True

    y.xori = y.xori - 100.0
    status = x.compare_topology(y)
    assert status is False


def test_similarity():
    """Testing similarity of two surfaces. 0.0 means identical in
    terms of mean value.
    """

    logger.info("Test if surfaces are similar...")

    mfile = TESTSET1

    x = xtgeo.RegularSurface(mfile)
    y = xtgeo.RegularSurface(mfile)

    si = x.similarity_index(y)
    tsetup.assert_equal(si, 0.0)

    y.values = y.values * 2

    si = x.similarity_index(y)
    tsetup.assert_equal(si, 1.0)


def test_irapbin_io_loop():
    """Do a loop over big Troll data set."""

    num = 10

    for _i in range(0, num):
        # print(i)
        x = xtgeo.RegularSurface()
        x.from_file(TESTSET1, fformat="irap_binary")

        m1 = x.values.mean()
        zval = x.values
        zval = zval + 300
        x.values = zval
        m2 = x.values.mean()
        x.to_file("TMP/troll.gri", fformat="irap_binary")

        assert m1 == pytest.approx(m2 - 300)


def test_irapbin_export_py():
    """Export Irapbin with pure python"""

    x = xtgeo.RegularSurface()
    x.from_file(TESTSET1, fformat="irap_binary")

    t0 = xtg.timer()
    for _ in range(10):
        x.to_file("TMP/purecx.gri", fformat="irap_binary", engine="cxtgeo")
    t1 = xtg.timer(t0)
    print("CXTGeo based write: {:3.4f}".format(t1))

    t0 = xtg.timer()
    for _ in range(10):
        x.to_file("TMP/purepy.gri", fformat="irap_binary", engine="python")
    t2 = xtg.timer(t0)
    print("Python based write: {:3.4f}".format(t2))
    print("Ratio python based / cxtgeo based {:3.4f}".format(t2 / t1))

    s1 = xtgeo.RegularSurface("TMP/purecx.gri")
    s2 = xtgeo.RegularSurface("TMP/purepy.gri")

    assert s1.values.mean() == s2.values.mean()

    assert s1.values[100, 100] == s2.values[100, 100]


def test_distance_from_point():
    """Distance from point."""

    x = xtgeo.RegularSurface()
    x.from_file(TESTSET1, fformat="irap_binary")

    x.distance_from_point(point=(464960, 7336900), azimuth=30)

    x.to_file("TMP/reek1_dist_point.gri", fformat="irap_binary")


def test_value_from_xy():
    """
    get Z value from XY point
    """

    x = xtgeo.RegularSurface()
    x.from_file(TESTSET1, fformat="irap_binary")

    z = x.get_value_from_xy(point=(460181.036, 5933948.386))

    tsetup.assert_almostequal(z, 1625.11, 0.01)

    # outside value shall return None
    z = x.get_value_from_xy(point=(0.0, 7337128.076))
    assert z is None


def test_fence():
    """Test sampling a fence from a surface."""

    myfence = np.array(
        [
            [462174.6191406, 5930073.3461914, 721.711059],
            [462429.4677734, 5930418.2055664, 720.909423],
            [462654.6738281, 5930883.9331054, 712.587158],
            [462790.8710937, 5931501.4443359, 676.873901],
            [462791.5273437, 5932040.4306640, 659.938476],
            [462480.2958984, 5932846.7387695, 622.102172],
            [462226.7070312, 5933397.8632812, 628.067138],
            [462214.4921875, 5933753.4936523, 593.260864],
            [462161.5048828, 5934327.8398437, 611.253540],
            [462325.0673828, 5934688.7519531, 626.485107],
            [462399.0429687, 5934975.2934570, 640.868774],
        ]
    )

    logger.debug("NP:")
    logger.debug(myfence)

    x = xtgeo.RegularSurface(TESTSET1)

    newfence = x.get_fence(myfence)

    logger.debug("updated NP:")
    logger.debug(newfence)

    tsetup.assert_almostequal(newfence[1][2], 1720.9094, 0.01)


def test_get_randomline_frompolygon():

    fence = xtgeo.Polygons(FENCE1)
    xs = xtgeo.RegularSurface(TESTSET1)

    # get the polygon
    fspec = fence.get_fence(distance=10, nextend=2, asnumpy=False)
    tsetup.assert_almostequal(fspec.dataframe[fspec.dhname][4], 10, 1)

    fspec = fence.get_fence(distance=20, nextend=5, asnumpy=True)

    arr = xs.get_randomline(fspec)

    x = arr[:, 0]
    y = arr[:, 1]

    print(arr)
    print(arr.shape)

    if XTGSHOW:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(x, y)
        plt.gca().invert_yaxis()
        plt.show()


def test_unrotate():
    """Change a rotated map to an unrotated instance"""

    x = xtgeo.RegularSurface()
    x.from_file(TESTSET1, fformat="irap_binary")

    logger.info(x)
    x.unrotate()
    logger.info(x)


def test_fill():
    """Fill the undefined values for the surface"""

    srf = xtgeo.RegularSurface()
    srf.from_file(TESTSET1, fformat="irap_binary")

    minv1 = srf.values.min()
    tsetup.assert_almostequal(srf.values.mean(), 1698.648, 0.001)

    srf.fill()
    minv2 = srf.values.min()
    tsetup.assert_almostequal(srf.values.mean(), 1705.201, 0.001)
    tsetup.assert_almostequal(minv1, minv2, 0.000001)

    srf = xtgeo.RegularSurface()
    srf.from_file(TESTSET1, fformat="irap_binary")
    srf.fill(444)
    tsetup.assert_almostequal(srf.values.mean(), 1342.10498, 0.001)


def test_smoothing():
    """Smooth the the surface"""

    srf = xtgeo.RegularSurface()
    srf.from_file(TESTSET1, fformat="irap_binary")

    mean1 = srf.values.mean()
    tsetup.assert_almostequal(mean1, 1698.65, 0.1)

    srf.smooth(iterations=1, width=5)

    mean2 = srf.values.mean()
    tsetup.assert_almostequal(mean2, 1698.65, 0.1)  # smoothed ~same mean

    assert mean1 != mean2  # but not exacly same
