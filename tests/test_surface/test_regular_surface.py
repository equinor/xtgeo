"""Testing RegularSurface class with methods."""
import sys
from os.path import join
from pathlib import Path

import pytest
import numpy as np

import xtgeo
from xtgeo import RegularSurface
from xtgeo.common import XTGeoDialog
from tests.conftest import assert_equal, assert_almostequal

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TPATH = xtg.testpathobj

# =============================================================================
# Do tests
# =============================================================================

TESTSET1 = TPATH / "surfaces/reek/1/topreek_rota.gri"
TESTSET1A = TPATH / "surfaces/reek/1/basereek_rota.gri"
TESTSET2 = TPATH / "surfaces/reek/1/topupperreek.gri"
TESTSET3 = TPATH / "surfaces/reek/1/topupperreek.fgr"
TESTSET4A = TPATH / "surfaces/etc/ib_test-horizon.map"  # IJXYZ table
TESTSET4B = TPATH / "surfaces/etc/ijxyz1.map"  # IJXYZ table
TESTSET4D = TPATH / "surfaces/etc/ijxyz1.dat"  # IJXYZ table OW
TESTSET4C = TPATH / "surfaces/etc/testx_1500_edit1.map"
TESTSET5 = TPATH / "surfaces/reek/2/02_midreek_rota.gri"
TESTSET6A = TPATH / "surfaces/etc/seabed_p.pmd"
TESTSET6B = TPATH / "surfaces/etc/seabed_p.gri"
TESTSET6C = TPATH / "surfaces/etc/seabed_p_v2.pmd"

FENCE1 = TPATH / "polygons/reek/1/fence.pol"


def test_create():
    """Create default surface."""
    logger.info("Simple case...")

    x = xtgeo.RegularSurface()
    assert_equal(x.ncol, 5, "NX")
    assert_equal(x.nrow, 3, "NY")
    val = x.values
    xdim, _ydim = val.shape
    assert_equal(xdim, 5, "NX from DIM")
    x.describe()


def test_values(default_surface):
    """Test behaviour of values attribute."""
    srf = xtgeo.RegularSurface(**default_surface)

    newvalues = srf.values.copy()
    srf.values = newvalues
    assert (srf.values.data == default_surface["values"]).all()

    srf.values = 44
    assert set(srf.values.data.flatten()) == {1e33, 44.0}

    srf = xtgeo.RegularSurface(**default_surface)
    newvalues = np.ones((srf.ncol, srf.nrow))
    srf.values = newvalues
    assert isinstance(srf.values, np.ma.MaskedArray)
    assert set(srf.values.data.flatten()) == {1.0}

    newvalues = np.arange(15).reshape(srf.ncol, srf.nrow)
    newvalues = np.ma.masked_where(newvalues < 3, newvalues)
    srf.values = newvalues
    assert (srf.values.data == newvalues.data).all()

    # list like input with undefined value
    newvalues = list(range(15))
    newvalues[2] = srf.undef
    newvalues[4] = float("nan")  # alternative
    srf.values = newvalues
    assert np.ma.count_masked(srf.values) == 2

    # list like input with wrong (non-broadcastable) length
    newvalues = list(range(14))
    newvalues[2] = srf.undef
    with pytest.raises(ValueError):
        srf.values = newvalues

    # plain wrong input
    with pytest.raises(ValueError):
        srf.values = "text"


def test_set_values1d(default_surface):
    """Test behaviour of set_values1d method."""
    srf = xtgeo.RegularSurface(**default_surface)
    new = srf.copy()
    assert np.ma.count_masked(new.values) == 1

    # values is a new np array; hence old mask shall not be reused
    vals = np.zeros((new.dimensions)) + 100
    vals[1, 1] = srf.undef

    new.set_values1d(vals.ravel())
    assert not np.array_equal(srf.values.mask, new.values.mask)
    assert np.ma.count_masked(new.values) == 1

    # values are modified from existing, hence mask will be additive
    new = srf.copy()
    vals = new.values.copy()
    vals[vals < 2] = new.undef
    new.values = vals
    assert np.ma.count_masked(new.values) == 2

    # values are modified from existing and set via set_values1d
    new = srf.copy()
    vals = new.values.copy()
    vals = vals.flatten()
    vals[vals < 2] = new.undef
    assert np.ma.count_masked(vals) == 1
    new.set_values1d(vals)
    assert np.ma.count_masked(new.values) == 2


def test_ijxyz_import1(tmpdir):
    """Import some IJ XYZ format, typical seismic."""
    logger.info("Import and export...")

    xsurf = xtgeo.RegularSurface()
    xsurf.from_file(TESTSET4A, fformat="ijxyz")
    xsurf.describe()
    assert_almostequal(xsurf.xori, 600413.048444, 0.0001)
    assert_almostequal(xsurf.xinc, 25.0, 0.0001)
    assert xsurf.ncol == 280
    assert xsurf.nrow == 1341
    xsurf.to_file(join(tmpdir, "ijxyz_set4a.gri"))


@pytest.mark.skipif(sys.platform == "win32", reason="divide by zero issue")
def test_ijxyz_import2(tmpdir):
    """Import some IJ XYZ small set with YFLIP -1."""
    logger.info("Import and export...")

    xsurf = xtgeo.RegularSurface()
    xsurf.from_file(TESTSET4B, fformat="ijxyz")
    xsurf.describe()
    assert_almostequal(xsurf.values.mean(), 5037.5840, 0.001)
    assert xsurf.ncol == 51
    assert xsurf.yflip == -1
    assert xsurf.nactive == 2578
    xsurf.to_file(join(tmpdir, "ijxyz_set4b.gri"))


@pytest.mark.skipif(sys.platform == "win32", reason="Unknown issue")
def test_ijxyz_import4_ow_messy_dat(tmpdir):
    """Import some IJ XYZ small set with YFLIP -1 from OW messy dat format."""
    logger.info("Import and export...")

    xsurf = xtgeo.RegularSurface()
    xsurf.from_file(TESTSET4D, fformat="ijxyz")
    xsurf.describe()
    assert_almostequal(xsurf.values.mean(), 5037.5840, 0.001)
    assert xsurf.ncol == 51
    assert xsurf.yflip == -1
    assert xsurf.nactive == 2578
    xsurf.to_file(join(tmpdir, "ijxyz_set4d.gri"))


@pytest.mark.skipif(sys.platform == "win32", reason="Unknown issue")
def test_ijxyz_import3(tmpdir):
    """Import some IJ XYZ small set yet again."""
    logger.info("Import and export...")

    xsurf = xtgeo.RegularSurface()
    xsurf.from_file(TESTSET4C, fformat="ijxyz")
    xsurf.describe()
    xsurf.to_file(join(tmpdir, "ijxyz_set4c.gri"))


def test_irapbin_import1():
    """Import Reek Irap binary."""
    logger.info("Import and export...")

    xsurf = xtgeo.RegularSurface(TESTSET2)
    assert xsurf.ncol == 1264
    assert xsurf.nrow == 2010
    assert_almostequal(xsurf.values[11, 0], 1678.89733887, 0.00001)
    assert_almostequal(xsurf.values[1263, 2009], 1893.75, 0.01)
    xsurf.describe()


def test_irapbin_import_use_pathib():
    """Import Reek Irap binary."""
    logger.info("Import and export...")

    pobj = Path(TESTSET2)

    xsurf = xtgeo.RegularSurface(pobj)
    assert xsurf.ncol == 1264
    assert xsurf.nrow == 2010


@pytest.mark.skipifroxar
def test_irapbin_import_quickplot(tmpdir, show_plot, generate_plot):
    """Import Reek Irap binary and do quickplot."""
    if not show_plot or generate_plot:
        pytest.skip()

    xsurf = xtgeo.RegularSurface(TESTSET2)
    cmap = "gist_earth"

    if show_plot:
        xsurf.quickplot(colormap=cmap)
    if generate_plot:
        xsurf.quickplot(join(tmpdir, "qplot.jpg"), colormap=cmap)


def test_irapbin_import_metadatafirst_simple():
    srf = xtgeo.RegularSurface(TESTSET2, values=False)
    assert set(srf.values.data.flatten().tolist()) == {0.0}
    assert srf.ncol == 1264

    srf.load_values()
    assert srf.values.mean() == pytest.approx(1672.8242448561361)


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
    assert_almostequal(sur[nsurf - 1].values[11, 0], 1678.89733887, 0.00001)


def test_irapbin_export_test(tmpdir):
    """Import Reek Irap binary using different numpy details, test timing"""
    logger.info("Import and export...")

    nsurf = 10
    surf = xtgeo.RegularSurface(TESTSET5)

    t1 = xtg.timer()
    for _ix in range(nsurf):
        surf.to_file(join(tmpdir, "tull1"), engine="cxtgeo")

    t2a = xtg.timer(t1)
    logger.info("Saving %s surfaces xtgeo %s secs.", nsurf, t2a)

    t2b = xtg.timer(t1)
    logger.info("TEST Saving %s surfaces xtgeo %s secs.", nsurf, t2b)

    gain = (t2a - t2b) / t2a
    logger.info("Speed gain %s percent", gain * 100)


def test_petromodbin_import_export(tmpdir):
    """Import Petromod PDM binary example."""
    logger.info("Import and export...")

    petromod = xtgeo.RegularSurface(TESTSET6A)
    irapbin = xtgeo.RegularSurface(TESTSET6B)
    assert petromod.ncol == irapbin.ncol
    assert petromod.nrow == irapbin.nrow
    assert petromod.values1d[200000] == irapbin.values1d[200000]

    testfile = join(tmpdir, "petromod.pmd")
    petromod.to_file(testfile, fformat="petromod")
    petromod_again = xtgeo.RegularSurface(testfile)
    assert petromod_again.values1d[200000] == irapbin.values1d[200000]

    # test with roation 0 and rotation origins 0
    petromod = xtgeo.RegularSurface(TESTSET6C)
    assert petromod.ncol == irapbin.ncol
    assert petromod.nrow == irapbin.nrow
    assert petromod.values1d[200000] == irapbin.values1d[200000]

    testfile = join(tmpdir, "petromod_other_units.pmd")
    petromod.to_file(testfile, fformat="petromod", pmd_dataunits=(16, 300))
    petromod.from_file(testfile, fformat="petromod")

    with pytest.raises(ValueError):
        petromod.to_file(
            join(tmpdir, "null"), fformat="petromod", pmd_dataunits=(-2, 999)
        )


def test_zmap_import_export(tmpdir, default_surface):
    """Import and export ZMAP ascii example."""
    logger.info("Import and export...")

    zmap = RegularSurface(**default_surface)
    zmap.to_file(join(tmpdir, "zmap1.zmap"), fformat="zmap_ascii")
    zmap2 = RegularSurface()
    zmap2.from_file(join(tmpdir, "zmap1.zmap"), fformat="zmap_ascii")

    assert zmap.values[0, 1] == zmap2.values[0, 1] == 6.0

    one1 = zmap.values.ravel()
    one2 = zmap2.values.ravel()
    assert one1.all() == one2.all()

    zmap.to_file(join(tmpdir, "zmap2.zmap"), fformat="zmap_ascii", engine="python")
    zmap3 = RegularSurface()
    zmap3.from_file(join(tmpdir, "zmap2.zmap"), fformat="zmap_ascii")
    one3 = zmap3.values.ravel()
    assert one1.all() == one3.all()


def test_swapaxes(tmpdir):
    """Import Reek Irap binary and swap axes."""
    xsurf = xtgeo.RegularSurface(TESTSET5)
    xsurf.describe()
    logger.info(xsurf.yflip)
    xsurf.to_file(join(tmpdir, "notswapped.gri"))
    val1 = xsurf.values.copy()
    xsurf.swapaxes()
    xsurf.describe()
    logger.info(xsurf.yflip)
    xsurf.to_file(join(tmpdir, "swapped.gri"))
    xsurf.swapaxes()
    val2 = xsurf.values.copy()
    xsurf.to_file(join(tmpdir, "swapped_reswapped.gri"))
    valdiff = val2 - val1
    assert_almostequal(valdiff.mean(), 0.0, 0.00001)
    assert_almostequal(valdiff.std(), 0.0, 0.00001)


def test_autocrop():
    """Import Reek Irap binary and autocrop surface"""
    xsurf = xtgeo.RegularSurface(TESTSET5)
    xcopy = xsurf.copy()

    xcopy.autocrop()

    assert xsurf.ncol > xcopy.ncol
    assert xcopy.nrow == 442
    assert xsurf.values.mean() == xcopy.values.mean()


def test_irapasc_import1():
    """Import Reek Irap ascii."""
    logger.info("Import and export...")

    xsurf = xtgeo.RegularSurface(TESTSET3, fformat="irap_ascii")
    assert xsurf.ncol == 1264
    assert xsurf.nrow == 2010
    assert_almostequal(xsurf.values[11, 0], 1678.89733887, 0.001)
    assert_almostequal(xsurf.values[1263, 2009], 1893.75, 0.01)


def test_irapasc_import1_engine_python():
    """Import Reek Irap ascii using python read engine"""
    logger.info("Import and export...")

    xsurf = xtgeo.RegularSurface(TESTSET3, fformat="irap_ascii", engine="python")
    assert xsurf.ncol == 1264
    assert xsurf.nrow == 2010
    assert_almostequal(xsurf.values[11, 0], 1678.89733887, 0.001)
    assert_almostequal(xsurf.values[1263, 2009], 1893.75, 0.01)


def test_irapbin_import1_engine_python():
    """Import Reek Irap binary using python read engine"""
    logger.info("Import and export...")

    xsurf = xtgeo.RegularSurface(TESTSET2, fformat="irap_binary", engine="python")
    assert xsurf.ncol == 1264
    assert xsurf.nrow == 2010
    assert_almostequal(xsurf.values[11, 0], 1678.89733887, 0.001)
    assert_almostequal(xsurf.values[1263, 2009], 1893.75, 0.01)

    xsurf2 = xtgeo.RegularSurface(TESTSET2, fformat="irap_binary", engine="cxtgeo")
    assert xsurf.values.mean() == xsurf2.values.mean()


def test_irapasc_io_engine_python(tmpdir):
    """Test IO using pure python read/write"""
    xsurf1 = xtgeo.RegularSurface()

    usefile1 = join(tmpdir, "surf3a.fgr")
    usefile2 = join(tmpdir, "surf3b.fgr")

    print(xsurf1[1, 1])

    xsurf1.to_file(usefile1, fformat="irap_ascii", engine="cxtgeo")
    xsurf1.to_file(usefile2, fformat="irap_ascii", engine="python")

    xsurf2 = xtgeo.RegularSurface(usefile2, fformat="irap_ascii", engine="python")
    xsurf3 = xtgeo.RegularSurface(usefile2, fformat="irap_ascii", engine="cxtgeo")

    assert xsurf1.ncol == xsurf3.ncol == xsurf3.ncol

    assert xsurf1.values[1, 1] == xsurf2.values[1, 1] == xsurf3.values[1, 1]


@pytest.mark.bigtest
def test_irapasc_import1_engine_compare():
    """Import Reek Irap ascii using python read engine"""
    logger.info("Import and export...")

    tt0 = xtg.timer()
    for _ in range(10):
        xtgeo.RegularSurface(TESTSET3, fformat="irap_ascii", engine="cxtgeo")
    print("CXTGeo engine for read:", xtg.timer(tt0))

    tt0 = xtg.timer()
    for _ in range(10):
        xtgeo.RegularSurface(TESTSET3, fformat="irap_ascii", engine="python")
    print("Python engine for read:", xtg.timer(tt0))


def test_minmax_rotated_map():
    """Min and max of rotated map"""
    logger.info("Import and export...")

    x = xtgeo.RegularSurface()
    x.from_file(TESTSET1, fformat="irap_binary")

    assert_almostequal(x.xmin, 454637.6, 0.1)
    assert_almostequal(x.xmax, 468895.1, 0.1)
    assert_almostequal(x.ymin, 5925995.0, 0.1)
    assert_almostequal(x.ymax, 5939998.7, 0.1)


def test_operator_overload(**default_surface):
    """Test operations between two surface in different ways"""

    surf1 = xtgeo.RegularSurface(ncol=100, nrow=50, rotation=10, values=100)
    assert surf1.values.mean() == 100.0
    id1 = id(surf1)
    id1v = id(surf1.values)

    surf2 = xtgeo.RegularSurface(ncol=100, nrow=50, rotation=0, values=100)
    diff = surf1 - surf2
    assert id(diff) != id1
    assert diff.values.mean() == 0.0

    assert id(surf1) == id1
    surf1 += diff
    assert id(surf1) == id1
    assert id(surf1.values) == id1v

    surf1 /= diff
    assert (surf1.values.count()) == 0


def test_twosurfaces_oper():
    """Test operations between two surface in more different ways"""

    surf1 = xtgeo.RegularSurface(TESTSET1)
    surf2 = xtgeo.RegularSurface(TESTSET1A)

    iso1 = surf2.copy()
    iso1.values -= surf1.values
    iso1mean = iso1.values.mean()
    assert_almostequal(iso1mean, 43.71, 0.01)

    iso2 = surf2.copy()
    iso2.subtract(surf1)
    iso2mean = iso2.values.mean()
    assert_almostequal(iso2mean, 43.71, 0.01)
    assert iso1.values.all() == iso2.values.all()

    iso3 = surf2 - surf1
    assert iso1.values.all() == iso3.values.all()
    assert isinstance(iso3, xtgeo.RegularSurface)

    sum1 = surf2.copy()
    sum1.values += surf1.values
    assert_almostequal(sum1.values.mean(), 3441.0, 0.01)

    sum2 = surf2.copy()
    sum2.add(surf1)
    assert sum1.values.all() == sum2.values.all()

    sum3 = surf1 + surf2
    assert sum1.values.all() == sum3.values.all()

    zrf2 = surf2.copy()
    zrf1 = surf1.copy()
    newzrf1 = surf1.copy()

    newzrf1.values = zrf2.values / zrf1.values
    assert newzrf1.values.mean() == pytest.approx(1.0257, abs=0.01)


def test_surface_comparisons():
    """Test the surface comparison overload"""
    surf1 = xtgeo.RegularSurface()
    id1 = id(surf1)

    surf2 = surf1.copy()

    cmp = surf1 == surf2
    np.testing.assert_equal(cmp, True)

    cmp = surf1 != surf2
    np.testing.assert_equal(cmp, False)

    cmp = surf1 <= surf2
    np.testing.assert_equal(cmp, True)

    cmp = surf1 < surf2
    np.testing.assert_equal(cmp, False)

    cmp = surf1 > surf2
    np.testing.assert_equal(cmp, False)

    surf2.values[0, 0] = -2
    cmp = surf1 == surf2
    assert bool(cmp[0, 0]) is False

    assert id(surf1) == id1


def test_surface_subtract_etc(default_surface):
    """Test the simple surf.subtract etc methods"""
    surf1 = xtgeo.RegularSurface(**default_surface)
    id1 = id(surf1)
    mean1 = surf1.values.mean()

    surf2 = surf1.copy()
    surf1.subtract(surf2)
    assert surf1.values.mean() == 0.0
    assert id(surf1) == id1

    surf1.add(surf2)
    assert surf1.values.mean() == mean1
    assert id(surf1) == id1

    surf1.multiply(surf2)
    surf1.divide(surf2)
    assert surf1.values.mean() == mean1
    assert id(surf1) == id1

    surf1.subtract(2)
    assert surf1.values.mean() == mean1 - 2
    assert id(surf1) == id1


@pytest.mark.bigtest
def test_irapbin_io(tmpdir):
    """Import and export Irap binary."""
    logger.info("Import and export...")

    x = xtgeo.RegularSurface()
    x.from_file(TESTSET1, fformat="irap_binary")

    x.to_file(join(tmpdir, "reek1_test.fgr"), fformat="irap_ascii")

    logger.debug("NX is %s", x.ncol)

    assert_equal(x.ncol, 554)

    # get the 1D numpy
    v1d = x.get_zval()

    logger.info("Mean VALUES are: %s", np.nanmean(v1d))

    zval = x.values

    # add value via numpy
    zval = zval + 300
    # update
    x.values = zval

    assert_almostequal(x.values.mean(), 1998.648, 0.01)

    x.to_file(join(tmpdir, "reek1_plus_300_a.fgr"), fformat="irap_ascii")
    x.to_file(join(tmpdir, "reek1_plus_300_b.gri"), fformat="irap_binary")

    mfile = TESTSET1

    # direct import
    y = xtgeo.RegularSurface(mfile)
    assert_equal(y.ncol, 554)

    # semidirect import
    cc = xtgeo.RegularSurface().from_file(mfile)
    assert_equal(cc.ncol, 554)


def test_get_values1d(default_surface):
    """Get the 1D array, different variants as masked, notmasked, order, etc"""

    xmap = xtgeo.RegularSurface(**default_surface)
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


def test_ij_map_indices(default_surface):
    """Get the IJ MAP indices"""

    xmap = xtgeo.RegularSurface(**default_surface)
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
    assert_almostequal(xxv[1], 0.0, 0.001)

    xcv, _ycv = xmap.get_xy_values(order="F")
    xxv = xcv.ravel(order="K")
    assert_almostequal(xxv[1], 25.0, 0.001)

    xcv, _ycv = xmap.get_xy_values(order="C", asmasked=True)

    xxv = xcv.ravel(order="K")
    assert_almostequal(xxv[1], 0.0, 0.001)

    xcv, _ycv = xmap.get_xy_values(order="F", asmasked=True)

    xxv = xcv.ravel(order="K")
    assert_almostequal(xxv[1], 25.0, 0.001)


def test_get_xy_values1d():
    """Get the XY coordinate values"""

    xmap = xtgeo.RegularSurface()

    xcv, _ycv = xmap.get_xy_values1d(activeonly=False, order="C")

    assert_almostequal(xcv[1], 0.0, 0.001)

    xcv, _ycv = xmap.get_xy_values1d(activeonly=False, order="F")

    assert_almostequal(xcv[1], 25.0, 0.001)

    xcv, _ycv = xmap.get_xy_values1d(activeonly=True, order="C")

    assert_almostequal(xcv[1], 0.0, 0.001)

    xcv, _ycv = xmap.get_xy_values1d(activeonly=True, order="F")

    assert_almostequal(xcv[1], 25.0, 0.001)


def test_dataframe_simple():
    """Get a pandas Dataframe object"""

    xmap = xtgeo.RegularSurface(TESTSET1)

    dfrc = xmap.dataframe(ijcolumns=True, order="C", activeonly=True)

    assert_almostequal(dfrc["X_UTME"][2], 465956.274, 0.01)

    xmap = xtgeo.surface_from_file(TESTSET2)

    dfrc = xmap.dataframe()

    assert_almostequal(dfrc["X_UTME"][2], 461582.562498, 0.01)

    xmap.coarsen(2)
    dfrc = xmap.dataframe()

    assert_almostequal(dfrc["X_UTME"][2], 461577.5575, 0.01)


@pytest.mark.bigtest
def test_dataframe_more(tmpdir):
    """Get a pandas Dataframe object, more detailed testing"""

    xmap = xtgeo.RegularSurface(TESTSET1)

    xmap.describe()

    dfrc = xmap.dataframe(ijcolumns=True, order="C", activeonly=True)
    dfrf = xmap.dataframe(ijcolumns=True, order="F", activeonly=True)

    dfrc.to_csv(join(tmpdir, "regsurf_df_c.csv"))
    dfrf.to_csv(join(tmpdir, "regsurf_df_f.csv"))
    xmap.to_file(join(tmpdir, "regsurf_df.ijxyz"), fformat="ijxyz")

    assert_almostequal(dfrc["X_UTME"][2], 465956.274, 0.01)
    assert_almostequal(dfrf["X_UTME"][2], 462679.773, 0.01)

    dfrcx = xmap.dataframe(ijcolumns=False, order="C", activeonly=True)
    dfrcx.to_csv(join(tmpdir, "regsurf_df_noij_c.csv"))
    dfrcy = xmap.dataframe(
        ijcolumns=False, order="C", activeonly=False, fill_value=None
    )
    dfrcy.to_csv(join(tmpdir, "regsurf_df_noij_c_all.csv"))


def test_get_xy_value_lists_small(default_surface):
    """Get the xy list and value list from small test case"""

    x = xtgeo.RegularSurface(**default_surface)  # default instance

    xylist, valuelist = x.get_xy_value_lists(valuefmt="8.3f", xyfmt="12.2f")

    logger.info(xylist[2])
    logger.info(valuelist[2])

    assert_equal(valuelist[2], 3.0)


@pytest.mark.bigtest
def test_get_xy_value_lists_reek():
    """Get the xy list and value list"""

    x = xtgeo.RegularSurface()
    x.from_file(TESTSET1, fformat="irap_binary")

    xylist, valuelist = x.get_xy_value_lists(valuefmt="8.3f", xyfmt="12.2f")

    logger.info(xylist[2])
    logger.info(valuelist[2])

    assert_equal(valuelist[2], 1910.445)


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
    """Check similarity of two surfaces.

    0.0 means identical in terms of mean value.
    """

    logger.info("Test if surfaces are similar...")

    mfile = TESTSET1

    x = xtgeo.RegularSurface(mfile)
    y = xtgeo.RegularSurface(mfile)

    si = x.similarity_index(y)
    assert_equal(si, 0.0)

    y.values = y.values * 2

    si = x.similarity_index(y)
    assert_equal(si, 1.0)


def test_irapbin_io_loop(tmpdir):
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
        x.to_file(join(tmpdir, "troll.gri"), fformat="irap_binary")

        assert m1 == pytest.approx(m2 - 300)


def test_irapbin_export_py(tmpdir):
    """Export Irapbin with pure python"""

    x = xtgeo.RegularSurface()
    x.from_file(TESTSET1, fformat="irap_binary")

    t0 = xtg.timer()
    for _ in range(10):
        x.to_file(join(tmpdir, "purecx.gri"), fformat="irap_binary", engine="cxtgeo")
    t1 = xtg.timer(t0)
    print("CXTGeo based write: {:3.4f}".format(t1))

    t0 = xtg.timer()
    for _ in range(10):
        x.to_file(join(tmpdir, "purepy.gri"), fformat="irap_binary", engine="python")
    t2 = xtg.timer(t0)
    print("Python based write: {:3.4f}".format(t2))
    print("Ratio python based / cxtgeo based {:3.4f}".format(t2 / t1))

    s1 = xtgeo.RegularSurface(join(tmpdir, "purecx.gri"))
    s2 = xtgeo.RegularSurface(join(tmpdir, "purepy.gri"))

    assert s1.values.mean() == s2.values.mean()

    assert s1.values[100, 100] == s2.values[100, 100]


def test_distance_from_point(tmpdir):
    """Distance from point."""

    x = xtgeo.RegularSurface()
    x.from_file(TESTSET1, fformat="irap_binary")

    x.distance_from_point(point=(464960, 7336900), azimuth=30)

    x.to_file(join(tmpdir, "reek1_dist_point.gri"), fformat="irap_binary")


def test_value_from_xy():
    """
    get Z value from XY point
    """

    x = xtgeo.RegularSurface()
    x.from_file(TESTSET1, fformat="irap_binary")

    z = x.get_value_from_xy(point=(460181.036, 5933948.386))

    assert_almostequal(z, 1625.11, 0.01)

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
    print(myfence)

    x = xtgeo.RegularSurface(TESTSET1)

    newfence = x.get_fence(myfence)

    logger.debug("updated NP:")
    logger.debug(newfence)
    print(newfence)

    assert_almostequal(newfence[1][2], 1720.9094, 0.01)


@pytest.mark.parametrize(
    "infence, sampling, expected",
    [
        (
            [
                [1.0, 25.0, -9],
                [20.0, 25.0, -9],
            ],
            "bilinear",
            [6.04, 6.8],
        ),
        (
            [
                [1.0, 25.0, -9],
                [20.0, 25.0, -9],
            ],
            "nearest",
            [6.0, 7.0],
        ),
        (
            [
                [26.0, 2.0, -9],
                [77.0, 22.0, -9],
                [99.0, 49.0, -9],
            ],
            "bilinear",
            [2.44, 8.48, 14.76],
        ),
        (
            [
                [26.0, 2.0, -9],
                [77.0, 22.0, -9],
                [99.0, 49.0, -9],
            ],
            "nearest",
            [2.0, 9.0, 15],
        ),
        (
            [
                [-1.0, 2.0, -9],
                [77.0, 13.0, -9],
                [49.0, 48.0, -9],
            ],
            "bilinear",
            [-999, 6.68, -999],
        ),
        (
            [
                [-1.0, 2.0, -9],
                [77.0, 13.0, -9],
                [49.0, 48.0, -9],
            ],
            "nearest",
            [-999, 9, -999],
        ),
    ],
)
def test_fence_sampling(infence, sampling, expected, default_surface):
    """Test a very simple fence with different sampling methods."""
    surf = xtgeo.RegularSurface(**default_surface)

    myfence = np.array(infence)
    myfence[myfence == -999] = np.nan
    newfence = surf.get_fence(myfence, sampling=sampling)
    assert np.allclose(newfence[:, 2], expected, equal_nan=True)


def test_get_randomline_frompolygon(show_plot):
    """Test randomline with both bilinear and nearest sampling for surfaces."""
    fence = xtgeo.Polygons(FENCE1)
    xs = xtgeo.RegularSurface(TESTSET1)

    # get the polygon
    fspec = fence.get_fence(distance=10, nextend=2, asnumpy=False)
    assert_almostequal(fspec.dataframe[fspec.dhname][4], 10, 1)

    fspec = fence.get_fence(distance=20, nextend=5, asnumpy=True)

    arr1 = xs.get_randomline(fspec)
    arr2 = xs.get_randomline(fspec, sampling="nearest")

    x = arr1[:, 0]
    y1 = arr1[:, 1]
    y2 = arr2[:, 1]

    assert y1.mean() == pytest.approx(1706.7514, abs=0.001)
    assert y2.mean() == pytest.approx(1706.6995, abs=0.001)

    if show_plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(x, y1)
        plt.plot(x, y2)
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
    assert_almostequal(srf.values.mean(), 1698.648, 0.001)

    srf.fill()
    minv2 = srf.values.min()
    assert_almostequal(srf.values.mean(), 1705.201, 0.001)
    assert_almostequal(minv1, minv2, 0.000001)

    srf = xtgeo.RegularSurface()
    srf.from_file(TESTSET1, fformat="irap_binary")
    srf.fill(444)
    assert_almostequal(srf.values.mean(), 1342.10498, 0.001)


def test_smoothing():
    """Smooth the the surface"""

    srf = xtgeo.RegularSurface()
    srf.from_file(TESTSET1, fformat="irap_binary")

    mean1 = srf.values.mean()
    assert_almostequal(mean1, 1698.65, 0.1)

    srf.smooth(iterations=1, width=5)

    mean2 = srf.values.mean()
    assert_almostequal(mean2, 1698.65, 0.3)  # smoothed ~same mean

    assert mean1 != mean2  # but not exacly same
