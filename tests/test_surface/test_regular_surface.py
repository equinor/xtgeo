"""Testing RegularSurface class with methods."""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

import xtgeo
from xtgeo import RegularSurface
from xtgeo.common import XTGeoDialog
from xtgeo.common.exceptions import InvalidFileFormatError

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

TESTSET1 = pathlib.Path("surfaces/reek/1/topreek_rota.gri")
TESTSET1A = pathlib.Path("surfaces/reek/1/basereek_rota.gri")
TESTSET2 = pathlib.Path("surfaces/reek/1/topupperreek.gri")
TESTSET3 = pathlib.Path("surfaces/reek/1/topupperreek.fgr")
TESTSET4A = pathlib.Path("surfaces/etc/ib_test-horizon.map")  # IJXYZ table
TESTSET4B = pathlib.Path("surfaces/etc/ijxyz1.map")  # IJXYZ table
TESTSET4D = pathlib.Path("surfaces/etc/ijxyz1.dat")  # IJXYZ table OW
TESTSET4C = pathlib.Path("surfaces/etc/testx_1500_edit1.map")
TESTSET5 = pathlib.Path("surfaces/reek/2/02_midreek_rota.gri")
TESTSET6A = pathlib.Path("surfaces/etc/seabed_p.pmd")
TESTSET6B = pathlib.Path("surfaces/etc/seabed_p.gri")
TESTSET6C = pathlib.Path("surfaces/etc/seabed_p_v2.pmd")
TESTSET7A = pathlib.Path("surfaces/etc/topvolantis_genhash.gri")
TESTSET7B = pathlib.Path("surfaces/etc/toptherys_genhash.gri")

FENCE1 = pathlib.Path("polygons/reek/1/fence.pol")


def test_surface_from_file_missing(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        xtgeo.surface_from_file(tmp_path / "nosuchfile", fformat="irap_binary")


@pytest.mark.filterwarnings("ignore:Default values*")
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


def test_regularsurface_copy():
    """Test copying a surface instance."""
    values = np.random.normal(2000, 50, size=12)
    srf = xtgeo.RegularSurface(ncol=3, nrow=4, xinc=20, yinc=20, values=values)
    assert srf.nactive == 12
    new = srf.copy()
    assert srf.nactive == new.nactive

    np.testing.assert_array_equal(srf.values, new.values)
    assert new.xori == srf.xori
    assert new.xinc == srf.xinc
    assert new.yori == srf.yori
    assert new.xinc == srf.xinc
    assert new.rotation == srf.rotation
    assert new.yflip == srf.yflip


@pytest.mark.filterwarnings("ignore:Default values*")
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


def test_ijxyz_import1(tmp_path, testdata_path):
    """Import some IJ XYZ format, typical seismic."""
    logger.info("Import and export...")

    xsurf = xtgeo.surface_from_file(testdata_path / TESTSET4A, fformat="ijxyz")
    xsurf.describe()
    assert xsurf.xori == pytest.approx(600413.048444, abs=0.0001)
    assert xsurf.xinc == pytest.approx(25.0, abs=0.0001)
    assert xsurf.ncol == 280
    assert xsurf.nrow == 1341
    xsurf.to_file(tmp_path / "ijxyz_set4a.gri")


@pytest.mark.skipif(sys.platform == "win32", reason="divide by zero issue")
def test_ijxyz_import2(tmp_path, testdata_path):
    """Import some IJ XYZ small set with YFLIP -1."""
    logger.info("Import and export...")

    xsurf = xtgeo.surface_from_file(testdata_path / TESTSET4B, fformat="ijxyz")
    xsurf.describe()
    assert xsurf.values.mean() == pytest.approx(5037.5840, abs=0.001)
    assert xsurf.ncol == 51
    assert xsurf.yflip == -1
    assert xsurf.nactive == 2578
    xsurf.to_file(tmp_path / "ijxyz_set4b.gri")


@pytest.mark.skipif(sys.platform == "win32", reason="Unknown issue")
def test_ijxyz_import4_ow_messy_dat(tmp_path, testdata_path):
    """Import some IJ XYZ small set with YFLIP -1 from OW messy dat format."""
    logger.info("Import and export...")

    xsurf = xtgeo.surface_from_file(testdata_path / TESTSET4D, fformat="ijxyz")
    xsurf.describe()
    assert xsurf.values.mean() == pytest.approx(5037.5840, 0.001)
    assert xsurf.ncol == 51
    assert xsurf.yflip == -1
    assert xsurf.nactive == 2578
    xsurf.to_file(tmp_path / "ijxyz_set4d.gri")


@pytest.mark.skipif(sys.platform == "win32", reason="Unknown issue")
def test_ijxyz_import3(tmp_path, testdata_path):
    """Import some IJ XYZ small set yet again."""
    logger.info("Import and export...")

    xsurf = xtgeo.surface_from_file(testdata_path / TESTSET4C, fformat="ijxyz")
    xsurf.describe()
    xsurf.to_file(tmp_path / "ijxyz_set4c.gri")


def test_irapbin_import1(testdata_path):
    """Import Reek Irap binary."""
    logger.info("Import and export...")

    xsurf = xtgeo.surface_from_file(testdata_path / TESTSET2)
    assert xsurf.ncol == 1264
    assert xsurf.nrow == 2010
    assert xsurf.values[11, 0] == pytest.approx(1678.89733887, abs=0.00001)
    assert xsurf.values[1263, 2009] == pytest.approx(1893.75, abs=0.01)
    xsurf.describe()


def test_irapbin_import_32bit(testdata_path):
    """Import Reek Irap binary, force 32 bit storage in import."""
    # setting dtype in importing surface
    xsurf = xtgeo.surface_from_file(testdata_path / TESTSET2, dtype=np.float32)
    assert xsurf.values.dtype == np.float32

    # setting dtype explicit as 64 bit
    xsurf.dtype = np.float64
    assert xsurf.values.dtype == np.float64

    # setting dtype explicit as 32 bit
    xsurf.dtype = np.float32
    assert xsurf.values.dtype == np.float32

    xsurf.dtype = "float64"
    assert xsurf.values.dtype == np.float64


def test_irapbin_import_invalid_dtype(testdata_path):
    """Import Reek Irap binary, invalid dtype."""
    # setting dtype in importing surface
    with pytest.raises(AttributeError):
        xtgeo.surface_from_file(testdata_path / TESTSET2, dtype=np.float33)


def test_irapbin_import_use_pathib(testdata_path):
    """Import Reek Irap binary."""
    logger.info("Import and export...")

    xsurf = xtgeo.surface_from_file(testdata_path / TESTSET2)
    assert xsurf.ncol == 1264
    assert xsurf.nrow == 2010


def test_irapbin_import_quickplot(tmp_path, show_plot, generate_plot, testdata_path):
    """Import Reek Irap binary and do quickplot."""
    if not show_plot and not generate_plot:
        pytest.skip()

    xsurf = xtgeo.surface_from_file(testdata_path / TESTSET2)
    cmap = "gist_earth"

    if show_plot:
        xsurf.quickplot(colormap=cmap)
    if generate_plot:
        xsurf.quickplot(tmp_path / "qplot.jpg", colormap=cmap)


def test_irapbin_import_metadatafirst_simple(testdata_path):
    srf = xtgeo.surface_from_file(testdata_path / TESTSET2, values=False)
    assert set(srf.values.data.flatten().tolist()) == {0.0}
    assert srf.ncol == 1264

    srf.load_values()
    assert srf.values.mean() == pytest.approx(1672.8242448561361)


def test_irapbin_import_metadatafirst(testdata_path):
    """Import Reek Irap binary, first with metadata only, then values."""
    logger.info("Import and export...")

    nsurf = 10
    sur = []
    t1 = xtg.timer()
    for ix in range(nsurf):
        sur.append(xtgeo.surface_from_file(testdata_path / TESTSET2, values=False))
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
    assert sur[nsurf - 1].values[11, 0] == pytest.approx(1678.89733887, abs=0.00001)


def test_irapbin_export_test(tmp_path, testdata_path):
    """Import Reek Irap binary using different numpy details, test timing"""
    logger.info("Import and export...")

    nsurf = 10
    surf = xtgeo.surface_from_file(testdata_path / TESTSET5)

    t1 = xtg.timer()
    for _ix in range(nsurf):
        surf.to_file(tmp_path / "tull1", engine="cxtgeo")

    t2a = xtg.timer(t1)
    logger.info("Saving %s surfaces xtgeo %s secs.", nsurf, t2a)

    t2b = xtg.timer(t1)
    logger.info("TEST Saving %s surfaces xtgeo %s secs.", nsurf, t2b)

    gain = (t2a - t2b) / t2a
    logger.info("Speed gain %s percent", gain * 100)


def test_petromodbin_import_export(tmp_path, testdata_path):
    """Import Petromod PDM binary example."""
    logger.info("Import and export...")

    petromod = xtgeo.surface_from_file(testdata_path / TESTSET6A)
    irapbin = xtgeo.surface_from_file(testdata_path / TESTSET6B)
    assert petromod.ncol == irapbin.ncol
    assert petromod.nrow == irapbin.nrow
    assert petromod.values1d[200000] == irapbin.values1d[200000]

    testfile = tmp_path / "petromod.pmd"
    petromod.to_file(testfile, fformat="petromod")
    petromod_again = xtgeo.surface_from_file(testfile)
    assert petromod_again.values1d[200000] == irapbin.values1d[200000]

    # test with roation 0 and rotation origins 0
    petromod = xtgeo.surface_from_file(testdata_path / TESTSET6C)
    assert petromod.ncol == irapbin.ncol
    assert petromod.nrow == irapbin.nrow
    assert petromod.values1d[200000] == irapbin.values1d[200000]

    testfile = tmp_path / "petromod_other_units.pmd"
    petromod.to_file(testfile, fformat="petromod", pmd_dataunits=(16, 300))
    petromod = xtgeo.surface_from_file(testfile, fformat="petromod")

    with pytest.raises(ValueError):
        petromod.to_file(tmp_path / "null", fformat="petromod", pmd_dataunits=(-2, 999))


@pytest.mark.filterwarnings("ignore:Default values*")
def test_zmap_import_export(tmp_path, default_surface):
    """Import and export ZMAP ascii example."""
    logger.info("Import and export...")

    zmap = RegularSurface(**default_surface)
    zmap.to_file(tmp_path / "zmap1.zmap", fformat="zmap_ascii")
    zmap2 = xtgeo.surface_from_file(tmp_path / "zmap1.zmap", fformat="zmap_ascii")

    assert zmap.values[0, 1] == zmap2.values[0, 1] == 6.0

    one1 = zmap.values.ravel()
    one2 = zmap2.values.ravel()
    assert one1.all() == one2.all()

    zmap.to_file(tmp_path / "zmap2.zmap", fformat="zmap_ascii", engine="python")
    zmap3 = xtgeo.surface_from_file(tmp_path / "zmap2.zmap", fformat="zmap_ascii")
    one3 = zmap3.values.ravel()
    assert one1.all() == one3.all()


@pytest.mark.filterwarnings("ignore:Default values*")
def test_zmap_larger_case_engine_integration(tmp_path, larger_surface):
    """Export ZMAP with various engines and check integration and detailed specs."""
    surf = RegularSurface(**larger_surface)

    surf.to_file(tmp_path / "cxtgeo.zmap", fformat="zmap")
    surf.to_file(tmp_path / "py.zmap", fformat="zmap", engine="python")
    surf.to_file(tmp_path / "irap.gri")

    isurf1 = xtgeo.surface_from_file(tmp_path / "cxtgeo.zmap")
    isurf2 = xtgeo.surface_from_file(tmp_path / "py.zmap")
    isurf3 = xtgeo.surface_from_file(tmp_path / "irap.gri")

    assert isurf1.values.tolist() == isurf2.values.tolist() == isurf3.values.tolist()

    # check values formatting in zmap files; the ZMAP format actually have a strict
    # setting spec where field width is hardcoded in header; here as fwidth * nfield
    # and less for incomplete lines. Also check number of decimals (fractional part)
    for fname in ["cxtgeo.zmap", "py.zmap"]:
        with open(tmp_path / fname, "r", encoding="ascii") as stream:
            buf = stream.readlines()
            _, _, nfield = [entry.strip() for entry in buf[1].split(",")]
            fwidth, _, _, ndeci, _ = [entry.strip() for entry in buf[2].split(",")]
            nrow, _, _, _, _, _ = [entry.strip() for entry in buf[3].split(",")]

            assert len(buf[12].rstrip()) == int(fwidth) * int(nfield)
            # incomplete line (in this particular case):
            assert len(buf[13].rstrip()) == (int(nrow) - int(nfield)) * int(fwidth)

            # check length of decimals (mantissa, fractional part) for first value
            firstnumber = buf[6].split()[0]
            _, fpart = firstnumber.split(".")
            assert len(fpart) == int(ndeci)


def test_swapaxes(tmp_path, testdata_path):
    """Import Reek Irap binary and swap axes."""
    xsurf = xtgeo.surface_from_file(testdata_path / TESTSET5)
    xsurf.describe()
    logger.info(xsurf.yflip)
    xsurf.to_file(tmp_path / "notswapped.gri")
    val1 = xsurf.values.copy()
    xsurf.swapaxes()
    xsurf.describe()
    logger.info(xsurf.yflip)
    xsurf.to_file(tmp_path / "swapped.gri")
    xsurf.swapaxes()
    val2 = xsurf.values.copy()
    xsurf.to_file(tmp_path / "swapped_reswapped.gri")
    valdiff = val2 - val1
    assert valdiff.mean() == pytest.approx(0.0, abs=0.00001)
    assert valdiff.std() == pytest.approx(0.0, abs=0.00001)


def test_make_lefthanded_simple(tmp_path):
    s1 = xtgeo.RegularSurface(
        ncol=7,
        nrow=3,
        xinc=1,
        yinc=2,
        values=np.arange(21),
        yflip=-1,
        rotation=30,
    )

    s1.to_file(tmp_path / "righthanded.gri")

    s2 = s1.copy()
    s2.make_lefthanded()
    s2.to_file(tmp_path / "lefthanded.gri")

    assert s1.values[6, 0] == s2.values[6, 2]


def test_make_righthanded_simple(tmp_path):
    s1 = xtgeo.RegularSurface(
        ncol=7,
        nrow=3,
        xinc=1,
        yinc=2,
        values=np.arange(21),
        yflip=1,
        rotation=30,
    )

    s1.to_file(tmp_path / "lefthanded.gri")

    s2 = s1.copy()
    s2.make_righthanded()
    s2.to_file(tmp_path / "righthanded.gri")

    assert s1.values[6, 0] == s2.values[6, 2]


def test_make_lefthanded_reek(tmp_path, testdata_path):
    """Import Reek Irap binary and make it lefthanded (pos axis down)."""
    s1 = xtgeo.surface_from_file(testdata_path / TESTSET5)

    # make the surface righthanded (pos axis down) for test
    s1._yflip = -1

    s1.to_file(tmp_path / "righthanded.gri")
    s2 = s1.copy()
    s2.make_lefthanded()
    assert s1.values.mean() == pytest.approx(s2.values.mean(), abs=0.001)
    assert s1.xmin == pytest.approx(s2.xmin, abs=0.001)
    assert s1.xmax == pytest.approx(s2.xmax, abs=0.001)
    assert s1.ymin == pytest.approx(s2.ymin, abs=0.001)
    assert s1.ymax == pytest.approx(s2.ymax, abs=0.001)
    assert s1.ncol == s2.ncol
    assert s1.nrow == s2.nrow


def test_autocrop(testdata_path):
    """Import Reek Irap binary and autocrop surface"""
    xsurf = xtgeo.surface_from_file(testdata_path / TESTSET5)
    xcopy = xsurf.copy()

    xcopy.autocrop()

    assert xsurf.ncol > xcopy.ncol
    assert xcopy.nrow == 442
    assert xsurf.values.mean() == xcopy.values.mean()


def test_irapasc_import1(testdata_path):
    """Import Reek Irap ascii."""
    logger.info("Import and export...")

    xsurf = xtgeo.surface_from_file(testdata_path / TESTSET3, fformat="irap_ascii")
    assert xsurf.ncol == 1264
    assert xsurf.nrow == 2010
    assert xsurf.values[11, 0] == pytest.approx(1678.89733887, abs=0.001)
    assert xsurf.values[1263, 2009] == pytest.approx(1893.75, abs=0.01)


def test_irapasc_import1_engine_python(testdata_path):
    """Import Reek Irap ascii using python read engine"""
    logger.info("Import and export...")

    xsurf = xtgeo.surface_from_file(
        testdata_path / TESTSET3, fformat="irap_ascii", engine="python"
    )
    assert xsurf.ncol == 1264
    assert xsurf.nrow == 2010
    assert xsurf.values[11, 0] == pytest.approx(1678.89733887, abs=0.001)
    assert xsurf.values[1263, 2009] == pytest.approx(1893.75, abs=0.01)


def test_irapbin_import1_engine_python(testdata_path):
    """Import Reek Irap binary using python read engine"""
    logger.info("Import and export...")

    xsurf = xtgeo.surface_from_file(
        testdata_path / TESTSET2, fformat="irap_binary", engine="python"
    )
    assert xsurf.ncol == 1264
    assert xsurf.nrow == 2010
    assert xsurf.values[11, 0] == pytest.approx(1678.89733887, abs=0.001)
    assert xsurf.values[1263, 2009] == pytest.approx(1893.75, abs=0.01)

    xsurf2 = xtgeo.surface_from_file(
        testdata_path / TESTSET2, fformat="irap_binary", engine="cxtgeo"
    )
    assert xsurf.values.mean() == xsurf2.values.mean()


@pytest.mark.filterwarnings("ignore:Default values*")
def test_irapasc_io_engine_python(tmp_path, default_surface):
    """Test IO using pure python read/write"""
    xsurf1 = xtgeo.RegularSurface(**default_surface)

    usefile1 = tmp_path / "surf3a.fgr"
    usefile2 = tmp_path / "surf3b.fgr"

    print(xsurf1[1, 1])

    xsurf1.to_file(usefile1, fformat="irap_ascii", engine="cxtgeo")
    xsurf1.to_file(usefile2, fformat="irap_ascii", engine="python")

    xsurf2 = xtgeo.surface_from_file(usefile2, fformat="irap_ascii", engine="python")
    xsurf3 = xtgeo.surface_from_file(usefile2, fformat="irap_ascii", engine="cxtgeo")

    assert xsurf1.ncol == xsurf3.ncol == xsurf3.ncol

    assert xsurf1.values[1, 1] == xsurf2.values[1, 1] == xsurf3.values[1, 1]


@pytest.mark.bigtest
@pytest.mark.benchmark(group="import/export surface")
@pytest.mark.parametrize("engine", ["cxtgeo", "python"])
def test_irapasc_import1_engine_compare(benchmark, engine, testdata_path):
    def read():
        xtgeo.surface_from_file(
            testdata_path / TESTSET3, fformat="irap_ascii", engine=engine
        )

    benchmark(read)


def test_minmax_rotated_map(testdata_path):
    """Min and max of rotated map"""
    logger.info("Import and export...")

    x = xtgeo.surface_from_file(testdata_path / TESTSET1, fformat="irap_binary")

    assert x.xmin == pytest.approx(454637.6, abs=0.1)
    assert x.xmax == pytest.approx(468895.1, abs=0.1)
    assert x.ymin == pytest.approx(5925995.0, abs=0.1)
    assert x.ymax == pytest.approx(5939998.7, abs=0.1)


def test_operator_overload():
    """Test operations between two surface in different ways"""

    surf1 = xtgeo.RegularSurface(
        ncol=100, nrow=50, yinc=25.0, xinc=25.0, rotation=10, values=100
    )
    assert surf1.values.mean() == 100.0
    id1 = id(surf1)
    id1v = id(surf1.values)

    surf2 = xtgeo.RegularSurface(
        ncol=100, nrow=50, yinc=25.0, xinc=25.0, rotation=0, values=100
    )
    diff = surf1 - surf2
    assert id(diff) != id1
    assert diff.values.mean() == 0.0

    assert id(surf1) == id1
    surf1 += diff
    assert id(surf1) == id1
    assert id(surf1.values) == id1v

    surf1 /= diff
    assert (surf1.values.count()) == 0


def test_twosurfaces_oper(testdata_path):
    """Test operations between two surface in more different ways"""

    surf1 = xtgeo.surface_from_file(testdata_path / TESTSET1)
    surf2 = xtgeo.surface_from_file(testdata_path / TESTSET1A)

    iso1 = surf2.copy()
    iso1.values -= surf1.values
    iso1mean = iso1.values.mean()
    assert iso1mean == pytest.approx(43.71, abs=0.01)

    iso2 = surf2.copy()
    iso2.subtract(surf1)
    iso2mean = iso2.values.mean()
    assert iso2mean == pytest.approx(43.71, abs=0.01)
    assert iso1.values.all() == iso2.values.all()

    iso3 = surf2 - surf1
    assert iso1.values.all() == iso3.values.all()
    assert isinstance(iso3, xtgeo.RegularSurface)

    sum1 = surf2.copy()
    sum1.values += surf1.values
    assert sum1.values.mean() == pytest.approx(3441.0, abs=0.01)

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


@pytest.mark.filterwarnings("ignore:Default values*")
def test_surface_comparisons(default_surface):
    """Test the surface comparison overload"""
    surf1 = xtgeo.RegularSurface(**default_surface)
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


@pytest.mark.filterwarnings("ignore:Default values*")
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


def test_irapbin_io(tmp_path, testdata_path):
    """Import and export Irap binary."""
    logger.info("Import and export...")

    x = xtgeo.surface_from_file(testdata_path / TESTSET1, fformat="irap_binary")

    x.to_file(tmp_path / "reek1_test.fgr", fformat="irap_ascii")

    logger.debug("NX is %s", x.ncol)

    assert x.ncol == 554

    # get the 1D numpy
    v1d = x.get_values1d()

    logger.info("Mean VALUES are: %s", np.nanmean(v1d))

    zval = x.values

    # add value via numpy
    zval = zval + 300
    # update
    x.values = zval

    assert x.values.mean() == pytest.approx(1998.648, abs=0.01)

    x.to_file(tmp_path / "reek1_plus_300_a.fgr", fformat="irap_ascii")
    x.to_file(tmp_path / "reek1_plus_300_b.gri", fformat="irap_binary")

    y = xtgeo.surface_from_file(testdata_path / TESTSET1)
    assert y.ncol == 554


@pytest.mark.filterwarnings("ignore:Default values*")
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


@pytest.mark.filterwarnings("ignore:Default values*")
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


@pytest.mark.filterwarnings("ignore:Default values*")
def test_get_xy_values(default_surface):
    """Get the XY coordinate values as 2D arrays"""

    xmap = xtgeo.RegularSurface(**default_surface)

    xcv, _ycv = xmap.get_xy_values(order="C")
    print(type(xcv))

    xxv = xcv.ravel()
    assert xxv[1] == pytest.approx(0.0, abs=0.001)

    xcv, _ycv = xmap.get_xy_values(order="F")
    xxv = xcv.ravel(order="F")
    assert xxv[1] == pytest.approx(25.0, abs=0.001)

    xcv, _ycv = xmap.get_xy_values(order="C", asmasked=True)

    xxv = xcv.ravel(order="C")
    assert xxv[1] == pytest.approx(0.0, abs=0.001)

    xcv, _ycv = xmap.get_xy_values(order="F", asmasked=True)

    xxv = xcv.ravel(order="F")
    assert xxv[1] == pytest.approx(25.0, abs=0.001)


@pytest.mark.filterwarnings("ignore:Default values*")
def test_get_xy_values1d(default_surface):
    """Get the XY coordinate values"""

    xmap = xtgeo.RegularSurface(**default_surface)

    xcv, _ycv = xmap.get_xy_values1d(activeonly=False, order="C")

    assert xcv[1] == pytest.approx(0.0, abs=0.001)

    xcv, _ycv = xmap.get_xy_values1d(activeonly=False, order="F")

    assert xcv[1] == pytest.approx(25.0, abs=0.001)

    xcv, _ycv = xmap.get_xy_values1d(activeonly=True, order="C")

    assert xcv[1] == pytest.approx(0.0, abs=0.001)

    xcv, _ycv = xmap.get_xy_values1d(activeonly=True, order="F")

    assert xcv[1] == pytest.approx(25.0, abs=0.001)


def test_dataframe_simple(testdata_path):
    """Get a pandas Dataframe object"""

    xmap = xtgeo.surface_from_file(testdata_path / TESTSET1)

    dfrc = xmap.get_dataframe(ijcolumns=True, order="C", activeonly=True)

    assert dfrc["X_UTME"][2] == pytest.approx(465956.274, abs=0.01)

    xmap = xtgeo.surface_from_file(testdata_path / TESTSET2)

    dfrc = xmap.get_dataframe()

    assert dfrc["X_UTME"][2] == pytest.approx(461582.562498, abs=0.01)

    xmap.coarsen(2)
    dfrc = xmap.get_dataframe()

    assert dfrc["X_UTME"][2] == pytest.approx(461577.5575, abs=0.01)


@pytest.mark.bigtest
def test_dataframe_more(tmp_path, testdata_path):
    """Get a pandas Dataframe object, more detailed testing"""

    xmap = xtgeo.surface_from_file(testdata_path / TESTSET1)

    xmap.describe()

    dfrc = xmap.get_dataframe(ijcolumns=True, order="C", activeonly=True)
    dfrf = xmap.get_dataframe(ijcolumns=True, order="F", activeonly=True)

    dfrc.to_csv(tmp_path / "regsurf_df_c.csv")
    dfrf.to_csv(tmp_path / "regsurf_df_f.csv")
    xmap.to_file(tmp_path / "regsurf_df.ijxyz", fformat="ijxyz")

    assert dfrc["X_UTME"][2] == pytest.approx(465956.274, abs=0.01)
    assert dfrf["X_UTME"][2] == pytest.approx(462679.773, abs=0.01)

    dfrcx = xmap.get_dataframe(ijcolumns=False, order="C", activeonly=True)
    dfrcx.to_csv(tmp_path / "regsurf_df_noij_c.csv")
    dfrcy = xmap.get_dataframe(
        ijcolumns=False, order="C", activeonly=False, fill_value=None
    )
    dfrcy.to_csv(tmp_path / "regsurf_df_noij_c_all.csv")


@pytest.mark.filterwarnings("ignore:Default values*")
def test_get_xy_value_lists_small(default_surface):
    """Get the xy list and value list from small test case"""

    x = xtgeo.RegularSurface(**default_surface)  # default instance

    xylist, valuelist = x.get_xy_value_lists(valuefmt="8.3f", xyfmt="12.2f")

    logger.info(xylist[2])
    logger.info(valuelist[2])

    assert valuelist[2] == 3.0


def test_get_xy_value_lists_reek(testdata_path):
    """Get the xy list and value list"""

    x = xtgeo.surface_from_file(testdata_path / TESTSET1, fformat="irap_binary")

    xylist, valuelist = x.get_xy_value_lists(valuefmt="8.3f", xyfmt="12.2f")

    logger.info(xylist[2])
    logger.info(valuelist[2])

    assert valuelist[2] == 1910.445


def test_topology(testdata_path):
    """Testing topology between two surfaces."""

    logger.info("Test if surfaces are similar...")

    x = xtgeo.surface_from_file(testdata_path / TESTSET1)
    y = xtgeo.surface_from_file(testdata_path / TESTSET1)

    status = x.compare_topology(y)
    assert status is True

    y.xori = y.xori - 100.0
    status = x.compare_topology(y)
    assert status is False


def test_similarity(testdata_path):
    """Check similarity of two surfaces.

    0.0 means identical in terms of mean value.
    """

    logger.info("Test if surfaces are similar...")

    x = xtgeo.surface_from_file(testdata_path / TESTSET1)
    y = xtgeo.surface_from_file(testdata_path / TESTSET1)

    si = x.similarity_index(y)
    assert si == 0.0

    y.values = y.values * 2

    si = x.similarity_index(y)
    assert si == 1.0


def test_irapbin_io_loop(tmp_path, testdata_path):
    """Do a loop over big Troll data set."""

    num = 10

    for _i in range(num):
        # print(i)
        x = xtgeo.surface_from_file(testdata_path / TESTSET1, fformat="irap_binary")

        m1 = x.values.mean()
        zval = x.values
        zval = zval + 300
        x.values = zval
        m2 = x.values.mean()
        x.to_file(tmp_path / "troll.gri", fformat="irap_binary")

        assert m1 == pytest.approx(m2 - 300)


def test_irapbin_export_py(tmp_path, testdata_path):
    """Export Irapbin with pure python"""

    x = xtgeo.surface_from_file(testdata_path / TESTSET1, fformat="irap_binary")

    t0 = xtg.timer()
    for _ in range(10):
        x.to_file(tmp_path / "purecx.gri", fformat="irap_binary", engine="cxtgeo")
    t1 = xtg.timer(t0)
    print(f"CXTGeo based write: {t1:3.4f}")

    t0 = xtg.timer()
    for _ in range(10):
        x.to_file(tmp_path / "purepy.gri", fformat="irap_binary", engine="python")
    t2 = xtg.timer(t0)
    print(f"Python based write: {t2:3.4f}")
    print(f"Ratio python based / cxtgeo based {t2 / t1:3.4f}")

    s1 = xtgeo.surface_from_file(tmp_path / "purecx.gri")
    s2 = xtgeo.surface_from_file(tmp_path / "purepy.gri")

    assert s1.values.mean() == s2.values.mean()

    assert s1.values[100, 100] == s2.values[100, 100]


def test_distance_from_point(tmp_path, testdata_path):
    """Distance from point."""

    x = xtgeo.surface_from_file(testdata_path / TESTSET1, fformat="irap_binary")

    x.distance_from_point(point=(464960, 7336900), azimuth=30)

    x.to_file(tmp_path / "reek1_dist_point.gri", fformat="irap_binary")


def test_value_from_xy(testdata_path):
    """
    get Z value from XY point
    """

    x = xtgeo.surface_from_file(testdata_path / TESTSET1, fformat="irap_binary")

    z = x.get_value_from_xy(point=(460181.036, 5933948.386))

    assert z == pytest.approx(1625.11, abs=0.01)

    # outside value shall return None
    z = x.get_value_from_xy(point=(0.0, 7337128.076))
    assert z is None


def test_fence(testdata_path):
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

    x = xtgeo.surface_from_file(testdata_path / TESTSET1)

    newfence = x.get_fence(myfence)

    logger.debug("updated NP:")
    logger.debug(newfence)
    print(newfence)

    assert newfence[1][2] == pytest.approx(1720.9094, abs=0.01)


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
@pytest.mark.filterwarnings("ignore:Default values*")
def test_fence_sampling(infence, sampling, expected, default_surface):
    """Test a very simple fence with different sampling methods."""
    surf = xtgeo.RegularSurface(**default_surface)

    myfence = np.array(infence)
    myfence[myfence == -999] = np.nan
    newfence = surf.get_fence(myfence, sampling=sampling)
    assert np.allclose(newfence[:, 2], expected, equal_nan=True)


def test_get_randomline_frompolygon(show_plot, testdata_path):
    """Test randomline with both bilinear and nearest sampling for surfaces."""
    fence = xtgeo.polygons_from_file(testdata_path / FENCE1)
    xs = xtgeo.surface_from_file(testdata_path / TESTSET1)

    # get the polygon
    fspec = fence.get_fence(distance=10, nextend=2, asnumpy=False)
    assert fspec.get_dataframe()[fspec.dhname][4] == pytest.approx(10, abs=1)

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


def test_unrotate(testdata_path):
    """Change a rotated map to an unrotated instance"""

    x = xtgeo.surface_from_file(testdata_path / TESTSET1, fformat="irap_binary")

    logger.info(x)
    x.unrotate()
    logger.info(x)


def test_fill(testdata_path):
    """Fill the undefined values for the surface"""

    srf = xtgeo.surface_from_file(testdata_path / TESTSET1, fformat="irap_binary")

    minv1 = srf.values.min()
    assert srf.values.mean() == pytest.approx(1698.648, abs=0.001)

    srf.fill()
    minv2 = srf.values.min()
    assert srf.values.mean() == pytest.approx(1705.201, abs=0.001)
    assert minv1 == pytest.approx(minv2, 0.000001)

    srf = xtgeo.surface_from_file(testdata_path / TESTSET1, fformat="irap_binary")
    srf.fill(444)
    assert srf.values.mean() == pytest.approx(1342.10498, abs=0.001)


def test_smoothing(testdata_path):
    """Smooth the surface using median/gaussian filter"""

    srf = xtgeo.surface_from_file(testdata_path / TESTSET1, fformat="irap_binary")

    mean1 = srf.values.mean()
    assert mean1 == pytest.approx(1698.65, abs=0.1)

    # test median smoothing methoid
    srf2 = srf.copy()
    srf2.smooth(method="median", iterations=1, width=5)
    mean2 = srf2.values.mean()
    assert mean2 == pytest.approx(1698.65, abs=0.3)  # smoothed ~same mean
    assert mean1 != mean2  # but not exacly same

    # test gaussian smoothing methoid
    srf3 = srf.copy()
    srf3.smooth(method="gaussian", iterations=1, width=5)
    mean3 = srf3.values.mean()
    assert mean3 == pytest.approx(1698.65, abs=0.3)  # smoothed ~same mean
    assert mean1 != mean3  # but not exacly same

    # check that the three surfaces have different min values
    min1 = srf.values.min()
    min2 = srf2.values.min()
    min3 = srf3.values.min()
    assert min1 == pytest.approx(1547.20, abs=0.1)
    assert min2 == pytest.approx(1553.62, abs=0.1)
    assert min3 == pytest.approx(1566.91, abs=0.1)


def test_loadvalues_after_remove(default_surface, testdata_path):
    default_surface.pop("values", None)
    with pytest.raises(InvalidFileFormatError, match="File format None is invalid"):
        srf = xtgeo.RegularSurface(filesrc=testdata_path / TESTSET1, **default_surface)
        srf.load_values()

    with pytest.raises(
        InvalidFileFormatError, match="File format irap_binary is invalid"
    ):
        srf = xtgeo.RegularSurface(
            filesrc=testdata_path / TESTSET1, fformat="irap_binary", **default_surface
        )
        srf.load_values()

    default_surface.pop("filesrc", None)
    srf = xtgeo.RegularSurface(**default_surface)
    with pytest.raises(
        ValueError, match="Can only load values into object initialised from file"
    ):
        srf.load_values()


def test_genhash_deterministic(testdata_path):
    """Check that generate_hash() is deterministic"""
    xsurf = xtgeo.surface_from_file(testdata_path / TESTSET7A)
    ysurf = xtgeo.surface_from_file(testdata_path / TESTSET7A)

    xhash = xsurf.generate_hash()
    yhash = ysurf.generate_hash()

    assert xhash == yhash


def test_genhash_same_mask(testdata_path):
    """
    Check if generate_hash() returns the same hash when properties are the
    same, actual values differ, but the masked values are equivalent.
    """
    xsurf = xtgeo.surface_from_file(testdata_path / TESTSET7A)
    ysurf = xtgeo.surface_from_file(testdata_path / TESTSET7B)
    zsurf = xtgeo.surface_from_file(testdata_path / TESTSET6A)

    xhash = xsurf.generate_hash()
    yhash = ysurf.generate_hash()
    zhash = zsurf.generate_hash()

    assert xhash != yhash
    assert xhash != zhash
    assert yhash != zhash


def test_get_boundary_polygons_simple(show_plot, tmp_path):
    """Test getting a boundary for a surface (very simple)."""
    values = np.ma.array([1, 2, 3, 4, 5, 6, 7, 8, 9], mask=[0, 1, 0, 1, 0, 0, 1, 0, 0])
    xs = xtgeo.RegularSurface(ncol=3, nrow=3, xinc=10, yinc=10, values=values)

    boundary = xs.get_boundary_polygons(alpha_factor=1.0, convex=False, simplify=False)

    if show_plot:
        # for debugging
        xs.quickplot(
            filename=tmp_path / "test.png",
            faults={
                "faults": boundary,
                "color": "red",
                "alpha": 0.1,
                "edgecolor": "black",
                "linewidth": 2,
            },
        )
    assert boundary.get_dataframe()[boundary.yname].values.tolist() == pytest.approx(
        [
            10,
            10,
            20,
            20,
            20,
            10,
        ]
    )


def test_get_boundary_polygons_complex(show_plot, testdata_path, tmp_path):
    """Test getting a boundary for a surface."""
    xs = xtgeo.surface_from_file(testdata_path / TESTSET1)
    xs.values = np.ma.masked_less(xs.values, 1700)
    xs.values = np.ma.masked_greater(xs.values, 1800)

    boundary = xs.get_boundary_polygons(alpha_factor=1, convex=False, simplify=False)

    if show_plot:
        # for debugging
        xs.quickplot(
            filename=tmp_path / "test.png",
            faults={
                "faults": boundary,
                "color": "red",
                "alpha": 0.1,
                "edgecolor": "black",
                "linewidth": 2,
            },
        )

    # reveal any major issues by asserting averages (polygons are checked visually)
    # for some reasons, macos tests gives slightly different result; that is why a large
    # tolerance is given
    assert boundary.get_dataframe()[boundary.xname].mean() == pytest.approx(
        462230.0, abs=3.0
    )
    assert boundary.get_dataframe()[boundary.yname].mean() == pytest.approx(
        5933457.0, abs=4.0
    )

    # get the first (largest) polygon
    boundary.filter_byid([0])

    if show_plot:
        # for debugging
        xs.quickplot(
            filename=tmp_path / "test2.png",
            faults={
                "faults": boundary,
                "color": "red",
                "alpha": 0.1,
                "edgecolor": "black",
                "linewidth": 2,
            },
        )


def test_boundary_polygons_are_sorted(testdata_path):
    """Test that boundary polygons are sorted from largest to smallest."""
    xs = xtgeo.surface_from_file(testdata_path / TESTSET1)
    xs.values = np.ma.masked_less(xs.values, 1700)
    xs.values = np.ma.masked_greater(xs.values, 1800)

    boundary = xs.get_boundary_polygons(simplify=False)

    df = boundary.get_dataframe(copy=False)

    # check that we have 7 unique boundaries for this surface
    assert df["POLY_ID"].nunique() == 7

    # check that the boundary are sorted from largest to smallest polygon
    pol_lengths = [len(poldf) for _, poldf in df.groupby("POLY_ID")]
    assert all(
        pol_lengths[i] >= pol_lengths[i + 1] for i in range(len(pol_lengths) - 1)
    )


def test_regsurface_get_dataframe(default_surface):
    """Test getting a dataframe from a surface."""

    surf = RegularSurface(**default_surface)
    dataframe = surf.get_dataframe()
    assert dataframe.VALUES.to_list() == [
        1.0,
        6.0,
        11.0,
        2.0,
        7.0,
        12.0,
        3.0,
        8.0,
        4.0,
        9.0,
        14.0,
        5.0,
        10.0,
        15.0,
    ]

    with pytest.warns(PendingDeprecationWarning):
        dataframe2 = surf.dataframe()

    pd.testing.assert_frame_equal(dataframe, dataframe2)
