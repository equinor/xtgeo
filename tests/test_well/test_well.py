# -*- coding: utf-8 -*-


import glob
from os.path import join
from collections import OrderedDict

import pytest
import numpy as np
import pandas as pd

from xtgeo.surface import RegularSurface
from xtgeo.well import Well
from xtgeo.xyz import Polygons
from xtgeo.common import XTGeoDialog

import tests.test_common.test_xtg as tsetup

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TMPD = xtg.tmpdir
TMPDX = xtg.tmpdirobj
TPATH = xtg.testpathobj

# =========================================================================
# Do tests
# =========================================================================

WFILE = join(TPATH, "wells/reek/1/OP_1.w")
WFILE_HOLES = join(TPATH, "wells/reek/1/OP_1_zholes.w")
WFILES = str(TPATH / "wells/reek/1/*")


WELL1 = join(TPATH, "wells/battle/1/WELL09.rmswell")
WELL2 = join(TPATH, "wells/battle/1/WELL36.rmswell")
WELL3 = join(TPATH, "wells/battle/1/WELL10.rmswell")

WELL4 = join(TPATH, "wells/drogon/1/55_33-1.rmswell")

SURF1 = TPATH / "surfaces/reek/1/topreek_rota.gri"
SURF2 = TPATH / "surfaces/reek/1/basereek_rota.gri"


@pytest.fixture(name="loadwell1")
def fixture_loadwell1():
    """Fixture for loading a well (pytest setup)."""
    return Well(WFILE)


@pytest.fixture(name="loadwell3")
def fixture_loadwell3():
    """Fixture for loading a well (pytest setup)."""
    return Well(WELL3)


def test_import(loadwell1):
    """Import well from file."""

    mywell = loadwell1

    logger.debug("True well name: %s", mywell.truewellname)
    tsetup.assert_equal(mywell.xpos, 461809.59, "XPOS")
    tsetup.assert_equal(mywell.ypos, 5932990.36, "YPOS")
    tsetup.assert_equal(mywell.wellname, "OP_1", "WNAME")

    logger.info(mywell.get_logtype("Zonelog"))
    logger.info(mywell.get_logrecord("Zonelog"))
    logger.info(mywell.lognames_all)
    logger.info(mywell.dataframe)

    # logger.info the numpy string of Poro...
    logger.info(type(mywell.dataframe["Poro"].values))


def test_import_long_well(loadwell3):
    """Import a longer well from file."""

    mywell = loadwell3

    logger.debug("True well name: %s", mywell.truewellname)

    mywell.geometrics()
    dfr = mywell.dataframe

    tsetup.assert_almostequal(dfr["Q_AZI"][27], 91.856158, 0.0001)


def test_import_well_selected_logs():
    """Import a well but restrict on lognames"""

    mywell = Well()
    mywell.from_file(WELL1, lognames="all")
    assert "ZONELOG" in mywell.dataframe

    mywell.from_file(WELL1, lognames="GR")
    assert "ZONELOG" not in mywell.dataframe

    mywell.from_file(WELL1, lognames=["GR"])
    assert "ZONELOG" not in mywell.dataframe

    mywell.from_file(WELL1, lognames=["DUMMY"])
    assert "ZONELOG" not in mywell.dataframe
    assert "GR" not in mywell.dataframe

    with pytest.raises(ValueError) as msg:
        logger.info(msg)
        mywell.from_file(WELL1, lognames=["DUMMY"], lognames_strict=True)

    mywell.from_file(WELL1, mdlogname="GR")
    assert mywell.mdlogname == "GR"

    with pytest.raises(ValueError) as msg:
        mywell.from_file(WELL1, mdlogname="DUMMY", strict=True)
        logger.info(msg)

    mywell.from_file(WELL1, mdlogname="DUMMY", strict=False)
    assert mywell.mdlogname is None


def test_change_a_lot_of_stuff(loadwell1):
    """Import well from file and try to change lognames etc."""

    mywell = loadwell1

    assert mywell.get_logtype("Poro") == "CONT"
    assert mywell.get_logrecord("Poro") is None

    with pytest.raises(ValueError) as vinfo:
        mywell.set_logrecord("Poro", {0: "null"})
    assert "Cannot set a log record for a continuous log" in str(vinfo.value)

    assert mywell.name == "OP_1"
    mywell.name = "OP_1_EDITED"
    assert mywell.name == "OP_1_EDITED"

    mywell.rename_log("Poro", "PORO")
    assert "PORO" in mywell.lognames

    with pytest.raises(ValueError) as vinfo:
        mywell.rename_log("PoroXX", "Perm")
    assert "Input log does not exist" in str(vinfo.value)

    with pytest.raises(ValueError) as vinfo:
        mywell.rename_log("PORO", "Perm")
    assert "New log name exists already" in str(vinfo.value)

    frec1 = mywell.get_logrecord("Facies")
    mywell.rename_log("Facies", "FACIES")
    frec2 = mywell.get_logrecord("FACIES")

    assert sorted(frec1) == sorted(frec2)


def test_import_export_many():
    """Import and export many wells (test speed)."""
    logger.debug(WFILES)

    for filename in sorted(glob.glob(WFILES)):
        mywell = Well(filename)
        logger.info(mywell.nrow)
        logger.info(mywell.ncol)
        logger.info(mywell.lognames)

        wname = join(TMPD, mywell.xwellname + ".w")
        mywell.to_file(wname)


def test_shortwellname():
    """Test that shortwellname gives wanted result."""
    mywell = Well()

    mywell._wname = "31/2-A-14 2H"
    short = mywell.shortwellname

    assert short == "A-142H"

    mywell._wname = "6412_2-A-14_2H"
    short = mywell.shortwellname

    assert short == "A-142H"


def test_hdf_io_single():
    """Test HDF io, single well."""
    mywell = Well(WELL1)

    wname = (TMPDX / "hdfwell").with_suffix(".hdf")
    mywell.to_hdf(wname)
    mywell2 = Well()
    mywell2.from_hdf(wname)


def test_import_as_rms_export_as_hdf_many():
    """Import RMS and export as HDF5 and RMS asc, many, and compare timings."""
    mywell = Well(WELL1)
    nmax = 50

    t0 = xtg.timer()
    wlist = []
    for _ in range(nmax):
        wname = (TMPDX / "$random").with_suffix(".hdf")
        wuse = mywell.to_hdf(wname, compression=None)
        wlist.append(wuse)
    print("Time for save HDF: ", xtg.timer(t0))

    t0 = xtg.timer()
    for wll in wlist:
        wname = (TMPDX / "$random").with_suffix(".hdf")
        wuse = mywell.from_hdf(wll)
    print("Time for load HDF: ", xtg.timer(t0))

    t0 = xtg.timer()
    wlist = []
    for _ in range(nmax):
        wname = (TMPDX / "$random").with_suffix(".rmsasc")
        wuse = mywell.to_file(wname)
        wlist.append(wuse)
    print("Time for save RMSASC: ", xtg.timer(t0))

    t0 = xtg.timer()
    for wll in wlist:
        wuse = mywell.from_file(wll)
    print("Time for load RMSASC: ", xtg.timer(t0))


def test_get_carr(loadwell1):
    """Get a C array pointer"""

    mywell = loadwell1

    dummy = mywell.get_carray("NOSUCH")

    tsetup.assert_equal(dummy, None, "Wrong log name")

    cref = mywell.get_carray("X_UTME")

    xref = str(cref)
    swig = False
    if "Swig" in xref and "double" in xref:
        swig = True

    tsetup.assert_equal(swig, True, "carray from log name, double")

    cref = mywell.get_carray("Zonelog")

    xref = str(cref)
    swig = False
    if "Swig" in xref and "int" in xref:
        swig = True

    tsetup.assert_equal(swig, True, "carray from log name, int")


def test_create_and_delete_logs(loadwell3):
    """Test create adn delete logs."""
    mywell = loadwell3

    status = mywell.create_log("NEWLOG")
    assert status is True

    status = mywell.create_log("NEWLOG", force=False)
    assert status is False

    status = mywell.create_log("NEWLOG", force=True, value=200)
    assert status is True
    assert mywell.dataframe.NEWLOG.mean() == 200.0

    ndeleted = mywell.delete_log("NEWLOG")

    assert ndeleted == 1
    status = mywell.create_log("NEWLOG", force=True, value=200)

    ndeleted = mywell.delete_log(["NEWLOG", "GR"])
    assert ndeleted == 2


def test_get_set_wlogs(loadwell3):
    """Test on getting ans setting a dictionary with some log attributes."""
    mywell = loadwell3

    mydict = mywell.get_wlogs()
    print(mydict)

    assert isinstance(mydict, OrderedDict)

    assert mydict["X_UTME"][0] == "CONT"
    assert mydict["ZONELOG"][0] == "DISC"
    assert mydict["ZONELOG"][1][24] == "ZONE_24"

    mydict["ZONELOG"][1][24] = "ZONE_24_EDITED"
    mywell.set_wlogs(mydict)

    mydict2 = mywell.get_wlogs()
    assert mydict2["X_UTME"][0] == "CONT"
    assert mydict2["ZONELOG"][0] == "DISC"
    assert mydict2["ZONELOG"][1][24] == "ZONE_24_EDITED"

    mydict2["EXTRA"] = None
    with pytest.raises(ValueError):
        mywell.set_wlogs(mydict2)


def test_make_hlen(loadwell1):
    """Create a hlen log."""

    mywell = loadwell1
    mywell.create_relative_hlen()

    logger.debug(mywell.dataframe)


def test_make_zqual_log(loadwell3):
    """Make a zonelog FLAG quality log."""
    mywell = loadwell3
    mywell.zonelogname = "ZONELOG"

    logger.debug("True well name: %s", mywell.truewellname)

    mywell.make_zone_qual_log("manamana")

    with pd.option_context("display.max_rows", 1000):
        print(mywell.dataframe)


@pytest.mark.parametrize(
    "logseries, nsamples, expected",
    [
        ([], 0, []),
        ([1], 0, [False]),
        ([1], 1, [False]),  # intentional
        ([1, 2], 0, [False, False]),
        ([0, 0], 1, [False, False]),
        ([1, 2], 1, [True, True]),
        ([1, 1, 1, 2, 2, 2], 1, [False, False, True, True, False, False]),
        ([1, 1, 1, 2, 2, 2], 2, [False, True, True, True, True, False]),
        ([10, 10, 10, 2, 2, 2], 1, [False, False, True, True, False, False]),
        ([1, 1, 1, 2, 2, 2], 10, [True, True, True, True, True, True]),
        ([np.nan, 1, 1, np.nan, 2, 2], 1, [False, False, False, False, False, False]),
        ([np.nan, 1, 1, 2, np.nan, 2], 1, [False, False, True, True, False, False]),
    ],
)
def test_mask_shoulderbeds_get_bseries(logseries, nsamples, expected):
    """Test for corner cases in _mask_shoulderbeds_logs."""
    from xtgeo.well._well_oper import _get_bseries

    logseries = pd.Series(logseries, dtype="float64")
    expected = pd.Series(expected, dtype="bool")

    results = _get_bseries(logseries, nsamples)
    if logseries.empty:
        assert expected.empty is True
    else:
        assert results.equals(expected)


@pytest.mark.parametrize(
    "dep, lseries, distance, expected",
    [
        (
            [1.0],
            [1],
            1.2,
            [False],
        ),
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1, 1, 1, 2, 2, 2],
            0.7,
            [False, False, True, True, False, False],
        ),
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1, 1, 1, 2, 3, 2],
            1.2,
            [False, False, True, True, True, True],
        ),
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1, 1, 1, 2, 3, 2],
            0.49999,
            [False, False, False, False, False, False],
        ),
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1, 1, 1, 2, 3, 2],
            0.50,
            [False, False, True, True, True, True],
        ),
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1, 1, 1, 2, np.nan, np.nan],
            1.2,
            [False, False, True, True, False, False],
        ),
    ],
)
def test_well_mask_shoulder_get_bseries_by_distance(dep, lseries, distance, expected):
    """Test for corner cases in get bseries by distance."""
    # note the cxtgeo function is also tested in test_cxtgeo_lowlevel!
    from xtgeo.well._well_oper import _get_bseries_by_distance

    dep = pd.Series(dep, dtype="float64")
    lseries = pd.Series(lseries, dtype="float64")
    expected = np.array(expected, dtype="bool")

    result = _get_bseries_by_distance(dep, lseries, distance)
    assert (result == expected).all()


def test_mask_shoulderbeds(loadwell3, loadwell1):
    """Test masking shoulderbeds effects."""
    mywell = loadwell3

    usewell = mywell.copy()
    usewell.mask_shoulderbeds(["ZONELOG"], ["GR"], nsamples=3)

    assert not np.isnan(mywell.dataframe.at[1595, "GR"])
    assert np.isnan(usewell.dataframe.at[1595, "GR"])

    # another well set with more discrete logs and several logs to modify
    mywell = loadwell1
    usewell = mywell.copy()
    usewell.mask_shoulderbeds(["Zonelog", "Facies"], ["Perm", "Poro"], nsamples=2)
    assert np.isnan(usewell.dataframe.at[4763, "Perm"])
    assert np.isnan(usewell.dataframe.at[4763, "Poro"])

    # corner cases
    with pytest.raises(ValueError):
        usewell.mask_shoulderbeds(
            ["Zonelog", "Facies"], ["NOPerm", "Poro"], nsamples=2, strict=True
        )

    with pytest.raises(ValueError):
        usewell.mask_shoulderbeds(["Perm", "Facies"], ["NOPerm", "Poro"], nsamples=2)

    assert usewell.mask_shoulderbeds(["Zonelog"], ["Perm", "Poro"]) is True
    assert usewell.mask_shoulderbeds(["Zonelog"], ["Dummy"]) is False
    assert usewell.mask_shoulderbeds(["Dummy"], ["Perm", "Poro"]) is False


def test_mask_shoulderbeds_use_tvd_md(loadwell3):
    """Test masking shoulderbeds effects using tvd or md distance."""
    mywell = loadwell3

    usewell = mywell.copy()
    # small distance for this test since almost horizontal well
    usewell.mask_shoulderbeds(["ZONELOG"], ["GR"], nsamples={"tvd": 0.01})

    assert not np.isnan(mywell.dataframe.at[1595, "GR"])
    assert np.isnan(usewell.dataframe.at[1595, "GR"])

    usewell = mywell.copy()

    with pytest.raises(ValueError) as verr:
        # since mdlogname does not exist
        usewell.mask_shoulderbeds(["ZONELOG"], ["GR"], nsamples={"md": 1.6})
    assert "no mdlogname attribute present" in str(verr)

    usewell.geometrics()  # to create Q_MDEPTH as mdlogname

    usewell.mask_shoulderbeds(["ZONELOG"], ["GR"], nsamples={"md": 1.6})
    assert not np.isnan(mywell.dataframe.at[1595, "GR"])
    assert np.isnan(usewell.dataframe.at[1595, "GR"])


def test_rescale_well(loadwell1):
    """Rescale (resample) a well to a finer increment"""

    mywell = loadwell1

    df1 = mywell.dataframe.copy()
    df1 = df1[(df1["Zonelog"] == 1)]

    mywell.rescale(delta=0.2)

    df2 = mywell.dataframe.copy()
    df2 = df2[(df2["Zonelog"] == 1)]

    tsetup.assert_almostequal(df1["Perm"].mean(), df2["Perm"].mean(), 20.0)
    tsetup.assert_almostequal(df1["Poro"].mean(), df2["Poro"].mean(), 0.001)


def test_rescale_well_tvdrange():
    """Rescale (resample) a well to a finer increment within a TVD range"""

    mywell = Well(WELL1)
    mywell.to_file(join(TMPD, "wll1_pre_rescale.w"))
    gr_avg1 = mywell.dataframe["GR"].mean()
    mywell.rescale(delta=2, tvdrange=(1286, 1333))
    mywell.to_file(join(TMPD, "wll1_post_rescale.w"))
    gr_avg2 = mywell.dataframe["GR"].mean()
    tsetup.assert_almostequal(gr_avg1, gr_avg2, 0.9)

    mywell1 = Well(WELL1)
    mywell1.rescale(delta=2)

    mywell2 = Well(WELL1)
    mywell2.rescale(delta=2, tvdrange=(0, 9999))
    assert mywell1.dataframe["GR"].mean() == mywell2.dataframe["GR"].mean()
    assert mywell1.dataframe["GR"].all() == mywell2.dataframe["GR"].all()


def test_rescale_well_tvdrange_coarsen_upper():
    """Rescale (resample) a well to a coarser increment in top part"""

    mywell = Well(WELL1)
    mywell.rescale(delta=20, tvdrange=(0, 1200))
    mywell.to_file(join(TMPD, "wll1_rescale_coarsen.w"))
    tsetup.assert_almostequal(mywell.dataframe.iat[10, 3], 365.8254, 0.1)


def test_fence():
    """Return a resampled fence."""

    mywell = Well(WFILE)
    pline = mywell.get_fence_polyline(nextend=10, tvdmin=1000)

    logger.debug(pline)


def test_fence_as_polygons():
    """Return a resampled fence as Polygons."""

    mywell = Well(WFILE)
    pline = mywell.get_fence_polyline(nextend=3, tvdmin=1000, asnumpy=False)

    assert isinstance(pline, Polygons)
    dfr = pline.dataframe
    print(dfr)
    tsetup.assert_almostequal(dfr["X_UTME"][5], 462569.00, 2.0)


def test_fence_as_polygons_drogon():
    """Return a resampled fence as Polygons for a 100% vertical."""

    mywell = Well(WELL4)
    pline = mywell.get_fence_polyline(
        nextend=3, tvdmin=1000, sampling=20, asnumpy=False
    )

    assert isinstance(pline, Polygons)
    dfr = pline.dataframe
    assert dfr.H_CUMLEN.max() == pytest.approx(60, 0.01)


def test_get_zonation_points():
    """Get zonations points (zone tops)"""

    mywell = Well(WFILE, zonelogname="Zonelog")
    mywell.get_zonation_points()


def test_get_zone_interval():
    """Get zonations points (zone tops)"""

    mywell = Well(WFILE, zonelogname="Zonelog")
    line = mywell.get_zone_interval(3)

    print(line)

    logger.info(type(line))

    tsetup.assert_almostequal(line.iat[0, 0], 462698.33299, 0.001)
    tsetup.assert_almostequal(line.iat[-1, 2], 1643.1618, 0.001)


def test_remove_parallel_parts():
    """Remove the part of the well thst is parallel with some other"""

    well1 = Well(WELL1)
    well2 = Well(WELL2)

    well1.truncate_parallel_path(well2)

    print(well1.dataframe)


def test_get_zonation_holes():
    """get a report of holes in the zonation, some samples with -999"""

    mywell = Well(WFILE_HOLES, zonelogname="Zonelog")
    report = mywell.report_zonation_holes()

    logger.info("\n%s", report)

    tsetup.assert_equal(report.iat[0, 0], 4193)  # first value for INDEX
    tsetup.assert_equal(report.iat[1, 3], 1609.5800)  # second value for Z


def test_get_filled_dataframe():
    """Get a filled DataFrame"""

    mywell = Well(WFILE)

    df1 = mywell.dataframe

    df2 = mywell.get_filled_dataframe(fill_value=-999, fill_value_int=-888)

    logger.info(df1)
    logger.info(df2)

    assert np.isnan(df1.iat[4860, 6])
    assert df2.iat[4860, 6] == -888


def test_create_surf_distance_log(loadwell1):
    """Test making a log which is distance to a surface."""

    well = loadwell1

    surf1 = RegularSurface(SURF1)
    surf2 = RegularSurface(SURF2)

    well.create_surf_distance_log(surf1, name="DIST_TOP")
    well.create_surf_distance_log(surf2, name="DIST_BASE")

    assert well.dataframe.loc[0, "DIST_TOP"] == pytest.approx(1653.303263)
    assert well.dataframe.loc[0, "DIST_BASE"] == pytest.approx(1696.573171)

    # moving the surface so it is outside the well
    surf1.translate_coordinates((10000, 10000, 0))
    well.create_surf_distance_log(surf1, name="DIST_BASE_NEW")
    assert np.isnan(well.dataframe.loc[0, "DIST_BASE_NEW"])


@pytest.mark.bigtest
def test_create_surf_distance_log_more(loadwell1):
    """Test making a log which is distance to a surface and do some operations.

    This is a prototype (need BIG_TEST env variable to run) exploring the possibility
    to run a check if zonelog is some % of being within surfaces. I.e. a surface
    version of:

    Grid().report_zone_mismatch() method

    When/if the prototype is incorporated this test should be removed.
    """

    well = loadwell1

    surf1 = RegularSurface(SURF1)
    surf2 = RegularSurface(SURF2)

    well.create_surf_distance_log(surf1, name="DIST_TOP")
    well.create_surf_distance_log(surf2, name="DIST_BASE")

    # for simplicity create one zone log
    lrec = {0: "ABOVE", 1: "IN", 2: "BELOW"}
    well.create_log("MEGAZONE1", logtype="DISC", logrecord=lrec)
    well.create_log("MEGAZONE2", logtype="DISC", logrecord=lrec)

    zl = well.dataframe["Zonelog"]
    well.dataframe["MEGAZONE1"][(zl > 0) & (zl < 4)] = 1
    well.dataframe["MEGAZONE1"][zl > 3] = 2
    well.dataframe["MEGAZONE1"][np.isnan(zl)] = np.nan

    # derive from distance log:
    d1 = well.dataframe["DIST_TOP"]
    d2 = well.dataframe["DIST_BASE"]
    well.dataframe["MEGAZONE2"][(d1 <= 0.0) & (d2 > 0)] = 1

    # now use logics from Grid() report_zone_mismatch()...
    # much coding pasting vvvvvv =======================================================

    zname = "MEGAZONE1"
    zmodel = "MEGAZONE2"
    zonelogname = "MEGAZONE1"
    depthrange = [1200, 3000]
    zonelogrange = [1, 1]

    well.to_file(TMPDX / "well_surf_dist.w")

    # get the IJK along the well as logs; use a copy of the well instance
    wll = well.copy()

    if depthrange:
        d1, d2 = depthrange
        wll._df = wll._df[(d1 < wll._df.Z_TVDSS) & (wll._df.Z_TVDSS < d2)]

    # from here, work with the dataframe only
    df = wll._df

    # zonelogrange
    z1, z2 = zonelogrange
    zmin = int(df[zonelogname].min())
    zmax = int(df[zonelogname].max())
    skiprange = list(range(zmin, z1)) + list(range(z2 + 1, zmax + 1))

    for zname in (zonelogname, zmodel):
        if skiprange:  # needed check; du to a bug in pandas version 0.21 .. 0.23
            df[zname].replace(skiprange, -888, inplace=True)
        df[zname].fillna(-999, inplace=True)
    # now there are various variotions on how to count mismatch:
    # dfuse 1: count matches when zonelogname is valid (exclude -888)
    # dfuse 2: count matches when zonelogname OR zmodel are valid (exclude < -888
    # or -999)
    # The first one is the original approach

    dfuse1 = df.copy(deep=True)
    dfuse1 = dfuse1.loc[dfuse1[zonelogname] > -888]

    dfuse1["zmatch1"] = np.where(dfuse1[zmodel] == dfuse1[zonelogname], 1, 0)
    mcount1 = dfuse1["zmatch1"].sum()
    tcount1 = dfuse1["zmatch1"].count()
    if not np.isnan(mcount1):
        mcount1 = int(mcount1)
    if not np.isnan(tcount1):
        tcount1 = int(tcount1)

    res1 = dfuse1["zmatch1"].mean() * 100

    dfuse2 = df.copy(deep=True)
    dfuse2 = dfuse2.loc[(df[zmodel] > -888) | (df[zonelogname] > -888)]
    dfuse2["zmatch2"] = np.where(dfuse2[zmodel] == dfuse2[zonelogname], 1, 0)
    mcount2 = dfuse2["zmatch2"].sum()
    tcount2 = dfuse2["zmatch2"].count()
    if not np.isnan(mcount2):
        mcount2 = int(mcount2)
    if not np.isnan(tcount2):
        tcount2 = int(tcount2)

    res2 = dfuse2["zmatch2"].mean() * 100

    # update Well() copy (segment only)
    wll.dataframe = dfuse2

    res = {
        "MATCH1": res1,
        "MCOUNT1": mcount1,
        "TCOUNT1": tcount1,
        "MATCH2": res2,
        "MCOUNT2": mcount2,
        "TCOUNT2": tcount2,
        "WELLINTV": wll,
    }
    print(res)

    assert res["MATCH2"] == pytest.approx(93.67, abs=0.03)
