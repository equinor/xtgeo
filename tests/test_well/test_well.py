# -*- coding: utf-8 -*-
from __future__ import division, absolute_import
from __future__ import print_function

import glob
from os.path import join

import pytest
import numpy as np
import pandas as pd

from xtgeo.well import Well
from xtgeo.xyz import Polygons
from xtgeo.common import XTGeoDialog

import test_common.test_xtg as tsetup

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TMPD = xtg.tmpdir
TESTPATH = xtg.testpath

# =========================================================================
# Do tests
# =========================================================================

WFILE = join(TESTPATH, "wells/reek/1/OP_1.w")
WFILE_HOLES = join(TESTPATH, "wells/reek/1/OP_1_zholes.w")
WFILES = join(TESTPATH, "wells/reek/1/*")


WELL1 = join(TESTPATH, "wells/battle/1/WELL09.rmswell")
WELL2 = join(TESTPATH, "wells/battle/1/WELL36.rmswell")
WELL3 = join(TESTPATH, "wells/battle/1/WELL10.rmswell")

WELL4 = join(TESTPATH, "wells/drogon/1/55_33-1.rmswell")


@pytest.fixture(name="loadwell1")
def fixture_loadwell1():
    """Fixture for loading a well (pytest setup)"""
    return Well(WFILE)


@pytest.fixture(name="loadwell3")
def fixture_loadwell3():
    """Fixture for loading a well (pytest setup)"""
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
    """ Import and export many wells (test speed)"""

    logger.debug(WFILES)

    for filename in sorted(glob.glob(WFILES)):
        mywell = Well(filename)
        logger.info(mywell.nrow)
        logger.info(mywell.ncol)
        logger.info(mywell.lognames)

        wname = join(TMPD, mywell.xwellname + ".w")
        mywell.to_file(wname)


def test_shortwellname():
    """Test that shortwellname gives wanted result"""

    mywell = Well()

    mywell._wname = "31/2-A-14 2H"
    short = mywell.shortwellname

    assert short == "A-142H"

    mywell._wname = "6412_2-A-14_2H"
    short = mywell.shortwellname

    assert short == "A-142H"


# def test_import_as_rms_export_as_hdf5_many():
#     """ Import RMS and export as HDF5, many"""

#     logger.debug(WFILES)

#     wfile = TMPD + "/mytest.h5"
#     for filename in glob.glob(WFILES):
#         logger.info("Importing " + filename)
#         mywell = Well(filename)
#         logger.info(mywell.nrow)
#         logger.info(mywell.ncol)
#         logger.info(mywell.lognames)

#         wname = TMPD + "/" + mywell.xwellname + ".h5"
#         logger.info("Exporting " + wname)
#         mywell.to_file(wfile, fformat='hdf5')


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


def test_make_hlen(loadwell1):
    """Create a hlen log."""

    mywell = loadwell1
    mywell.create_relative_hlen()

    logger.debug(mywell.dataframe)


def test_make_zqual_log(loadwell3):
    """Make a zonelog FLAG quality log"""

    mywell = loadwell3
    mywell.zonelogname = "ZONELOG"

    logger.debug("True well name: %s", mywell.truewellname)

    mywell.make_zone_qual_log("manamana")

    with pd.option_context("display.max_rows", 1000):
        print(mywell.dataframe)


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
    pline = mywell.get_fence_polyline(nextend=3, tvdmin=1000, asnumpy=False)

    # assert isinstance(pline, Polygons)
    # dfr = pline.dataframe
    # print(dfr)
    # tsetup.assert_almostequal(dfr["X_UTME"][5], 462569.00, 2.0)


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
    """get a report of holes in the zonation, some samples with -999 """

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
