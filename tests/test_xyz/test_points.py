import os
from os.path import join
import pandas as pd

import pytest

from xtgeo.xyz import Points

from xtgeo.common import XTGeoDialog
import tests.test_common.test_xtg as tsetup

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TMPD = xtg.tmpdir
TSTPATH = xtg.testpathobj

XTGSHOW = False
if "XTG_SHOW" in os.environ:
    XTGSHOW = True

# =========================================================================
# Do tests
# =========================================================================

PFILE = join(TSTPATH, "points/eme/1/emerald_10_random.poi")
POINTSET2 = join(TSTPATH, "points/reek/1/pointset2.poi")
POINTSET3 = join(TSTPATH, "points/battle/1/many.rmsattr")
POINTSET4 = join(TSTPATH, "points/reek/1/poi_attr.rmsattr")
CSV1 = join(TSTPATH, "3dgrids/etc/gridqc1_rms_cellcenter.csv")


def test_custom_points():
    """Make points from list of tuples."""

    plist = [(234, 556, 11), (235, 559, 14), (255, 577, 12)]

    mypoints = Points(plist)

    x0 = mypoints.dataframe["X_UTME"].values[0]
    z2 = mypoints.dataframe["Z_TVDSS"].values[2]
    assert x0 == 234
    assert z2 == 12


def test_import():
    """Import XYZ points from file."""

    mypoints = Points(PFILE)  # should guess based on extesion

    x0 = mypoints.dataframe["X_UTME"].values[0]
    tsetup.assert_almostequal(x0, 460842.434326, 0.001)


def test_import_from_dataframe():
    """Import Points via Pandas dataframe."""

    mypoints = Points()
    dfr = pd.read_csv(CSV1, skiprows=3)
    attr = {"IX": "I", "JY": "J", "KZ": "K"}
    mypoints.from_dataframe(dfr, east="X", north="Y", tvdmsl="Z", attributes=attr)

    assert mypoints.dataframe.X_UTME.mean() == dfr.X.mean()

    with pytest.raises(ValueError):
        mypoints.from_dataframe(
            dfr, east="NOTTHERE", north="Y", tvdmsl="Z", attributes=attr
        )


def test_export_points():
    """Export XYZ points to file, various formats."""

    mypoints = Points(POINTSET4)  # should guess based on extesion

    print(mypoints.dataframe)
    mypoints.to_file(join(TMPD, "poi_export1.rmsattr"), fformat="rms_attr")


def test_import_rmsattr_format():
    """Import points with attributes from RMS attr format."""

    mypoi = Points()

    mypoi.from_file(POINTSET3, fformat="rms_attr")

    print(mypoi.dataframe["VerticalSep"].dtype)
    mypoi.to_file("TMP/attrs.rmsattr", fformat="rms_attr")


def test_export_points_rmsattr():
    """Export XYZ points to file, as rmsattr."""

    mypoints = Points(POINTSET4)  # should guess based on extesion
    logger.info(mypoints.dataframe)
    mypoints.to_file(join(TMPD, "poi_export1.rmsattr"), fformat="rms_attr")
    mypoints2 = Points(join(TMPD, "poi_export1.rmsattr"))

    assert mypoints.dataframe["Seg"].equals(mypoints2.dataframe["Seg"])
    assert mypoints.dataframe["MyNum"].equals(mypoints2.dataframe["MyNum"])
