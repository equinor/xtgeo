import os
from os.path import join
import pandas as pd
import numpy as np

import pytest

from xtgeo.xyz import Polygons

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

PFILE1A = join(TSTPATH, "polygons/reek/1/top_upper_reek_faultpoly.zmap")
PFILE1B = join(TSTPATH, "polygons/reek/1/top_upper_reek_faultpoly.xyz")
PFILE1C = join(TSTPATH, "polygons/reek/1/top_upper_reek_faultpoly.pol")
PFILE = join(TSTPATH, "points/eme/1/emerald_10_random.poi")
POLSET2 = join(TSTPATH, "polygons/reek/1/polset2.pol")
POLSET3 = join(TSTPATH, "polygons/etc/outline.pol")
POLSET4 = join(TSTPATH, "polygons/etc/well16.pol")
POINTSET2 = join(TSTPATH, "points/reek/1/pointset2.poi")


def test_import_zmap_and_xyz():
    """Import XYZ polygons on ZMAP and XYZ format from file"""

    mypol2a = Polygons()
    mypol2b = Polygons()
    mypol2c = Polygons()

    mypol2a.from_file(PFILE1A, fformat="zmap")
    mypol2b.from_file(PFILE1B)
    mypol2c.from_file(PFILE1C)

    assert mypol2a.nrow == mypol2b.nrow
    assert mypol2b.nrow == mypol2c.nrow

    for col in ["X_UTME", "Y_UTMN", "Z_TVDSS", "POLY_ID"]:
        status = np.allclose(
            mypol2a.dataframe[col].values, mypol2b.dataframe[col].values
        )

        assert status is True


def test_import_export_polygons():
    """Import XYZ polygons from file. Modify, and export."""

    mypoly = Polygons()

    mypoly.from_file(PFILE, fformat="xyz")

    z0 = mypoly.dataframe["Z_TVDSS"].values[0]

    tsetup.assert_almostequal(z0, 2266.996338, 0.001)

    mypoly.dataframe["Z_TVDSS"] += 100

    mypoly.to_file(TMPD + "/polygon_export.xyz", fformat="xyz")

    # reimport and check
    mypoly2 = Polygons(TMPD + "/polygon_export.xyz")

    tsetup.assert_almostequal(z0 + 100, mypoly2.dataframe["Z_TVDSS"].values[0], 0.001)


def test_polygon_boundary():
    """Import XYZ polygons from file and test boundary function."""

    mypoly = Polygons()

    mypoly.from_file(PFILE, fformat="xyz")

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


def test_polygon_tlen_hlen():
    """Test the tlen and hlen operations"""

    pol = Polygons(POLSET3)
    pol.tlen()
    pol.hlen()
    print(pol.dataframe)

    assert pol.dataframe[pol.hname].all() <= pol.dataframe[pol.tname].all()
    assert pol.dataframe[pol.hname].any() <= pol.dataframe[pol.tname].any()

    pol.filter_byid(0)
    hlen = pol.get_shapely_objects()[0].length  # shapely length is 2D!
    print(pol.dataframe[pol.tname].iloc[-1])
    assert (abs(pol.dataframe[pol.hname].iloc[-1] - hlen)) < 0.001
    assert (abs(pol.dataframe[pol.dhname].iloc[0] - 1761.148)) < 0.01


def test_rescale_polygon():
    """Take a polygons set and rescale/resample"""

    pol = Polygons(POLSET4)

    oldpol = pol.copy()
    oldpol.name = "ORIG"
    oldpol.hlen()
    pol.rescale(100)
    pol.hlen()

    pol2 = Polygons(POLSET4)

    pol2.rescale(100, kind="slinear")
    pol2.name = "slinear"
    pol2.hlen()

    arr1 = pol.dataframe[pol.dhname].values
    arr2 = pol2.dataframe[pol2.dhname].values

    logger.info(
        "avg, min, max: POL simple %s %s %s vs slinear %s %s %s",
        arr1.mean(),
        arr1.min(),
        arr1.max(),
        arr2.mean(),
        arr2.min(),
        arr2.max(),
    )

    if XTGSHOW:
        pol.quickplot(others=[oldpol, pol2])

    assert oldpol.dataframe.H_CUMLEN.max() == pytest.approx(5335, rel=0.02)
    assert pol.dataframe.H_CUMLEN.max() == pytest.approx(5335, rel=0.02)
    assert pol2.dataframe.H_CUMLEN.max() == pytest.approx(5335, rel=0.02)


def test_fence_from_polygon():
    """Test polygons get_fence method"""

    pol = Polygons(POLSET2)

    df = pol.dataframe[0:3]

    df.at[0, "X_UTME"] = 0.0
    df.at[1, "X_UTME"] = 100.0
    df.at[2, "X_UTME"] = 100.0

    df.at[0, "Y_UTMN"] = 20.0
    df.at[1, "Y_UTMN"] = 20.0
    df.at[2, "Y_UTMN"] = 100.0

    df.at[0, "Z_TVDSS"] = 0.0
    df.at[1, "Z_TVDSS"] = 1000.0
    df.at[2, "Z_TVDSS"] = 2000.0

    pol.dataframe = df

    fence = pol.get_fence(
        distance=100, nextend=4, name="SOMENAME", asnumpy=False, atleast=10
    )
    logger.info(fence.dataframe)
    assert fence.dataframe.H_DELTALEN.mean() == pytest.approx(19.98, abs=0.01)
    assert fence.dataframe.H_DELTALEN.std() <= 0.05


def test_fence_from_vertical_polygon():
    """Test fence from polygon which only has vertical samples, e.g. a vertical well"""

    pol = Polygons()

    mypoly = {
        pol.xname: [0.0, 0.0, 0.0],
        pol.yname: [100.0, 100.0, 100.0],
        pol.zname: [0, 50, 60],
        pol.pname: [1, 1, 1],
    }
    pol.dataframe = pd.DataFrame(mypoly)

    fence = pol.get_fence(
        distance=10, nextend=1, name="SOMENAME", asnumpy=False, atleast=3
    )
    logger.info(fence.dataframe.to_string())

    assert len(fence.dataframe) == 161
    assert fence.dataframe.H_DELTALEN.mean() == 0.125
    assert fence.dataframe.H_DELTALEN.std() <= 0.001
    assert fence.dataframe.H_CUMLEN.max() == 10
    assert fence.dataframe.H_CUMLEN.min() == -10.0


def test_fence_from_almost_vertical_polygon():
    """Test fence from polygon which only has close to vertical samples"""

    pol = Polygons()

    mypoly = {
        pol.xname: [0.1, 0.2, 0.3],
        pol.yname: [100.1, 100.2, 100.3],
        pol.zname: [0, 50, 60],
        pol.pname: [1, 1, 1],
    }
    pol.dataframe = pd.DataFrame(mypoly)

    fence = pol.get_fence(
        distance=10, nextend=1, name="SOMENAME", asnumpy=False, atleast=3
    )
    logger.info(fence.dataframe.to_string())

    assert len(fence.dataframe) == 145
    assert fence.dataframe.H_DELTALEN.mean() == pytest.approx(0.1414, abs=0.01)
    assert fence.dataframe.H_DELTALEN.std() <= 0.001
    assert fence.dataframe.H_CUMLEN.max() == pytest.approx(10.0, abs=0.5)
    assert fence.dataframe.H_CUMLEN.min() == pytest.approx(-10.0, abs=0.5)


def test_fence_from_slanted_polygon():
    """Test fence from polygon which is slanted; but total HLEN is less than distance"""

    pol = Polygons()

    mypoly = {
        pol.xname: [0.0, 3.0, 6.0],
        pol.yname: [100.0, 102.0, 104.0],
        pol.zname: [0, 50, 60],
        pol.pname: [1, 1, 1],
    }
    pol.dataframe = pd.DataFrame(mypoly)

    fence = pol.get_fence(
        distance=10, nextend=1, name="SOMENAME", asnumpy=False, atleast=3
    )
    logger.info(fence.dataframe.to_string())

    assert len(fence.dataframe) == 9
    assert fence.dataframe.H_DELTALEN.mean() == pytest.approx(3.6, abs=0.02)
    assert fence.dataframe.H_DELTALEN.std() <= 0.001


def test_fence_from_more_slanted_polygon():
    """Test fence from poly which is slanted; and total HLEN is > than distance"""

    pol = Polygons()

    mypoly = {
        pol.xname: [0.0, 7.0, 15.0],
        pol.yname: [100.0, 110.0, 120.0],
        pol.zname: [0, 50, 60],
        pol.pname: [1, 1, 1],
    }
    pol.dataframe = pd.DataFrame(mypoly)

    fence = pol.get_fence(
        distance=10, nextend=1, name="SOMENAME", asnumpy=False, atleast=3
    )
    logger.info(fence.dataframe.to_string())

    assert len(fence.dataframe) == 5
    assert fence.dataframe.H_DELTALEN.mean() == pytest.approx(12.49, abs=0.02)
    assert fence.dataframe.H_DELTALEN.std() <= 0.001


def test_rename_columns():
    """Renaming xname, yname, zname"""

    pol = Polygons(POLSET2)
    assert pol.xname == "X_UTME"

    pol.xname = "NEWX"
    assert pol.xname == "NEWX"

    assert "NEWX" in pol.dataframe

    pol.yname = "NEWY"
    assert pol.yname == "NEWY"
    assert pol.xname != "NEWY"

    assert "NEWY" in pol.dataframe


def test_empty_polygon_has_default_name():
    pol = Polygons()
    assert pol.dtname == "T_DELTALEN"


def test_check_column_names():
    pol = Polygons()
    data = pd.DataFrame({"T_DELTALEN": [1, 2]})
    pol.dataframe = data
    assert pol.dtname == "T_DELTALEN"
    assert pol.dhname is None


def test_delete_from_empty_polygon_does_not_fail(recwarn):
    pol = Polygons()
    pol.delete_columns(pol.dtname)
    assert len(recwarn) == 1
    assert "Trying to delete" in str(recwarn.list[0].message)


@pytest.mark.parametrize("test_name", [(1), ({}), ([]), (2.3)])
@pytest.mark.parametrize(
    "name_attribute", [("hname"), ("dhname"), ("tname"), ("dtname"), ("pname")]
)
def test_raise_incorrect_name_type(test_name, name_attribute):
    pol = Polygons()
    data = pd.DataFrame({"ANYTHING": [1, 2]})
    pol.dataframe = data

    with pytest.raises(ValueError) as execinfo:
        setattr(pol, name_attribute, test_name)
    assert "Wrong type of input" in str(execinfo.value)


@pytest.mark.parametrize(
    "name_attribute", [("hname"), ("dhname"), ("tname"), ("dtname"), ("pname")]
)
def test_raise_non_existing_name(name_attribute):
    pol = Polygons()
    data = pd.DataFrame({"ANYTHING": [1, 2]})
    pol.dataframe = data

    with pytest.raises(ValueError) as execinfo:
        setattr(pol, name_attribute, "NON_EXISTING")
    assert "does not exist" in str(execinfo.value)
