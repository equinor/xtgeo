import pathlib

import numpy as np
import pandas as pd
import pytest

from xtgeo.xyz import Polygons

PFILE1A = pathlib.Path("polygons/reek/1/top_upper_reek_faultpoly.zmap")
PFILE1B = pathlib.Path("polygons/reek/1/top_upper_reek_faultpoly.xyz")
PFILE1C = pathlib.Path("polygons/reek/1/top_upper_reek_faultpoly.pol")
PFILE = pathlib.Path("points/eme/1/emerald_10_random.poi")
POLSET2 = pathlib.Path("polygons/reek/1/polset2.pol")
POLSET3 = pathlib.Path("polygons/etc/outline.pol")
POLSET4 = pathlib.Path("polygons/etc/well16.pol")
POINTSET2 = pathlib.Path("points/reek/1/pointset2.poi")


def test_import_zmap_and_xyz(testpath):
    """Import XYZ polygons on ZMAP and XYZ format from file"""

    mypol2a = Polygons()
    mypol2b = Polygons()
    mypol2c = Polygons()

    mypol2a.from_file(testpath / PFILE1A, fformat="zmap")
    mypol2b.from_file(testpath / PFILE1B)
    mypol2c.from_file(testpath / PFILE1C)

    assert mypol2a.nrow == mypol2b.nrow
    assert mypol2b.nrow == mypol2c.nrow

    for col in ["X_UTME", "Y_UTMN", "Z_TVDSS", "POLY_ID"]:
        assert np.allclose(mypol2a.dataframe[col].values, mypol2b.dataframe[col].values)


def test_import_export_polygons(testpath, tmp_path):
    """Import XYZ polygons from file. Modify, and export."""

    mypoly = Polygons()

    mypoly.from_file(testpath / PFILE, fformat="xyz")

    z0 = mypoly.dataframe["Z_TVDSS"].values[0]

    assert z0 == pytest.approx(2266.996338, abs=0.001)

    mypoly.dataframe["Z_TVDSS"] += 100

    mypoly.to_file(tmp_path / "polygon_export.xyz", fformat="xyz")

    # reimport and check
    mypoly2 = Polygons(tmp_path / "polygon_export.xyz")

    assert z0 + 100 == pytest.approx(mypoly2.dataframe["Z_TVDSS"].values[0], 0.001)


def test_polygon_boundary(testpath):
    """Import XYZ polygons from file and test boundary function."""

    mypoly = Polygons()

    mypoly.from_file(testpath / PFILE, fformat="xyz")

    boundary = mypoly.get_boundary()

    assert boundary[0] == pytest.approx(460595.6036, abs=0.0001)
    assert boundary[4] == pytest.approx(2025.952637, abs=0.0001)
    assert boundary[5] == pytest.approx(2266.996338, abs=0.0001)


def test_polygon_filter_byid(testpath):
    """Filter a Polygon by a list of ID's"""

    pol = Polygons(testpath / POLSET3)

    assert pol.dataframe["POLY_ID"].iloc[0] == 0
    assert pol.dataframe["POLY_ID"].iloc[-1] == 3

    pol.filter_byid()
    assert pol.dataframe["POLY_ID"].iloc[-1] == 0

    pol = Polygons(testpath / POLSET3)
    pol.filter_byid([1, 3])

    assert pol.dataframe["POLY_ID"].iloc[0] == 1
    assert pol.dataframe["POLY_ID"].iloc[-1] == 3

    pol = Polygons(testpath / POLSET3)
    pol.filter_byid(2)

    assert pol.dataframe["POLY_ID"].iloc[0] == 2
    assert pol.dataframe["POLY_ID"].iloc[-1] == 2

    pol = Polygons(testpath / POLSET3)
    pol.filter_byid(99)  # not present; should remove all rows
    assert pol.nrow == 0


def test_polygon_tlen_hlen(testpath):
    """Test the tlen and hlen operations"""

    pol = Polygons(testpath / POLSET3)
    pol.tlen()
    pol.hlen()

    assert pol.dataframe[pol.hname].all() <= pol.dataframe[pol.tname].all()
    assert pol.dataframe[pol.hname].any() <= pol.dataframe[pol.tname].any()

    pol.filter_byid(0)
    hlen = pol.get_shapely_objects()[0].length  # shapely length is 2D!
    assert (abs(pol.dataframe[pol.hname].iloc[-1] - hlen)) < 0.001
    assert (abs(pol.dataframe[pol.dhname].iloc[0] - 1761.148)) < 0.01


def test_rescale_polygon(testpath):
    """Take a polygons set and rescale/resample"""

    pol = Polygons(testpath / POLSET4)

    oldpol = pol.copy()
    oldpol.name = "ORIG"
    oldpol.hlen()
    pol.rescale(100)
    pol.hlen()

    pol2 = Polygons(testpath / POLSET4)

    pol2.rescale(100, kind="slinear")
    pol2.name = "slinear"
    pol2.hlen()

    assert oldpol.dataframe.H_CUMLEN.max() == pytest.approx(5335, rel=0.02)
    assert pol.dataframe.H_CUMLEN.max() == pytest.approx(5335, rel=0.02)
    assert pol2.dataframe.H_CUMLEN.max() == pytest.approx(5335, rel=0.02)


def test_fence_from_polygon(testpath):
    """Test polygons get_fence method"""

    pol = Polygons(testpath / POLSET2)

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

    assert len(fence.dataframe) == 5
    assert fence.dataframe.H_DELTALEN.mean() == pytest.approx(12.49, abs=0.02)
    assert fence.dataframe.H_DELTALEN.std() <= 0.001


def test_rename_columns(testpath):
    """Renaming xname, yname, zname"""

    pol = Polygons(testpath / POLSET2)
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

    with pytest.raises(ValueError, match="Wrong type of input"):
        setattr(pol, name_attribute, test_name)


@pytest.mark.parametrize(
    "name_attribute", [("hname"), ("dhname"), ("tname"), ("dtname"), ("pname")]
)
def test_raise_non_existing_name(name_attribute):
    pol = Polygons()
    data = pd.DataFrame({"ANYTHING": [1, 2]})
    pol.dataframe = data

    with pytest.raises(ValueError, match="does not exist"):
        setattr(pol, name_attribute, "NON_EXISTING")
