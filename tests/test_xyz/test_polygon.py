import pathlib

import numpy as np
import pandas as pd
import pytest
import xtgeo
from xtgeo.xyz import Points, Polygons

PFILE1A = pathlib.Path("polygons/reek/1/top_upper_reek_faultpoly.zmap")
PFILE1B = pathlib.Path("polygons/reek/1/top_upper_reek_faultpoly.xyz")
PFILE1C = pathlib.Path("polygons/reek/1/top_upper_reek_faultpoly.pol")
PFILE = pathlib.Path("points/eme/1/emerald_10_random.poi")
POLSET2 = pathlib.Path("polygons/reek/1/polset2.pol")
POLSET3 = pathlib.Path("polygons/etc/outline.pol")
POLSET4 = pathlib.Path("polygons/etc/well16.pol")
POINTSET2 = pathlib.Path("points/reek/1/pointset2.poi")


@pytest.mark.parametrize(
    "filename, fformat",
    [
        (PFILE1A, "zmap"),
        (PFILE1B, "xyz"),
        (PFILE1C, "pol"),
    ],
)
def test_polygons_from_file_alternatives(testpath, filename, fformat):
    polygons1 = xtgeo.polygons_from_file(testpath / filename, fformat=fformat)
    polygons2 = xtgeo.polygons_from_file(testpath / filename)

    pd.testing.assert_frame_equal(polygons1.dataframe, polygons2.dataframe)


def test_polygons_from_lists():
    plist = [(234, 556, 11), (235, 559, 14), (255, 577, 12)]

    mypol = Polygons(plist)

    assert mypol.dataframe["X_UTME"].values[0] == 234
    assert mypol.dataframe["Z_TVDSS"].values[2] == 12

    plist = [
        (234, 556, 11, 0),
        (235, 559, 14, 1),
        (255, 577, 12, 1),
    ]

    mypol = Polygons(plist)
    assert mypol.dataframe["POLY_ID"].values[2] == 1

    somedf = mypol.dataframe.copy()
    mypol2 = Polygons(somedf)
    assert mypol.dataframe.equals(mypol2.dataframe)


def test_polygons_from_list_and_attrs():
    plist = [
        (234, 556, 11, 0, "some", 1.0),
        (235, 559, 14, 1, "attr", 1.1),
        (255, 577, 12, 1, "here", 1.2),
    ]
    attrs = dict()
    attrs["sometxt"] = "str"
    attrs["somefloat"] = "float"

    mypol = Polygons(plist, attributes=attrs)
    assert mypol.dataframe["POLY_ID"].values[2] == 1

    somedf = mypol.dataframe.copy()
    mypol2 = Polygons(somedf, attributes=attrs)
    assert mypol.dataframe.equals(mypol2.dataframe)


def test_polygons_from_attrs_not_ordereddict():
    """Make polygons with attrs from list of tuples ndarray or dataframe.

    It seems that python 3.6 dicts are actually ordered "but cannot be trusted"?
    In python 3.7+ it is a feature.
    """

    plist = [
        (234, 556, 11, 0, "some", 1.0),
        (235, 559, 14, 1, "attr", 1.1),
        (255, 577, 12, 1, "here", 1.2),
    ]
    attrs = {}
    attrs["sometxt"] = "str"
    attrs["somefloat"] = "float"

    mypol = Polygons(plist, attributes=attrs)
    assert mypol.dataframe["POLY_ID"].values[2] == 1


def test_import_zmap_and_xyz(testpath):
    """Import XYZ polygons on ZMAP and XYZ format from file"""

    mypol2a = xtgeo.polygons_from_file(testpath / PFILE1A, fformat="zmap")
    mypol2b = xtgeo.polygons_from_file(testpath / PFILE1B)
    mypol2c = xtgeo.polygons_from_file(testpath / PFILE1C)

    assert mypol2a.nrow == mypol2b.nrow
    assert mypol2b.nrow == mypol2c.nrow

    for col in ["X_UTME", "Y_UTMN", "Z_TVDSS", "POLY_ID"]:
        assert np.allclose(mypol2a.dataframe[col].values, mypol2b.dataframe[col].values)


def test_import_export_polygons(testpath, tmp_path):
    """Import XYZ polygons from file. Modify, and export."""

    mypoly = xtgeo.polygons_from_file(testpath / PFILE, fformat="xyz")

    z0 = mypoly.dataframe["Z_TVDSS"].values[0]

    assert z0 == pytest.approx(2266.996338, abs=0.001)

    mypoly.dataframe["Z_TVDSS"] += 100

    mypoly.to_file(tmp_path / "polygon_export.xyz", fformat="xyz")

    # reimport and check
    mypoly2 = xtgeo.polygons_from_file(tmp_path / "polygon_export.xyz")

    assert z0 + 100 == pytest.approx(mypoly2.dataframe["Z_TVDSS"].values[0], 0.001)


def test_polygon_boundary(testpath):
    """Import XYZ polygons from file and test boundary function."""

    mypoly = xtgeo.polygons_from_file(testpath / PFILE, fformat="xyz")

    boundary = mypoly.get_boundary()

    assert boundary[0] == pytest.approx(460595.6036, abs=0.0001)
    assert boundary[4] == pytest.approx(2025.952637, abs=0.0001)
    assert boundary[5] == pytest.approx(2266.996338, abs=0.0001)


def test_polygon_filter_byid(testpath):
    """Filter a Polygon by a list of ID's"""

    pol = xtgeo.polygons_from_file(testpath / POLSET3)

    assert pol.dataframe["POLY_ID"].iloc[0] == 0
    assert pol.dataframe["POLY_ID"].iloc[-1] == 3

    pol.filter_byid()
    assert pol.dataframe["POLY_ID"].iloc[-1] == 0

    pol = xtgeo.polygons_from_file(testpath / POLSET3)
    pol.filter_byid([1, 3])

    assert pol.dataframe["POLY_ID"].iloc[0] == 1
    assert pol.dataframe["POLY_ID"].iloc[-1] == 3

    pol = xtgeo.polygons_from_file(testpath / POLSET3)
    pol.filter_byid(2)

    assert pol.dataframe["POLY_ID"].iloc[0] == 2
    assert pol.dataframe["POLY_ID"].iloc[-1] == 2

    pol = xtgeo.polygons_from_file(testpath / POLSET3)
    pol.filter_byid(99)  # not present; should remove all rows
    assert pol.nrow == 0


def test_polygon_tlen_hlen(testpath):
    """Test the tlen and hlen operations"""

    pol = xtgeo.polygons_from_file(testpath / POLSET3)
    pol.tlen()
    pol.hlen()

    assert pol.dataframe[pol.hname].all() <= pol.dataframe[pol.tname].all()
    assert pol.dataframe[pol.hname].any() <= pol.dataframe[pol.tname].any()

    pol.filter_byid(0)
    hlen = pol.get_shapely_objects()[0].length  # shapely length is 2D!
    assert (abs(pol.dataframe[pol.hname].iloc[-1] - hlen)) < 0.001
    assert (abs(pol.dataframe[pol.dhname].iloc[0] - 1761.148)) < 0.01


@pytest.mark.parametrize(
    "dorescale, kind, expectmax, expectlen",
    [
        (False, None, 5335, 5429),
        (True, None, 5335, 54),
        (True, "slineaer", 5335, 54),
        (True, "cubic", 5335, 53),
    ],
)
def test_rescale_polygon(testpath, dorescale, kind, expectmax, expectlen):
    """Take a polygons set and rescale/resample."""

    pol = xtgeo.polygons_from_file(testpath / POLSET4)

    if not dorescale:
        pol.name = "ORIG"
        pol.hlen()
    else:
        pol.rescale(100, kind=kind)
        pol.name = kind if kind else "none"
        pol.hlen()

    assert pol.dataframe.H_CUMLEN.max() == pytest.approx(expectmax, rel=0.02)
    assert pol.dataframe.shape == (expectlen, 6)


def test_fence_from_polygon(testpath):
    """Test polygons get_fence method"""

    pol = xtgeo.polygons_from_file(testpath / POLSET2)

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

    pol = xtgeo.polygons_from_file(testpath / POLSET2)
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


def test_delete_from_empty_polygon_shall_not_fail(recwarn):
    pol = Polygons()
    pol.delete_columns([pol.dtname])
    assert len(recwarn) == 0


def test_delete_columns_protected_columns():
    pol = Polygons([(1, 2, 3)])
    with pytest.warns(UserWarning, match="protected and will not be deleted"):
        pol.delete_columns([pol.xname])
    with pytest.warns(UserWarning, match="protected and will not be deleted"):
        pol.delete_columns([pol.yname])
    with pytest.warns(UserWarning, match="protected and will not be deleted"):
        pol.delete_columns([pol.zname])

    assert pol.xname in pol.dataframe
    assert pol.yname in pol.dataframe
    assert pol.zname in pol.dataframe


def test_delete_columns_strict_raises():
    pol = Polygons()
    with pytest.raises(ValueError, match="not present"):
        pol.delete_columns([pol.tname], strict=True)


@pytest.mark.parametrize("test_name", [(1), ({}), ([]), (2.3)])
def test_raise_incorrect_name_type(test_name):
    pol = Polygons()
    data = pd.DataFrame(
        {
            "X_UTME": [1.0, 2.0],
            "Y_UTMN": [2.0, 1.0],
            "Z_TVDSS": [3.0, 3.0],
            "POLY_ID": [3, 3],
            "T_DELTALEN": [2.0, 1.0],
        }
    )
    pol.dataframe = data
    pol._pname = "POLY_ID"
    pol._tname = "T_DELTALEN"

    with pytest.raises(ValueError, match="Wrong type of input"):
        setattr(pol, "tname", test_name)


@pytest.mark.parametrize(
    "name_attribute", [("hname"), ("dhname"), ("tname"), ("dtname")]
)
def test_raise_special_name_name_type(name_attribute):
    pol = Polygons()
    data = pd.DataFrame(
        {
            "X_UTME": [1.0, 2.0],
            "Y_UTMN": [2.0, 1.0],
            "Z_TVDSS": [3.0, 3.0],
            "POLY_ID": [3, 3],
            "T_DELTALEN": [2.0, 1.0],
        }
    )
    pol._pname = "POLY_ID"
    pol.dataframe = data

    with pytest.raises(ValueError, match="does not exist as a column"):
        setattr(pol, name_attribute, "anyname")


@pytest.mark.parametrize(
    "func, where, value, expected_result",
    [
        ("add", "inside", 44.0, [47.0, 47.0, 3.0]),
        ("add", "outside", 44.0, [3.0, 3.0, 47.0]),
        ("mul", "inside", 2.0, [6.0, 6.0, 3.0]),
        ("div", "inside", 2.0, [1.5, 1.5, 3.0]),
        ("sub", "outside", 3.0, [3.0, 3.0, 0.0]),
        ("set", "outside", 88.0, [3.0, 3.0, 88.0]),
        ("eli", "outside", 0, [3.0, 3.0]),
    ],
)
@pytest.mark.parametrize("shorthand", [True, False])
@pytest.mark.parametrize("constructor", [Points, Polygons])
def test_polygons_operation_in_polygons(
    func, where, value, expected_result, shorthand, constructor
):
    """Test what happens to point belonging to polygons for add_inside etc."""

    closed_poly = Polygons(
        pd.DataFrame(
            {
                "X_UTME": [0.0, 100.0, 100.0, 0.0, 0.0],
                "Y_UTMN": [0.0, 0.0, 200.0, 200.0, 0.0],
                "Z_TVDSS": [3.0, 3.0, 3.0, 3.0, 3.0],
                "POLY_ID": [0, 0, 0, 0, 0],
            }
        )
    )

    # this set will first point on border, middle point inside and last point outside
    # the closed_poly defined above
    pointset = constructor(
        pd.DataFrame(
            {
                "X_UTME": [0.0, 50.0, 200.0],
                "Y_UTMN": [0.0, 100.0, 300.0],
                "Z_TVDSS": [3.0, 3.0, 3.0],
                "POLY_ID": [0, 0, 0],
            }
        )
    )

    inside = True if "inside" in where else False

    if shorthand:
        if func == "eli":
            getattr(pointset, f"eli_{where}")(closed_poly)
        else:
            getattr(pointset, f"{func}_{where}")(closed_poly, value)
    else:
        pointset.operation_polygons(closed_poly, value, opname=func, inside=inside)
    assert list(pointset.dataframe["Z_TVDSS"]) == expected_result


@pytest.mark.parametrize(
    "functionname, expected",
    [
        ("add_inside", [3.0, 5.0, 3.0, 1.0]),
        ("add_outside", [3.0, 1.0, 3.0, 5.0]),
        ("sub_inside", [-1.0, -3.0, -1.0, 1.0]),
        ("sub_outside", [-1.0, 1.0, -1.0, -3.0]),
        ("mul_inside", [2.0, 4.0, 2.0, 1.0]),
        ("mul_outside", [2.0, 1.0, 2.0, 4.0]),
        ("div_inside", [0.5, 0.25, 0.5, 1.0]),
        ("div_outside", [0.5, 1.0, 0.5, 0.25]),
        ("set_inside", [2.0, 2.0, 2.0, 1.0]),
        ("set_outside", [2.0, 1.0, 2.0, 2.0]),
        ("eli_inside", [1.0]),
        ("eli_outside", [1.0]),
    ],
)
def test_shortform_polygons_overlap(functionname, expected):
    inner_polygon = [
        (3.0, 3.0, 0.0, 0),
        (5.0, 3.0, 0.0, 0),
        (5.0, 5.0, 0.0, 0),
        (3.0, 5.0, 0.0, 0),
        (3.0, 3.0, 0.0, 0),
    ]

    overlap_polygon = [
        (4.0, 4.0, 0.0, 2),
        (6.0, 4.0, 0.0, 2),
        (6.0, 6.0, 0.0, 2),
        (4.0, 6.0, 0.0, 2),
        (4.0, 4.0, 0.0, 2),
    ]

    pol = Polygons(inner_polygon + overlap_polygon)
    # The Four points are placed: within both polygons
    poi = Points([(3.5, 3.5, 1.0), (4.5, 4.5, 1.0), (5.5, 5.5, 1.0), (6.5, 6.5, 1.0)])
    if "eli" in functionname:
        getattr(poi, functionname)(pol)
    else:
        getattr(poi, functionname)(pol, 2.0)

    assert list(poi.dataframe[poi.zname].values) == expected


def test_polygons_simplify_preserve_topology():
    polygon = [
        (3.0, 3.0, 0.0, 0),
        (5.0, 3.0, 0.0, 0),
        (5.0, 5.0, 0.0, 0),
        (4.9, 5.0, 0.0, 0),
        (5.0, 5.0, 0.0, 0),
        (3.0, 5.0, 0.0, 0),
        (3.0, 3.0, 0.0, 0),
    ]
    pol = Polygons(polygon)

    status = pol.simplify(tolerance=0.3)
    print(pol.dataframe)
    assert status is True

    assert pol.dataframe[pol.xname].values.tolist() == [3.0, 5.0, 5.0, 3.0, 3.0]


def test_polygons_simplify_not_preserve_topology():
    polygon = [
        (3.0, 3.0, 0.0, 0),
        (5.0, 3.0, 0.0, 0),
        (5.0, 5.0, 0.0, 0),
        (4.9, 5.0, 0.0, 0),
        (5.0, 5.0, 0.0, 0),
        (3.0, 5.0, 0.0, 0),
        (3.0, 3.0, 0.0, 0),
    ]
    pol = Polygons(polygon)

    status = pol.simplify(tolerance=0.3, preserve_topology=False)
    print(pol.dataframe)
    assert status is True

    assert pol.dataframe[pol.xname].values.tolist() == [3.0, 5.0, 5.0, 3.0, 3.0]
