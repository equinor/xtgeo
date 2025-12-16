import itertools
import pathlib

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st
from pandas.testing import assert_frame_equal

import xtgeo
from xtgeo.xyz import Points

PFILE = pathlib.Path("points/eme/1/emerald_10_random.poi")
PFILE2 = pathlib.Path("polygons/reek/1/top_upper_reek_faultpoly.zmap")
POINTSET2 = pathlib.Path("points/reek/1/pointset2.poi")
POINTSET3 = pathlib.Path("points/battle/1/many.rmsattr")
POINTSET4 = pathlib.Path("points/reek/1/poi_attr.rmsattr")
POINTSET4_CSV = pathlib.Path("points/reek/1/poi_attr.csv")
CSV1 = pathlib.Path("3dgrids/etc/gridqc1_rms_cellcenter.csv")


@pytest.fixture
def points_with_attrs():
    plist = [
        (234.0, 556.0, 11.0, 0, "some", 1.0),
        (235.0, 559.0, 14.0, 1, "attr", 1.1),
        (255.0, 577.0, 12.0, 1, "here", 1.2),
    ]
    attrs = {
        "some_int": "int",
        "sometxt": "str",
        "somefloat": "float",
    }
    return plist, attrs


@pytest.mark.parametrize(
    "filename, fformat",
    [
        (PFILE, "poi"),
        (POINTSET2, "poi"),
        (PFILE2, "zmap"),
        (POINTSET3, "rmsattr"),
        (POINTSET4, "rmsattr"),
        (POINTSET4_CSV, "csv"),
    ],
)
def test_points_from_file_alternatives(testdata_path, filename, fformat):
    # deprecated
    points1 = xtgeo.points_from_file(testdata_path / filename, fformat=fformat)
    points2 = xtgeo.points_from_file(testdata_path / filename)

    points3 = xtgeo.points_from_file(testdata_path / filename, fformat=fformat)
    points4 = xtgeo.points_from_file(testdata_path / filename)

    pd.testing.assert_frame_equal(points1.get_dataframe(), points2.get_dataframe())
    pd.testing.assert_frame_equal(points2.get_dataframe(), points3.get_dataframe())
    pd.testing.assert_frame_equal(points3.get_dataframe(), points4.get_dataframe())


def test_points_rmsattr_vs_csv(testdata_path):
    """Test reading points from RMS attr and CSV files."""

    # read from file
    mypoints_rmsattr = xtgeo.points_from_file(
        testdata_path / POINTSET4, fformat="rms_attr"
    )
    mypoints_csv = xtgeo.points_from_file(testdata_path / POINTSET4_CSV, fformat="csv")

    # compare dataframes
    pd.testing.assert_frame_equal(
        mypoints_rmsattr.get_dataframe(),
        mypoints_csv.get_dataframe(),
        check_dtype=False,
    )


def test_points_read_write_csv(testdata_path, tmp_path):
    """Test reading and writing points to CSV file."""

    # read from file
    mypoints = xtgeo.points_from_file(testdata_path / CSV1, fformat="csv")

    # write to file
    usefile = tmp_path / "test_points.csv"
    mypoints.to_file(usefile, fformat="csv")

    # read back from file
    mypoints2 = xtgeo.points_from_file(usefile, fformat="csv")

    pd.testing.assert_frame_equal(
        mypoints.get_dataframe(), mypoints2.get_dataframe(), check_dtype=False
    )


def test_points_io_with_attrs(points_with_attrs, tmp_path):
    """Test roundtrip of polygons with attributes from file and back.

    Note that some formats do not store attributes.
    """

    plist, attrs = points_with_attrs

    mypoi = Points(plist, attributes=attrs)
    mypoi_no_attrs = mypoi.copy()
    mypoi_no_attrs.set_dataframe(
        mypoi_no_attrs.get_dataframe()[["X_UTME", "Y_UTMN", "Z_TVDSS"]]
    )

    # poi
    usefile = tmp_path / "test_roundtrip_no_attrs.poi"
    mypoi_no_attrs.to_file(usefile, fformat="poi")
    mypoi2 = xtgeo.points_from_file(usefile, fformat="poi")
    pd.testing.assert_frame_equal(
        mypoi_no_attrs.get_dataframe(), mypoi2.get_dataframe(), check_dtype=False
    )

    # csv, no attributes
    usefile = tmp_path / "test_roundtrip_no_attrs.csv"
    mypoi.to_file(usefile, fformat="csv", attributes=False)
    mypoi = xtgeo.points_from_file(usefile, fformat="csv")
    pd.testing.assert_frame_equal(
        mypoi_no_attrs.get_dataframe(), mypoi.get_dataframe(), check_dtype=False
    )

    # csv, with attributes
    usefile = tmp_path / "test_roundtrip_with_attrs.csv"
    print("Using", usefile)
    mypoi.to_file(usefile, fformat="csv", attributes=True)
    mypoi2 = xtgeo.points_from_file(usefile, fformat="csv")
    pd.testing.assert_frame_equal(
        mypoi.get_dataframe(), mypoi2.get_dataframe(), check_dtype=False
    )


def test_points_io_table_with_some_attrs(points_with_attrs, tmp_path):
    """Test roundtrip with selected attributes from file and back for csv/pq."""

    plist, attrs = points_with_attrs

    mypoints = Points(plist, attributes=attrs)
    mypoints_no_attrs = mypoints.copy()
    mypoints_no_attrs.set_dataframe(
        mypoints_no_attrs.get_dataframe()[["X_UTME", "Y_UTMN", "Z_TVDSS"]]
    )

    # csv/parquet, "somefloat" only as attr
    for fmt in ["csv", "parquet"]:
        usefile = tmp_path / f"test_roundtrip_some_attrs.{fmt}"
        mypoints.to_file(usefile, fformat=fmt, attributes=["somefloat"])
        mypoints2 = xtgeo.points_from_file(usefile, fformat=fmt)

        pd.testing.assert_frame_equal(
            mypoints.get_dataframe().iloc[:, :3],
            mypoints2.get_dataframe().iloc[:, :3],
            check_dtype=False,
        )

        assert "somefloat" in mypoints2.get_dataframe().columns
        assert "sometxt" not in mypoints2.get_dataframe().columns
        assert "some_int" not in mypoints2.get_dataframe().columns


def test_points_from_list_of_tuples():
    plist = [(234, 556, 11), (235, 559, 14), (255, 577, 12)]

    mypoints = Points(plist)

    x0 = mypoints.get_dataframe()["X_UTME"].values[0]
    z2 = mypoints.get_dataframe()["Z_TVDSS"].values[2]
    assert x0 == 234
    assert z2 == 12


def test_points_with_attrs_copy():
    plist = [
        (234, 556, 11, "some", 1.0),
        (235, 559, 14, "attr", 1.1),
        (255, 577, 12, "here", 1.2),
    ]
    attrs = {}
    attrs["sometxt"] = "str"
    attrs["somefloat"] = "float"

    mypoi = Points(plist, attributes=attrs)

    cppoi = mypoi.copy()

    assert_frame_equal(mypoi.get_dataframe(), cppoi.get_dataframe())
    assert "somefloat" in mypoi.get_dataframe().columns


@st.composite
def list_of_equal_length_lists(draw):
    list_len = draw(st.integers(min_value=3, max_value=3))
    fixed_len_list = st.lists(
        st.floats(allow_nan=False, allow_infinity=False),
        min_size=list_len,
        max_size=list_len,
    )
    return draw(st.lists(fixed_len_list, min_size=1))


@given(list_of_equal_length_lists())
def test_create_pointset(points):
    """Create randomly generated points and verify content."""
    pointset = Points(points)
    points = np.array(points)

    assert len(points) == pointset.nrow

    np.testing.assert_array_almost_equal(
        pointset.get_dataframe()["X_UTME"], points[:, 0]
    )
    np.testing.assert_array_almost_equal(
        pointset.get_dataframe()["Y_UTMN"], points[:, 1]
    )
    np.testing.assert_array_almost_equal(
        pointset.get_dataframe()["Z_TVDSS"], points[:, 2]
    )


def test_import(testdata_path):
    """Import XYZ points from file."""

    mypoints = xtgeo.points_from_file(
        testdata_path / PFILE
    )  # should guess based on extesion

    x0 = mypoints.get_dataframe()["X_UTME"].values[0]
    assert x0 == pytest.approx(460842.434326, 0.001)


def test_import_from_dataframe(testdata_path):
    """Import Points via Pandas dataframe."""

    dfr = pd.read_csv(testdata_path / CSV1, skiprows=3)

    attr = {"I": "int", "J": "int", "K": "int"}
    mypoints = xtgeo.Points(
        values=dfr, xname="X", yname="Y", zname="Z", attributes=attr
    )

    assert mypoints.get_dataframe().X.mean() == dfr.X.mean()

    with pytest.raises(ValueError):
        mypoints = Points(dfr, xname="NOTTHERE", yname="Y", zname="Z", attributes=attr)


def test_export_and_load_points(tmp_path):
    """Export XYZ points to file."""
    plist = [(1.0, 1.0, 1.0), (2.0, 3.0, 4.0), (5.0, 6.0, 7.0)]
    test_points = Points(plist)

    export_path = tmp_path / "test_points.xyz"
    test_points.to_file(export_path)

    exported_points = xtgeo.points_from_file(export_path)

    pd.testing.assert_frame_equal(
        test_points.get_dataframe(), exported_points.get_dataframe()
    )
    assert list(itertools.chain.from_iterable(plist)) == list(
        test_points.get_dataframe().values.flatten()
    )


def test_export_load_rmsformatted_points(testdata_path, tmp_path):
    """Export XYZ points to file, various formats."""

    test_points_path = testdata_path / POINTSET4
    orig_points = xtgeo.points_from_file(
        test_points_path
    )  # should guess based on extesion

    export_path = tmp_path / "attrs.rmsattr"
    orig_points.to_file(export_path, fformat="rms_attr")

    reloaded_points = xtgeo.points_from_file(export_path)

    pd.testing.assert_frame_equal(
        orig_points.get_dataframe().iloc[:, :3],
        reloaded_points.get_dataframe().iloc[:, :3],
        check_dtype=False,
    )


def test_io_rms_attrs(points_with_attrs, tmp_path):
    """Test points with attributes from file and back using rms_attrs fmt."""

    plist, attrs = points_with_attrs

    mypoints = Points(plist, attributes=attrs)

    # export to rmsattr
    usefile = tmp_path / "test_roundtrip.rmsattr"
    mypoints.to_file(usefile, fformat="rms_attr", attributes=True)

    # read back from file
    mypoints2 = xtgeo.points_from_file(usefile, fformat="rms_attr")

    pd.testing.assert_frame_equal(
        mypoints.get_dataframe(), mypoints2.get_dataframe(), check_dtype=False
    )

    # just a few attrs
    usefile = tmp_path / "test_roundtrip2.rmsattr"
    mypoints.to_file(usefile, fformat="rms_attr", attributes=["somefloat"])

    # read back from file
    mypoints2 = xtgeo.points_from_file(usefile, fformat="rms_attr")

    pd.testing.assert_frame_equal(
        mypoints.get_dataframe().iloc[:, :3],
        mypoints2.get_dataframe().iloc[:, :3],
        check_dtype=False,
    )
    assert "some_int" not in mypoints2.get_dataframe().columns


def test_io_rms_wrong_attrs(points_with_attrs, tmp_path):
    """Test points with attributes from file and back using rms_attrs fmt."""

    plist, attrs = points_with_attrs

    mypoints = Points(plist, attributes=attrs)

    # export to rmsattr
    usefile = tmp_path / "test_roundtrip.rmsattr"
    mypoints.to_file(usefile, fformat="rms_attr", attributes=True)

    usefile = tmp_path / "test_roundtrip2.rmsattr"

    with pytest.raises(ValueError):
        # should raise ValueError if attributes not in dataframe
        mypoints.to_file(usefile, fformat="rms_attr", attributes=["nosuchattr"])

    with pytest.raises(TypeError):
        mypoints.to_file(usefile, fformat="rms_attr", attributes="nosuchattr")


@pytest.mark.bigtest
def test_import_rmsattr_format(testdata_path, tmp_path):
    """Import points with attributes from RMS attr format."""

    test_points_path = testdata_path / POINTSET3
    orig_points = xtgeo.points_from_file(test_points_path, fformat="rms_attr")

    export_path = tmp_path / "attrs.rmsattr"
    orig_points.to_file(export_path, fformat="rms_attr")

    reloaded_points = xtgeo.points_from_file(export_path, fformat="rms_attr")
    pd.testing.assert_frame_equal(
        orig_points.get_dataframe(), reloaded_points.get_dataframe()
    )


def test_export_points_rmsattr(testdata_path, tmp_path):
    """Export XYZ points to file, as rmsattr."""

    mypoints = xtgeo.points_from_file(
        testdata_path / POINTSET4
    )  # should guess based on extesion
    output_path = tmp_path / "poi_export.rmsattr"

    mypoints.to_file(output_path, fformat="rms_attr")
    mypoints2 = xtgeo.points_from_file(output_path)

    assert mypoints.get_dataframe()["Seg"].equals(mypoints2.get_dataframe()["Seg"])

    np.testing.assert_array_almost_equal(
        mypoints.get_dataframe()["MyNum"].values,
        mypoints2.get_dataframe()["MyNum"].values,
    )
