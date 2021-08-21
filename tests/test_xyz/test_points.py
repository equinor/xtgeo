import itertools
import pathlib
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
import xtgeo
from hypothesis import given, settings
from hypothesis import strategies as st
from xtgeo.xyz import Points

PFILE = pathlib.Path("points/eme/1/emerald_10_random.poi")
POINTSET2 = pathlib.Path("points/reek/1/pointset2.poi")
POINTSET3 = pathlib.Path("points/battle/1/many.rmsattr")
POINTSET4 = pathlib.Path("points/reek/1/poi_attr.rmsattr")
CSV1 = pathlib.Path("3dgrids/etc/gridqc1_rms_cellcenter.csv")


def test_initialise_simple_listlike():
    """Make points list-like list of tuples or list of lists"""

    plist1 = [(234, 556, 11), (235, 559, 14), (255, 577, 12)]
    plist2 = [[234, 556, 11], [235, 559, 14], [255, 577, 12]]

    mypoints1 = Points(values=plist1)
    mypoints2 = Points(plist2)

    assert mypoints1.dataframe["X_UTME"].values[0] == 234
    assert mypoints1.dataframe["Z_TVDSS"].values[2] == 12

    assert mypoints2.dataframe["X_UTME"].values[0] == 234
    assert mypoints2.dataframe["Z_TVDSS"].values[2] == 12

    assert mypoints1.dataframe.equals(mypoints2.dataframe)


def test_initialise_simple_numpy_or_dataframe():
    """Make points from numpy or pandas dataframe"""

    nparr = np.array([(234, 556, 11), (235, 559, 14), (255, 577, 12)])
    dfr = pd.DataFrame(nparr)
    list_np = [
        np.array((234, 556, 11)),
        np.array((235, 559, 14)),
        np.array((255, 577, 12)),
    ]

    mypoints1 = Points(values=nparr)
    mypoints2 = Points(values=dfr)
    mypoints3 = Points(values=list_np)

    # without 'values' key is also allowed
    mypoints4 = Points(nparr)
    mypoints5 = Points(dfr)
    mypoints6 = Points(list_np)

    assert mypoints1.dataframe["X_UTME"].values[0] == 234
    assert mypoints1.dataframe["Z_TVDSS"].values[2] == 12

    assert mypoints1.dataframe.equals(mypoints2.dataframe)
    assert mypoints1.dataframe.equals(mypoints3.dataframe)
    assert mypoints1.dataframe.equals(mypoints4.dataframe)
    assert mypoints1.dataframe.equals(mypoints5.dataframe)
    assert mypoints1.dataframe.equals(mypoints6.dataframe)


def test_initialise_from_file(testpath):
    """Make points from a file, internally a class method"""
    sp1 = xtgeo.points_from_file(testpath / PFILE)
    assert sp1.dataframe["X_UTME"].values[0] == pytest.approx(460842.434326)

    # rms_attrs format
    sp2 = xtgeo.points_from_file(testpath / POINTSET4)
    assert sp2.dataframe["X_UTME"].values[0] == pytest.approx(461288.81485)
    assert sp2.dataframe["MyNum"].values[3] == 22.0

    # old style (shall give DeprecationWarning)
    mypoints = Points(testpath / PFILE)

    assert mypoints.dataframe["X_UTME"].values[0] == pytest.approx(460842.434326)


@st.composite
def list_of_equal_length_lists(draw):
    list_len = draw(st.integers(min_value=3, max_value=3))
    fixed_len_list = st.lists(
        st.floats(allow_nan=False, allow_infinity=False),
        min_size=list_len,
        max_size=list_len,
    )
    return draw(st.lists(fixed_len_list, min_size=1))


@settings(deadline=400)
@given(list_of_equal_length_lists())
def test_create_pointset(points):
    """Create randomly generated points and verify content."""
    points = np.array(points)
    pointset = Points(points)

    assert len(points) == pointset.nrow

    np.testing.assert_array_almost_equal(pointset.dataframe["X_UTME"], points[:, 0])
    np.testing.assert_array_almost_equal(pointset.dataframe["Y_UTMN"], points[:, 1])
    np.testing.assert_array_almost_equal(pointset.dataframe["Z_TVDSS"], points[:, 2])


def test_import_from_dataframe_old(testpath):
    """Import Points via Pandas dataframe."""

    mypoints = Points()
    dfr = pd.read_csv(testpath / CSV1, skiprows=3)
    attr = {"IX": "I", "JY": "J", "KZ": "K"}
    mypoints.from_dataframe(dfr, east="X", north="Y", tvdmsl="Z", attributes=attr)

    assert mypoints.dataframe.X_UTME.mean() == dfr.X.mean()

    with pytest.raises(ValueError):
        mypoints.from_dataframe(
            dfr, east="NOTTHERE", north="Y", tvdmsl="Z", attributes=attr
        )


def test_import_from_dataframe_classmethod(testpath):
    """Import Points via Pandas dataframe using classmethod from v 2.16."""

    dfr = pd.read_csv(testpath / CSV1, skiprows=3)
    attr = {"IX": "I", "JY": "J", "KZ": "K"}
    mypoints = xtgeo.points_from_dataframe(
        dfr, east="X", north="Y", tvdmsl="Z", zname="SomeZ", attributes=attr
    )
    assert mypoints.dataframe.X_UTME.mean() == dfr.X.mean()

    with pytest.raises(KeyError):
        mypoints = xtgeo.points_from_dataframe(
            dfr, east="NOTTHERE", north="Y", tvdmsl="Z", attributes=attr
        )


def test_export_and_load_points(tmp_path):
    """Export XYZ points to file."""
    plist = [(1.0, 1.0, 1.0), (2.0, 3.0, 4.0), (5.0, 6.0, 7.0)]
    test_points = Points(plist)

    export_path = tmp_path / "test_points.xyz"
    test_points.to_file(export_path)

    exported_points = Points(export_path)

    pd.testing.assert_frame_equal(test_points.dataframe, exported_points.dataframe)
    assert list(itertools.chain.from_iterable(plist)) == list(
        test_points.dataframe.values.flatten()
    )


def test_export_load_rmsformatted_points(testpath, tmp_path):
    """Export XYZ points to file, various formats."""

    test_points_path = testpath / POINTSET4
    orig_points = Points(test_points_path)  # should guess based on extesion

    export_path = tmp_path / "attrs.rmsattr"
    orig_points.to_file(export_path, fformat="rms_attr")

    reloaded_points = Points(export_path)

    pd.testing.assert_frame_equal(orig_points.dataframe, reloaded_points.dataframe)


def test_import_rmsattr_format(testpath, tmp_path):
    """Import points with attributes from RMS attr format."""

    orig_points = Points()

    test_points_path = testpath / POINTSET3
    orig_points.from_file(test_points_path, fformat="rms_attr")

    # export_path = tmp_path / "attrs.rmsattr"
    export_path = "attrs.rmsattr"
    orig_points.to_file(export_path, fformat="rms_attr")

    # reloaded_points = Points()
    # reloaded_points.from_file(export_path, fformat="rms_attr")
    # pd.testing.assert_frame_equal(orig_points.dataframe, reloaded_points.dataframe)

    # assert list(orig_points.attributes) == ["FaultBlock", "FaultTag", "VerticalSep"]


def test_export_points_rmsattr(testpath, tmp_path):
    """Export XYZ points to file, as rmsattr."""

    mypoints = Points(testpath / POINTSET4)  # should guess based on extesion
    output_path = tmp_path / "poi_export.rmsattr"

    mypoints.to_file(output_path, fformat="rms_attr")
    mypoints2 = Points(output_path)

    assert mypoints.dataframe["Seg"].equals(mypoints2.dataframe["Seg"])
    assert mypoints.dataframe["MyNum"].equals(mypoints2.dataframe["MyNum"])


def test_get_points_set2(tmp_path):
    """Points with values and an attribute."""
    values = [
        (1.0, 2.0, 44.0, "some"),
        (1.1, 2.1, 45.0, "attr"),
        (1.2, 2.2, 46.0, "here"),
        (1.3, 2.3, 47.0, "my"),
        (1.4, 2.4, 48.0, "friend"),
    ]
    attrs = OrderedDict()
    attrs["some"] = "str"
    poi = xtgeo.Points(values=values, attributes=attrs)
    assert poi.dataframe["some"][3] == "my"

    mynumpy = np.array(values)
    poi = xtgeo.Points(values=mynumpy, attributes=attrs)
    assert poi.dataframe["some"][3] == "my"

    poi.to_file(tmp_path / "poifile.rms_attr", fformat="rms_attr")
    poi2 = xtgeo.points_from_file(tmp_path / "poifile.rms_attr", fformat="rms_attr")

    assert poi.dataframe["some"][3] == "my"
    assert poi.dataframe.equals(poi2.dataframe)

    dataframe = poi.dataframe.copy()
    dataframe.rename(columns={dataframe.columns[0]: "myx"}, inplace=True)
    dataframe.astype({"myx": np.float32})
    poi3 = xtgeo.Points(values=dataframe, attributes=attrs)
    assert poi.dataframe.equals(poi3.dataframe)


def test_append(testpath):
    """Append two points sets."""

    test_points_path = testpath / POINTSET3
    psetx = xtgeo.points_from_file(test_points_path, fformat="rms_attr")
    pset1 = psetx.copy()
    pset2 = psetx.copy()

    pset1.dataframe = psetx.dataframe.take([0, 1, 2, 3, 4, 5])
    pset2.dataframe = psetx.dataframe.take([50, 51, 52, 53, 54, 55])

    pset1.append(pset2)
    assert pset1.dataframe.iat[10, 2] == 61.042
    assert "FaultBlock" not in pset1.dataframe


def test_append_with_attrs(testpath):
    """Append two points sets."""

    test_points_path = testpath / POINTSET3
    psetx = xtgeo.points_from_file(test_points_path, fformat="rms_attr")
    pset1 = psetx.copy()
    pset2 = psetx.copy()

    pset1.dataframe = psetx.dataframe.take([0, 1, 2, 3, 4, 5])
    pset2.dataframe = psetx.dataframe.take([50, 51, 52, 53, 54, 55])

    pset1.append(pset2, attributes=["FaultBlock", "FaultTag"])
    assert pset1.dataframe.iat[10, 2] == 61.042
    assert "FaultBlock" in pset1.dataframe
