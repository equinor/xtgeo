import itertools
import pathlib

import numpy as np
import pandas as pd
import pytest
import xtgeo
from hypothesis import given, settings
from hypothesis import strategies as st
from xtgeo.xyz import Points

PFILE = pathlib.Path("points/eme/1/emerald_10_random.poi")
PFILE2 = pathlib.Path("polygons/reek/1/top_upper_reek_faultpoly.zmap")
POINTSET2 = pathlib.Path("points/reek/1/pointset2.poi")
POINTSET3 = pathlib.Path("points/battle/1/many.rmsattr")
POINTSET4 = pathlib.Path("points/reek/1/poi_attr.rmsattr")
CSV1 = pathlib.Path("3dgrids/etc/gridqc1_rms_cellcenter.csv")


@pytest.mark.parametrize(
    "filename, fformat",
    [
        (PFILE, "poi"),
        (POINTSET2, "poi"),
        (PFILE2, "zmap"),
        (POINTSET3, "rmsattr"),
        (POINTSET4, "rmsattr"),
    ],
)
def test_points_from_file_alternatives(testpath, filename, fformat):
    # deprecated
    points1 = Points(testpath / filename, fformat=fformat)
    points2 = Points()
    points2.from_file(testpath / filename, fformat=fformat)

    points3 = xtgeo.points_from_file(testpath / filename, fformat=fformat)
    points4 = xtgeo.points_from_file(testpath / filename)

    pd.testing.assert_frame_equal(points1.dataframe, points2.dataframe)
    pd.testing.assert_frame_equal(points2.dataframe, points3.dataframe)
    pd.testing.assert_frame_equal(points3.dataframe, points4.dataframe)


def test_points_from_list_of_tuples():
    plist = [(234, 556, 11), (235, 559, 14), (255, 577, 12)]

    mypoints = Points(plist)

    x0 = mypoints.dataframe["X_UTME"].values[0]
    z2 = mypoints.dataframe["Z_TVDSS"].values[2]
    assert x0 == 234
    assert z2 == 12


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
    pointset = Points(points)
    points = np.array(points)

    assert len(points) == pointset.nrow

    np.testing.assert_array_almost_equal(pointset.dataframe["X_UTME"], points[:, 0])
    np.testing.assert_array_almost_equal(pointset.dataframe["Y_UTMN"], points[:, 1])
    np.testing.assert_array_almost_equal(pointset.dataframe["Z_TVDSS"], points[:, 2])


def test_import(testpath):
    """Import XYZ points from file."""

    mypoints = Points(testpath / PFILE)  # should guess based on extesion

    x0 = mypoints.dataframe["X_UTME"].values[0]
    assert x0 == pytest.approx(460842.434326, 0.001)


def test_import_from_dataframe(testpath):
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


@pytest.mark.bigtest
def test_import_rmsattr_format(testpath, tmp_path):
    """Import points with attributes from RMS attr format."""

    orig_points = Points()

    test_points_path = testpath / POINTSET3
    orig_points.from_file(test_points_path, fformat="rms_attr")

    export_path = tmp_path / "attrs.rmsattr"
    orig_points.to_file(export_path, fformat="rms_attr")

    reloaded_points = Points()
    reloaded_points.from_file(export_path, fformat="rms_attr")
    pd.testing.assert_frame_equal(orig_points.dataframe, reloaded_points.dataframe)


def test_export_points_rmsattr(testpath, tmp_path):
    """Export XYZ points to file, as rmsattr."""

    mypoints = Points(testpath / POINTSET4)  # should guess based on extesion
    output_path = tmp_path / "poi_export.rmsattr"

    mypoints.to_file(output_path, fformat="rms_attr")
    mypoints2 = Points(output_path)

    assert mypoints.dataframe["Seg"].equals(mypoints2.dataframe["Seg"])
    assert mypoints.dataframe["MyNum"].equals(mypoints2.dataframe["MyNum"])
