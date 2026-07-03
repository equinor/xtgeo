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
    """Test that explicit and auto-detected ``fformat`` produce identical results.

    What is tested:
        ``points_from_file`` is called four times for each parametrized file: twice
        with the format specified explicitly and twice relying on auto-detection. All
        four resulting dataframes must be identical, confirming that format sniffing
        agrees with the explicit argument.

    Expected behaviour:
        All four dataframes compare equal.
    """
    # deprecated
    points1 = xtgeo.points_from_file(testdata_path / filename, fformat=fformat)
    points2 = xtgeo.points_from_file(testdata_path / filename)

    points3 = xtgeo.points_from_file(testdata_path / filename, fformat=fformat)
    points4 = xtgeo.points_from_file(testdata_path / filename)

    pd.testing.assert_frame_equal(points1.get_dataframe(), points2.get_dataframe())
    pd.testing.assert_frame_equal(points2.get_dataframe(), points3.get_dataframe())
    pd.testing.assert_frame_equal(points3.get_dataframe(), points4.get_dataframe())


def test_points_rmsattr_vs_csv(testdata_path):
    """Test reading points from RMS attr and CSV files.

    What is tested:
        The same point-set (``POINTSET4``) is loaded once with ``fformat="rms_attr"``
        and once with ``fformat="csv"`` (from a pre-exported CSV counterpart).
        Both readers must yield numerically equivalent dataframes.

    Expected behaviour:
        Both dataframes are numerically equivalent.
    """

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
    """Test reading and writing points to CSV file.

    What is tested:
        A CSV file is read with ``points_from_file``, written back to a new CSV with
        ``to_file``, then re-read. The initial and final dataframes are compared.

    Expected behaviour:
        Every row and column value is preserved through the write–read cycle.
    """

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
    What is tested:
        A ``Points`` instance with three attributes (``some_int``, ``sometxt``,
        ``somefloat``) is round-tripped through two formats:

        * ``poi`` (no attribute support) — only the XYZ columns survive.
        * ``csv`` — tested both without attributes and with all attributes.

    Expected behaviour:
        * The ``poi`` roundtrip preserves only the three coordinate columns.
        * The ``csv`` roundtrip without attributes produces the same XYZ-only frame.
        * The ``csv`` roundtrip with all attributes produces a frame equal to the
          original including all attribute columns.

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
    """Test roundtrip with selected attributes from file and back for csv/pq.

    What is tested:
        When ``attributes=["somefloat"]`` is passed to ``to_file`` for both ``csv``
        and ``parquet`` formats, only the requested attribute column is written. The
        other two attributes (``some_int``, ``sometxt``) must be absent from the
        reloaded dataframe.

    Expected behaviour:
        * The XYZ coordinate columns are identical to the original.
        * ``somefloat`` is present in the reloaded dataframe.
        * ``sometxt`` and ``some_int`` are absent from the reloaded dataframe.
    """

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
    """Test that a plain list of 3-tuples creates a valid ``Points`` instance.

    What is tested:
        The ``Points`` constructor accepts a list of ``(x, y, z)`` tuples and maps
        them directly into the internal dataframe without any conversion errors.

    Expected behaviour:
        The first X value equals ``234`` and the third Z value equals ``12``,
        confirming the mapping is positional and preserves the input order.
    """
    plist = [(234, 556, 11), (235, 559, 14), (255, 577, 12)]

    mypoints = Points(plist)

    x0 = mypoints.get_dataframe()["X_UTME"].values[0]
    z2 = mypoints.get_dataframe()["Z_TVDSS"].values[2]
    assert x0 == 234
    assert z2 == 12


def test_points_with_attrs_copy(points_with_attrs):
    """Test that ``copy()`` produces a fully independent deep copy.

    What is tested:
        A ``Points`` instance carrying three attribute columns (``some_int``,
        ``sometxt``, ``somefloat``) is duplicated with ``copy()``. The resulting
        object must hold a dataframe that is value-equal to the original and must
        include all attribute columns.

    Expected behaviour:
        ``assert_frame_equal`` passes for the original and the copy, and
        ``somefloat`` remains present in the copy's dataframe.
    """
    plist, attrs = points_with_attrs

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
    """Test that a ``Points`` instance stores exactly the input coordinates.

    What is tested:
        Property-based testing with Hypothesis generates random lists of equal-length
        rows (always 3 columns: X, Y, Z). The ``Points`` constructor must accept any
        such input and map the three columns to ``X_UTME``, ``Y_UTMN``, and
        ``Z_TVDSS`` without loss.

    Expected behaviour:
        ``pointset.nrow`` equals the number of input rows, and each coordinate column
        matches the corresponding column of the raw numpy array to floating-point
        precision.
    """
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
    """Verify that a file is read correctly from disk.

    What is tested:
        ``points_from_file`` loads a ``.poi`` file, relying on the file extension to
        auto-detect the format.

    Expected behaviour:
        The first X coordinate (``X_UTME``) matches the known reference value of
        approximately ``460842.434326`` to three decimal places.
    """

    mypoints = xtgeo.points_from_file(
        testdata_path / PFILE
    )  # should guess based on extesion

    x0 = mypoints.get_dataframe()["X_UTME"].values[0]
    assert x0 == pytest.approx(460842.434326, 0.001)


def test_import_from_dataframe(testdata_path):
    """Verify that a ``Points`` instance can be constructed from a pandas DataFrame.

    What is tested:
        A CSV file is read into a raw ``pd.DataFrame``, which is then passed to the
        ``Points`` constructor together with explicit ``xname``, ``yname``, ``zname``
        and ``attributes`` keyword arguments.

    Expected behaviour:
        * The resulting ``Points`` dataframe has the same X-column mean as the raw
          ``DataFrame``, confirming no rows were dropped or reordered.
        * Passing a non-existent ``xname`` raises a ``ValueError``.
    """

    dfr = pd.read_csv(testdata_path / CSV1, skiprows=3)

    attr = {"I": "int", "J": "int", "K": "int"}
    mypoints = xtgeo.Points(
        values=dfr, xname="X", yname="Y", zname="Z", attributes=attr
    )

    assert mypoints.get_dataframe().X.mean() == dfr.X.mean()

    with pytest.raises(ValueError):
        mypoints = Points(dfr, xname="NOTTHERE", yname="Y", zname="Z", attributes=attr)


def test_export_and_load_points(tmp_path):
    """Export XYZ points to file. Write to file and read back.

    What is tested:
        Three points are written to an ``.xyz`` file with ``to_file`` and then
        re-read with ``points_from_file``. Both the dataframe equality and the raw
        numeric values are checked.

    Expected behaviour:
        * The reloaded dataframe equals the original via
          ``pd.testing.assert_frame_equal``.
        * Flattening the dataframe produces the same sequence of numbers as
          flattening the original list of tuples.
    """
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
    """Export XYZ points to file, various formats

    What is tested:
        A real-world ``.rmsattr`` file is loaded, exported to a new file using
        ``fformat="rms_attr"``, and then reloaded. The XYZ columns of the original
        and the reloaded instance are compared.

    Expected behaviour:
        The first three columns (X, Y, Z) of the original and reloaded dataframes
        are equal, confirming coordinate data is preserved through the export.
    """

    test_points_path = testdata_path / POINTSET4
    orig_points = xtgeo.points_from_file(
        test_points_path
    )  # should guess based on extension

    export_path = tmp_path / "attrs.rmsattr"
    orig_points.to_file(export_path, fformat="rms_attr")

    reloaded_points = xtgeo.points_from_file(export_path)

    pd.testing.assert_frame_equal(
        orig_points.get_dataframe().iloc[:, :3],
        reloaded_points.get_dataframe().iloc[:, :3],
        check_dtype=False,
    )


def test_io_rms_attrs(points_with_attrs, tmp_path):
    """Test points with attributes from file and back using rms_attrs fmt.

    What is tested:
        A ``Points`` instance with three attributes is written and re-read using
        ``fformat="rms_attr"`` in two scenarios:

        * All attributes exported (``attributes=True``) — the full dataframe must
          survive the roundtrip.
        * A single attribute exported (``attributes=["somefloat"]``) — the reloaded
          file must contain only the XYZ columns plus ``somefloat``; ``some_int``
          must be absent.

    Expected behaviour:
        * Full-attribute roundtrip: dataframes are equal (``check_dtype=False``).
        * Partial-attribute roundtrip: XYZ columns match; ``some_int`` is absent.
    """

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
    """Test points with attributes from file and back using rms_attrs fmt.

    What is tested:
        After a valid export of all attributes, two invalid ``to_file`` calls are
        made:

        * ``attributes=["nosuchattr"]`` — references a column that does not exist.
        * ``attributes="nosuchattr"`` — passes a bare string instead of a list.

    Expected behaviour:
        * A non-existent attribute name raises ``ValueError``.
        * A string (not a list) passed as ``attributes`` raises ``TypeError``.
    """

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
    """Import points with attributes from RMS attr format.

    What is tested:
        A larger real-world ``.rmsattr`` file (``POINTSET3``) is imported, exported,
        and re-imported. The complete dataframe — including all attribute columns —
        must survive the roundtrip without any modification.

    Expected behaviour:
        ``pd.testing.assert_frame_equal`` passes for the original and reloaded
        dataframes, confirming exact preservation of all rows and columns.
    """

    test_points_path = testdata_path / POINTSET3
    orig_points = xtgeo.points_from_file(test_points_path, fformat="rms_attr")

    export_path = tmp_path / "attrs.rmsattr"
    orig_points.to_file(export_path, fformat="rms_attr")

    reloaded_points = xtgeo.points_from_file(export_path, fformat="rms_attr")
    pd.testing.assert_frame_equal(
        orig_points.get_dataframe(), reloaded_points.get_dataframe()
    )


def test_export_points_rmsattr(testdata_path, tmp_path):
    """Export XYZ points to file, as rmsattr..

    What is tested:
        ``POINTSET4`` is loaded (format auto-detected), exported as
        ``fformat="rms_attr"``, and re-read. Two specific attribute columns are
        checked: a categorical column (``Seg``) and a numeric column (``MyNum``).

    Expected behaviour:
        * The ``Seg`` column is identical in both instances.
        * The ``MyNum`` values match to floating-point precision via
          ``np.testing.assert_array_almost_equal``.
    """

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


def test_points_comparison_operators_compare_z_column():
    """Test that the overloaded rich-comparison operators act on the Z column only.

    What is tested:
        ``Points`` overrides ``__eq__``, ``__gt__``, ``__ge__``, ``__lt__`` and
        ``__le__`` so that comparing an instance against a scalar is interpreted as
        an element-wise comparison of the Z (depth) values. Three points are created
        with Z values 1.0, 3.0 and 5.0 (X and Y are constant and irrelevant here).

    Expected behavior:
        Each operator returns a pandas boolean Series aligned to the rows, comparing
        the scalar ``3.0`` against the Z column:
            * ``== 3.0`` -> [False, True, False]
            * ``>  3.0`` -> [False, False, True]
            * ``>= 3.0`` -> [False, True, True]
            * ``<  3.0`` -> [True, False, False]
            * ``<= 3.0`` -> [True, True, False]
        The result is a mask usable for boolean indexing, not a single bool.
    """
    points = Points([(0.0, 0.0, 1.0), (0.0, 0.0, 3.0), (0.0, 0.0, 5.0)])

    assert list(points == 3.0) == [False, True, False]
    assert list(points > 3.0) == [False, False, True]
    assert list(points >= 3.0) == [False, True, True]
    assert list(points < 3.0) == [True, False, False]
    assert list(points <= 3.0) == [True, True, False]


def test_points_random_populates_dataframe():
    """Test the private ``_random`` helper fills the instance with random points.

    What is tested:
        ``Points._random(nrandom)`` replaces the internal dataframe with ``nrandom``
        rows of pseudo-random coordinates generated via ``numpy.random.rand``. The
        test starts from an empty ``Points()`` and requests four random points.

    Expected behaviour:
        After the call the dataframe has exactly four rows, the three default XYZ
        column names (``X_UTME``, ``Y_UTMN``, ``Z_TVDSS``), and because the values
        come from ``rand`` every coordinate lies within the half-open range [0, 1).
    """
    points = Points()

    points._random(4)

    dataframe = points.get_dataframe()
    assert len(dataframe) == 4
    assert list(dataframe.columns) == ["X_UTME", "Y_UTMN", "Z_TVDSS"]
    assert dataframe.ge(0.0).all().all()
    assert dataframe.lt(1.0).all().all()


def test_points_get_boundary_delegates_to_base_implementation():
    """Test ``Points.get_boundary`` returns the 3D bounding box of all points.

    Expected behaviour:
        The method returns a six-tuple ordered as
        ``(xmin, xmax, ymin, ymax, zmin, zmax)``.
    """
    points = Points([(1.0, 8.0, 9.0), (-1.0, 3.0, 4.0), (2.0, 5.0, 7.0)])

    assert points.get_boundary() == (-1.0, 2.0, 3.0, 8.0, 4.0, 9.0)


def test_points_from_numpy_array_2d():
    """Test a 2D ``(n, 3)`` numpy array is accepted directly as XYZ input.

    What is tested:
        ``_from_list_like`` detects a numpy array of shape ``(n, 3)`` (exactly the
        three coordinate columns and no attributes) and maps the columns to
        X_UTME, Y_UTMN, Z_TVDSS.

    Expected behavior:
        Two rows are stored and the Z column reflects the third array column,
        i.e. ``[3.0, 6.0]``.
    """
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    points = Points(arr)

    assert points.nrow == 2
    assert points.get_dataframe()["X_UTME"].tolist() == [1.0, 4.0]
    assert points.get_dataframe()["Y_UTMN"].tolist() == [2.0, 5.0]
    assert points.get_dataframe()["Z_TVDSS"].tolist() == [3.0, 6.0]


def test_points_from_1d_numpy_array_raises():
    """Test a one-dimensional numpy array is rejected as invalid input.

    What is tested:
        ``_from_list_like`` requires a two-dimensional array (rows x columns). A flat
        1D array is ambiguous and must be refused.

    Expected behaviour:
        A ``ValueError`` is raised.
    """
    with pytest.raises(ValueError, match="two-dimensional"):
        Points(np.array([1.0, 2.0, 3.0]))


@pytest.mark.parametrize(
    "dataframe, match",
    [
        (
            pd.DataFrame({"X_UTME": [1.0, 2.0], "Y_UTMN": [20.0, 30.0]}),
            "zname",
        ),
        (
            pd.DataFrame(
                {"X_UTME": [1.0, 2.0], "Z_TVDSS": [3.0, 4.0], "Other": [2.0, 3.0]}
            ),
            "yname",
        ),
        (
            pd.DataFrame(
                {"Y_UTMN": [20.0, 30.0], "Z_TVDSS": [3.0, 4.0], "Other": [2.0, 3.0]}
            ),
            "xname",
        ),
    ],
)
def test_points_dataframe_missing_required_column_raises(dataframe, match):
    """Test that a dataframe missing a required coordinate column fails.

    What is tested:
        When a ``pandas.DataFrame`` is passed directly, the constructor requires all
        three configured ``xname``, ``yname``, and ``zname`` columns to exist.  Three
        cases are covered:

        * Only ``X_UTME`` and ``Y_UTMN`` present → missing ``zname`` fires.
        * Only ``X_UTME`` and ``Z_TVDSS`` present (plus an unrelated column) →
          missing ``yname`` fires.
        * Only ``Y_UTMN`` and ``Z_TVDSS`` present (plus an unrelated column) →
          missing ``xname`` fires.

    Expected behaviour:
        A ``ValueError`` is raised referencing the missing column name.
    """
    with pytest.raises(ValueError, match=match):
        Points(values=dataframe)


def test_points_get_dataframe_copy_semantics(points_with_attrs):
    """Test the ``copy`` flag of ``get_dataframe`` controls view vs. copy.

    What is tested:
        ``get_dataframe(copy=False)`` returns the live internal dataframe (a view),
        while ``get_dataframe(copy=True)`` (the default) returns an independent deep
        copy.

    Expected behaviour:
        Mutating the ``copy=False`` result is reflected back in the instance (Z
        becomes 999.0). Mutating a subsequent ``copy=True`` result does NOT change the
        instance, so the instance keeps the original value.
    """
    plist, attrs = points_with_attrs
    points = Points(plist, attributes=attrs)

    view = points.get_dataframe(copy=False)
    view.loc[0, "Z_TVDSS"] = 999.0
    assert points.get_dataframe()["Z_TVDSS"][0] == 999.0

    independent = points.get_dataframe(copy=True)
    independent.loc[0, "Z_TVDSS"] = 5.0
    assert points.get_dataframe()["Z_TVDSS"][0] == 999.0


def test_points_delete_columns_protects_coordinates(points_with_attrs):
    """Test ``delete_columns`` never removes any of the protected coordinate columns.

    What is tested:
        The coordinate columns returned by ``protected_columns`` — ``xname``
        (``X_UTME``), ``yname`` (``Y_UTMN``), and ``zname`` (``Z_TVDSS``) — are all
        guarded against deletion. The test attempts to delete all three at once.

    Expected behaviour:
        All three requests are silently ignored (a user warning is emitted internally
        for each) and every coordinate column remains present in the dataframe.
    """
    plist, attrs = points_with_attrs
    points = Points(plist, attributes=attrs)

    with pytest.warns(UserWarning) as warning_records:
        points.delete_columns([points.xname, points.yname, points.zname])

    warned_messages = [str(w.message) for w in warning_records]
    assert any(points.xname in msg for msg in warned_messages)
    assert any(points.yname in msg for msg in warned_messages)
    assert any(points.zname in msg for msg in warned_messages)

    columns = points.get_dataframe().columns
    assert points.xname in columns
    assert points.yname in columns
    assert points.zname in columns


def test_file_importer_drops_nan_rows(tmp_path):
    """Test undefined rows are dropped when importing a points file.

    What is tested:
        The XYZ reader treats ``999.00`` as the undefined sentinel (``na_values``),
        converting it to NaN, and ``_file_importer`` then drops any row containing
        NaN.

    Expected behaviour:
        Only the two fully-defined rows survive (``nrow == 2``) and the value 999.0
        never appears in the resulting Z column.
    """
    poifile = tmp_path / "with_undef.poi"
    # three rows, the middle one carrying a 999.0 (undefined) Z value
    poifile.write_text("1.0 2.0 3.0\n4.0 5.0 999.0\n7.0 8.0 9.0\n")

    points = xtgeo.points_from_file(poifile, fformat="poi")

    assert points.nrow == 2
    assert 999.0 not in points.get_dataframe()["Z_TVDSS"].values


def test_to_file_pfilter_selects_matching_rows(tmp_path):
    """Test ``pfilter`` keeps only rows whose attribute matches the listed values.

    What is tested:
        ``pfilter={col: [values]}`` keeps only the rows whose attribute column
        matches one of the listed values. Three points are tagged with a ``Region``
        of A, B, A; the filter selects only ``Region == "A"``.

    Expected behaviour:
        ``to_file`` returns 2 (the number of matching rows), and re-reading the
        file confirms only the two ``A`` rows were persisted.
    """
    plist = [
        (1.0, 2.0, 3.0, "A"),
        (4.0, 5.0, 6.0, "B"),
        (7.0, 8.0, 9.0, "A"),
    ]
    points = Points(plist, attributes={"Region": "str"})

    outfile = tmp_path / "filtered.rmsattr"
    ncount = points.to_file(
        outfile, fformat="rms_attr", attributes=True, pfilter={"Region": ["A"]}
    )

    assert ncount == 2
    reloaded = xtgeo.points_from_file(outfile, fformat="rms_attr")
    assert reloaded.nrow == 2


def test_to_file_pfilter_invalid_key_raises(tmp_path):
    """Test ``pfilter`` referencing a non-existent column raises ``KeyError``.

    What is tested:
        A ``pfilter`` referencing a column that does not exist raises ``KeyError``
        rather than silently producing an unfiltered or empty file.

    Expected behaviour:
        A ``KeyError`` is raised when the filter key ``NoSuch`` is not a valid
        column.
    """
    plist = [
        (1.0, 2.0, 3.0, "A"),
        (4.0, 5.0, 6.0, "B"),
        (7.0, 8.0, 9.0, "A"),
    ]
    points = Points(plist, attributes={"Region": "str"})

    with pytest.raises(KeyError):
        points.to_file(
            tmp_path / "bad.rmsattr",
            fformat="rms_attr",
            attributes=True,
            pfilter={"NoSuch": ["x"]},
        )


def test_data_reader_factory_unsupported_format_raises():
    """Test ``_data_reader_factory`` raises for an unsupported ``FileFormat``.

    What is tested:
        ``_data_reader_factory`` contains an explicit fallback that raises
        ``InvalidFileFormatError`` when it receives a ``FileFormat`` enum member
        that has no handler registered for the Points type (e.g.
        ``FileFormat.RMSWELL``).

    Expected behaviour:
        An ``InvalidFileFormatError`` is raised whose message contains the phrase
        ``"invalid for type Points"``.
    """
    from xtgeo.common.exceptions import InvalidFileFormatError
    from xtgeo.io._file import FileFormat
    from xtgeo.xyz.points import _data_reader_factory

    with pytest.raises(InvalidFileFormatError, match="invalid for type Points"):
        _data_reader_factory(FileFormat.RMSWELL)


def test_wells_importer_skips_none_wells_and_object_dtype_attrs():
    """Test ``_wells_importer`` skips ``None`` wells and maps object-dtype attrs.

    What is tested:
        1. A well whose ``get_zonation_points`` returns ``None`` is silently skipped
           (the ``if wp is not None:`` branch stays False) while a second well
           provides valid data, confirming the loop continues without appending.
        2. An object-dtype column named ``"zone"`` is assigned type ``"int"`` in
           the returned attributes dict.
        3. An object-dtype column whose name is neither ``"zone"`` nor contains
           ``"name"`` falls through to the ``"float"`` fallback in the attrs dict.

    Expected behaviour:
        * The returned dataframe contains only the rows from the non-``None`` well.
        * ``result["attributes"]["zone"]`` is ``"int"``.
        * ``result["attributes"]["Category"]`` is ``"float"``.
    """
    from unittest.mock import MagicMock

    from xtgeo.xyz.points import _wells_importer

    well_none = MagicMock()
    well_none.get_zonation_points.return_value = None

    df = pd.DataFrame(
        {
            "X_UTME": [1.0],
            "Y_UTMN": [2.0],
            "Z_TVDSS": [3.0],
            # object dtype + "zone" name maps to "int"
            "zone": pd.array(["shallow"], dtype=object),
            # object dtype without "name" in column maps to "float"
            "Category": pd.array(["sand"], dtype=object),
        }
    )
    well_valid = MagicMock()
    well_valid.get_zonation_points.return_value = df

    result = _wells_importer([well_none, well_valid])

    assert len(result["values"]) == 1
    assert result["attributes"]["zone"] == "int"
    assert result["attributes"]["Category"] == "float"


def test_wells_dfrac_importer_skips_none_wells():
    """Test ``_wells_dfrac_importer`` silently skips wells that return ``None``.

    What is tested:
        When ``get_fraction_per_zone`` returns ``None`` for a well, the importer
        skips that well and continues the loop rather than appending or raising.
        A second well provides valid data so ``pd.concat`` can succeed.

    Expected behaviour:
        The returned dataframe contains only the rows from the non-``None`` well.
    """
    from unittest.mock import MagicMock

    from xtgeo.xyz.points import _wells_dfrac_importer

    well_none = MagicMock()
    well_none.get_fraction_per_zone.return_value = None

    df = pd.DataFrame({"X_UTME": [1.0], "Y_UTMN": [2.0], "DFRAC": [0.5]})
    well_valid = MagicMock()
    well_valid.get_fraction_per_zone.return_value = df

    result = _wells_dfrac_importer(
        [well_none, well_valid],
        dlogname="Facies",
        dcodes=[1],
    )

    assert len(result["values"]) == 1


def test_points_get_xyz_arrays_returns_n3_numpy_array():
    """Test ``get_xyz_arrays`` returns coordinates as a ``(n, 3)`` numpy array.

    What is tested:
        ``Points.get_xyz_arrays`` delegates to the base ``XYZ`` implementation,
        which stacks the X, Y, and Z columns into a two-dimensional numpy array
        of shape ``(n, 3)``.

    Expected behaviour:
        For two points the result has shape ``(2, 3)`` and each column matches the
        original X, Y, Z values in order.
    """
    points = Points([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)])

    arr = points.get_xyz_arrays()

    assert arr.shape == (2, 3)
    np.testing.assert_array_equal(arr[:, 0], [1.0, 4.0])
    np.testing.assert_array_equal(arr[:, 1], [2.0, 5.0])
    np.testing.assert_array_equal(arr[:, 2], [3.0, 6.0])


def test_points_dataframe_property_getter_warns_and_returns_dataframe():
    """Test deprecated ``dataframe`` getter emits warning and returns dataframe.

    What is tested:
        Accessing ``Points.dataframe`` triggers a ``PendingDeprecationWarning``
        and still returns the underlying dataframe with the expected values.

    Expected behaviour:
        The warning is emitted and the returned dataframe contains the original
        coordinates.
    """
    points = Points([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)])

    with pytest.warns(PendingDeprecationWarning, match="get_dataframe"):
        dataframe = points.dataframe

    assert list(dataframe.columns) == ["X_UTME", "Y_UTMN", "Z_TVDSS"]
    assert dataframe["Z_TVDSS"].tolist() == [3.0, 6.0]


def test_points_dataframe_property_setter_warns_and_sets_deep_copy():
    """Test deprecated ``dataframe`` setter emits warning and updates data.

    What is tested:
        Assigning through ``Points.dataframe`` triggers a
        ``PendingDeprecationWarning`` and routes through ``set_dataframe``.

    Expected behaviour:
        The warning is emitted, the points dataframe is updated to assigned
        values, and later mutation of the source dataframe does not affect the
        stored dataframe (deep-copy semantics).
    """
    points = Points([(1.0, 2.0, 3.0)])
    new_df = pd.DataFrame({"X_UTME": [10.0], "Y_UTMN": [11.0], "Z_TVDSS": [12.0]})

    with pytest.warns(PendingDeprecationWarning, match="set_dataframe"):
        points.dataframe = new_df

    assert points.get_dataframe(copy=False)["Z_TVDSS"].tolist() == [12.0]

    new_df.loc[0, "Z_TVDSS"] = 99.0
    assert points.get_dataframe(copy=False)["Z_TVDSS"].tolist() == [12.0]
