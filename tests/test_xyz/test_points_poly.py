import pathlib
import sys

import pytest

import xtgeo
from xtgeo.xyz import Points, Polygons

POLSET2 = pathlib.Path("polygons/reek/1/polset2.pol")
POINTSET2 = pathlib.Path("points/reek/1/pointset2.poi")

SMALL_POLY_INNER = [
    (3.0, 3.0, 0.0, 0),
    (5.0, 3.0, 0.0, 0),
    (5.0, 5.0, 0.0, 0),
    (3.0, 5.0, 0.0, 0),
    (3.0, 3.0, 0.0, 0),
]

SMALL_POLY_OUTER = [
    (2.0, 2.0, 0.0, 1),
    (6.0, 2.0, 0.0, 1),
    (6.0, 6.0, 0.0, 1),
    (2.0, 6.0, 0.0, 1),
    (2.0, 2.0, 0.0, 1),
]

SMALL_POLY_OVERLAP_INNER = [
    (4.0, 4.0, 0.0, 2),
    (6.0, 4.0, 0.0, 2),
    (6.0, 6.0, 0.0, 2),
    (4.0, 6.0, 0.0, 2),
    (4.0, 4.0, 0.0, 2),
]

LARGE_POLY_ENCOMPASS = [
    (0.0, 0.0, 0.0, 10),
    (10.0, 0.0, 0.0, 10),
    (10.0, 10.0, 0.0, 10),
    (0.0, 10.0, 0.0, 10),
    (0.0, 0.0, 0.0, 10),
]

LARGE_POLY_SHIFTED = [
    (15.0, 0.0, 0.0, 11),
    (30.0, 0.0, 0.0, 11),
    (30.0, 10.0, 0.0, 11),
    (15.0, 10.0, 0.0, 11),
    (15.0, 0.0, 0.0, 11),
]


@pytest.fixture(name="reekset")
def fixture_reekset(testdata_path):
    """Read a test set from disk."""
    poi = xtgeo.points_from_file(testdata_path / POINTSET2)
    pol = xtgeo.polygons_from_file(testdata_path / POLSET2)
    return poi, pol


def test_mark_points_polygon(reekset):
    """Import XYZ points and get marks of points inside or outside"""
    poi, pol = reekset
    poi2 = poi.copy()
    # remove points in polygons
    poi2.mark_in_polygons(pol, name="_ptest", inside_value=1, outside_value=0)
    assert poi2.get_dataframe()["_ptest"].values.sum() == 11

    # a.a, but allow multiple polygons (reuse same here)
    poi3 = poi.copy()
    poi3.mark_in_polygons([pol, pol], name="_ptest", inside_value=1, outside_value=0)
    assert poi3.get_dataframe()["_ptest"].values.sum() == 11


def test_mark_points_polygon_protected(reekset):
    """Raise a ValuError if a protected name is attempted"""
    poi, pol = reekset
    poi2 = poi.copy()
    with pytest.raises(ValueError, match="is protected and cannot be used"):
        poi2.mark_in_polygons(pol, name=poi.xname, inside_value=1, outside_value=0)


def test_points_in_polygon(reekset):
    """Import XYZ points and do operations if inside or outside"""

    _poi, pol = reekset
    poi = _poi.copy()
    assert poi.nrow == 30

    # remove points in polygon
    poi.operation_polygons(pol, 0, opname="eli", version=2)

    assert poi.nrow == 19

    # remove points outside polygon
    poi.operation_polygons(pol, 0, opname="eli", inside=False, version=1)
    assert poi.nrow == 0

    # new behaviour with version = 2:
    poi = _poi.copy()  # refresh
    poi.operation_polygons(pol, 0, opname="eli", inside=False, version=2)

    assert poi.nrow == 11


def test_add_inside_old_new_behaviour(reekset):
    """Add inside using old and new class behaviour"""
    _poi, pol = reekset
    poi1 = _poi.copy()
    dataframe1 = poi1.get_dataframe()
    dataframe1.Z_TVDSS = 0.0
    poi2 = _poi.copy()
    dataframe2 = poi2.get_dataframe()
    dataframe2.Z_TVDSS = 0.0
    poi1.set_dataframe(dataframe1)
    poi2.set_dataframe(dataframe2)

    with pytest.warns(DeprecationWarning):
        poi1.add_inside(pol, 1)

    zvec = poi1.get_dataframe()["Z_TVDSS"].values
    assert 2.0 in zvec.tolist()  # will be doubling where polygons overlap!
    zvec = zvec[zvec < 1]
    assert zvec.size == 19

    poi2.add_inside_polygons(pol, 1)
    zvec = poi2.get_dataframe()["Z_TVDSS"].values
    assert 2.0 not in zvec.tolist()  # here NO doubling where polygons overlap!
    zvec = zvec[zvec < 1]
    assert zvec.size == 19


def test_check_single_point_in_polygon():
    for ver in (1, 2):
        pol = Polygons(SMALL_POLY_INNER)
        poi = Points([(4.0, 4.0, 4.0)])
        poi.operation_polygons(pol, value=1, opname="eli", inside=False, version=ver)
        assert len(poi.get_dataframe()) == 1


def test_check_multi_point_single_polygon():
    for ver in (1, 2):
        pol = Polygons(SMALL_POLY_INNER)
        poi = Points([(6.0, 4.0, 4.0), (4.0, 4.0, 4.0), (4.0, 6.0, 4.0)])
        assert len(poi.get_dataframe()) == 3

        poi.operation_polygons(pol, value=1, opname="eli", inside=False, version=ver)
        assert len(poi.get_dataframe()) == 1

        poi.operation_polygons(pol, value=1, opname="eli", inside=True, version=ver)
        assert len(poi.get_dataframe()) == 0


def test_check_multi_point_single_polygon_zdir():
    pol = Polygons(SMALL_POLY_INNER)
    poi = Points([(4.0, 4.0, 0.0), (4.0, 4.0, 4.0), (4.0, 4.0, 8.0)])
    assert len(poi.get_dataframe()) == 3

    # Note z-direction has no effect. All points are deleted.
    poi.operation_polygons(pol, value=1, opname="eli", inside=True)
    assert len(poi.get_dataframe()) == 0


def test_check_multi_point_multi_polyon_inside_op():
    pol = Polygons(SMALL_POLY_INNER + LARGE_POLY_SHIFTED)
    # Two points in small cube, one in large cube, one outside
    for ver in (1, 2):
        poi = Points(
            [(4.0, 4.0, 0.0), (4.5, 4.0, 0.0), (7.0, 7.0, 0.0), (20.0, 5.0, 0.0)]
        )
        assert len(poi.get_dataframe()) == 4

        poi.operation_polygons(pol, value=1, opname="eli", inside=True, version=ver)
        assert len(poi.get_dataframe()) == 1

        poi = Points(
            [(4.0, 4.0, 0.0), (4.5, 4.0, 0.0), (7.0, 7.0, 0.0), (20.0, 5.0, 0.0)]
        )
        poi.operation_polygons(pol, value=1, opname="eli", inside=False, version=ver)
        assert len(poi.get_dataframe()) == 0 if ver == 1 else 3  # dependent on version!


def test_check_multi_point_multi_polyon_outside_op():
    pol = Polygons(SMALL_POLY_INNER + LARGE_POLY_SHIFTED)
    # Two points in small cube, one in large cube, one outside
    poi = Points([(4.0, 4.0, 0.0), (4.5, 4.0, 0.0), (7.0, 7.0, 0.0), (20.0, 5.0, 0.0)])
    assert len(poi.get_dataframe()) == 4

    # Note the operation will loop over the polygons if version 1, and hence remove the
    # points in the small polygon when considering the large polygon, and vice versa
    poi.operation_polygons(pol, value=1, opname="eli", inside=False, version=1)
    assert len(poi.get_dataframe()) == 0

    # Fixed in version 2
    poi = Points([(4.0, 4.0, 0.0), (4.5, 4.0, 0.0), (7.0, 7.0, 0.0), (20.0, 5.0, 0.0)])
    poi.operation_polygons(pol, value=1, opname="eli", inside=False, version=2)
    assert len(poi.get_dataframe()) == 3


def test_check_single_polygon_in_single_polygon():
    for ver in (1, 2):
        inner_pol = Polygons(SMALL_POLY_INNER)
        outer_pol = Polygons(SMALL_POLY_OUTER)

        # Do not delete inner_pol when specified to delete poly outside outer polygon
        inner_pol.operation_polygons(
            outer_pol, value=1, opname="eli", inside=False, version=ver
        )
        assert len(inner_pol.get_dataframe()) == 5

        # Do not delete outer_pol when specified to delete polygons inside inner polygon
        outer_pol.operation_polygons(
            inner_pol, value=1, opname="eli", inside=True, version=ver
        )
        assert len(outer_pol.get_dataframe()) == 5

        inner_pol.operation_polygons(
            outer_pol, value=1, opname="eli", inside=True, version=ver
        )
        assert len(inner_pol.get_dataframe()) == 0

        inner_pol = Polygons(SMALL_POLY_INNER)

        outer_pol.operation_polygons(
            inner_pol, value=1, opname="eli", inside=False, version=ver
        )
        assert len(outer_pol.get_dataframe()) == 0


def test_check_multi_polygon_in_single_polygon():
    for ver in (1, 2):
        inner_pols = Polygons(SMALL_POLY_INNER + SMALL_POLY_OVERLAP_INNER)
        outer_pol = Polygons(SMALL_POLY_OUTER)

        inner_pols.operation_polygons(
            outer_pol, value=1, opname="eli", inside=True, version=ver
        )
        assert len(inner_pols.get_dataframe()) == 0


def test_operation_inclusive_polygon():
    for ver in (1, 2):
        pol = Polygons(SMALL_POLY_INNER)
        # We place a point on the edge of a polygon
        poi = Points([(4.0, 4.0, 0.0)])
        poi.operation_polygons(pol, value=1, opname="eli", inside=True, version=ver)
        assert len(poi.get_dataframe()) == 0

        # We place a point on a corner of a polygon
        poi = Points([(3.0, 3.0, 0.0)])
        poi.operation_polygons(pol, value=1, opname="eli", inside=True)
        assert len(poi.get_dataframe()) == 0


def test_polygons_overlap():
    for ver in (1, 2):
        pol = Polygons(SMALL_POLY_INNER + SMALL_POLY_OVERLAP_INNER)
        # The Four points are placed: within first poly, within the overlap, within the
        # second poly, outside both poly
        poi = Points(
            [(3.5, 3.5, 0.0), (4.5, 4.5, 0.0), (5.5, 5.5, 0.0), (6.5, 6.5, 0.0)]
        )
        poi.operation_polygons(pol, value=1, opname="eli", inside=True, version=ver)
        assert len(poi.get_dataframe()) == 1


@pytest.mark.parametrize(
    "oper, expected", [("add", 12), ("sub", 8), ("div", 5), ("mul", 20), ("set", 2)]
)
def test_oper_single_point_in_polygon(oper, expected):
    for ver in (1, 2):
        pol = Polygons(SMALL_POLY_INNER)
        poi = Points([(4.0, 4.0, 10.0)])
        # operators work on z-values
        poi.operation_polygons(pol, value=2, opname=oper, inside=True, version=ver)
        assert poi.get_dataframe()[poi.zname].values[0] == expected


@pytest.mark.parametrize(
    "oper, expected",
    [
        ("add", [12, 14, 12, 10]),
        ("sub", [8, 6, 8, 10]),
        ("div", [5, 2.5, 5, 10]),
        ("mul", [20, 40, 20, 10]),
        ("set", [2, 2, 2, 10]),
    ],
)
def test_oper_points_outside_overlapping_polygon_v1(oper, expected):
    pol = Polygons(SMALL_POLY_INNER + SMALL_POLY_OVERLAP_INNER)
    # The Four points are placed: within first poly, within the overlap, within the
    # second poly, outside both poly
    poi = Points(
        [
            (3.5, 3.5, 10.0),
            (4.5, 4.5, 10.0),
            (5.5, 5.5, 10.0),
            (6.5, 6.5, 10.0),
        ]
    )
    # Operations are performed n times, where n reflects the number of polygons a
    # point is within
    poi.operation_polygons(pol, value=2, opname=oper, inside=True, version=1)
    assert list(poi.get_dataframe()[poi.zname].values) == expected


@pytest.mark.parametrize(
    "oper, expected",
    [
        ("add", [12, 12, 12, 10]),
        ("sub", [8, 8, 8, 10]),
        ("div", [5, 5, 5, 10]),
        ("mul", [20, 20, 20, 10]),
        ("set", [2, 2, 2, 10]),
    ],
)
def test_oper_points_outside_overlapping_polygon_v2(oper, expected):
    # different expected values her for version 2, compared with version 1
    pol = Polygons(SMALL_POLY_INNER + SMALL_POLY_OVERLAP_INNER)
    # The Four points are placed: within first poly, within the overlap, within the
    # second poly, outside both poly
    poi = Points(
        [
            (3.5, 3.5, 10.0),
            (4.5, 4.5, 10.0),
            (5.5, 5.5, 10.0),
            (6.5, 6.5, 10.0),
        ]
    )
    # Operations are performed n times, where n reflects the number of polygons a
    # point is within
    poi.operation_polygons(pol, value=2, opname=oper, inside=True, version=2)
    assert list(poi.get_dataframe()[poi.zname].values) == expected


@pytest.mark.parametrize(
    "oper, expected",
    [
        ("add", [12, 10, 12, 14]),
        ("sub", [8, 10, 8, 6]),
        ("div", [5, 10, 5, 2.5]),
        ("mul", [20, 10, 20, 40]),
        ("set", [2, 10, 2, 2]),
    ],
)
def test_oper_points_inside_overlapping_polygon_v1(oper, expected):
    pol = Polygons(SMALL_POLY_INNER + SMALL_POLY_OVERLAP_INNER)
    # The Four points are placed: within first poly, within the overlap, within the
    # second poly, outside both poly
    poi = Points(
        [(3.5, 3.5, 10.0), (4.5, 4.5, 10.0), (5.5, 5.5, 10.0), (6.5, 6.5, 10.0)]
    )
    # Operations are performed n times, where n reflects the number of polygons a
    # point is outside
    poi.operation_polygons(pol, value=2, opname=oper, inside=False, version=1)
    assert list(poi.get_dataframe()[poi.zname].values) == expected


@pytest.mark.parametrize(
    "oper, expected",
    [
        ("add", [10, 10, 10, 12]),
        ("sub", [10, 10, 10, 8]),
        ("div", [10, 10, 10, 5]),
        ("mul", [10, 10, 10, 20]),
        ("set", [10, 10, 10, 2]),
    ],
)
def test_oper_points_inside_overlapping_polygon_v2(oper, expected):
    pol = Polygons(SMALL_POLY_INNER + SMALL_POLY_OVERLAP_INNER)
    # The Four points are placed: within first poly, within the overlap, within the
    # second poly, outside both poly
    poi = Points(
        [(3.5, 3.5, 10.0), (4.5, 4.5, 10.0), (5.5, 5.5, 10.0), (6.5, 6.5, 10.0)]
    )
    # Operations are performed n times, where n reflects the number of polygons a
    # point is outside
    poi.operation_polygons(pol, value=2, opname=oper, inside=False, version=2)
    assert list(poi.get_dataframe()[poi.zname].values) == expected


def test_add_inside_polygons_etc():
    pol = Polygons(SMALL_POLY_INNER + SMALL_POLY_OVERLAP_INNER)
    # The Four points are placed: within first poly, within the overlap, within the
    # second poly, outside both poly
    _poi = Points(
        [(3.5, 3.5, 10.0), (4.5, 4.5, 10.0), (5.5, 5.5, 10.0), (6.5, 6.5, 10.0)]
    )
    poi = _poi.copy()
    poi.add_inside_polygons(pol, 10.0)
    assert list(poi.get_dataframe()[poi.zname].values) == [20.0, 20.0, 20.0, 10.0]

    poi = _poi.copy()
    poi.add_outside_polygons(pol, 10.0)
    assert list(poi.get_dataframe()[poi.zname].values) == [10.0, 10.0, 10.0, 20.0]

    poi = _poi.copy()
    poi.sub_inside_polygons(pol, 10.0)
    assert list(poi.get_dataframe()[poi.zname].values) == [0.0, 0.0, 0.0, 10.0]

    poi = _poi.copy()
    poi.sub_outside_polygons(pol, 10.0)
    assert list(poi.get_dataframe()[poi.zname].values) == [10.0, 10.0, 10.0, 0.0]

    poi = _poi.copy()
    poi.mul_inside_polygons(pol, 10.0)
    assert list(poi.get_dataframe()[poi.zname].values) == [100.0, 100.0, 100.0, 10.0]

    poi = _poi.copy()
    poi.mul_outside_polygons(pol, 10.0)
    assert list(poi.get_dataframe()[poi.zname].values) == [10.0, 10.0, 10.0, 100.0]

    poi = _poi.copy()
    poi.div_inside_polygons(pol, 10.0)
    assert list(poi.get_dataframe()[poi.zname].values) == [1.0, 1.0, 1.0, 10.0]

    poi = _poi.copy()
    poi.div_outside_polygons(pol, 10.0)
    assert list(poi.get_dataframe()[poi.zname].values) == [10.0, 10.0, 10.0, 1.0]

    # zero division
    poi = _poi.copy()
    poi.div_outside_polygons(pol, 0.0)
    assert list(poi.get_dataframe()[poi.zname].values) == [10.0, 10.0, 10.0, 0.0]

    # close to zero division
    poi = _poi.copy()
    poi.div_outside_polygons(pol, 1e-10)
    assert list(poi.get_dataframe()[poi.zname].values) == [10.0, 10.0, 10.0, 1e11]

    poi = _poi.copy()
    poi.eli_inside_polygons(pol)
    assert list(poi.get_dataframe()[poi.zname].values) == [10.0]

    poi = _poi.copy()
    poi.eli_outside_polygons(pol)
    assert list(poi.get_dataframe()[poi.zname].values) == [10.0, 10.0, 10.0]


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Different order in python 3.7")
def test_boundary_from_points_simple():
    """Test deriving a boundary around points (classmethod)."""

    points = Points(
        [
            (0.0, 0.0, 10.0),
            (4.5, 4.5, 10.0),
            (2.5, 2.5, 10.0),
            (1.5, 8.5, 10.0),
            (12.5, 1.5, 10.0),
            (1.5, 1.5, 10.0),
            (11.5, 12.5, 10.0),
        ]
    )
    boundary = xtgeo.Polygons.boundary_from_points(points, alpha_factor=0.75)
    assert boundary.get_dataframe()[boundary.xname].values.tolist() == [
        4.5,
        2.5,
        1.5,
        1.5,
        11.5,
        4.5,
    ]


def test_boundary_from_points_simple_estimate_alpha():
    """Test deriving a boundary around points (classmethod)."""

    points = Points(
        [
            (0.0, 0.0, 10.0),
            (4.5, 4.5, 10.0),
            (2.5, 2.5, 10.0),
            (1.5, 8.5, 10.0),
            (12.5, 1.5, 10.0),
            (1.5, 1.5, 10.0),
            (11.5, 12.5, 10.0),
        ]
    )
    boundary = xtgeo.Polygons.boundary_from_points(points, alpha=None)
    assert len(boundary.get_dataframe()) == 6


def test_boundary_from_points_too_few():
    """Test deriving a boundary around points (classmethod), too few points."""

    points = Points(
        [
            (0.0, 0.0, 10.0),
            (4.5, 4.5, 10.0),
            (2.5, 2.5, 10.0),
        ]
    )
    with pytest.raises(ValueError, match="Too few points"):
        xtgeo.Polygons.boundary_from_points(points, alpha_factor=1, alpha=None)


def test_boundary_from_points_more_data(testdata_path):
    """Test deriving a boundary around points (classmethod)."""

    points = xtgeo.points_from_file(testdata_path / POINTSET2)
    boundary = xtgeo.Polygons.boundary_from_points(points, alpha=2000)
    assert boundary.get_dataframe()[boundary.xname].values[
        0:5
    ].tolist() == pytest.approx(
        [464023.440918, 462452.241211, 461325.12793, 460761.571045, 460068.859375]
    )

    assert len(boundary.get_dataframe()) == 15


def test_boundary_from_points_more_data_guess_alpha(testdata_path):
    """Test deriving a boundary around points (classmethod)."""

    points = xtgeo.points_from_file(testdata_path / POINTSET2)
    boundary = xtgeo.Polygons.boundary_from_points(points, alpha=None)

    assert len(boundary.get_dataframe()) == 15


def test_boundary_from_points_more_data_convex_alpha0(testdata_path):
    """Test deriving a boundary around points, convex."""

    points = xtgeo.points_from_file(testdata_path / POINTSET2)
    with pytest.raises(ValueError, match="The alpha value must be greater than 0.0"):
        xtgeo.Polygons.boundary_from_points(points, alpha=0)


def test_boundary_from_points_alpha_factor_none(testdata_path):
    """Test deriving a boundary around points, with alpha_factor None."""

    points = xtgeo.points_from_file(testdata_path / POINTSET2)
    with pytest.warns(UserWarning, match="The alpha_factor is None"):
        boundary = xtgeo.Polygons.boundary_from_points(points, alpha_factor=None)
        assert len(boundary.get_dataframe()) == 15
