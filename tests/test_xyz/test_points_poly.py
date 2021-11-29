import pathlib

import pytest

from xtgeo.xyz import Points
from xtgeo.xyz import Polygons

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


def test_points_in_polygon(testpath):
    """Import XYZ points and do operations if inside or outside"""

    poi = Points(testpath / POINTSET2)
    pol = Polygons(testpath / POLSET2)
    assert poi.nrow == 30

    # remove points in polygon
    poi.operation_polygons(pol, 0, opname="eli")

    assert poi.nrow == 19

    poi = Points(testpath / POINTSET2)
    # remove points outside polygon
    poi.operation_polygons(pol, 0, opname="eli", inside=False)
    assert poi.nrow == 1


def test_check_single_point_in_polygon():
    pol = Polygons(SMALL_POLY_INNER)
    poi = Points([(4.0, 4.0, 4.0)])
    poi.operation_polygons(pol, value=1, opname="eli", inside=False)
    assert len(poi.dataframe) == 1

    poi.operation_polygons(pol, value=1, opname="eli", inside=True)
    assert len(poi.dataframe) == 0


def test_check_multi_point_single_polygon():
    pol = Polygons(SMALL_POLY_INNER)
    poi = Points([(6.0, 4.0, 4.0), (4.0, 4.0, 4.0), (4.0, 6.0, 4.0)])
    assert len(poi.dataframe) == 3

    poi.operation_polygons(pol, value=1, opname="eli", inside=False)
    assert len(poi.dataframe) == 1

    poi.operation_polygons(pol, value=1, opname="eli", inside=True)
    assert len(poi.dataframe) == 0


def test_check_multi_point_single_polygon_zdir():
    pol = Polygons(SMALL_POLY_INNER)
    poi = Points([(4.0, 4.0, 0.0), (4.0, 4.0, 4.0), (4.0, 4.0, 8.0)])
    assert len(poi.dataframe) == 3

    # Note z-direction has no effect. All points are deleted.
    poi.operation_polygons(pol, value=1, opname="eli", inside=True)
    assert len(poi.dataframe) == 0


def test_check_multi_point_multi_polyon_inside_op():
    pol = Polygons(SMALL_POLY_INNER + LARGE_POLY_SHIFTED)
    # Two points in small cube, one in large cube, one outside
    poi = Points([(4.0, 4.0, 0.0), (4.5, 4.0, 0.0), (7.0, 7.0, 0.0), (20.0, 5.0, 0.0)])
    assert len(poi.dataframe) == 4

    poi.operation_polygons(pol, value=1, opname="eli", inside=True)
    assert len(poi.dataframe) == 1

    poi = Points([(4.0, 4.0, 0.0), (4.5, 4.0, 0.0), (7.0, 7.0, 0.0), (20.0, 5.0, 0.0)])
    poi.operation_polygons(pol, value=1, opname="eli", inside=False)
    assert len(poi.dataframe) == 0


def test_check_multi_point_multi_polyon_outside_op():
    pol = Polygons(SMALL_POLY_INNER + LARGE_POLY_SHIFTED)
    # Two points in small cube, one in large cube, one outside
    poi = Points([(4.0, 4.0, 0.0), (4.5, 4.0, 0.0), (7.0, 7.0, 0.0), (20.0, 5.0, 0.0)])
    assert len(poi.dataframe) == 4

    # Note the operation will loop over the polygons, and hence remove the points
    # in the small polygon when considering the large polygon, and vice versa
    poi.operation_polygons(pol, value=1, opname="eli", inside=False)
    assert len(poi.dataframe) == 0


def test_check_single_polygon_in_single_polygon():
    inner_pol = Polygons(SMALL_POLY_INNER)
    outer_pol = Polygons(SMALL_POLY_OUTER)

    # Do not delete inner_pol when specified to delete polygons inside inner polygon
    inner_pol.operation_polygons(outer_pol, value=1, opname="eli", inside=False)
    assert len(inner_pol.dataframe) == 5

    # Do not delete outer_pol when specified to delete polygons outside outer polygon
    outer_pol.operation_polygons(inner_pol, value=1, opname="eli", inside=True)
    assert len(outer_pol.dataframe) == 5

    inner_pol.operation_polygons(outer_pol, value=1, opname="eli", inside=True)
    assert len(inner_pol.dataframe) == 0

    inner_pol = Polygons(SMALL_POLY_INNER)

    outer_pol.operation_polygons(inner_pol, value=1, opname="eli", inside=False)
    assert len(outer_pol.dataframe) == 0


def test_check_multi_polygon_in_single_polygon():
    inner_pols = Polygons(SMALL_POLY_INNER + SMALL_POLY_OVERLAP_INNER)
    outer_pol = Polygons(SMALL_POLY_OUTER)

    inner_pols.operation_polygons(outer_pol, value=1, opname="eli", inside=True)
    assert len(inner_pols.dataframe) == 0


def test_operation_inclusive_polygon():
    pol = Polygons(SMALL_POLY_INNER)
    # We place a point on the edge of a polygon
    poi = Points([(4.0, 4.0, 0.0)])
    poi.operation_polygons(pol, value=1, opname="eli", inside=True)
    assert len(poi.dataframe) == 0

    # We place a point on a corner of a polygon
    poi = Points([(3.0, 3.0, 0.0)])
    poi.operation_polygons(pol, value=1, opname="eli", inside=True)
    assert len(poi.dataframe) == 0


def test_polygons_overlap():
    pol = Polygons(SMALL_POLY_INNER + SMALL_POLY_OVERLAP_INNER)
    # The Four points are placed: within first poly, within the overlap, within the
    # second poly, outside both poly
    poi = Points([(3.5, 3.5, 0.0), (4.5, 4.5, 0.0), (5.5, 5.5, 0.0), (6.5, 6.5, 0.0)])
    poi.operation_polygons(pol, value=1, opname="eli", inside=True)
    assert len(poi.dataframe) == 1


@pytest.mark.parametrize(
    "oper, expected", [("add", 12), ("sub", 8), ("div", 5), ("mul", 20), ("set", 2)]
)
def test_oper_single_point_in_polygon(oper, expected):
    pol = Polygons(SMALL_POLY_INNER)
    poi = Points([(4.0, 4.0, 10.0)])
    # operators work on z-values
    poi.operation_polygons(pol, value=2, opname=oper, inside=True)
    assert poi.dataframe[poi.zname].values[0] == expected


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
def test_oper_points_outside_overlapping_polygon(oper, expected):
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
    poi.operation_polygons(pol, value=2, opname=oper, inside=True)
    assert list(poi.dataframe[poi.zname].values) == expected


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
def test_oper_points_inside_overlapping_polygon(oper, expected):
    pol = Polygons(SMALL_POLY_INNER + SMALL_POLY_OVERLAP_INNER)
    # The Four points are placed: within first poly, within the overlap, within the
    # second poly, outside both poly
    poi = Points(
        [(3.5, 3.5, 10.0), (4.5, 4.5, 10.0), (5.5, 5.5, 10.0), (6.5, 6.5, 10.0)]
    )
    # Operations are performed n times, where n reflects the number of polygons a
    # point is outside
    poi.operation_polygons(pol, value=2, opname=oper, inside=False)
    assert list(poi.dataframe[poi.zname].values) == expected
