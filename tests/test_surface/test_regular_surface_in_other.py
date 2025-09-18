import logging
import pathlib

import numpy as np
import numpy.ma as ma
import pytest

import xtgeo

logger = logging.getLogger(__name__)

SURF1 = pathlib.Path("surfaces/reek/1/topreek_rota.gri")
POLY1 = pathlib.Path("polygons/reek/1/closedpoly1.pol")

# a surface with nodes in X position 0.0 3.0 6.0 and Y 0.0 3.0 6.0
# and 12 values
SF1 = xtgeo.RegularSurface(
    xori=0,
    yori=0,
    ncol=3,
    nrow=3,
    xinc=3.0,
    yinc=3.0,
    values=np.array([10, 10, 10, 20, 0, 10, 10, 10, 1]),
)

SF2 = xtgeo.RegularSurface(
    xori=0,
    yori=0,
    ncol=3,
    nrow=3,
    xinc=3.0,
    yinc=3.0,
    values=np.array([100, 200, 300, 400, 0, 500, 600, 700, 800]),
)

SMALL_POLY = [
    (2.9, 2.9, 0.0, 0),
    (4.9, 2.9, 0.0, 0),
    (6.1, 6.1, 0.0, 0),
    (2.9, 4.0, 0.0, 0),
    (2.9, 2.9, 0.0, 0),
]

SMALL_POLY_OVERLAP = [
    (3.0, 0.0, 0.0, 2),
    (6.1, -1.0, 0.0, 2),
    (6.1, 3.2, 0.0, 2),
    (2.0, 4.0, 0.0, 2),
    (3.0, 0.0, 0.0, 2),
]


@pytest.fixture(name="reekset")
def fixture_reekset(testdata_path):
    """Read a test set from disk."""
    srf = xtgeo.surface_from_file(testdata_path / SURF1)
    pol = xtgeo.polygons_from_file(testdata_path / POLY1)
    return srf, pol


@pytest.mark.parametrize(
    "oper, value, inside, expected",
    [
        #                           O   O   O   I  I   O   I   I  I
        #                orig     [10, 10, 10, 20, 0, 10, 10, 10, 1]
        ("add", 1, True, ma.array([10, 10, 10, 21, 1, 10, 11, 11, 2])),
        ("add", 1, False, ma.array([11, 11, 11, 20, 0, 11, 10, 10, 1])),
        ("add", SF2, True, ma.array([10, 10, 10, 420, 0, 10, 610, 710, 801])),
        ("add", SF2, False, ma.array([110, 210, 310, 20, 0, 510, 10, 10, 1])),
        ("sub", 1, True, ma.array([10, 10, 10, 19, -1, 10, 9, 9, 0])),
        ("sub", 1, False, ma.array([9, 9, 9, 20, 0, 9, 10, 10, 1])),
        ("sub", SF2, True, ma.array([10, 10, 10, -380, 0, 10, -590, -690, -799])),
        ("sub", SF2, False, ma.array([-90, -190, -290, 20, 0, -490, 10, 10, 1])),
        ("mul", 2, True, ma.array([10, 10, 10, 40, 0, 10, 20, 20, 2])),
        ("mul", 2, False, ma.array([20, 20, 20, 20, 0, 20, 10, 10, 1])),
        ("div", 2, True, ma.array([10, 10, 10, 10, 0, 10, 5, 5, 0.5])),
        ("div", 2, False, ma.array([5, 5, 5, 20, 0, 5, 10, 10, 1])),
        (
            "eli",
            0,
            True,
            ma.array([False, False, False, True, True, False, True, True, True]),
        ),
        (
            "eli",
            0,
            False,
            ma.array([True, True, True, False, False, True, False, False, False]),
        ),
    ],
)
def test_operations_inside_outside_polygon_minimal(oper, value, inside, expected):
    """Do operations in relation to polygons, minimal example"""

    poly = xtgeo.Polygons(SMALL_POLY + SMALL_POLY_OVERLAP)

    surf = SF1.copy()
    surf.operation_polygons(poly, value, inside=inside, opname=oper)
    if oper == "eli":
        np.testing.assert_array_equal(surf.values.mask.ravel(), expected)
    else:
        np.testing.assert_array_equal(surf.values.ravel(), expected)


@pytest.mark.parametrize(
    "oper, inside, expected",
    [
        ("add", True, 31.2033),
        ("add", False, 70.7966),
        ("sub", True, -29.2033),
        ("sub", False, -68.7966),
        ("mul", True, 30.9013),
        ("mul", False, 70.0988),
        ("div", True, 0.7010),
        ("div", False, 0.3090),
        ("set", True, 30.9013),
        ("set", False, 70.0987),
        ("eli", True, 1.0),
        ("eli", False, 1.0),
    ],
)
def test_operations_inside_outside_polygon_generic(reekset, oper, inside, expected):
    """Do operations in relation to polygons in a generic way"""

    logger.info("Simple case...")

    _surf, poly = reekset
    _surf.values = 1.0

    surf = _surf.copy()
    surf.operation_polygons(poly, 100.0, inside=inside, opname=oper)
    assert surf.values.mean() == pytest.approx(expected, abs=0.001)


def test_operations_inside_outside_polygon_shortforms(tmp_path, testdata_path):
    """Various shortforms for operations in polygons"""

    # assert values are checked in RMS

    zurf = xtgeo.surface_from_file(testdata_path / SURF1)
    poly = xtgeo.polygons_from_file(testdata_path / POLY1)

    surf = zurf.copy()
    surf.add_inside(poly, 200)
    assert surf.values.mean() == pytest.approx(1759.06, abs=0.01)

    surf = zurf.copy()
    surf.add_outside(poly, 200)
    assert surf.values.mean() == pytest.approx(1838.24, abs=0.01)

    # add another surface inside polygon (here itself)
    surf = zurf.copy()
    surf2 = zurf.copy()
    surf.add_inside(poly, surf2)
    assert surf.values.mean() == pytest.approx(2206.20, abs=0.01)

    # divide on zero
    surf = zurf.copy()
    surf.div_inside(poly, 0.0)
    surf.to_file(tmp_path / "div2.gri")
    surf.to_file(tmp_path / "div2.fgr", fformat="irap_ascii")

    # set inside
    surf = zurf.copy()
    surf.set_inside(poly, 700)
    assert surf.values.mean() == pytest.approx(1402.52, abs=0.01)

    # eliminate inside
    surf = zurf.copy()
    surf.eli_inside(poly)
    assert surf.values.mean() == pytest.approx(1706.52, abs=0.01)
