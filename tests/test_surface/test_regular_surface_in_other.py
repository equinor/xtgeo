# coding: utf-8

from os.path import join

import pytest
import xtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

# =============================================================================
# Do tests
# =============================================================================

TPATH = xtg.testpathobj

SURF1 = TPATH / "surfaces/reek/1/topreek_rota.gri"
POLY1 = TPATH / "polygons/reek/1/closedpoly1.pol"


def test_operations_inside_outside_polygon_generic():
    """Do perations in relation to polygons in a generic way"""

    logger.info("Simple case...")

    surf = xtgeo.surface_from_file(SURF1)
    assert surf.values.mean() == pytest.approx(1698.65, abs=0.01)
    poly = xtgeo.polygons_from_file(POLY1)

    surf.operation_polygons(poly, 100.0)  # add 100 inside poly
    assert surf.values.mean() == pytest.approx(1728.85, abs=0.01)


def test_operations_inside_outside_polygon_shortforms(tmpdir):
    """Various shortforms for operations in polygons"""

    # assert values are checked in RMS

    zurf = xtgeo.surface_from_file(SURF1)
    poly = xtgeo.polygons_from_file(POLY1)

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
    surf.to_file(join(tmpdir, "div2.gri"))
    surf.to_file(join(tmpdir, "div2.fgr"), fformat="irap_ascii")

    # set inside
    surf = zurf.copy()
    surf.set_inside(poly, 700)
    assert surf.values.mean() == pytest.approx(1402.52, abs=0.01)

    # eliminate inside
    surf = zurf.copy()
    surf.eli_inside(poly)
    assert surf.values.mean() == pytest.approx(1706.52, abs=0.01)
