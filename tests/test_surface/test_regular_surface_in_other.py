# coding: utf-8

from os.path import join

import tests.test_common.test_xtg as tsetup
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
    tsetup.assert_almostequal(surf.values.mean(), 1698.65, 0.01)
    poly = xtgeo.xyz.Polygons(POLY1)

    surf.operation_polygons(poly, 100.0)  # add 100 inside poly
    tsetup.assert_almostequal(surf.values.mean(), 1728.85, 0.01)


def test_operations_inside_outside_polygon_shortforms(tmpdir):
    """Various shortforms for operations in polygons"""

    # assert values are checked in RMS

    zurf = xtgeo.surface_from_file(SURF1)
    poly = xtgeo.xyz.Polygons(POLY1)

    surf = zurf.copy()
    surf.add_inside(poly, 200)
    tsetup.assert_almostequal(surf.values.mean(), 1759.06, 0.01)

    surf = zurf.copy()
    surf.add_outside(poly, 200)
    tsetup.assert_almostequal(surf.values.mean(), 1838.24, 0.01)

    # add another surface inside polygon (here itself)
    surf = zurf.copy()
    surf2 = zurf.copy()
    surf.add_inside(poly, surf2)
    tsetup.assert_almostequal(surf.values.mean(), 2206.20, 0.01)

    # divide on zero
    surf = zurf.copy()
    surf.div_inside(poly, 0.0)
    surf.to_file(join(tmpdir, "div2.gri"))
    surf.to_file(join(tmpdir, "div2.fgr"), fformat="irap_ascii")

    # set inside
    surf = zurf.copy()
    surf.set_inside(poly, 700)
    tsetup.assert_almostequal(surf.values.mean(), 1402.52, 0.01)

    # eliminate inside
    surf = zurf.copy()
    surf.eli_inside(poly)
    tsetup.assert_almostequal(surf.values.mean(), 1706.52, 0.01)
