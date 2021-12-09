# -*- coding: utf-8 -*-
import os
import sys
from os.path import join

import xtgeo

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    sys.exit(-9)

TPATH = xtg.testpathobj

reekgrid = TPATH / "3dgrids/reek/REEK.EGRID"
reekpoly = TPATH / "polygons/reek/1/mypoly.pol"


def test_grid_inactivate_inside(tmpdir):
    """Inactivate a grid inside polygons"""
    g1 = xtgeo.grid_from_file(reekgrid)

    p1 = xtgeo.xyz.Polygons(reekpoly)

    act1 = g1.get_actnum().values3d
    n1 = act1[7, 55, 1]
    assert n1 == 1

    try:
        g1.inactivate_inside(p1, layer_range=(1, 4))
    except RuntimeError as rw:
        print(rw)

    g1.to_file(join(tmpdir, "reek_inact_ins_pol.roff"))

    act2 = g1.get_actnum().values3d
    n2 = act2[7, 55, 1]
    assert n2 == 0


def test_grid_inactivate_outside(tmpdir):
    """Inactivate a grid outside polygons"""
    g1 = xtgeo.grid_from_file(reekgrid)

    p1 = xtgeo.xyz.Polygons(reekpoly)

    act1 = g1.get_actnum().values3d
    n1 = act1[3, 56, 1]
    assert n1 == 1

    try:
        g1.inactivate_outside(p1, layer_range=(1, 4))
    except RuntimeError as rw:
        print(rw)

    g1.to_file(os.path.join(tmpdir, "reek_inact_out_pol.roff"))

    act2 = g1.get_actnum().values3d
    n2 = act2[3, 56, 1]
    assert n2 == 0

    assert int(act1[20, 38, 4]) == int(act2[20, 38, 4])
