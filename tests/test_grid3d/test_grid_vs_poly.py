# -*- coding: utf-8 -*-
import sys
import os

import xtgeo

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    sys.exit(-9)

td = xtg.tmpdir
TPATH = xtg.testpathobj

if "XTG_SHOW" in os.environ:
    xtgshow = True
else:
    xtgshow = False


# =============================================================================
# Do tests
# =============================================================================

reekgrid = TPATH / "3dgrids/reek/REEK.EGRID"
reekpoly = TPATH / "polygons/reek/1/mypoly.pol"


def test_grid_inactivate_inside():
    """Inactivate a grid inside polygons"""
    g1 = xtgeo.grid3d.Grid(reekgrid)

    p1 = xtgeo.xyz.Polygons(reekpoly)

    act1 = g1.get_actnum().values3d
    n1 = act1[7, 55, 1]
    assert n1 == 1

    try:
        g1.inactivate_inside(p1, layer_range=(1, 4))
    except RuntimeError as rw:
        print(rw)

    g1.to_file(os.path.join(td, "reek_inact_ins_pol.roff"))

    # geom = g1.get_geometrics(return_dict=True)

    # myprop = g1.get_actnum()
    # layslice = xtgeo.plot.Grid3DSlice()
    # layslice.canvas(title="Layer 1")
    # layslice.plot_gridslice(myprop, window=(geom['xmin'], geom['xmax'],
    #                                         geom['ymin'], geom['ymax']))

    # if xtgshow:
    #     layslice.show()
    # else:
    #     print('Output to screen disabled (will plotto screen); '
    #           'use XTG_SHOW env variable')
    #     layslice.savefig('TMP/inact_inside.png')

    act2 = g1.get_actnum().values3d
    n2 = act2[7, 55, 1]
    assert n2 == 0

    # assert int(act1[20, 38, 4]) == int(act2[20, 38, 4])

    # print(np.sum(act1), np.sum(act2))


def test_grid_inactivate_outside():
    """Inactivate a grid outside polygons"""
    g1 = xtgeo.grid3d.Grid(reekgrid)

    p1 = xtgeo.xyz.Polygons(reekpoly)

    act1 = g1.get_actnum().values3d
    n1 = act1[3, 56, 1]
    assert n1 == 1

    try:
        g1.inactivate_outside(p1, layer_range=(1, 4))
    except RuntimeError as rw:
        print(rw)

    g1.to_file(os.path.join(td, "reek_inact_out_pol.roff"))

    act2 = g1.get_actnum().values3d
    n2 = act2[3, 56, 1]
    assert n2 == 0

    assert int(act1[20, 38, 4]) == int(act2[20, 38, 4])

    # logger.info(np.sum(act1), np.sum(act2))
