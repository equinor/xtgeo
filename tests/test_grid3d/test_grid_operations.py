# -*- coding: utf-8 -*-
"""Testing: test_grid_operations"""
from collections import OrderedDict
from os.path import join

import xtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TPATH = xtg.testpathobj

# pylint: disable=logging-format-interpolation

# =============================================================================
# Do tests
# =============================================================================


DUAL = TPATH / "3dgrids/etc/dual_distorted2.grdecl"


def test_hybridgrid1(tmpdir, snapshot):
    grd = xtgeo.create_box_grid(
        (4, 3, 5),
        flip=1,
        oricenter=False,
        origin=(10.0, 20.0, 1000.0),
        rotation=30.0,
        increment=(100, 150, 5),
    )
    grd.subgrids = OrderedDict({"name1": [1], "name2": [2, 3], "name3": [4, 5]})
    assert grd.subgrids is not None  # initially, prior to subgrids

    logger.info("Read grid... done, NZ is %s", grd.nlay)
    grd.to_file(join(tmpdir, "test_hybridgrid1_asis.bgrdecl"), fformat="bgrdecl")

    logger.info("Convert...")
    nhdiv = 40
    newnlay = grd.nlay * 2 + nhdiv
    snapshot.assert_match(
        grd.dataframe(activeonly=False).tail(50).round().to_csv(line_terminator="\n"),
        "grid_pre_hybrid.csv",
    )
    grd.convert_to_hybrid(nhdiv=nhdiv, toplevel=1700, bottomlevel=1740)
    snapshot.assert_match(
        grd.dataframe(activeonly=False).tail(50).round().to_csv(line_terminator="\n"),
        "grid_post_hybrid.csv",
    )

    assert grd.nlay == newnlay, "New NLAY number"
    assert grd.subgrids is None

    dzv = grd.get_dz()

    assert dzv.values3d.mean() == 5.0

    grd2 = xtgeo.grid_from_file(join(tmpdir, "test_hybridgrid1_asis.bgrdecl"))
    snapshot.assert_match(
        grd2.dataframe(activeonly=False).tail(50).round().to_csv(line_terminator="\n"),
        "grid_pre_hybrid.csv",
    )


def test_reverse_row_axis_box(tmpdir):
    """Crop a grid."""

    grd = xtgeo.create_box_grid(
        origin=(1000, 4000, 300),
        increment=(100, 100, 2),
        dimension=(2, 3, 1),
        rotation=0,
    )

    assert grd.ijk_handedness == "left"
    grd.to_file(join(tmpdir, "reverse_left.grdecl"), fformat="grdecl")
    grd.reverse_row_axis()
    assert grd.ijk_handedness == "right"
    grd.to_file(join(tmpdir, "reverse_right.grdecl"), fformat="grdecl")


def test_reverse_row_axis_dual(tmpdir):
    """Reverse axis for distorted but small grid"""

    grd = xtgeo.grid_from_file(DUAL)

    assert grd.ijk_handedness == "left"
    grd.to_file(join(tmpdir, "dual_left.grdecl"), fformat="grdecl")
    cellcorners1 = grd.get_xyz_cell_corners((5, 1, 1))
    grd.reverse_row_axis()
    assert grd.ijk_handedness == "right"
    grd.to_file(join(tmpdir, "dual_right.grdecl"), fformat="grdecl")
    cellcorners2 = grd.get_xyz_cell_corners((5, 3, 1))

    assert cellcorners1[7] == cellcorners2[1]
