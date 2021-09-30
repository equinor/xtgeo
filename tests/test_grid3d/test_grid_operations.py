# -*- coding: utf-8 -*-
"""Testing: test_grid_operations"""
from collections import OrderedDict
from os.path import join

import pytest

import tests.test_common.test_xtg as tsetup
from xtgeo.common import XTGeoDialog
from xtgeo.grid3d import Grid, GridProperty

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TPATH = xtg.testpathobj

# pylint: disable=logging-format-interpolation

# =============================================================================
# Do tests
# =============================================================================
EMEGFILE = TPATH / "3dgrids/eme/1/emerald_hetero_grid.roff"
EMERFILE = TPATH / "3dgrids/eme/1/emerald_hetero_region.roff"

EMEGFILE2 = TPATH / "3dgrids/eme/2/emerald_hetero_grid.roff"
EMEZFILE2 = TPATH / "3dgrids/eme/2/emerald_hetero.roff"

DUAL = TPATH / "3dgrids/etc/dual_distorted2.grdecl"
DUALPROPS = TPATH / "3dgrids/etc/DUAL"


def test_hybridgrid1(tmpdir, snapshot):
    grd = Grid()
    grd.subgrids = OrderedDict({"name1": [1], "name2": [2, 3], "name3": [4, 5]})
    assert grd.subgrids is not None  # initially, prior to subgrids

    logger.info("Read grid... done, NZ is %s", grd.nlay)
    grd.to_file(join(tmpdir, "test_hybridgrid1_asis.bgrdecl"), fformat="bgrdecl")

    logger.info("Convert...")
    nhdiv = 40
    newnlay = grd.nlay * 2 + nhdiv
    print(grd.dataframe(activeonly=False))
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

    grd2 = Grid(join(tmpdir, "test_hybridgrid1_asis.bgrdecl"))
    snapshot.assert_match(
        grd2.dataframe(activeonly=False).tail(50).round().to_csv(line_terminator="\n"),
        "grid_pre_hybrid.csv",
    )


def test_hybridgrid2(tmpdir):
    """Making a hybridgrid for Emerald case in region"""

    logger.info("Read grid...")
    grd = Grid()
    logger.info("Read grid... done, NLAY is {}".format(grd.nlay))

    reg = GridProperty()
    reg.from_file(EMERFILE, name="REGION")

    nhdiv = 40

    grd.convert_to_hybrid(
        nhdiv=nhdiv, toplevel=1650, bottomlevel=1690, region=reg, region_number=1
    )

    grd.to_file(join(tmpdir, "test_hybridgrid2.roff"))


def test_inactivate_thin_cells(tmpdir):
    """Make hybridgrid for Emerald case in region, and inactive thin cells"""

    logger.info("Read grid...")
    grd = Grid(EMEGFILE)
    logger.info("Read grid... done, NLAY is {}".format(grd.nlay))

    reg = GridProperty()
    reg.from_file(EMERFILE, name="REGION")

    nhdiv = 40

    grd.convert_to_hybrid(
        nhdiv=nhdiv, toplevel=1650, bottomlevel=1690, region=reg, region_number=1
    )

    grd.inactivate_by_dz(0.001)

    grd.to_file(join(tmpdir, "test_hybridgrid2_inact_thin.roff"))


def test_refine_vertically(tmpdir):
    """Do a grid refinement vertically."""

    logger.info("Read grid...")

    grd = Grid(EMEGFILE)
    logger.info("Read grid... done, NLAY is {}".format(grd.nlay))
    logger.info("Subgrids before: %s", grd.get_subgrids())

    avg_dz1 = grd.get_dz().values3d.mean()

    # idea; either a scalar (all cells), or a dictionary for zone wise
    grd.refine_vertically(3)

    avg_dz2 = grd.get_dz().values3d.mean()

    assert avg_dz1 == pytest.approx(3 * avg_dz2, abs=0.0001)

    logger.info("Subgrids after: %s", grd.get_subgrids())
    grd.inactivate_by_dz(0.001)

    grd.to_file(join(tmpdir, "test_refined_by_3.roff"))


def test_refine_vertically_per_zone(tmpdir):
    """Do a grid refinement vertically, via a dict per zone."""

    logger.info("Read grid...")

    grd_orig = Grid(EMEGFILE2)
    grd = grd_orig.copy()

    logger.info("Read grid... done, NLAY is {}".format(grd.nlay))
    grd.to_file(join(tmpdir, "test_refined_by_dict_initial.roff"))

    logger.info("Subgrids before: %s", grd.get_subgrids())

    zone = GridProperty(EMEZFILE2, grid=grd, name="Zone")
    logger.info("Zone values min max: %s %s", zone.values.min(), zone.values.max())

    logger.info("Subgrids list: %s", grd.subgrids)

    refinement = {1: 4, 2: 2}
    grd.refine_vertically(refinement, zoneprop=zone)

    grd1s = grd.get_subgrids()
    logger.info("Subgrids after: %s", grd1s)

    grd = grd_orig.copy()
    grd.refine_vertically(refinement)  # no zoneprop
    grd2s = grd.get_subgrids()
    logger.info("Subgrids after: %s", grd2s)
    assert list(grd1s.values()) == list(grd2s.values())


def test_reverse_row_axis_box(tmpdir):
    """Crop a grid."""

    grd = Grid()
    grd.create_box(
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

    grd = Grid(DUAL)

    assert grd.ijk_handedness == "left"
    grd.to_file(join(tmpdir, "dual_left.grdecl"), fformat="grdecl")
    cellcorners1 = grd.get_xyz_cell_corners((5, 1, 1))
    grd.reverse_row_axis()
    assert grd.ijk_handedness == "right"
    grd.to_file(join(tmpdir, "dual_right.grdecl"), fformat="grdecl")
    cellcorners2 = grd.get_xyz_cell_corners((5, 3, 1))

    assert cellcorners1[7] == cellcorners2[1]


def test_reverse_row_axis_dualprops():
    """Reverse axis for distorted but small grid with props"""

    grd = Grid(DUALPROPS, fformat="eclipserun", initprops=["PORO", "PORV"])

    poro = grd.gridprops.props[0]
    logger.info(grd.gridprops.describe())
    porowas = poro.copy()
    tsetup.assert_almostequal(poro.values[1, 0, 0], 0.17777, 0.01)
    assert grd.ijk_handedness == "left"

    grd.reverse_row_axis()
    tsetup.assert_almostequal(poro.values[1, 2, 0], 0.17777, 0.01)
    assert poro.values[1, 2, 0] == porowas.values[1, 0, 0]

    grd.reverse_row_axis()
    assert poro.values[1, 0, 0] == porowas.values[1, 0, 0]
    assert grd.ijk_handedness == "left"

    grd.reverse_row_axis(ijk_handedness="left")  # ie do nothing in this case
    assert poro.values[1, 0, 0] == porowas.values[1, 0, 0]
    assert grd.ijk_handedness == "left"


def test_reverse_row_axis_eme(tmpdir):
    """Reverse axis for emerald grid"""

    grd1 = Grid(EMEGFILE)

    assert grd1.ijk_handedness == "left"
    grd1.to_file(join(tmpdir, "eme_left.roff"), fformat="roff")
    grd2 = grd1.copy()
    geom1 = grd1.get_geometrics(return_dict=True)
    grd2.reverse_row_axis()
    assert grd2.ijk_handedness == "right"
    grd2.to_file(join(tmpdir, "eme_right.roff"), fformat="roff")
    geom2 = grd2.get_geometrics(return_dict=True)

    tsetup.assert_almostequal(geom1["avg_rotation"], geom2["avg_rotation"], 0.01)


def test_copy_grid(tmpdir):
    """Copy a grid."""

    grd = Grid(EMEGFILE2)
    grd2 = grd.copy()

    grd.to_file(join(tmpdir, "gcp1.roff"))
    grd2.to_file(join(tmpdir, "gcp2.roff"))

    xx1 = Grid(join(tmpdir, "gcp1.roff"))
    xx2 = Grid(join(tmpdir, "gcp2.roff"))

    assert xx1._zcornsv.mean() == xx2._zcornsv.mean()
    assert xx1._actnumsv.mean() == xx2._actnumsv.mean()


def test_crop_grid(tmpdir):
    """Crop a grid."""

    logger.info("Read grid...")

    grd = Grid(EMEGFILE2)
    zprop = GridProperty(EMEZFILE2, name="Zone", grid=grd)

    logger.info("Read grid... done, NLAY is {}".format(grd.nlay))
    logger.info(
        "Read grid...NCOL, NROW, NLAY is {} {} {}".format(grd.ncol, grd.nrow, grd.nlay)
    )

    grd.crop((30, 60), (20, 40), (1, 46), props=[zprop])

    grd.to_file(join(tmpdir, "grid_cropped.roff"))

    grd2 = Grid(join(tmpdir, "grid_cropped.roff"))

    assert grd2.ncol == 31


def test_crop_grid_after_copy():
    """Copy a grid, then crop and check number of active cells."""

    logger.info("Read grid...")

    grd = Grid(EMEGFILE2)
    grd.describe()
    zprop = GridProperty(EMEZFILE2, name="Zone", grid=grd)
    grd.describe(details=True)

    logger.info(grd.dimensions)

    grd2 = grd.copy()
    grd2.describe(details=True)

    logger.info("GRD2 props: %s", grd2.props)
    assert grd.propnames == grd2.propnames

    logger.info("GRD2 number of active cells: %s", grd2.nactive)
    act = grd.get_actnum()
    logger.info(act.values.shape)
    logger.info("ZPROP: %s", zprop.values.shape)

    grd2.crop((1, 30), (40, 80), (23, 46))

    grd2.describe(details=True)


def test_reduce_to_one_layer():
    """Reduce grid to one layer"""

    logger.info("Read grid...")

    grd1 = Grid(EMEGFILE2)
    grd1.reduce_to_one_layer()

    assert grd1.nlay == 1
