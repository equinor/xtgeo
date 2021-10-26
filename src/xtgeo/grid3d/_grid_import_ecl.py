# -*- coding: utf-8 -*-

"""Grid import functions for Eclipse, new approach (i.e. version 2)."""


import xtgeo
from xtgeo.grid3d._egrid import EGrid, RockModel
from xtgeo.grid3d._grdecl_grid import GrdeclGrid, GridRelative

xtg = xtgeo.XTGeoDialog()

logger = xtg.functionlogger(__name__)


def import_ecl_egrid(
    gfile,
    relative_to=GridRelative.MAP,
    fileformat="egrid",
):
    egrid = EGrid.from_file(gfile._file, fileformat=fileformat)

    result = grid_from_ecl_grid(egrid, relative_to=relative_to)

    if egrid.egrid_head.file_head.rock_model == RockModel.DUAL_POROSITY:
        result["dualporo"] = True
        result["dualperm"] = False
    elif egrid.egrid_head.file_head.rock_model == RockModel.DUAL_PERMEABILITY:
        result["dualporo"] = True
        result["dualperm"] = True

    return result


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import eclipse run suite: EGRID + properties from INIT and UNRST
# For the INIT and UNRST, props dates shall be selected
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def import_ecl_run(
    groot, ecl_grid, initprops=None, restartprops=None, restartdates=None
):
    """Import combo ECL runs."""
    ecl_init = groot + ".INIT"
    ecl_rsta = groot + ".UNRST"

    ecl_init = xtgeo._XTGeoFile(ecl_init)
    ecl_rsta = xtgeo._XTGeoFile(ecl_rsta)

    grdprops = xtgeo.grid3d.GridProperties()

    # import the init properties unless list is empty
    if initprops:
        grdprops.from_file(
            ecl_init.name, names=initprops, fformat="init", dates=None, grid=ecl_grid
        )

    # import the restart properties for dates unless lists are empty
    if restartprops and restartdates:
        grdprops.from_file(
            ecl_rsta.name,
            names=restartprops,
            fformat="unrst",
            dates=restartdates,
            grid=ecl_grid,
        )
    ecl_grid.gridprops = grdprops


def import_ecl_grdecl(gfile, relative_to=GridRelative.MAP):
    """Import grdecl format."""

    grdecl_grid = GrdeclGrid.from_file(gfile._file, fileformat="grdecl")
    return grid_from_ecl_grid(grdecl_grid, relative_to=relative_to)


def import_ecl_bgrdecl(gfile, relative_to=GridRelative.MAP):
    """Import binary files with GRDECL layout."""

    grdecl_grid = GrdeclGrid.from_file(gfile._file, fileformat="bgrdecl")
    return grid_from_ecl_grid(grdecl_grid, relative_to=relative_to)


def grid_from_ecl_grid(ecl_grid, relative_to=GridRelative.MAP):
    result = dict()
    result["coordsv"] = ecl_grid.xtgeo_coord(relative_to=relative_to)
    result["zcornsv"] = ecl_grid.xtgeo_zcorn(relative_to=relative_to)
    result["actnumsv"] = ecl_grid.xtgeo_actnum()
    if relative_to == GridRelative.MAP and ecl_grid.map_axis_units is not None:
        result["units"] = ecl_grid.map_axis_units
    else:
        result["units"] = ecl_grid.grid_units

    return result
