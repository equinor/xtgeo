# -*- coding: utf-8 -*-

"""Grid import functions for Eclipse, new approach (i.e. version 2)."""


import xtgeo
from xtgeo.grid3d._egrid import EGrid, RockModel
from xtgeo.grid3d._grdecl_grid import GrdeclGrid, GridRelative

xtg = xtgeo.XTGeoDialog()

logger = xtg.functionlogger(__name__)


def import_ecl_egrid(
    gfile,
    units=None,
    coordinates=GridRelative.MAP,
    fileformat="egrid",
):
    egrid = EGrid.from_file(gfile._file, fileformat=fileformat)
    if units is not None:
        egrid.convert_units(units)

    result = dict()

    result["coordsv"] = egrid.xtgeo_coord(coordinates=coordinates)
    result["zcornsv"] = egrid.xtgeo_zcorn()
    result["actnumsv"] = egrid.xtgeo_actnum()

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


def import_ecl_grdecl(gfile, units=None, coordinates=GridRelative.MAP):
    """Import grdecl format."""

    grdecl_grid = GrdeclGrid.from_file(gfile._file, fileformat="grdecl")
    if units is not None:
        grdecl_grid.convert_units(units)

    return {
        "coordsv": grdecl_grid.xtgeo_coord(coordinates=coordinates),
        "zcornsv": grdecl_grid.xtgeo_zcorn(),
        "actnumsv": grdecl_grid.xtgeo_actnum(),
        "subgrids": None,
    }


def import_ecl_bgrdecl(gfile, units=None, coordinates=GridRelative.MAP):
    """Import binary files with GRDECL layout."""

    grdecl_grid = GrdeclGrid.from_file(gfile._file, fileformat="bgrdecl")
    if units is not None:
        grdecl_grid.convert_units(units)

    return {
        "coordsv": grdecl_grid.xtgeo_coord(coordinates=coordinates),
        "zcornsv": grdecl_grid.xtgeo_zcorn(),
        "actnumsv": grdecl_grid.xtgeo_actnum(),
        "subgrids": None,
    }
