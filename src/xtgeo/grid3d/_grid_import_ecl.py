"""Grid import functions for Eclipse, new approach (i.e. version 2)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xtgeo.common import null_logger
from xtgeo.grid3d._egrid import EGrid, RockModel
from xtgeo.grid3d._grdecl_grid import GrdeclGrid, GridRelative
from xtgeo.grid3d.grid_properties import GridProperties, gridproperties_from_file
from xtgeo.io._file import FileFormat, FileWrapper

logger = null_logger(__name__)


if TYPE_CHECKING:
    from xtgeo.grid3d.grid import Grid


def import_ecl_egrid(
    gfile: FileWrapper,
    relative_to: GridRelative = GridRelative.MAP,
    fileformat: FileFormat = FileFormat.EGRID,
) -> dict[str, Any]:
    egrid = EGrid.from_file(gfile.file, fileformat=fileformat)
    result = grid_from_ecl_grid(egrid, relative_to=relative_to)

    if egrid.egrid_head.file_head.rock_model == RockModel.DUAL_POROSITY:
        result["dualporo"] = True
        result["dualperm"] = False
    elif egrid.egrid_head.file_head.rock_model == RockModel.DUAL_PERMEABILITY:
        result["dualporo"] = True
        result["dualperm"] = True

    return result


def import_ecl_run(
    groot: str,
    ecl_grid: Grid,
    initprops: list[str] | None = None,
    restartprops: list[str] | None = None,
    restartdates: list[str] | None = None,
) -> None:
    """Import Eclipse run suite: EGrid and properties from INIT and UNRST.
    For the INIT and UNRST files, property dates shall be selected."""
    ecl_init = FileWrapper(f"{groot}.INIT")
    ecl_rsta = FileWrapper(f"{groot}.UNRST")
    grdprops = GridProperties()

    # import the init properties unless list is empty
    if initprops:
        initprops_ = gridproperties_from_file(
            ecl_init.name, names=initprops, fformat="init", dates=None, grid=ecl_grid
        )
        if initprops_.props:
            grdprops.append_props(initprops_.props)

    # import the restart properties for dates unless lists are empty
    if restartprops and restartdates:
        restartprops_ = gridproperties_from_file(
            ecl_rsta.name,
            names=restartprops,
            fformat="unrst",
            dates=restartdates,
            grid=ecl_grid,
        )
        if restartprops_.props:
            grdprops.append_props(restartprops_.props)
    ecl_grid.gridprops = grdprops


def import_ecl_grdecl(
    gfile: FileWrapper, relative_to: GridRelative = GridRelative.MAP
) -> dict[str, Any]:
    """Import grdecl format."""
    grdecl_grid = GrdeclGrid.from_file(gfile.file, fileformat=FileFormat.GRDECL)
    return grid_from_ecl_grid(grdecl_grid, relative_to=relative_to)


def import_ecl_bgrdecl(
    gfile: FileWrapper, relative_to: GridRelative = GridRelative.MAP
) -> dict[str, Any]:
    """Import binary files with GRDECL layout."""
    grdecl_grid = GrdeclGrid.from_file(gfile.file, fileformat=FileFormat.BGRDECL)
    return grid_from_ecl_grid(grdecl_grid, relative_to=relative_to)


def grid_from_ecl_grid(
    ecl_grid: EGrid, relative_to: GridRelative = GridRelative.MAP
) -> dict[str, Any]:
    result = {}
    result["coordsv"] = ecl_grid.xtgeo_coord(relative_to=relative_to)
    result["zcornsv"] = ecl_grid.xtgeo_zcorn(relative_to=relative_to)
    result["actnumsv"] = ecl_grid.xtgeo_actnum()
    if relative_to == GridRelative.MAP and ecl_grid.map_axis_units is not None:
        result["units"] = ecl_grid.map_axis_units
    else:
        result["units"] = ecl_grid.grid_units

    return result
