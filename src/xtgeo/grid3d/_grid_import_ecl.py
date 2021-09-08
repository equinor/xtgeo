# -*- coding: utf-8 -*-

"""Grid import functions for Eclipse, new approach (i.e. version 2)."""


import xtgeo
from xtgeo.grid3d._egrid import EGrid, RockModel
from xtgeo.grid3d._grdecl_grid import GrdeclGrid, GridRelative

xtg = xtgeo.XTGeoDialog()

logger = xtg.functionlogger(__name__)


def import_ecl_egrid(
    self,
    gfile,
    relative_to=GridRelative.MAP,
    fileformat="egrid",
):
    egrid = EGrid.from_file(gfile._file, fileformat=fileformat)

    self._ncol, self._nrow, self._nlay = egrid.dimensions

    self._coordsv = egrid.xtgeo_coord(relative_to=relative_to)
    self._zcornsv = egrid.xtgeo_zcorn(relative_to=relative_to)
    self._actnumsv = egrid.xtgeo_actnum()
    self._subgrids = None
    self._xtgformat = 2
    if relative_to == GridRelative.MAP and egrid.map_axis_units is not None:
        self.units = egrid.map_axis_units
    else:
        self.units = egrid.grid_units

    if egrid.egrid_head.file_head.rock_model == RockModel.DUAL_POROSITY:
        self._dualporo = True
        self._dualperm = False
    elif egrid.egrid_head.file_head.rock_model == RockModel.DUAL_PERMEABILITY:
        self._dualporo = True
        self._dualperm = True

    if self._dualporo:
        self._dualactnum = self.get_actnum(name="DUALACTNUM")
        acttmp = self._dualactnum.copy()
        acttmp.values[acttmp.values >= 1] = 1
        self.set_actnum(acttmp)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import eclipse run suite: EGRID + properties from INIT and UNRST
# For the INIT and UNRST, props dates shall be selected
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def import_ecl_run(self, groot, initprops=None, restartprops=None, restartdates=None):
    """Import combo ECL runs."""
    ecl_grid = groot + ".EGRID"
    ecl_init = groot + ".INIT"
    ecl_rsta = groot + ".UNRST"

    ecl_grid = xtgeo._XTGeoFile(ecl_grid)
    ecl_init = xtgeo._XTGeoFile(ecl_init)
    ecl_rsta = xtgeo._XTGeoFile(ecl_rsta)

    # import the grid
    import_ecl_egrid(self, ecl_grid)

    grdprops = xtgeo.grid3d.GridProperties()

    # import the init properties unless list is empty
    if initprops:
        grdprops.from_file(
            ecl_init.name, names=initprops, fformat="init", dates=None, grid=self
        )

    # import the restart properties for dates unless lists are empty
    if restartprops and restartdates:
        grdprops.from_file(
            ecl_rsta.name,
            names=restartprops,
            fformat="unrst",
            dates=restartdates,
            grid=self,
        )

    self.gridprops = grdprops


def import_ecl_grdecl(self, gfile, relative_to=GridRelative.MAP):
    """Import grdecl format."""

    grdecl_grid = GrdeclGrid.from_file(gfile._file, fileformat="grdecl")
    grid_from_grdecl(self, grdecl_grid, relative_to=relative_to)


def import_ecl_bgrdecl(self, gfile, relative_to=GridRelative.MAP):
    """Import binary files with GRDECL layout."""

    grdecl_grid = GrdeclGrid.from_file(gfile._file, fileformat="bgrdecl")
    grid_from_grdecl(self, grdecl_grid, relative_to=relative_to)


def grid_from_grdecl(self, grdecl_grid, relative_to=GridRelative.MAP):

    self._ncol, self._nrow, self._nlay = grdecl_grid.dimensions

    self._coordsv = grdecl_grid.xtgeo_coord(relative_to=relative_to)
    self._zcornsv = grdecl_grid.xtgeo_zcorn(relative_to=relative_to)
    self._actnumsv = grdecl_grid.xtgeo_actnum()
    self._subgrids = None
    self._xtgformat = 2
    if relative_to == GridRelative.MAP and grdecl_grid.map_axis_units is not None:
        self.units = grdecl_grid.map_axis_units
    else:
        self.units = grdecl_grid.grid_units
