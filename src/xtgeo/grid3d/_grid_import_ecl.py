# -*- coding: utf-8 -*-

"""Grid import functions for Eclipse, new approach (i.e. version 2)."""

import numpy as np

import xtgeo
import xtgeo.cxtgeo._cxtgeo as _cxtgeo
from xtgeo.grid3d._grdecl_grid import GrdeclGrid
from xtgeo.grid3d._grid_eclbin_record import eclbin_record

from . import _grid3d_utils as utils

xtg = xtgeo.XTGeoDialog()

logger = xtg.functionlogger(__name__)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import Eclipse result .EGRID
# See notes in grid.py on dual porosity / dual perm scheme.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def import_ecl_egrid(self, gfile):
    """Import, private to this routine."""
    # scan file for property
    logger.info("Make kwlist by scanning")
    kwlist = utils.scan_keywords(
        gfile, fformat="xecl", maxkeys=1000, dataframe=False, dates=False
    )
    bpos = {}
    for name in ("COORD", "ZCORN", "ACTNUM", "MAPAXES"):
        bpos[name] = -1  # initially

    self._dualporo = False
    for kwitem in kwlist:
        kwname, kwtype, kwlen, kwbyte = kwitem
        if kwname == "FILEHEAD":
            # read FILEHEAD record:
            filehead = eclbin_record(gfile, "FILEHEAD", kwlen, kwtype, kwbyte)
            dualp = filehead[5].tolist()
            logger.info("Dual porosity flag is %s", dualp)
            if dualp == 1:
                self._dualporo = True
                self._dualperm = False
            elif dualp == 2:
                self._dualporo = True
                self._dualperm = True
        elif kwname == "GRIDHEAD":
            # read GRIDHEAD record:
            gridhead = eclbin_record(gfile, "GRIDHEAD", kwlen, kwtype, kwbyte)
            self._ncol, self._nrow, self._nlay = gridhead[1:4].tolist()
        elif kwname in ("COORD", "ZCORN", "ACTNUM"):
            bpos[kwname] = kwbyte
        elif kwname == "MAPAXES":  # not always present
            bpos[kwname] = kwbyte

    logger.info(
        "Grid dimensions in EGRID file: %s %s %s", self._ncol, self._nrow, self._nlay
    )

    # allocate dimensions:
    ncoord, nzcorn, ntot = self.vectordimensions

    self._coordsv = np.zeros(ncoord, dtype=np.float64)
    self._zcornsv = np.zeros(nzcorn, dtype=np.float64)
    self._actnumsv = np.zeros(ntot, dtype=np.int32)
    p_nact = _cxtgeo.new_longpointer()

    option = 0
    if self._dualporo:
        option = 1

    cfhandle = gfile.get_cfhandle()
    ier = _cxtgeo.grd3d_imp_ecl_egrid(
        cfhandle,
        self._ncol,
        self._nrow,
        self._nlay,
        bpos["MAPAXES"],
        bpos["COORD"],
        bpos["ZCORN"],
        bpos["ACTNUM"],
        self._coordsv,
        self._zcornsv,
        self._actnumsv,
        p_nact,
        option,
    )

    gfile.cfclose()

    logger.info("Reading ECL EGRID (C code) done")
    if ier == -1:
        raise RuntimeError("Error code -1 from _cxtgeo.grd3d_imp_ecl_egrid")

    self._nactive = _cxtgeo.longpointer_value(p_nact)
    self._xtgformat = 1

    # in case of DUAL PORO/PERM ACTNUM will be 0..3; need to convert
    if self._dualporo:
        self._dualactnum = self.get_actnum(name="DUALACTNUM")
        acttmp = self._dualactnum.copy()
        acttmp.values[acttmp.values >= 1] = 1
        self.set_actnum(acttmp)

    logger.info("File is closed")


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


def import_ecl_grdecl(self, gfile):
    """Import grdecl format."""

    grdecl_grid = GrdeclGrid.from_file(gfile._file)

    self._ncol, self._nrow, self._nlay = grdecl_grid.dimensions

    self._coordsv = grdecl_grid.xtgeo_coord()
    self._zcornsv = grdecl_grid.xtgeo_zcorn()
    self._actnumsv = grdecl_grid.xtgeo_actnum()
    self._subgrids = None
    self._xtgformat = 2


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import Eclipse binary GRDECL format
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def import_ecl_bgrdecl(self, gfile):
    """Import binary files with GRDECL layout."""
    cfhandle = gfile.get_cfhandle()

    # scan file for properties; these have similar binary format as e.g. EGRID
    logger.info("Make kwlist by scanning")
    kwlist = utils.scan_keywords(
        gfile, fformat="xecl", maxkeys=1000, dataframe=False, dates=False
    )

    bpos = {}
    needkwlist = ["SPECGRID", "COORD", "ZCORN", "ACTNUM"]
    optkwlist = ["MAPAXES"]
    for name in needkwlist + optkwlist:
        bpos[name] = -1  # initially

    for kwitem in kwlist:
        kwname, kwtype, kwlen, kwbyte = kwitem
        if kwname == "SPECGRID":
            # read grid geometry record:
            specgrid = eclbin_record(gfile, "SPECGRID", kwlen, kwtype, kwbyte)
            self._ncol, self._nrow, self._nlay = specgrid[0:3].tolist()
        elif kwname in needkwlist:
            bpos[kwname] = kwbyte
        elif kwname == "MAPAXES":  # not always present
            bpos[kwname] = kwbyte

    # allocate dimensions:
    ncoord, nzcorn, ntot = self.vectordimensions

    self._coordsv = np.zeros(ncoord, dtype=np.float64)
    self._zcornsv = np.zeros(nzcorn, dtype=np.float64)
    self._actnumsv = np.zeros(ntot, dtype=np.int32)

    p_nact = _cxtgeo.new_longpointer()

    ier = _cxtgeo.grd3d_imp_ecl_egrid(
        cfhandle,
        self._ncol,
        self._nrow,
        self._nlay,
        bpos["MAPAXES"],
        bpos["COORD"],
        bpos["ZCORN"],
        bpos["ACTNUM"],
        self._coordsv,
        self._zcornsv,
        self._actnumsv,
        p_nact,
        0,
    )

    if ier == -1:
        raise RuntimeError("Error code -1 from _cxtgeo.grd3d_imp_ecl_egrid")

    self._nactive = _cxtgeo.longpointer_value(p_nact)
    self._xtgformat = 1

    # if local_fhandle:
    #     gfile.close(cond=local_fhandle)
    if gfile.cfclose():
        logger.info("Closed SWIG C file")
