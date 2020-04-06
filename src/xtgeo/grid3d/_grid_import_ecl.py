# -*- coding: utf-8 -*-

"""Grid import functions for Eclipse, new approach (i.e. version 2)"""

from __future__ import print_function, absolute_import

import re
import os
from tempfile import mkstemp
import numpy as np

import xtgeo
import xtgeo.cxtgeo._cxtgeo as _cxtgeo

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
            ncol, nrow, nlay = gridhead[1:4].tolist()
            logger.info("%s %s %s", ncol, nrow, nlay)
        elif kwname in ("COORD", "ZCORN", "ACTNUM"):
            bpos[kwname] = kwbyte
        elif kwname == "MAPAXES":  # not always present
            bpos[kwname] = kwbyte

    self._ncol = ncol
    self._nrow = nrow
    self._nlay = nlay

    logger.info("Grid dimensions in EGRID file: %s %s %s", ncol, nrow, nlay)

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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import eclipse input .GRDECL
# Uses a tmp file so not very efficient
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def import_ecl_grdecl(self, gfile):

    # make a temporary file
    fds, tmpfile = mkstemp(prefix="tmpxtgeo")
    os.close(fds)

    with open(gfile.name) as oldfile, open(tmpfile, "w") as newfile:
        for line in oldfile:
            if not (re.search(r"^--", line) or re.search(r"^\s+$", line)):
                newfile.write(line)

    newfile.close()
    oldfile.close()

    # find ncol nrow nz
    mylist = []
    found = False
    with open(tmpfile) as xfile:
        for line in xfile:
            if found:
                logger.info(line)
                mylist = line.split()
                break
            if re.search(r"^SPECGRID", line):
                found = True

    if not found:
        logger.error("SPECGRID not found. Nothing imported!")
        return
    xfile.close()

    self._ncol, self._nrow, self._nlay = int(mylist[0]), int(mylist[1]), int(mylist[2])

    logger.info("NX NY NZ in grdecl file: %s %s %s", self._ncol, self._nrow, self._nlay)

    ncoord, nzcorn, ntot = self.vectordimensions

    logger.info("Reading...")

    self._coordsv = np.zeros(ncoord, dtype=np.float64)
    self._zcornsv = np.zeros(nzcorn, dtype=np.float64)
    self._actnumsv = np.zeros(ntot, dtype=np.int32)

    ptr_num_act = _cxtgeo.new_intpointer()

    cfhandle = gfile.get_cfhandle()

    _cxtgeo.grd3d_import_grdecl(
        cfhandle,
        self._ncol,
        self._nrow,
        self._nlay,
        self._coordsv,
        self._zcornsv,
        self._actnumsv,
        ptr_num_act,
    )

    # close and remove tmpfile
    gfile.cfclose()
    os.remove(tmpfile)

    nact = _cxtgeo.intpointer_value(ptr_num_act)

    logger.info("Number of active cells: %s", nact)
    self._subgrids = None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import Eclipse binary GRDECL format
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def import_ecl_bgrdecl(self, gfile):
    """Import binary files with GRDECL layout"""

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
            ncol, nrow, nlay = specgrid[0:3].tolist()
            logger.info("%s %s %s", ncol, nrow, nlay)
        elif kwname in needkwlist:
            bpos[kwname] = kwbyte
        elif kwname == "MAPAXES":  # not always present
            bpos[kwname] = kwbyte

    self._ncol = ncol
    self._nrow = nrow
    self._nlay = nlay

    logger.info("Grid dimensions in binary GRDECL file: %s %s %s", ncol, nrow, nlay)

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

    # if local_fhandle:
    #     gfile.close(cond=local_fhandle)
    if gfile.cfclose():
        logger.info("Closed SWIG C file")
