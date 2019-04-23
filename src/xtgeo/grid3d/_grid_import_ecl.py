# -*- coding: utf-8 -*-

"""Grid import functions for Eclipse, new approach (i.e. version 2)"""

from __future__ import print_function, absolute_import

import re
import os
from tempfile import mkstemp

import xtgeo
import xtgeo.cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

from xtgeo.grid3d._gridprop_import import eclbin_record

from xtgeo.common import _get_fhandle, _close_fhandle

from . import _grid3d_utils as utils

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

_cxtgeo.xtg_verbose_file("NONE")
XTGDEBUG = xtg.get_syslevel()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import Eclipse result .EGRID
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def import_ecl_egrid(self, gfile):
    """Import, private to this routine.

    """
    fhandle, pclose = _get_fhandle(gfile)

    # scan file for property
    logger.info("Make kwlist by scanning")
    kwlist = utils.scan_keywords(
        fhandle, fformat="xecl", maxkeys=1000, dataframe=False, dates=False
    )
    bpos = {}
    for name in ("COORD", "ZCORN", "ACTNUM", "MAPAXES"):
        bpos[name] = -1  # initially

    for kwitem in kwlist:
        kwname, kwtype, kwlen, kwbyte = kwitem
        if kwname == "GRIDHEAD":
            # read GRIDHEAD record:
            gridhead = eclbin_record(fhandle, "GRIDHEAD", kwlen, kwtype, kwbyte)
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
    ntot = self._ncol * self._nrow * self._nlay
    ncoord = (self._ncol + 1) * (self._nrow + 1) * 2 * 3
    nzcorn = self._ncol * self._nrow * (self._nlay + 1) * 4

    self._p_coord_v = _cxtgeo.new_doublearray(ncoord)
    self._p_zcorn_v = _cxtgeo.new_doublearray(nzcorn)
    self._p_actnum_v = _cxtgeo.new_intarray(ntot)

    nact = _cxtgeo.grd3d_imp_ecl_egrid(
        fhandle,
        self._ncol,
        self._nrow,
        self._nlay,
        bpos["MAPAXES"],
        bpos["COORD"],
        bpos["ZCORN"],
        bpos["ACTNUM"],
        self._p_coord_v,
        self._p_zcorn_v,
        self._p_actnum_v,
        XTGDEBUG,
    )

    self._nactive = nact

    _close_fhandle(fhandle, pclose)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import eclipse run suite: EGRID + properties from INIT and UNRST
# For the INIT and UNRST, props dates shall be selected
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def import_ecl_run(self, groot, initprops=None, restartprops=None, restartdates=None):

    ecl_grid = groot + ".EGRID"
    ecl_init = groot + ".INIT"
    ecl_rsta = groot + ".UNRST"

    # import the grid
    import_ecl_egrid(self, ecl_grid)

    grdprops = xtgeo.grid3d.GridProperties()

    # import the init properties unless list is empty
    if initprops:
        grdprops.from_file(
            ecl_init, names=initprops, fformat="init", dates=None, grid=self
        )

    # import the restart properties for dates unless lists are empty
    if restartprops and restartdates:
        grdprops.from_file(
            ecl_rsta, names=restartprops, fformat="unrst", dates=restartdates, grid=self
        )

    self.gridprops = grdprops


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import eclipse input .GRDECL
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def import_ecl_grdecl(self, gfile):

    # make a temporary file
    fds, tmpfile = mkstemp()
    # make a temporary

    with open(gfile) as oldfile, open(tmpfile, "w") as newfile:
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

    ntot = self._ncol * self._nrow * self._nlay
    ncoord = (self._ncol + 1) * (self._nrow + 1) * 2 * 3
    nzcorn = self._ncol * self._nrow * (self._nlay + 1) * 4

    logger.info("Reading...")

    ptr_num_act = _cxtgeo.new_intpointer()
    self._p_coord_v = _cxtgeo.new_doublearray(ncoord)
    self._p_zcorn_v = _cxtgeo.new_doublearray(nzcorn)
    self._p_actnum_v = _cxtgeo.new_intarray(ntot)

    _cxtgeo.grd3d_import_grdecl(
        self._ncol,
        self._nrow,
        self._nlay,
        self._p_coord_v,
        self._p_zcorn_v,
        self._p_actnum_v,
        ptr_num_act,
        tmpfile,
        XTGDEBUG,
    )

    # remove tmpfile
    os.close(fds)
    os.remove(tmpfile)

    nact = _cxtgeo.intpointer_value(ptr_num_act)

    logger.info("Number of active cells: %s", nact)
    self._subgrids = None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import Eclipse binary GRDECL format
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def import_ecl_bgrdecl(self, gfile):
    """Import binary files with GRDECL layout"""

    fhandle, pclose = _get_fhandle(gfile)

    # scan file for properties; these have similar binary format as e.g. EGRID
    logger.info("Make kwlist by scanning")
    kwlist = utils.scan_keywords(
        fhandle, fformat="xecl", maxkeys=1000, dataframe=False, dates=False
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
            specgrid = eclbin_record(fhandle, "SPECGRID", kwlen, kwtype, kwbyte)
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
    ntot = self._ncol * self._nrow * self._nlay
    ncoord = (self._ncol + 1) * (self._nrow + 1) * 2 * 3
    nzcorn = self._ncol * self._nrow * (self._nlay + 1) * 4

    self._p_coord_v = _cxtgeo.new_doublearray(ncoord)
    self._p_zcorn_v = _cxtgeo.new_doublearray(nzcorn)
    self._p_actnum_v = _cxtgeo.new_intarray(ntot)

    nact = _cxtgeo.grd3d_imp_ecl_egrid(
        fhandle,
        self._ncol,
        self._nrow,
        self._nlay,
        bpos["MAPAXES"],
        bpos["COORD"],
        bpos["ZCORN"],
        bpos["ACTNUM"],
        self._p_coord_v,
        self._p_zcorn_v,
        self._p_actnum_v,
        XTGDEBUG,
    )

    self._nactive = nact

    _close_fhandle(fhandle, pclose)
