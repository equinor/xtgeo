# -*- coding: utf-8 -*-

"""Grid import functions for Eclipse, new approach (i.e. version 2)."""

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


def vectordimensions(ncol, nrow, nlay):
    ncoord = (ncol + 1) * (nrow + 1) * 2 * 3
    nzcorn = ncol * nrow * (nlay + 1) * 4
    ntot = ncol * nrow * nlay

    return (ncoord, nzcorn, ntot)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import Eclipse result .EGRID
# See notes in grid.py on dual porosity / dual perm scheme.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def import_ecl_egrid(gfile):
    """Import, private to this routine."""
    # scan file for property
    logger.info("Make kwlist by scanning")
    kwlist = utils.scan_keywords(
        gfile, fformat="xecl", maxkeys=1000, dataframe=False, dates=False
    )
    bpos = {}
    for name in ("COORD", "ZCORN", "ACTNUM", "MAPAXES"):
        bpos[name] = -1  # initially

    dualporo = False
    dualperm = False
    for kwitem in kwlist:
        kwname, kwtype, kwlen, kwbyte = kwitem
        if kwname == "FILEHEAD":
            # read FILEHEAD record:
            filehead = eclbin_record(gfile, "FILEHEAD", kwlen, kwtype, kwbyte)
            dualp = filehead[5].tolist()
            logger.info("Dual porosity flag is %s", dualp)
            if dualp == 1:
                dualporo = True
                dualperm = False
            elif dualp == 2:
                dualporo = True
                dualperm = True
        elif kwname == "GRIDHEAD":
            # read GRIDHEAD record:
            gridhead = eclbin_record(gfile, "GRIDHEAD", kwlen, kwtype, kwbyte)
            ncol, nrow, nlay = gridhead[1:4].tolist()
        elif kwname in ("COORD", "ZCORN", "ACTNUM"):
            bpos[kwname] = kwbyte
        elif kwname == "MAPAXES":  # not always present
            bpos[kwname] = kwbyte

    logger.info("Grid dimensions in EGRID file: %s %s %s", ncol, nrow, nlay)

    # allocate dimensions:
    ncoord, nzcorn, ntot = vectordimensions(ncol, nrow, nlay)

    coordsv = np.zeros(ncoord, dtype=np.float64)
    zcornsv = np.zeros(nzcorn, dtype=np.float64)
    actnumsv = np.zeros(ntot, dtype=np.int32)

    p_nact = _cxtgeo.new_longpointer()

    option = 0
    if dualporo:
        option = 1

    cfhandle = gfile.get_cfhandle()
    ier = _cxtgeo.grd3d_imp_ecl_egrid(
        cfhandle,
        ncol,
        nrow,
        nlay,
        bpos["MAPAXES"],
        bpos["COORD"],
        bpos["ZCORN"],
        bpos["ACTNUM"],
        coordsv,
        zcornsv,
        actnumsv,
        p_nact,
        option,
    )

    gfile.cfclose()

    logger.info("Reading ECL EGRID (C code) done")
    if ier == -1:
        raise RuntimeError("Error code -1 from _cxtgeo.grd3d_imp_ecl_egrid")
    args = {
        "xtgformat": 1,
        "ncol": ncol,
        "nrow": nrow,
        "nlay": nlay,
        "coordsv": coordsv,
        "zcornsv": zcornsv,
        "actnumsv": actnumsv,
        "dualporo": dualporo,
        "dualperm": dualperm,
    }

    # in case of DUAL PORO/PERM ACTNUM will be 0..3; need to convert
    if dualporo:
        act = xtgeo.grid3d.GridProperty(
            ncol=ncol,
            nrow=nrow,
            nlay=nlay,
            values=np.zeros((ncol, nrow, nlay), dtype=np.int32),
            name=name,
            discrete=True,
        )
        val = np.reshape(actnumsv, (ncol, nrow, nlay), order="F")
        val = np.asanyarray(val, order="C")
        val = val.ravel(order="K")

        act.values = val
        act.mask_undef()
        act.codes = {0: "0", 1: "1"}
        args["dualactnum"] = act

        acttmp = act.copy()
        acttmp.values[acttmp.values >= 1] = 1
        val1d = acttmp.values.ravel(order="K")

        val = np.reshape(val1d, (ncol, nrow, nlay), order="C")
        val = np.asanyarray(val, order="F")
        val = val.ravel(order="K")
        args["actnumsv"] = val
    logger.info("File is closed")
    return args


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import eclipse run suite: EGRID + properties from INIT and UNRST
# For the INIT and UNRST, props dates shall be selected
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def import_ecl_run(groot, initprops=None, restartprops=None, restartdates=None):
    """Import combo ECL runs."""
    ecl_grid = groot + ".EGRID"
    ecl_init = groot + ".INIT"
    ecl_rsta = groot + ".UNRST"

    ecl_grid = xtgeo._XTGeoFile(ecl_grid)
    ecl_init = xtgeo._XTGeoFile(ecl_init)
    ecl_rsta = xtgeo._XTGeoFile(ecl_rsta)
    # import the grid
    args = import_ecl_egrid(ecl_grid)

    grdprops = xtgeo.grid3d.GridProperties()

    # import the init properties unless list is empty
    grid = xtgeo.Grid(**args)
    if initprops:
        grdprops.from_file(
            ecl_init.name, names=initprops, fformat="init", dates=None, grid=grid
        )

    # import the restart properties for dates unless lists are empty
    if restartprops and restartdates:
        grdprops.from_file(
            ecl_rsta.name,
            names=restartprops,
            fformat="unrst",
            dates=restartdates,
            grid=grid,
        )
    args["props"] = grdprops
    return args


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import eclipse input .GRDECL
# Uses a tmp file so not very efficient
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def import_ecl_grdecl(gfile):
    """Import grdecl format."""
    # make a temporary file
    fds, tmpfile = mkstemp(prefix="tmpxtgeo")
    os.close(fds)

    with open(gfile.name) as oldfile, open(tmpfile, "w") as newfile:
        for line in oldfile:
            if not (re.search(r"^--", line) or re.search(r"^\s+$", line)):
                newfile.write(line)

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
    ncol, nrow, nlay = int(mylist[0]), int(mylist[1]), int(mylist[2])

    logger.info("NX NY NZ in grdecl file: %s %s %s", ncol, nrow, nlay)

    ncoord, nzcorn, ntot = vectordimensions(ncol, nrow, nlay)

    coordsv = np.zeros(ncoord, dtype=np.float64)
    zcornsv = np.zeros(nzcorn, dtype=np.float64)
    actnumsv = np.zeros(ntot, dtype=np.int32)

    _cxtgeo.grd3d_import_grdecl(
        gfile.get_cfhandle(),
        ncol,
        nrow,
        nlay,
        coordsv,
        zcornsv,
        actnumsv,
        _cxtgeo.new_intpointer(),
    )

    # close and remove tmpfile
    gfile.cfclose()
    os.remove(tmpfile)

    args = {
        "ncol": ncol,
        "nrow": nrow,
        "nlay": nlay,
        "coordsv": coordsv,
        "zcornsv": zcornsv,
        "actnumsv": actnumsv,
        "xtgformat": 1,
    }

    return args


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import Eclipse binary GRDECL format
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def import_ecl_bgrdecl(gfile):
    """Import binary files with GRDECL layout."""

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
        elif kwname in needkwlist:
            bpos[kwname] = kwbyte
        elif kwname == "MAPAXES":  # not always present
            bpos[kwname] = kwbyte

    # allocate dimensions:
    ncoord, nzcorn, ntot = vectordimensions(ncol, nrow, nlay)

    coordsv = np.zeros(ncoord, dtype=np.float64)
    zcornsv = np.zeros(nzcorn, dtype=np.float64)
    actnumsv = np.zeros(ntot, dtype=np.int32)

    ier = _cxtgeo.grd3d_imp_ecl_egrid(
        gfile.get_cfhandle(),
        ncol,
        nrow,
        nlay,
        bpos["MAPAXES"],
        bpos["COORD"],
        bpos["ZCORN"],
        bpos["ACTNUM"],
        coordsv,
        zcornsv,
        actnumsv,
        _cxtgeo.new_longpointer(),
        0,
    )

    if ier == -1:
        raise RuntimeError("Error code -1 from _cxtgeo.grd3d_imp_ecl_egrid")

    if gfile.cfclose():
        logger.info("Closed SWIG C file")

    args = {
        "ncol": ncol,
        "nrow": nrow,
        "nlay": nlay,
        "coordsv": coordsv,
        "zcornsv": zcornsv,
        "actnumsv": actnumsv,
        "xtgformat": 1,
    }

    return args
