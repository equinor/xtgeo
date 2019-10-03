"""Importing grid props from ECL runs, e,g, INIT, UNRST"""

from __future__ import print_function, absolute_import

import numpy as np
import numpy.ma as ma

import xtgeo
import xtgeo.cxtgeo.cxtgeo as _cxtgeo

from xtgeo.common import xtgeo_system as xsys

from . import _grid_eclbin_record as _eclbin
from . import _grid3d_utils as utils

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)

_cxtgeo.xtg_verbose_file("NONE")
XTGDEBUG = xtg.get_syslevel()


def import_eclbinary(
    self, pfile, name=None, etype=1, date=None, grid=None, fracture=False
):

    # if pfile is a file, then the file is opened/closed here; otherwise, the
    # "outer" routine must handle that

    local_fhandle = not xsys.is_fhandle(pfile)
    fhandle = xsys.get_fhandle(pfile)

    if name == "SOIL":
        # some recursive magic here
        logger.info("Making SOIL from SWAT and SGAS ...")
        logger.info("PFILE is %s", pfile)

        swat = self.__class__()
        swat.from_file(fhandle, name="SWAT", grid=grid, date=date, fformat="unrst")

        sgas = self.__class__()
        sgas.from_file(fhandle, name="SGAS", grid=grid, date=date, fformat="unrst")

        self.name = "SOIL" + "_" + str(date)
        self._nrow = swat.nrow
        self._ncol = swat.ncol
        self._nlay = swat.nlay
        self._date = date
        self._values = swat._values * -1 - sgas._values + 1.0

        if self._dualporo and fracture:
            # if fracture then self._dualactnum will be 1 at inactive fracture cells
            self._values[self._dualactnum == 1] = 0.0

        if self._dualporo and not fracture:
            # if matrix then self._dualactnum will be 2 at inactive matrix cells
            self._values[self._dualactnum == 2] = 0.0

        del swat
        del sgas

    else:
        logger.info("Importing %s", name)

        kwlist, date = _import_eclbinary_meta(self, fhandle, etype, date, grid)

        _import_eclbinary_checks1(self, grid)

        kwlist, kwname, kwlen, kwtype, kwbyte = _import_eclbinary_checks2(
            kwlist, name, etype, date
        )

        if grid._dualporo:  # _dualporo shall always be True if _dualperm is True
            _import_eclbinary_dualporo(
                self,
                grid,
                fhandle,
                kwname,
                kwlen,
                kwtype,
                kwbyte,
                name,
                date,
                etype,
                fracture,
            )
        else:
            _import_eclbinary_prop(
                self, grid, fhandle, kwname, kwlen, kwtype, kwbyte, name, date, etype
            )

    if not xsys.close_fhandle(fhandle, cond=local_fhandle):
        raise RuntimeError("Error in closing file handle for binary Eclipse file")


def _import_eclbinary_meta(self, fhandle, etype, date, grid):
    """Find settings and metadata, private to this routine.

    Returns:
        A file scan as a kwlist

    """
    nentry = 0

    datefound = True
    if etype == 5:
        datefound = False
        logger.info("Look for date %s", date)

        # scan for date and find SEQNUM entry number; also potentially update date!
        dtlist = utils.scan_dates(fhandle)
        if date == 0:
            date = dtlist[0][1]
        elif date == 9:
            date = dtlist[-1][1]

        logger.info("Redefined date is %s", date)

        for ientry, dtentry in enumerate(dtlist):
            if str(dtentry[1]) == str(date):
                datefound = True
                nentry = ientry
                break

        if not datefound:
            msg = "Date {} not found, nentry={}".format(date, nentry)
            xtg.warn(msg)
            raise xtgeo.DateNotFoundError(msg)

    # scan file for property
    logger.info("Make kwlist")
    kwlist = utils.scan_keywords(
        fhandle, fformat="xecl", maxkeys=100000, dataframe=False, dates=True
    )

    # INTEHEAD is needed to verify grid dimensions:
    for kwitem in kwlist:
        if kwitem[0] == "INTEHEAD":
            kwname, kwtype, kwlen, kwbyte, _kwdate = kwitem
            break

    # LOGIHEAD item [14] should be True, if dualporo model...
    # LOGIHEAD item [14] and [15] should be True, if dualperm (+ dualporo) model.
    #
    # However, skip this test; now just assume double number if the grid says so,
    # which kind if doubles (not exact!) the layers when reading,
    # and assign first half* to Matrix (M) and second half to Fractures (F).
    # *half: The number of active cells per half may NOT be equal

    if grid.dualporo:
        self._dualporo = True

    if grid.dualperm:
        self._dualporo = True
        self._dualperm = True

    # read INTEHEAD record:
    intehead = _eclbin.eclbin_record(fhandle, kwname, kwlen, kwtype, kwbyte)
    ncol, nrow, nlay = intehead[8:11].tolist()

    self._ncol = ncol
    self._nrow = nrow
    self._nlay = nlay

    return kwlist, date


def _import_eclbinary_checks1(self, grid):
    """Do some validations/checks"""

    if self._dualporo:
        logger.info(
            "Grid dimensions in INIT or RESTART file seems: %s %s %s",
            self._ncol,
            self._nrow,
            self._nlay,
        )
        logger.info(
            "Note! This is DUAL POROSITY grid, so actual size: %s %s %s",
            self._ncol,
            self._nrow,
            grid.nlay,
        )
    else:
        logger.info(
            "Grid dimensions in INIT or RESTART file: %s %s %s",
            self._ncol,
            self._nrow,
            self._nlay,
        )

    logger.info(
        "Grid dimensions from GRID file: %s %s %s", grid.ncol, grid.nrow, grid.nlay
    )

    if not grid.dualporo and (
        grid.ncol != self._ncol or grid.nrow != self._nrow or grid.nlay != self._nlay
    ):
        msg = "Errors in dimensions prop: {} {} {} vs grid: {} {} {} ".format(
            self._ncol, self._nrow, self._nlay, grid.ncol, grid.ncol, grid.nlay
        )
        raise RuntimeError(msg)

    if grid.dualporo and (
        grid.ncol != self._ncol
        or grid.nrow != self._nrow
        or grid.nlay * 2 != self._nlay
    ):
        msg = "Errors in dimensions prop: {} {} {} vs grid: {} {} {} ".format(
            self._ncol, self._nrow, self._nlay, grid.ncol, grid.ncol, 2 * grid.nlay
        )
        raise RuntimeError(msg)

    # reset nlay for property when dualporo
    if grid.dualporo:
        self._nlay = grid.nlay


def _import_eclbinary_checks2(kwlist, name, etype, date):
    """More checks, and returns what's needed for actual import"""

    datefound = True

    kwfound = False
    datefoundhere = False
    usedate = "0"
    restart = False

    if etype == 5:
        usedate = str(date)
        restart = True

    for kwitem in kwlist:
        kwname, kwtype, kwlen, kwbyte, kwdate = kwitem
        logger.debug("Keyword %s -  date: %s usedate: %s", kwname, kwdate, usedate)
        if name == kwname:
            kwfound = True

        if name == kwname and usedate == str(kwdate):
            logger.info("Keyword %s ok at date %s", name, usedate)
            kwname, kwtype, kwlen, kwbyte, kwdate = kwitem
            datefoundhere = True
            break

    if restart:
        if datefound and not kwfound:
            msg = "Date <{}> is found, but not keyword <{}>".format(date, name)
            xtg.warn(msg)
            raise xtgeo.KeywordNotFoundError(msg)

        if not datefoundhere and kwfound:
            msg = "The keyword <{}> exists but not for " "date <{}>".format(name, date)
            xtg.warn(msg)
            raise xtgeo.KeywordFoundNoDateError(msg)
    else:
        if not kwfound:
            msg = "The keyword <{}> is not found".format(name)
            xtg.warn(msg)
            raise xtgeo.KeywordNotFoundError(msg)

    return kwlist, kwname, kwlen, kwtype, kwbyte


def _import_eclbinary_prop(
    self, grid, fhandle, kwname, kwlen, kwtype, kwbyte, name, date, etype
):
    """Import the actual record"""

    values = _eclbin.eclbin_record(fhandle, kwname, kwlen, kwtype, kwbyte)

    self._isdiscrete = False
    use_undef = xtgeo.UNDEF
    self.codes = {}

    if kwtype == "INTE":
        self._isdiscrete = True
        use_undef = xtgeo.UNDEF_INT

        # make the code list
        uniq = np.unique(values).tolist()
        codes = dict(zip(uniq, uniq))
        codes = {key: str(val) for key, val in codes.items()}  # val: strings
        self.codes = codes

    else:
        values = values.astype(np.float64)  # cast REAL (float32) to float64

    # arrays from Eclipse INIT or UNRST are usually for inactive values only.
    # Use the ACTNUM index array for vectorized numpy remapping (need both C
    # and F order)
    gactnum = grid.get_actnum().values
    gactindc = grid.actnum_indices
    gactindf = grid.get_actnum_indices(order="F")

    allvalues = (
        np.zeros((self._ncol * self._nrow * self._nlay), dtype=values.dtype) + use_undef
    )

    msg = "\n"
    msg = msg + "grid.actnum_indices.shape[0] = {}\n".format(
        grid.actnum_indices.shape[0]
    )
    msg = msg + "values.shape[0] = {}\n".format(values.shape[0])
    msg = msg + "ncol nrow nlay {} {} {}, nrow*nrow*nlay = {}\n".format(
        self._ncol, self._nrow, self._nlay, self._ncol * self._nrow * self._nlay
    )

    logger.info(msg)

    if gactindc.shape[0] == values.shape[0]:
        allvalues[gactindf] = values
    elif values.shape[0] == self._ncol * self._nrow * self._nlay:  # often case for PORV
        allvalues = values.copy()
    else:

        msg = (
            "BUG somehow... Is the file corrupt? If not contact "
            "the library developer(s)!\n" + msg
        )
        raise SystemExit(msg)

    allvalues = allvalues.reshape((self._ncol, self._nrow, self._nlay), order="F")
    allvalues = np.asanyarray(allvalues, order="C")
    allvalues = ma.masked_where(gactnum < 1, allvalues)

    self._values = allvalues

    if etype == 1:
        self._name = name
    else:
        self._name = name + "_" + str(date)
        self._date = date


def _import_eclbinary_dualporo(
    self, grid, fhandle, kwname, kwlen, kwtype, kwbyte, name, date, etype, fracture
):
    """Import the actual record for dual poro scheme"""

    #
    # TODO, merge this with routine above!
    # A lot of code duplication here, as this is under testing
    #

    values = _eclbin.eclbin_record(fhandle, kwname, kwlen, kwtype, kwbyte)

    # arrays from Eclipse INIT or UNRST are usually for inactive values only.
    # Use the ACTNUM index array for vectorized numpy remapping (need both C
    # and F order)

    gactnum = grid.get_actnum().values
    gactindc = grid.get_dualactnum_indices(order="C", fracture=fracture)
    gactindf = grid.get_dualactnum_indices(order="F", fracture=fracture)

    indsize = gactindc.size
    if kwlen == 2 * grid.ntotal:
        indsize = grid.ntotal  # in case of e.g. PORV which is for all cells

    if not fracture:
        values = values[:indsize]
    else:
        values = values[values.size - indsize :]

    self._isdiscrete = False
    self.codes = {}

    if kwtype == "INTE":
        self._isdiscrete = True

        # make the code list
        uniq = np.unique(values).tolist()
        codes = dict(zip(uniq, uniq))
        codes = {key: str(val) for key, val in codes.items()}  # val: strings
        self.codes = codes

    else:
        values = values.astype(np.float64)  # cast REAL (float32) to float64

    allvalues = (
        np.zeros((self._ncol * self._nrow * self._nlay), dtype=values.dtype)
        + self.undef
    )

    if gactindc.shape[0] == values.shape[0]:
        allvalues[gactindf] = values
    elif values.shape[0] == self._ncol * self._nrow * self._nlay:  # often case for PORV
        allvalues = values.copy()
    else:

        msg = (
            "BUG somehow reading binary Eclipse! Is the file corrupt? If not contact "
            "the library developer(s)!\n"
        )
        raise SystemExit(msg)

    allvalues = allvalues.reshape((self._ncol, self._nrow, self.nlay), order="F")
    allvalues = np.asanyarray(allvalues, order="C")
    allvalues = ma.masked_where(gactnum < 1, allvalues)

    if self._dualporo:
        # set values which are tecnically ACTNUM active but still "UNDEF" to 0
        allvalues[allvalues > self.undef_limit] = 0

    self._values = allvalues

    append = ""
    if self._dualporo:
        append = "M"
        if fracture:
            append = "F"

    if etype == 1:
        self._name = name + append
    else:
        self._name = name + append + "_" + str(date)
        self._date = date
