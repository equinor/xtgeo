"""GridProperty (not GridProperies) import functions"""

from __future__ import print_function, absolute_import

import re
import os
from tempfile import mkstemp

import numpy as np
import numpy.ma as ma

import xtgeo.cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.common.exceptions import DateNotFoundError
from xtgeo.common.exceptions import KeywordFoundNoDateError
from xtgeo.common.exceptions import KeywordNotFoundError

from xtgeo.common import _get_fhandle, _close_fhandle

from . import _gridprop_lowlevel
from . import _grid3d_utils as utils

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

_cxtgeo.xtg_verbose_file("NONE")
XTGDEBUG = xtg.get_syslevel()


def from_file(
    self, pfile, fformat=None, name="unknown", grid=None, date=None, _roffapiv=1
):  # _roffapiv for devel.
    """Import grid property from file, and makes an instance of this."""

    # pylint: disable=too-many-branches, too-many-statements

    self._filesrc = pfile

    # it may be that pfile already is an open file; hence a filehandle
    # instead. Check for this, and skip tests if so
    pfile_is_not_fhandle = True
    _fhandle, pclose = _get_fhandle(pfile)
    if not pclose:
        pfile_is_not_fhandle = False

    if pfile_is_not_fhandle:
        if os.path.isfile(pfile):
            logger.debug("File %s exists OK", pfile)
        else:
            raise IOError("No such file: {}".format(pfile))

        # work on file extension
        _froot, fext = os.path.splitext(pfile)
        if fformat is None or fformat == "guess":
            if not fext:
                raise ValueError("File extension missing. STOP")

            fformat = fext.lower().replace(".", "")

        logger.debug("File name to be used is %s", pfile)
        logger.debug("File format is %s", fformat)

    ier = 0
    if fformat == "roff":
        logger.info("Importing ROFF...")
        ier = import_roff(self, pfile, name, grid=grid, _roffapiv=_roffapiv)

    elif fformat.lower() == "init":
        ier = import_eclbinary(self, pfile, name=name, etype=1, date=None, grid=grid)

    elif fformat.lower() == "unrst":
        if date is None:
            raise ValueError("Restart file, but no date is given")

        if isinstance(date, str):
            if "-" in date:
                date = int(date.replace("-", ""))
            elif date == "first":
                date = 0
            elif date == "last":
                date = 9
            else:
                date = int(date)

        if not isinstance(date, int):
            raise RuntimeError("Date is not int format")

        ier = import_eclbinary(self, pfile, name=name, etype=5, date=date, grid=grid)

    elif fformat.lower() == "grdecl":
        ier = import_grdecl_prop(self, pfile, name=name, grid=grid)

    elif fformat.lower() == "bgrdecl":
        ier = import_bgrdecl_prop(self, pfile, name=name, grid=grid)
    else:
        logger.warning("Invalid file format")
        raise SystemExit("Invalid file format")

    # if grid, then append this gridprop to the current grid object
    if ier == 0:
        if grid:
            grid.append_prop(self)
    elif ier == 22:
        raise DateNotFoundError(
            "Date {} not found when importing {}".format(date, name)
        )
    elif ier == 23:
        raise KeywordNotFoundError(
            "Keyword {} not found for date {} when importing".format(name, date)
        )
    elif ier == 24:
        raise KeywordFoundNoDateError(
            "Keyword {} found but not for date " "{} when importing".format(name, date)
        )
    elif ier == 25:
        raise KeywordNotFoundError("Keyword {} not found when importing".format(name))
    else:
        raise RuntimeError("Something went wrong, code {}".format(ier))

    return self


def import_eclbinary(self, pfile, name=None, etype=1, date=None, grid=None):
    ios = 0
    if name == "SOIL":
        # some recursive magic here
        logger.info("Making SOIL from SWAT and SGAS ...")
        logger.info("PFILE is %s", pfile)

        swat = self.__class__()
        swat.from_file(pfile, name="SWAT", grid=grid, date=date, fformat="unrst")

        sgas = self.__class__()
        sgas.from_file(pfile, name="SGAS", grid=grid, date=date, fformat="unrst")

        self.name = "SOIL" + "_" + str(date)
        self._nrow = swat.nrow
        self._ncol = swat.ncol
        self._nlay = swat.nlay
        self._date = date
        self._values = swat._values * -1 - sgas._values + 1.0

        del swat
        del sgas

    else:
        logger.info("Importing %s", name)
        ios = _import_eclbinary(
            self, pfile, name=name, etype=etype, date=date, grid=grid
        )

    return ios


def eclbin_record(fhandle, kwname, kwlen, kwtype, kwbyte):
    # read a binary Eclipse record via cxtgeo

    ilen = flen = dlen = 1

    if kwtype == "INTE":
        ilen = kwlen
        kwntype = 1
    elif kwtype == "REAL":
        flen = kwlen
        kwntype = 2
    elif kwtype == "DOUB":
        dlen = kwlen
        kwntype = 3
    else:
        raise ValueError(
            "Wrong type of kwtype {} for {}, must be INTE, REAL "
            "or DOUB".format(kwtype, kwname)
        )

    npint = np.zeros((ilen), dtype=np.int32)
    npflt = np.zeros((flen), dtype=np.float32)
    npdbl = np.zeros((dlen), dtype=np.float64)

    _cxtgeo.grd3d_read_eclrecord(
        fhandle, kwbyte, kwntype, npint, npflt, npdbl, XTGDEBUG
    )

    npuse = None
    if kwtype == "INTE":
        npuse = npint
        del npflt
        del npdbl
    if kwtype == "REAL":
        npuse = npflt
        del npint
        del npdbl
    if kwtype == "DOUB":
        npuse = npdbl
        del npint
        del npflt

    return npuse


def _import_eclbinary(self, pfile, name=None, etype=1, date=None, grid=None):
    """Import, private to this routine.

    Raises:
        DateNotFoundError: If restart do not contain requested date.
        KeywordFoundNoDateError: If keyword is found but not at given date.
        KeywordNotFoundError: If Keyword is not found.
        RuntimeError: Mismatch in grid vs property, etc.

    """
    # This function requires simplification!
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements

    fhandle, pclose = _get_fhandle(pfile)

    nentry = 0

    datefound = True
    if etype == 5:
        datefound = False
        logger.info("Look for date %s", date)

        # scan for date and find SEQNUM entry number
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
            msg = "In {}: Date {} not found, nentry={}".format(pfile, date, nentry)
            xtg.warn(msg)
            raise DateNotFoundError(msg)

    # scan file for property
    logger.info("Make kwlist")
    kwlist = utils.scan_keywords(
        fhandle, fformat="xecl", maxkeys=100000, dataframe=False, dates=True
    )

    # first INTEHEAD is needed to verify grid dimensions:
    for kwitem in kwlist:
        if kwitem[0] == "INTEHEAD":
            kwname, kwtype, kwlen, kwbyte, kwdate = kwitem
            break

    # read INTEHEAD record:
    intehead = eclbin_record(fhandle, kwname, kwlen, kwtype, kwbyte)
    ncol, nrow, nlay = intehead[8:11].tolist()

    self._ncol = ncol
    self._nrow = nrow
    self._nlay = nlay

    logger.info("Grid dimensions in INIT or RESTART file: %s %s %s", ncol, nrow, nlay)

    logger.info(
        "Grid dimensions from GRID file: %s %s %s", grid.ncol, grid.nrow, grid.nlay
    )

    if grid.ncol != ncol or grid.nrow != nrow or grid.nlay != nlay:
        msg = "In {}: Errors in dimensions prop: {} {} {} vs grid: {} {} {} ".format(
            pfile, ncol, nrow, nlay, grid.ncol, grid.ncol, grid.nlay
        )
        raise RuntimeError(msg)

    # Restarts (etype == 5):
    # there are cases where keywords do not exist for all dates, e.g .'RV'.
    # The trick is to check for dates also...

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
            msg = "For {}: Date <{}> is found, but not keyword <{}>".format(
                pfile, date, name
            )
            xtg.warn(msg)
            raise KeywordNotFoundError(msg)

        if not datefoundhere and kwfound:
            msg = "For {}: The keyword <{}> exists but not for " "date <{}>".format(
                pfile, name, date
            )
            xtg.warn(msg)
            raise KeywordFoundNoDateError(msg)
    else:
        if not kwfound:
            msg = "For {}: The keyword <{}> is not found".format(pfile, name)
            xtg.warn(msg)
            raise KeywordNotFoundError(msg)

    # read record:
    values = eclbin_record(fhandle, kwname, kwlen, kwtype, kwbyte)

    if kwtype == "INTE":
        self._isdiscrete = True
        use_undef = self._undef_i

        # make the code list
        uniq = np.unique(values).tolist()
        codes = dict(zip(uniq, uniq))
        codes = {key: str(val) for key, val in codes.items()}  # val: strings
        self.codes = codes

    else:
        self._isdiscrete = False
        values = values.astype(np.float64)  # cast REAL (float32) to float64
        use_undef = self._undef
        self.codes = {}

    # arrays from Eclipse INIT or UNRST are usually for inactive values only.
    # Use the ACTNUM index array for vectorized numpy remapping
    actnum = grid.get_actnum().values
    allvalues = np.zeros((ncol * nrow * nlay), dtype=values.dtype) + use_undef

    msg = "\n"
    msg = msg + "grid.actnum_indices.shape[0] = {}\n".format(
        grid.actnum_indices.shape[0]
    )
    msg = msg + "values.shape[0] = {}\n".format(values.shape[0])
    msg = msg + "ncol nrow nlay {} {} {}, nrow*nrow*nlay = {}\n".format(
        ncol, nrow, nlay, ncol * nrow * nlay
    )

    logger.info(msg)

    if grid.actnum_indices.shape[0] == values.shape[0]:
        allvalues[grid.get_actnum_indices(order="F")] = values
    elif values.shape[0] == ncol * nrow * nlay:  # often case for PORV array
        allvalues = values.copy()
    else:
        msg = (
            "BUG somehow... Is the file corrupt? If not contact "
            "the library developer(s)!\n" + msg
        )
        raise SystemExit(msg)

    allvalues = allvalues.reshape((ncol, nrow, nlay), order="F")
    allvalues = np.asanyarray(allvalues, order="C")
    allvalues = ma.masked_where(actnum < 1, allvalues)

    _close_fhandle(fhandle, pclose)

    self._values = allvalues

    if etype == 1:
        self._name = name
    else:
        self._name = name + "_" + str(date)
        self._date = date

    return 0


def import_bgrdecl_prop(self, pfile, name="unknown", grid=None):
    """Import property from binary files with GRDECL layout"""

    fhandle, pclose = _get_fhandle(pfile)

    # scan file for properties; these have similar binary format as e.g. EGRID
    logger.info("Make kwlist by scanning")
    kwlist = utils.scan_keywords(
        fhandle, fformat="xecl", maxkeys=1000, dataframe=False, dates=False
    )
    bpos = {}
    bpos[name] = -1

    for kwitem in kwlist:
        kwname, kwtype, kwlen, kwbyte = kwitem
        logger.info("KWITEM: %s", kwitem)
        if name == kwname:
            bpos[name] = kwbyte
            break

    if bpos[name] == -1:
        raise KeywordNotFoundError(
            "Cannot find property name {} in file {}".format(name, pfile)
        )
    self._ncol = grid.ncol
    self._nrow = grid.nrow
    self._nlay = grid.nlay

    values = eclbin_record(fhandle, kwname, kwlen, kwtype, kwbyte)
    if kwtype == "INTE":
        self._isdiscrete = True
        # make the code list
        uniq = np.unique(values).tolist()
        codes = dict(zip(uniq, uniq))
        codes = {key: str(val) for key, val in codes.items()}  # val: strings
        self.codes = codes

    else:
        self._isdiscrete = False
        values = values.astype(np.float64)  # cast REAL (float32) to float64
        self.codes = {}

    # property arrays from binary GRDECL will be for all cells, but they
    # are in Fortran order, so need to convert...

    actnum = grid.get_actnum().values
    allvalues = values.reshape(self.dimensions, order="F")
    allvalues = np.asanyarray(allvalues, order="C")
    allvalues = ma.masked_where(actnum < 1, allvalues)
    self.values = allvalues
    self._name = name

    _close_fhandle(fhandle, pclose)
    return 0


def import_grdecl_prop(self, pfile, name="unknown", grid=None):
    """Read a GRDECL ASCII property record"""

    if grid is None:
        raise ValueError("A grid instance is required as argument")

    self._ncol = grid.ncol
    self._nrow = grid.nrow
    self._nlay = grid.nlay
    self._name = name
    self._filesrc = pfile
    actnumv = grid.get_actnum().values

    # This requires that the Python part clean up comments
    # etc, and make a tmp file.

    # make a temporary file
    fds, tmpfile = mkstemp()
    # make a temporary

    with open(pfile) as oldfile, open(tmpfile, "w") as newfile:
        for line in oldfile:
            if not (re.search(r"^--", line) or re.search(r"^\s+$", line)):
                newfile.write(line)

    newfile.close()
    oldfile.close()

    # now read the property
    nlen = self._ncol * self._nrow * self._nlay
    ier, values = _cxtgeo.grd3d_import_grdecl_prop(
        tmpfile, self._ncol, self._nrow, self._nlay, name, nlen, 0, XTGDEBUG
    )
    # remove tmpfile
    os.close(fds)
    os.remove(tmpfile)

    if ier != 0:
        raise KeywordNotFoundError(
            "Cannot import {}, not present in file {}?".format(name, pfile)
        )

    self.values = values.reshape(self.dimensions)

    self.values = ma.masked_equal(self.values, actnumv == 0)

    return 0


def import_roff(self, pfile, name, grid=None, _roffapiv=1):
    """Import ROFF format"""

    logger.info("Keyword grid is inactive, values is: %s", grid)

    if _roffapiv <= 1:
        status = _import_roff_v1(self, pfile, name)
        if status != 0:
            raise SystemExit("Error from ROFF import")
    elif _roffapiv == 2:
        status = _import_roff_v2(self, pfile, name)

    return status


def _import_roff_v1(self, pfile, name):
    """Import ROFF format, version 1"""
    # pylint: disable=too-many-locals

    # there is a todo here to get it more robust for various cases,
    # e.g. that a ROFF file may contain both a grid an numerous
    # props

    logger.info("Looking for %s in file %s", name, pfile)

    ptr_ncol = _cxtgeo.new_intpointer()
    ptr_nrow = _cxtgeo.new_intpointer()
    ptr_nlay = _cxtgeo.new_intpointer()
    ptr_ncodes = _cxtgeo.new_intpointer()
    ptr_type = _cxtgeo.new_intpointer()

    ptr_idum = _cxtgeo.new_intpointer()
    ptr_ddum = _cxtgeo.new_doublepointer()

    # read with mode 0, to scan for ncol, nrow, nlay and ndcodes, and if
    # property is found...
    ier, _codenames = _cxtgeo.grd3d_imp_prop_roffbin(
        pfile,
        0,
        ptr_type,
        ptr_ncol,
        ptr_nrow,
        ptr_nlay,
        ptr_ncodes,
        name,
        ptr_idum,
        ptr_ddum,
        ptr_idum,
        0,
        XTGDEBUG,
    )

    if ier == -1:
        msg = "Cannot find property name {}".format(name)
        logger.warning(msg)
        return ier

    self._ncol = _cxtgeo.intpointer_value(ptr_ncol)
    self._nrow = _cxtgeo.intpointer_value(ptr_nrow)
    self._nlay = _cxtgeo.intpointer_value(ptr_nlay)
    self._ncodes = _cxtgeo.intpointer_value(ptr_ncodes)

    ptype = _cxtgeo.intpointer_value(ptr_type)

    ntot = self._ncol * self._nrow * self._nlay

    if self._ncodes <= 1:
        self._ncodes = 1
        self._codes = {0: "undef"}

    logger.debug("Number of codes: %s", self._ncodes)

    # allocate

    if ptype == 1:  # float, assign to double
        ptr_pval_v = _cxtgeo.new_doublearray(ntot)
        ptr_ival_v = _cxtgeo.new_intarray(1)
        self._isdiscrete = False
        self._dtype = "float64"

    elif ptype > 1:
        ptr_pval_v = _cxtgeo.new_doublearray(1)
        ptr_ival_v = _cxtgeo.new_intarray(ntot)
        self._isdiscrete = True
        self._dtype = "int32"

    # number of codes and names
    ptr_ccodes_v = _cxtgeo.new_intarray(self._ncodes)

    # NB! note the SWIG trick to return modified char values; use cstring.i
    # inn the config and %cstring_bounded_output(char *p_codenames_v, NN);
    # Then the argument for *p_codevalues_v in C is OMITTED here!

    ier, cnames = _cxtgeo.grd3d_imp_prop_roffbin(
        pfile,
        1,
        ptr_type,
        ptr_ncol,
        ptr_nrow,
        ptr_nlay,
        ptr_ncodes,
        name,
        ptr_ival_v,
        ptr_pval_v,
        ptr_ccodes_v,
        0,
        XTGDEBUG,
    )

    if self._isdiscrete:
        _gridprop_lowlevel.update_values_from_carray(
            self, ptr_ival_v, np.int32, delete=True
        )
    else:
        _gridprop_lowlevel.update_values_from_carray(
            self, ptr_pval_v, np.float64, delete=True
        )

    # now make dictionary of codes
    if self._isdiscrete:
        cnames = cnames.replace(";", "")
        cname_list = cnames.split("|")
        cname_list.pop()  # some rubbish as last entry
        ccodes = []
        for ino in range(0, self._ncodes):
            ccodes.append(_cxtgeo.intarray_getitem(ptr_ccodes_v, ino))

        self._codes = dict(zip(ccodes, cname_list))

    self._name = name

    return 0


def _import_roff_v2(self, pfile, name):
    """Import ROFF format, version 2 (improved version)"""

    # This routine do first a scan for all keywords. Then it grabs
    # the relevant data by only reading relevant portions of the input file

    fhandle, _pclose = _get_fhandle(pfile)

    kwords = utils.scan_keywords(fhandle, fformat="roff")

    for kwd in kwords:
        logger.info(kwd)

    # byteswap:
    byteswap = _rkwquery(fhandle, kwords, "filedata!byteswaptest", -1)

    ncol = _rkwquery(fhandle, kwords, "dimensions!nX", byteswap)
    nrow = _rkwquery(fhandle, kwords, "dimensions!nY", byteswap)
    nlay = _rkwquery(fhandle, kwords, "dimensions!nZ", byteswap)
    logger.info("Dimensions in ROFF file %s %s %s", ncol, nrow, nlay)

    # get the actual parameter:
    vals = _rarraykwquery(
        self, fhandle, kwords, "parameter!name!" + name, byteswap, ncol, nrow, nlay
    )

    self._values = vals
    self._name = name

    return 0


def _rkwquery(fhandle, kws, name, swap):
    """Local function for _import_roff_v2, single data"""

    kwtypedict = {"int": 1, "float": 2}
    iresult = _cxtgeo.new_intpointer()
    presult = _cxtgeo.new_floatpointer()

    dtype = 0
    reclen = 0
    bytepos = 1
    for items in kws:
        if name in items[0]:
            dtype = kwtypedict.get(items[1])
            reclen = items[2]
            bytepos = items[3]
            break

    if dtype == 0:
        raise ValueError("Cannot find property <{}> in file".format(name))

    if reclen != 1:
        raise SystemError("Stuff is rotten here...")

    _cxtgeo.grd3d_imp_roffbin_data(
        fhandle, swap, dtype, bytepos, iresult, presult, XTGDEBUG
    )

    # -1 indicates that it is the swap flag which is looked for!
    if dtype == 1:
        xresult = _cxtgeo.intpointer_value(iresult)
        print("xresult = {}".format(xresult))
        if swap == -1:
            if xresult == 1:
                return 0
            return 1

    return xresult


def _rarraykwquery(self, fhandle, kws, name, swap, ncol, nrow, nlay):
    """Local function for _import_roff_v2, 3D parameter arrays.

    This parameters are translated to numpy data for the values
    attribute usage.

    Note from scan:
    parameter!name!PORO   char        1            310
    parameter!data        float   35840            336

    Hence it is the parameter!data which comes after parameter!name!PORO which
    is releveant here, given that name = PORO.

    """

    kwtypedict = {"int": 1, "float": 2, "double": 3, "byte": 5}

    dtype = 0
    reclen = 0
    bytepos = 1
    namefound = False
    for items in kws:
        if name in items[0]:
            dtype = kwtypedict.get(items[1])
            reclen = items[2]
            bytepos = items[3]
            namefound = True
        if "parameter!data" in items[0] and namefound:
            dtype = kwtypedict.get(items[1])
            reclen = items[2]
            bytepos = items[3]
            break

    if dtype == 0:
        raise ValueError("Cannot find property <{}> in file".format(name))

    if reclen <= 1:
        raise SystemError("Stuff is rotten here...")

    inumpy = np.zeros(ncol * nrow * nlay, dtype=np.int32)
    fnumpy = np.zeros(ncol * nrow * nlay, dtype=np.float32)

    _cxtgeo.grd3d_imp_roffbin_arr(
        fhandle, swap, ncol, nrow, nlay, bytepos, dtype, fnumpy, inumpy, XTGDEBUG
    )

    # remember that for grid props, order=F in CXTGEO, while order=C
    # in xtgeo-python!
    if dtype == 1:
        vals = inumpy
        # vals = inumpy.reshape((ncol, nrow, nlay), order='F')
        # vals = np.asanyarray(vals, order='C')
        vals = ma.masked_greater(vals, self._undef_ilimit)
        del fnumpy
        del inumpy
    elif dtype == 2:
        vals = fnumpy
        # vals = fnumpy.reshape((ncol, nrow, nlay), order='F')
        # vals = np.asanyarray(vals, order='C')
        vals = ma.masked_greater(vals, self._undef_limit)
        vals = vals.astype(np.float64)
        del fnumpy
        del inumpy

    vals = vals.reshape(ncol, nrow, nlay)
    print(vals.min(), vals.max())
    return vals
