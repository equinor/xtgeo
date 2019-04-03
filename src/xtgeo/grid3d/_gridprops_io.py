# -*- coding: utf-8 -*-
"""Import/export or scans of grid properties (cf GridProperties class"""
import pandas as pd

import xtgeo
import xtgeo.cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.common import _get_fhandle, _close_fhandle

from xtgeo.grid3d import _gridprop_import

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

XTGDEBUG = xtg.get_syslevel()
_cxtgeo.xtg_verbose_file("NONE")


def scan_keywords(pfile, fformat="xecl", maxkeys=100000, dataframe=False, dates=False):

    if fformat == "xecl":
        if dates:
            data = _scan_ecl_keywords_w_dates(
                pfile, maxkeys=maxkeys, dataframe=dataframe
            )
        else:
            data = _scan_ecl_keywords(pfile, maxkeys=maxkeys, dataframe=dataframe)

    else:
        data = _scan_roff_keywords(pfile, maxkeys=maxkeys, dataframe=dataframe)
    return data


def _scan_ecl_keywords(pfile, maxkeys=100000, dataframe=False):

    # In case pfile is not a file name but a swig pointer to a file handle,
    # the file must not be closed

    ultramax = int(1000000 / 9)  # cf *swig_bnd_char_1m in cxtgeo.i
    if maxkeys > ultramax:
        raise ValueError("maxkeys value is too large, must be < {}".format(ultramax))

    rectypes = _cxtgeo.new_intarray(maxkeys)
    reclens = _cxtgeo.new_longarray(maxkeys)
    recstarts = _cxtgeo.new_longarray(maxkeys)

    fhandle, pclose = _get_fhandle(pfile)

    nkeys, keywords = _cxtgeo.grd3d_scan_eclbinary(
        fhandle, rectypes, reclens, recstarts, maxkeys, XTGDEBUG
    )

    _close_fhandle(fhandle, pclose)

    keywords = keywords.replace(" ", "")
    keywords = keywords.split("|")

    # record types translation (cf: grd3d_scan_eclbinary.c in cxtgeo)
    rct = {
        "1": "INTE",
        "2": "REAL",
        "3": "DOUB",
        "4": "CHAR",
        "5": "LOGI",
        "6": "MESS",
        "-1": "????",
    }

    rc = []
    rl = []
    rs = []
    for i in range(nkeys):
        rc.append(rct[str(_cxtgeo.intarray_getitem(rectypes, i))])
        rl.append(_cxtgeo.longarray_getitem(reclens, i))
        rs.append(_cxtgeo.longarray_getitem(recstarts, i))

    _cxtgeo.delete_intarray(rectypes)
    _cxtgeo.delete_longarray(reclens)
    _cxtgeo.delete_longarray(recstarts)

    result = list(zip(keywords, rc, rl, rs))

    if dataframe:
        cols = ["KEYWORD", "TYPE", "NITEMS", "BYTESTART"]
        df = pd.DataFrame.from_records(result, columns=cols)
        return df

    return result


def _scan_ecl_keywords_w_dates(pfile, maxkeys=100000, dataframe=False):

    """Add a date column to the keyword"""

    xkeys = _scan_ecl_keywords(pfile, maxkeys=maxkeys, dataframe=False)

    xdates = scan_dates(pfile, maxdates=maxkeys, dataframe=False)

    result = []
    # now merge these two:
    nv = -1
    date = 0
    for item in xkeys:
        name, dtype, reclen, bytepos = item
        if name == "SEQNUM":
            nv += 1
            date = xdates[nv][1]

        entry = (name, dtype, reclen, bytepos, date)
        result.append(entry)

    if dataframe:
        cols = ["KEYWORD", "TYPE", "NITEMS", "BYTESTART", "DATE"]
        df = pd.DataFrame.from_records(result, columns=cols)
        return df

    return result


def _scan_roff_keywords(pfile, maxkeys=100000, dataframe=False):

    # In case pfile is not a file name but a swig pointer to a file handle,
    # the file must not be closed

    ultramax = int(1000000 / 9)  # cf *swig_bnd_char_1m in cxtgeo.i
    if maxkeys > ultramax:
        raise ValueError("maxkeys value is too large, must be < {}".format(ultramax))

    rectypes = _cxtgeo.new_intarray(maxkeys)
    reclens = _cxtgeo.new_longarray(maxkeys)
    recstarts = _cxtgeo.new_longarray(maxkeys)

    fhandle, pclose = _get_fhandle(pfile)

    nkeys, _tmp1, keywords = _cxtgeo.grd3d_scan_roffbinary(
        fhandle, rectypes, reclens, recstarts, maxkeys, XTGDEBUG
    )

    _close_fhandle(fhandle, pclose)

    keywords = keywords.replace(" ", "")
    keywords = keywords.split("|")

    # record types translation (cf: grd3d_scan_eclbinary.c in cxtgeo)
    rct = {
        "1": "int",
        "2": "float",
        "3": "double",
        "4": "char",
        "5": "bool",
        "6": "byte",
    }

    rc = []
    rl = []
    rs = []
    for i in range(nkeys):
        rc.append(rct[str(_cxtgeo.intarray_getitem(rectypes, i))])
        rl.append(_cxtgeo.longarray_getitem(reclens, i))
        rs.append(_cxtgeo.longarray_getitem(recstarts, i))

    _cxtgeo.delete_intarray(rectypes)
    _cxtgeo.delete_longarray(reclens)
    _cxtgeo.delete_longarray(recstarts)

    result = list(zip(keywords, rc, rl, rs))

    if dataframe:
        cols = ["KEYWORD", "TYPE", "NITEMS", "BYTESTARTDATA"]
        df = pd.DataFrame.from_records(result, columns=cols)
        return df

    return result


def scan_dates(pfile, maxdates=1000, dataframe=False):
    """Scan DATES in Eclipse OUTPUT files (UNRST)"""

    seq = _cxtgeo.new_intarray(maxdates)
    day = _cxtgeo.new_intarray(maxdates)
    mon = _cxtgeo.new_intarray(maxdates)
    yer = _cxtgeo.new_intarray(maxdates)

    fhandle, pclose = _get_fhandle(pfile)

    nstat = _cxtgeo.grd3d_ecl_tsteps(fhandle, seq, day, mon, yer, maxdates, XTGDEBUG)

    _close_fhandle(fhandle, pclose)

    sq = []
    da = []
    for i in range(nstat):
        sq.append(_cxtgeo.intarray_getitem(seq, i))
        dday = _cxtgeo.intarray_getitem(day, i)
        dmon = _cxtgeo.intarray_getitem(mon, i)
        dyer = _cxtgeo.intarray_getitem(yer, i)
        date = "{0:4}{1:02}{2:02}".format(dyer, dmon, dday)
        da.append(int(date))

    for item in [seq, day, mon, yer]:
        _cxtgeo.delete_intarray(item)

    zdates = list(zip(sq, da))  # list for PY3

    if dataframe:
        cols = ["SEQNUM", "DATE"]
        df = pd.DataFrame.from_records(zdates, columns=cols)
        return df

    return zdates


def import_ecl_output(
    props, pfile, names=None, dates=None, grid=None, namestyle=0
):  # pylint: disable=too-many-locals, too-many-branches, too-many-statements

    logger.debug("'namestyle' is %s (not in use)", namestyle)

    if not grid:
        raise ValueError("Grid Geometry object is missing")

    if not names:
        raise ValueError("Name list is empty (None)")

    fhandle, pclose = _get_fhandle(pfile)

    # scan valid keywords
    kwlist = props.scan_keywords(fhandle)

    usenames = list()

    if names == "all":
        nact = grid.nactive
        ntot = grid.ntotal

        for kw in kwlist:
            kwname, _tmp1, nlen, _bs1 = kw
            if nlen in (nact, ntot):
                usenames.append(kwname)
    else:
        usenames = list(names)

    logger.info("NAMES are %s", usenames)

    lookfornames = list(set(usenames))

    possiblekw = []
    for name in lookfornames:
        namefound = False
        for kwitem in kwlist:
            possiblekw.append(kwitem[0])
            if name == kwitem[0]:
                namefound = True
        if not namefound:
            if name == "SOIL":
                pass  # will check for SWAT and SGAS later
            else:
                raise ValueError(
                    "Keyword {} not found. Possible list: {}".format(name, possiblekw)
                )

    # check valid dates, and remove invalid entries (allowing that user
    # can be a bit sloppy on DATES)

    validdates = [None]
    if dates:
        dlist = props.scan_dates(fhandle)

        validdates = []
        alldates = []
        for date in dates:
            for ditem in dlist:
                alldates.append(str(ditem[1]))
                if str(date) == str(ditem[1]):
                    validdates.append(date)

        if not validdates:
            msg = "No valid dates given (dates: {} vs {})".format(dates, alldates)
            xtg.error(msg)
            raise ValueError(msg)

        if len(dates) > len(validdates):
            invalidddates = list(set(dates).difference(validdates))
            msg = (
                "In file {}: Some dates not found: {}, but will continue "
                "with dates: {}".format(pfile, invalidddates, validdates)
            )
            xtg.warn(msg)
            # raise DateNotFoundError(msg)

    use2names = list(usenames)  # to make copy

    logger.info("Use names: %s", use2names)
    logger.info("Valid dates: %s", validdates)

    # now import each property
    firstproperty = True

    for date in validdates:
        # xprop = dict()
        # soil_ok = False

        for name in use2names:

            if date is None:
                date = None
                propname = name
                etype = 1
            else:
                propname = name + "_" + str(date)
                etype = 5

            prop = xtgeo.grid3d.GridProperty()

            # use a private GridProperty function here, for convinience
            # (since filehandle)
            ier = _gridprop_import.import_eclbinary(
                prop, fhandle, name=name, date=date, grid=grid, etype=etype
            )
            if ier != 0:
                raise ValueError(
                    "Something went wrong, IER = {} while "
                    "name={}, date={}, etype={}, propname={}".format(
                        ier, name, date, etype, propname
                    )
                )

            if firstproperty:
                ncol = prop.ncol
                nrow = prop.nrow
                nlay = prop.nlay
                firstproperty = False

            logger.info("Appended property %s", propname)
            props._names.append(propname)
            props._props.append(prop)

    props._ncol = ncol
    props._nrow = nrow
    props._nlay = nlay

    if validdates[0] != 0:
        props._dates = validdates

    _close_fhandle(fhandle, pclose)
