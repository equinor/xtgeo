# -*- coding: utf-8 -*-

"""Some grid utilities, file scanning etc (methods with no class)"""
import pandas as pd
import xtgeo
import xtgeo.cxtgeo._cxtgeo as _cxtgeo
from xtgeo import XTGeoCLibError
from xtgeo.common.constants import MAXDATES, MAXKEYWORDS

xtg = xtgeo.XTGeoDialog()
logger = xtg.functionlogger(__name__)


def scan_keywords(
    pfile, fformat="xecl", maxkeys=MAXKEYWORDS, dataframe=False, dates=False
):
    """Quick scan of keywords in Eclipse binary restart/init/... file,
    or ROFF binary files.

    Cf. grid_properties.py description
    """

    pfile.get_cfhandle()  # just to keep cfhanclecounter correct

    if fformat == "xecl":
        if dates:
            data = _scan_ecl_keywords_w_dates(
                pfile, maxkeys=maxkeys, dataframe=dataframe
            )
        else:
            data = _scan_ecl_keywords(pfile, maxkeys=maxkeys, dataframe=dataframe)

    else:
        data = _scan_roff_keywords(pfile, maxkeys=maxkeys, dataframe=dataframe)

    pfile.cfclose()

    return data


def scan_dates(pfile, maxdates=MAXDATES, dataframe=False):
    """Quick scan dates in a simulation restart file.

    Cf. grid_properties.py description
    """

    seq = _cxtgeo.new_intarray(maxdates)
    day = _cxtgeo.new_intarray(maxdates)
    mon = _cxtgeo.new_intarray(maxdates)
    yer = _cxtgeo.new_intarray(maxdates)

    cfhandle = pfile.get_cfhandle()

    nstat = _cxtgeo.grd3d_ecl_tsteps(cfhandle, seq, day, mon, yer, maxdates)

    pfile.cfclose()

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


def _scan_ecl_keywords(pfile, maxkeys=MAXKEYWORDS, dataframe=False):
    logger.info("Scanning ECL keywords...")
    cfhandle = pfile.get_cfhandle()

    # maxkeys*10 is used for 1D kewords; 10 => max 8 letters in eclipse +
    # "|" + "extra buffer"
    nkeys, keywords, rectypes, reclens, recstarts = _cxtgeo.grd3d_scan_eclbinary(
        cfhandle, maxkeys * 10, maxkeys, maxkeys, maxkeys
    )

    pfile.cfclose()

    if nkeys == -1:
        raise XTGeoCLibError(f"scanning ecl keywords exited with error code {nkeys}")

    if nkeys == -2:
        raise XTGeoCLibError(
            f"Number of keywords seems greater than {maxkeys} which is currently a "
            "hard limit"
        )

    keywords = keywords.replace(" ", "")
    keywords = keywords.split("|")

    # record types translation (cf: grd3d_scan_eclbinary.c in cxtgeo)
    rct = {
        1: "INTE",
        2: "REAL",
        3: "DOUB",
        4: "CHAR",
        5: "LOGI",
        6: "MESS",
        -1: "????",
    }

    rectypes = rectypes.tolist()
    rc = [rct[key] for nn, key in enumerate(rectypes) if nn < nkeys]
    rl = reclens[0:nkeys].tolist()
    rs = recstarts[0:nkeys].tolist()

    result = list(zip(keywords, rc, rl, rs))

    if dataframe:
        cols = ["KEYWORD", "TYPE", "NITEMS", "BYTESTART"]
        df = pd.DataFrame.from_records(result, columns=cols)
        return df

    return result


def _scan_ecl_keywords_w_dates(pfile, maxkeys=MAXKEYWORDS, dataframe=False):
    """Add a date column to the keyword"""

    logger.info("Scan keywords with dates...")
    xkeys = _scan_ecl_keywords(pfile, maxkeys=maxkeys, dataframe=False)

    xdates = scan_dates(pfile, maxdates=MAXDATES, dataframe=False)

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


def _scan_roff_keywords(pfile, maxkeys=MAXKEYWORDS, dataframe=False):

    rectypes = _cxtgeo.new_intarray(maxkeys)
    reclens = _cxtgeo.new_longarray(maxkeys)
    recstarts = _cxtgeo.new_longarray(maxkeys)

    cfhandle = pfile.get_cfhandle()

    # maxkeys*32 is just to give sufficient allocated character space
    nkeys, _tmp1, keywords = _cxtgeo.grd3d_scan_roffbinary(
        cfhandle, maxkeys * 32, rectypes, reclens, recstarts, maxkeys
    )

    pfile.cfclose()

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
