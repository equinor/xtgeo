"""Some grid utilities, file scanning etc (methods with no class)"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

import pandas as pd

import xtgeo
import xtgeo.cxtgeo._cxtgeo as _cxtgeo
from xtgeo import XTGeoCLibError
from xtgeo.common.constants import MAXDATES, MAXKEYWORDS

if TYPE_CHECKING:
    from xtgeo.common.sys import _XTGeoFile

    from .grid_properties import GridPropertiesKeywords, KeywordDateTuple, KeywordTuple

xtg = xtgeo.XTGeoDialog()
from xtgeo.common import logger


def scan_keywords(
    pfile: _XTGeoFile,
    fformat: Literal["xecl", "roff"] = "xecl",
    maxkeys: int = MAXKEYWORDS,
    dataframe: bool = False,
    dates: bool = False,
) -> GridPropertiesKeywords:
    """Quick scan of keywords in Eclipse binary restart/init/... file,
    or ROFF binary files.

    Cf. grid_properties.py description
    """

    pfile.get_cfhandle()  # just to keep cfhanclecounter correct

    if fformat == "xecl":
        if dates:
            keywords = _scan_ecl_keywords_w_dates(
                pfile, maxkeys=maxkeys, dataframe=dataframe
            )
        else:
            keywords = _scan_ecl_keywords(pfile, maxkeys=maxkeys, dataframe=dataframe)
    elif fformat == "roff":
        keywords = _scan_roff_keywords(pfile, maxkeys=maxkeys, dataframe=dataframe)
    else:
        raise ValueError(f"File format can be either `roff` or `xecl`, given {fformat}")
    pfile.cfclose()
    return keywords


def scan_dates(pfile: _XTGeoFile, maxdates: int = MAXDATES, dataframe: bool = False):
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
        date = f"{dyer:4}{dmon:02}{dday:02}"
        da.append(int(date))

    for item in [seq, day, mon, yer]:
        _cxtgeo.delete_intarray(item)

    zdates = list(zip(sq, da))  # list for PY3

    if dataframe:
        cols = ["SEQNUM", "DATE"]
        df = pd.DataFrame.from_records(zdates, columns=cols)
        return df

    return zdates


def _scan_ecl_keywords(
    pfile: _XTGeoFile, maxkeys: int = MAXKEYWORDS, dataframe: bool = False
) -> list[KeywordTuple] | pd.DataFrame:
    cfhandle = pfile.get_cfhandle()

    # maxkeys*10 is used for 1D keywords; 10 => max 8 letters in eclipse +
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

    keywords = re.sub(r"\s+\|", "|", keywords)
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


def _scan_ecl_keywords_w_dates(
    pfile: _XTGeoFile, maxkeys: int = MAXKEYWORDS, dataframe: bool = False
) -> list[KeywordDateTuple] | pd.DataFrame:
    """Add a date column to the keyword"""
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


def _scan_roff_keywords(
    pfile: _XTGeoFile, maxkeys: int = MAXKEYWORDS, dataframe: bool = False
) -> list[KeywordTuple] | pd.DataFrame:
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
