"""Some grid utilities, file scanning etc (methods with no class)"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import roffio

from xtgeo import XTGeoCLibError, _cxtgeo
from xtgeo.common import null_logger
from xtgeo.common.constants import MAXDATES, MAXKEYWORDS

if TYPE_CHECKING:
    from xtgeo.common.sys import _XTGeoFile

    from .grid_properties import GridPropertiesKeywords, KeywordDateTuple, KeywordTuple

logger = null_logger(__name__)


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
    if fformat not in ("xecl", "roff"):
        raise ValueError(f"File format can be either `roff` or `xecl`, given {fformat}")

    if fformat == "xecl":
        pfile.get_cfhandle()  # just to keep cfhanclecounter correct
        if dates:
            keywords = _scan_ecl_keywords_w_dates(
                pfile,
                maxkeys=maxkeys,
                dataframe=dataframe,
            )
        else:
            keywords = _scan_ecl_keywords(
                pfile,
                maxkeys=maxkeys,
                dataframe=dataframe,
            )
        pfile.cfclose()
    elif fformat == "roff":
        keywords = _scan_roff_keywords(pfile, maxkeys=maxkeys, dataframe=dataframe)
    return keywords


def scan_dates(
    pfile: _XTGeoFile,
    maxdates: int = MAXDATES,
    dataframe: bool = False,
) -> list | pd.DataFrame:
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

    return (
        pd.DataFrame.from_records(zdates, columns=["SEQNUM", "DATE"])
        if dataframe
        else zdates
    )


def _scan_ecl_keywords(
    pfile: _XTGeoFile,
    maxkeys: int = MAXKEYWORDS,
    dataframe: bool = False,
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
    pfile: _XTGeoFile,
    maxkeys: int = MAXKEYWORDS,
    dataframe: bool = False,
) -> list[KeywordDateTuple] | pd.DataFrame:
    """Add a date column to the keyword"""
    xkeys = _scan_ecl_keywords(pfile, maxkeys=maxkeys, dataframe=False)
    assert isinstance(xkeys, list)
    xdates = scan_dates(pfile, maxdates=MAXDATES, dataframe=False)
    assert isinstance(xdates, list)

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
    pfile: _XTGeoFile,
    maxkeys: int = MAXKEYWORDS,
    dataframe: bool = False,
) -> list[KeywordTuple] | pd.DataFrame:
    with open(pfile.file, "rb") as fin:
        is_binary = fin.read(8) == b"roff-bin"

    keywords = []
    with roffio.lazy_read(pfile.file) as roff_iter:
        SPACE_OR_NUL = 1
        TAG = 3 + SPACE_OR_NUL  # "tag"
        ENDTAG = 6 + SPACE_OR_NUL  # "endtag"
        ARRAY_AND_SIZE = 5 + SPACE_OR_NUL + 4  # "array", 4 byte int

        count = 0
        done = False
        # 81 is where the standard RMS exported header size ends.
        # This offset won't be correct for non-RMS exported roff files,
        # but it is a compromise to keep the old functionality of byte
        # counting _close enough_ because this data is not made available
        # from roffio.
        byte_pos = 81

        for tag_name, tag_group in roff_iter:
            byte_pos += TAG
            byte_pos += len(tag_name) + SPACE_OR_NUL

            for keyword, value in tag_group:
                if isinstance(value, (np.ndarray, bytes)):
                    byte_pos += ARRAY_AND_SIZE
                dtype, size, offset = _get_roff_type_and_size(value, is_binary)

                byte_pos += len(dtype) + SPACE_OR_NUL
                byte_pos += len(keyword) + SPACE_OR_NUL

                keyword = f"{tag_name}!{keyword}"
                if tag_name == "parameter" and keyword == "name":
                    keyword += f"!{value}"
                keywords.append((keyword, dtype, size, byte_pos))

                byte_pos += offset
                count += 1
                if count == maxkeys:
                    done = True
                    break

            byte_pos += ENDTAG
            if done:
                break

    if dataframe:
        cols = ["KEYWORD", "TYPE", "NITEMS", "BYTESTARTDATA"]
        return pd.DataFrame.from_records(keywords, columns=cols)

    return keywords


def _get_roff_type_and_size(
    value: str | bool | bytes | np.ndarray, is_binary: bool
) -> tuple[str, int, int]:
    # If is_binary is False add a multiplier because values will
    # be separated by spaces in the case of numerical/boolean
    # data, as opposed to buffer packed, while strings will be
    # quoted and not just NUL delimited
    if isinstance(value, str):
        return "char", 1, len(value) + (1 if is_binary else 3)
    if isinstance(value, bool):
        return "bool", 1, 1 if is_binary else 2
    if isinstance(value, bytes):
        return "byte", len(value), len(value) * (1 if is_binary else 2)
    if np.issubdtype(value.dtype, np.bool_):
        return "bool", value.size, value.size * (1 if is_binary else 2)
    if np.issubdtype(value.dtype, np.int8) or np.issubdtype(value.dtype, np.uint8):
        return "byte", value.size, value.size * (1 if is_binary else 2)
    if np.issubdtype(value.dtype, np.integer):
        return "int", value.size, value.size * (4 if is_binary else 5)
    if np.issubdtype(value.dtype, np.float32):
        return "float", value.size, value.size * (4 if is_binary else 5)
    if np.issubdtype(value.dtype, np.double):
        return "double", value.size, value.size * (8 if is_binary else 9)
    if np.issubdtype(value.dtype, np.unicode_):
        total_bytes = sum(len(val) + (1 if is_binary else 3) for val in value)
        return "char", value.size, total_bytes
    raise ValueError(f"Could not find suitable roff type for {type(value)}")
