"""Some grid utilities, file scanning etc (methods with no class)"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import resfo
import roffio

from xtgeo.common import null_logger
from xtgeo.common.constants import MAXDATES, MAXKEYWORDS

if TYPE_CHECKING:
    from xtgeo.io._file import FileWrapper

    from .grid_properties import GridPropertiesKeywords, KeywordDateTuple, KeywordTuple

logger = null_logger(__name__)


def scan_keywords(
    pfile: FileWrapper,
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

    if fformat == "roff":
        return _scan_roff_keywords(pfile, maxkeys=maxkeys, dataframe=dataframe)
    return (
        _scan_ecl_keywords_w_dates(pfile, maxkeys=maxkeys, dataframe=dataframe)
        if dates
        else _scan_ecl_keywords(pfile, maxkeys=maxkeys, dataframe=dataframe)
    )


def scan_dates(
    pfile: FileWrapper,
    maxdates: int = MAXDATES,
    dataframe: bool = False,
) -> list | pd.DataFrame:
    """Quick scan dates in a simulation restart file.

    Cf. grid_properties.py description
    """
    dates = []
    seqnum = -1
    for item in resfo.lazy_read(pfile.file):
        kw = item.read_keyword().strip()
        data = item.read_array()

        if kw == "SEQNUM":
            seqnum = data[0]
            continue

        # With LGRs multiple INTEHEADs may occur. Ensure we get the date
        # from the first INTEHEAD after a SEQNUM.
        if kw == "INTEHEAD" and seqnum != -1:
            # Index 66 = year, 65 = month, 64 = day
            date = int(f"{data[66]}{data[65]:02d}{data[64]:02d}")
            dates.append((seqnum, date))
            seqnum = -1

    return (
        pd.DataFrame.from_records(dates, columns=["SEQNUM", "DATE"])
        if dataframe
        else dates
    )


def _scan_ecl_keywords(
    pfile: FileWrapper,
    maxkeys: int = MAXKEYWORDS,
    dataframe: bool = False,
) -> list[KeywordTuple] | pd.DataFrame:
    keywords = []
    for item in resfo.lazy_read(pfile.file):
        keywords.append(
            (
                item.read_keyword().strip(),
                item.read_type().strip(),
                item.read_length(),
                item.stream.tell(),
            )
        )

    return (
        pd.DataFrame.from_records(
            keywords,
            columns=["KEYWORD", "TYPE", "NITEMS", "BYTESTART"],
        )
        if dataframe
        else keywords
    )


def _scan_ecl_keywords_w_dates(
    pfile: FileWrapper,
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

    return (
        pd.DataFrame.from_records(
            result,
            columns=["KEYWORD", "TYPE", "NITEMS", "BYTESTART", "DATE"],
        )
        if dataframe
        else result
    )


def _scan_roff_keywords(
    pfile: FileWrapper,
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

    return (
        pd.DataFrame.from_records(
            keywords,
            columns=["KEYWORD", "TYPE", "NITEMS", "BYTESTARTDATA"],
        )
        if dataframe
        else keywords
    )


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
