"""GridProperty import function of xtgcpprop format."""

from __future__ import annotations

import json
from contextlib import contextmanager
from io import BytesIO, StringIO
from pathlib import Path
from struct import unpack
from typing import TYPE_CHECKING, Any, Generator

import numpy as np

import xtgeo.common.sys as xsys
from xtgeo.common import null_logger
from xtgeo.common.constants import UNDEF, UNDEF_INT
from xtgeo.metadata.metadata import MetaDataCPProperty

logger = null_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import DTypeLike

    from xtgeo.io._file import FileWrapper


@contextmanager
def _read_from_stream(
    stream: BytesIO | StringIO,
    size: int | None,
    seek: int | None,
) -> Generator[bytes, None, None]:
    """Helper function to read from a stream with optional seeking."""
    was_at = stream.tell()
    if seek is not None:
        stream.seek(seek)
    try:
        data = stream.read(size)
        yield data if isinstance(data, bytes) else data.encode()
    finally:
        stream.seek(was_at)


@contextmanager
def _read_filelike(
    filelike: Path | BytesIO | StringIO,
    size: int | None = None,
    seek: int | None = None,
) -> Generator[bytes, None, None]:
    """Context manager for reading a specified number of bytes from a file-like object.

    Accepts either a Path, BytesIO, or StringIO object as input. The function reads
    up to 'offset' bytes from the file-like object and yields these bytes.

    For BytesIO and StringIO, the read operation preserves the original
    file cursor position.
    """

    if isinstance(filelike, Path):
        with filelike.open("rb") as f:
            if seek is not None:
                f.seek(seek)
            yield f.read(size)
    elif isinstance(filelike, (BytesIO, StringIO)):
        with _read_from_stream(stream=filelike, size=size, seek=seek) as f:
            yield f
    else:
        raise TypeError("Filelike must be one of: Path, BytesIO or StringIO.")


def import_xtgcpprop(
    mfile: FileWrapper,
    ijrange: Sequence[int] | None = None,
    zerobased: bool = False,
) -> dict[str, Any]:
    """Using pure python for experimental xtgcpprop import.

    Args:
        mfile (FileWrapper): Input file reference
        ijrange (list-like): List or tuple with 4 members [i_from, i_to, j_from, j_to]
            where cell indices are zero based (starts with 0)
        zerobased (bool): If ijrange basis is zero or one.

    """
    offset = 36

    with _read_filelike(mfile.file, size=offset) as header:
        # unpack header
        swap, magic, nbyte, ncol, nrow, nlay = unpack("= i i i q q q", header)

    if swap != 1 or magic not in (1351, 1352):
        raise ValueError("Invalid file format (wrong swap id or magic number).")

    if magic == 1351:
        dtype: DTypeLike = np.float32 if nbyte == 4 else np.float64
    else:
        dtype = f"int{nbyte * 8}"

    narr = ncol * nrow * nlay

    ncolnew = nrownew = 0

    if ijrange:
        vals, ncolnew, nrownew = _import_xtgcpprop_partial(
            mfile, nbyte, dtype, offset, ijrange, zerobased, ncol, nrow, nlay
        )

    else:
        vals = xsys.npfromfile(mfile.file, dtype=dtype, count=narr, offset=offset)

    # read metadata which will be at position offet + nfloat*narr +13
    with _read_filelike(
        mfile.file,
        seek=offset + nbyte * narr + 13,
    ) as _meta:
        meta = json.loads(_meta, object_pairs_hook=dict)

    req = meta["_required_"]

    result = {att: req[att] for att in MetaDataCPProperty.REQUIRED}

    if ijrange:
        result["ncol"] = ncolnew
        result["nrow"] = nrownew

    result["values"] = np.ma.masked_equal(
        vals.reshape((result["ncol"], result["nrow"], result["nlay"])),
        UNDEF_INT if result["discrete"] else UNDEF,
    )
    return result


def _import_xtgcpprop_partial(
    mfile: FileWrapper,
    nbyte: int,
    dtype: DTypeLike,
    offset: int,
    ijrange: Sequence[int],
    zerobased: bool,
    ncol: int,
    nrow: int,
    nlay: int,
) -> tuple[np.ndarray, int, int]:
    """Partial import of a property."""
    i1, i2, j1, j2 = ijrange
    if not zerobased:
        i1 -= 1
        i2 -= 1
        j1 -= 1
        j2 -= 1

    ncolnew = i2 - i1 + 1
    nrownew = j2 - j1 + 1

    if ncolnew < 1 or ncolnew > ncol or nrownew < 1 or nrownew > nrow:
        raise ValueError("The ijrange spesification is invalid.")

    vals = np.zeros(ncolnew * nrownew * nlay, dtype=dtype)

    for newnum, inum in enumerate(range(i1, i2 + 1)):
        newpos = offset + (inum * nrow * nlay + j1 * nlay) * nbyte
        ncount = nrownew * nlay
        xvals = xsys.npfromfile(mfile.file, dtype=dtype, count=ncount, offset=newpos)
        vals[newnum * ncount : newnum * ncount + ncount] = xvals

    return vals, ncolnew, nrownew
