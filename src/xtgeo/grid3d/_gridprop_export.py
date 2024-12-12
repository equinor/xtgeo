"""GridProperty export functions."""

from __future__ import annotations

import io
import json
import os
import struct
from contextlib import ExitStack
from typing import IO, TYPE_CHECKING, Any, Literal

import numpy as np
import resfo
import roffio

from xtgeo.common.exceptions import InvalidFileFormatError
from xtgeo.common.log import null_logger
from xtgeo.io._file import FileFormat, FileWrapper

from ._roff_parameter import RoffParameter

if TYPE_CHECKING:
    from xtgeo.common.types import FileLike

    from .grid_property import GridProperty

logger = null_logger(__name__)


def to_file(
    gridprop: GridProperty,
    pfile: FileLike,
    fformat: Literal["roff", "roffasc", "grdecl", "bgrdecl", "xtgcpprop"] = "roff",
    name: str | None = None,
    append: bool = False,
    dtype: type[np.float32] | type[np.float64] | type[np.int32] | None = None,
    fmt: str | None = None,
) -> None:
    """Export the grid property to file."""
    logger.debug("Export property to file %s as %s", pfile, fformat)

    binary = fformat in (
        FileFormat.ROFF_BINARY.value + FileFormat.BGRDECL.value + ["xtgcpprop"]
    )

    if not name:
        name = gridprop.name or ""

    xtg_file = FileWrapper(pfile, mode="rb")
    xtg_file.check_folder(raiseerror=OSError)

    if fformat in FileFormat.ROFF_BINARY.value + FileFormat.ROFF_ASCII.value:
        _export_roff(gridprop, xtg_file.name, name, binary, append=append)
    elif fformat in FileFormat.GRDECL.value + FileFormat.BGRDECL.value:
        _export_grdecl(
            gridprop,
            xtg_file.name,
            name,
            dtype=dtype or np.int32 if gridprop.isdiscrete else np.float32,
            append=append,
            binary=binary,
            fmt=fmt,
        )
    elif fformat == "xtgcpprop":
        _export_xtgcpprop(gridprop, xtg_file.name)
    else:
        extensions = FileFormat.extensions_string(
            [
                FileFormat.ROFF_BINARY,
                FileFormat.ROFF_ASCII,
                FileFormat.GRDECL,
                FileFormat.BGRDECL,
            ]
        )
        raise InvalidFileFormatError(
            f"File format {fformat} is invalid for type GridProperty. "
            f"Supported formats are {extensions}."
        )


def _export_roff(
    gridprop: GridProperty,
    pfile: str | io.BytesIO | io.StringIO,
    name: str,
    binary: bool,
    append: bool = False,
) -> None:
    if append:
        logger.warning(
            "Append is not implemented for roff format, defaulting to write."
        )
    roff = RoffParameter.from_xtgeo_grid_property(gridprop)
    roff.name = name
    roff.to_file(pfile, roffio.Format.BINARY if binary else roffio.Format.ASCII)


def _export_grdecl(
    gridprop: GridProperty,
    pfile: str | io.BytesIO | io.StringIO,
    name: str,
    dtype: type[np.float32] | type[np.float64] | type[np.int32],
    append: bool = False,
    binary: bool = False,
    fmt: str | None = None,
) -> None:
    """Export ascii or binary GRDECL"""
    vals = gridprop.values.ravel(order="F")
    if np.ma.isMaskedArray(vals):
        undef_export = 1 if gridprop.isdiscrete or "int" in str(dtype) else 0.0
        vals = np.ma.filled(vals, fill_value=undef_export)

    mode = "a" if append else "w"
    if binary:
        mode += "b"

    with ExitStack() as stack:
        fout = (
            stack.enter_context(open(pfile, mode)) if isinstance(pfile, str) else pfile
        )

        if append:
            fout.seek(os.SEEK_END)

        if binary:
            resfo.write(
                fout,
                [(name.ljust(8), vals.astype(dtype))],
                fileformat=resfo.Format.UNFORMATTED,
            )
        else:
            # Always the case when not binary
            assert isinstance(fout, io.TextIOWrapper)
            fout.write(name)
            fout.write("\n")
            for i, v in enumerate(vals):
                fout.write(" ")
                if fmt:
                    fout.write(fmt % v)
                elif gridprop.isdiscrete:
                    fout.write(str(v))
                else:
                    fout.write(f"{v:3e}")
                if i % 6 == 5:
                    fout.write("\n")
            fout.write(" /\n")


def _export_xtgcpprop(
    gridprop: GridProperty, pfile: str | io.BytesIO | io.StringIO
) -> None:
    """Export to experimental xtgcpproperty format, python version."""
    logger.debug("Export as xtgcpprop...")
    gridprop._metadata.required = gridprop

    magic = 1352 if gridprop.isdiscrete else 1351
    prevalues = (1, magic, 4, gridprop.ncol, gridprop.nrow, gridprop.nlay)
    mystruct = struct.Struct("= i i i q q q")
    pre = mystruct.pack(*prevalues)

    meta = gridprop.metadata.get_metadata()

    # Convert StringIO to BytesIO as this is a binary format
    with ExitStack() as stack:
        if isinstance(pfile, io.StringIO):
            data = pfile.getvalue().encode("utf-8")
            fout: IO[Any] = io.BytesIO(data)
        elif isinstance(pfile, io.BytesIO):
            fout = pfile
        else:
            fout = stack.enter_context(open(pfile, "wb"))
        fout.write(pre)
        gridprop.get_npvalues1d(fill_value=gridprop.undef).astype(np.float32).tofile(
            fout
        )
        fout.write("\nXTGMETA.v01\n".encode())
        fout.write(json.dumps(meta).encode())

    logger.debug("Export as xtgcpprop... done")
