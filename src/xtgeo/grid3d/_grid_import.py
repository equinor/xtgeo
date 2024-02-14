"""Grid import functions for various formats."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xtgeo.common.exceptions import InvalidFileFormatError
from xtgeo.common.log import null_logger
from xtgeo.io._file import FileFormat, FileWrapper

from . import _grid_import_ecl, _grid_import_roff, _grid_import_xtgcpgeom

logger = null_logger(__name__)


def from_file(
    gfile: FileWrapper,
    fformat: FileFormat,
    **kwargs: Any,
) -> dict[str, Any]:
    """Import grid geometry from file, and makes an instance of this class.

    Returns:
    dictionary of keyword arguments to be used in Grid constructor.
    """
    if not isinstance(gfile, FileWrapper):
        raise RuntimeError("Error gfile must be a FileWrapper instance")

    result: dict[str, Any] = {
        "filesrc": gfile.name,
    }
    gfile.check_file(raiseerror=IOError, raisetext=f"Cannot access file {gfile.name}")

    if fformat in (FileFormat.ROFF_BINARY, FileFormat.ROFF_ASCII):
        result.update(_grid_import_roff.import_roff(gfile, **kwargs))
    elif fformat in (FileFormat.EGRID, FileFormat.FEGRID):
        result.update(
            _grid_import_ecl.import_ecl_egrid(
                gfile,
                fileformat=fformat,
                **kwargs,
            )
        )
    elif fformat == FileFormat.GRDECL:
        result.update(_grid_import_ecl.import_ecl_grdecl(gfile, **kwargs))
    elif fformat == FileFormat.BGRDECL:
        result.update(_grid_import_ecl.import_ecl_bgrdecl(gfile, **kwargs))
    elif fformat == FileFormat.XTG:
        result.update(_grid_import_xtgcpgeom.import_xtgcpgeom(gfile, **kwargs))
    elif fformat == FileFormat.HDF:
        result.update(_grid_import_xtgcpgeom.import_hdf5_cpgeom(gfile, **kwargs))
    else:
        extensions = FileFormat.extensions_string(
            [
                FileFormat.ROFF_BINARY,
                FileFormat.ROFF_ASCII,
                FileFormat.EGRID,
                FileFormat.FEGRID,
                FileFormat.GRDECL,
                FileFormat.BGRDECL,
                FileFormat.XTG,
                FileFormat.HDF,
            ]
        )
        raise InvalidFileFormatError(
            f"File format {fformat} is invalid for type Grid. "
            f"Supported formats are {extensions}."
        )

    if gfile.memstream:
        result["name"] = "unknown"
    else:
        # Mypy does not know that if gfile.memstream -> False
        # then .file must be Path.
        assert isinstance(gfile.file, Path)
        result["name"] = gfile.file.stem

    return result
