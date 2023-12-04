"""Grid import functions for various formats."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from xtgeo.common import null_logger
from xtgeo.common.sys import _XTGeoFile
from xtgeo.grid3d import _grid_import_ecl, _grid_import_roff

from . import _grid_import_xtgcpgeom

logger = null_logger(__name__)


def from_file(
    gfile: _XTGeoFile,
    fformat: Literal[
        "bgrdecl",
        "egrid",
        "fegrid",
        "grdecl",
        "guess",
        "hdf",
        "roff_ascii",
        "roff_binary",
        "xtg",
    ]
    | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Import grid geometry from file, and makes an instance of this class.

    Returns:
    dictionary of keyword arguments to be used in Grid constructor.
    """
    if not isinstance(gfile, _XTGeoFile):
        raise RuntimeError("Error gfile must be a _XTGeoFile instance")

    result: dict[str, Any] = {}

    result["filesrc"] = gfile.name

    _fformat = (
        gfile.detect_fformat()
        if fformat is None or fformat == "guess"
        else gfile.generic_format_by_proposal(fformat)  # default
    )

    gfile.check_file(raiseerror=IOError, raisetext=f"Cannot access file {gfile.name}")

    if _fformat in ["roff_binary", "roff_ascii"]:
        result.update(_grid_import_roff.import_roff(gfile, **kwargs))
    elif _fformat in ["egrid", "fegrid"]:
        result.update(
            _grid_import_ecl.import_ecl_egrid(gfile, fileformat=_fformat, **kwargs)
        )
    elif _fformat == "grdecl":
        result.update(_grid_import_ecl.import_ecl_grdecl(gfile, **kwargs))
    elif _fformat == "bgrdecl":
        result.update(_grid_import_ecl.import_ecl_bgrdecl(gfile, **kwargs))
    elif _fformat == "xtg":
        result.update(_grid_import_xtgcpgeom.import_xtgcpgeom(gfile, **kwargs))
    elif _fformat == "hdf":
        result.update(_grid_import_xtgcpgeom.import_hdf5_cpgeom(gfile, **kwargs))
    else:
        raise ValueError(f"Invalid file format: {_fformat}")

    if gfile.memstream:
        result["name"] = "unknown"
    else:
        # Mypy does not know that if gfile.memstream -> False
        # then .file must be Path.
        assert isinstance(gfile.file, Path)
        result["name"] = gfile.file.stem

    return result
