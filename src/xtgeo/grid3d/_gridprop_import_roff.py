"""Importing grid props from ROFF, binary"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from xtgeo.common import null_logger

from ._roff_parameter import RoffParameter

if TYPE_CHECKING:
    from xtgeo.grid3d import Grid
    from xtgeo.io._file import FileWrapper

logger = null_logger(__name__)


def import_roff(
    pfile: FileWrapper,
    name: str | None = None,
    grid: Grid | None = None,
) -> dict[str, Any]:
    """Import ROFF format"""
    result: dict[str, Any] = {}
    roff_param = RoffParameter.from_file(pfile._file, name)
    result["codes"] = roff_param.xtgeo_codes()
    result["name"] = roff_param.name
    result["ncol"] = int(roff_param.nx)
    result["nrow"] = int(roff_param.ny)
    result["nlay"] = int(roff_param.nz)
    result["discrete"] = roff_param.is_discrete
    result["values"] = roff_param.xtgeo_values()

    if grid is not None and (actnum := grid.get_actnum()):
        result["values"] = np.ma.masked_where(
            actnum.values < 1,
            result["values"],
        )

    roff_val = roff_param.values
    if isinstance(roff_val, bytes) or np.issubdtype(roff_val.dtype, np.uint8):
        result["roxar_dtype"] = np.uint8
    elif np.issubdtype(roff_val.dtype, np.integer):
        result["roxar_dtype"] = np.uint16
    elif np.issubdtype(roff_val.dtype, np.floating):
        result["roxar_dtype"] = np.float32
    else:
        raise ValueError(f"Could not deduce roxar type of {roff_val.dtype}")

    return result
