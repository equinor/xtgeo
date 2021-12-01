"""Importing grid props from ROFF, binary"""


import warnings

import numpy as np
import xtgeo

from ._roff_parameter import RoffParameter

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)


def import_roff(pfile, name=None, grid=None):
    """Import ROFF format"""

    if name == "unknown":
        warnings.warn(
            "Using name='unknown' to select first property in roff file"
            " is deprecated, use name=None instead",
            DeprecationWarning,
        )
        name = None

    result = dict()
    roff_param = RoffParameter.from_file(pfile._file, name)
    result["codes"] = roff_param.xtgeo_codes()
    result["name"] = roff_param.name
    result["ncol"] = int(roff_param.nx)
    result["nrow"] = int(roff_param.ny)
    result["nlay"] = int(roff_param.nz)
    result["discrete"] = roff_param.is_discrete
    result["values"] = roff_param.xtgeo_values()
    if grid is not None:
        result["values"] = np.ma.masked_where(
            grid.get_actnum().values < 1,
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
