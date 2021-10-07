"""Importing grid props from GRDECL, ascii or binary"""


import ecl_data_io as eclio
import numpy as np
import numpy.ma as ma

import xtgeo

from ._grdecl_format import match_keyword, open_grdecl

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)


def import_bgrdecl_prop(pfile, name, grid):
    """Import prop for binary files with GRDECL layout.

    Args:
        pfile (_XTgeoCFile): xtgeo file instance
        name (str): Name of parameter.
        grid (Grid()): XTGeo Grid instance.

    Raises:
        xtgeo.KeywordNotFoundError: Cannot find property...
    """
    result = dict()
    result["ncol"] = grid.ncol
    result["nrow"] = grid.nrow
    result["nlay"] = grid.nlay
    result["name"] = name
    result["filesrc"] = pfile

    for entry in eclio.lazy_read(pfile.file):
        if match_keyword(entry.read_keyword(), name):
            values = entry.read_array()
            result["discrete"] = np.issubdtype(values.dtype, np.integer)
            if result["discrete"]:
                uniq = np.unique(values).tolist()
                codes = dict(zip(uniq, uniq))
                codes = {key: str(val) for key, val in codes.items()}  # val: strings
                result["codes"] = codes
                values = values.astype(np.int32)
                result["roxar_dtype"] = np.uint16
            else:
                values = values.astype(np.float64)
                result["codes"] = {}
                result["roxar_dtype"] = np.float32
            result["values"] = ma.masked_where(
                grid.get_actnum().values < 1, values.reshape(grid.dimensions, order="F")
            )
            return result

    raise xtgeo.KeywordNotFoundError(
        "Cannot find property name {} in file {}".format(name, pfile.name)
    )


def read_grdecl_3d_property(filename, keyword, dimensions, dtype=float):
    """
    Read a 3d grid property from a grdecl file, see open_grdecl for description
    of format.

    Args:
        filename (pathlib.Path or str): File in grdecl format.
        keyword (str): The keyword of the property in the file
        dimensions ((int,int,int)): Triple of the size of grid.
        dtype (function): The datatype to be read, ie., float.

    Raises:
        xtgeo.KeywordNotFoundError: If keyword is not found in the file.

    Returns:
        numpy array with given dimensions and data type read
        from the grdecl file.
    """
    result = None

    with open_grdecl(filename, keywords=[], simple_keywords=(keyword,)) as kw_generator:
        try:
            _, result = next(kw_generator)
        except StopIteration as si:
            raise xtgeo.KeywordNotFoundError(
                f"Cannot import {keyword}, not present in file {filename}?"
            ) from si

    # The values are stored in F order in the grdecl file
    f_order_values = np.array([dtype(v) for v in result])
    return np.ascontiguousarray(f_order_values.reshape(dimensions, order="F"))


def import_grdecl_prop(pfile, name, grid):
    """Read a GRDECL ASCII property record"""
    result = dict()
    result["ncol"] = grid.ncol
    result["nrow"] = grid.nrow
    result["nlay"] = grid.nlay
    result["name"] = name
    result["filesrc"] = pfile
    actnumv = grid.get_actnum().values

    result["values"] = ma.masked_where(
        actnumv == 0, read_grdecl_3d_property(pfile.file, name, grid.dimensions, float)
    )
    return result
