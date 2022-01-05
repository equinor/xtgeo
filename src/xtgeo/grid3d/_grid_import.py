# -*- coding: utf-8 -*-
"""Grid import functions for various formats."""

import xtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.grid3d import _grid_import_ecl, _grid_import_roff

from . import _grid_import_xtgcpgeom

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


def from_file(gfile, fformat=None, **kwargs):  # pylint: disable=too-many-branches
    """Import grid geometry from file, and makes an instance of this class.

    Returns:
    dictionary of keyword arguments to be used in Grid constructor.
    """
    if not isinstance(gfile, xtgeo._XTGeoFile):
        raise RuntimeError("Error gfile must be a _XTGeoFile instance")

    result = {}

    result["filesrc"] = gfile.name

    if fformat is None or fformat == "guess":
        fformat = gfile.detect_fformat()
    else:
        fformat = gfile.generic_format_by_proposal(fformat)  # default

    gfile.check_file(raiseerror=IOError, raisetext=f"Cannot access file {gfile.name}")

    if fformat in ["roff_binary", "roff_ascii"]:
        result.update(_grid_import_roff.import_roff(gfile, **kwargs))
    elif fformat in ["egrid", "fegrid"]:
        result.update(
            _grid_import_ecl.import_ecl_egrid(gfile, fileformat=fformat, **kwargs)
        )
    elif fformat == "grdecl":
        result.update(_grid_import_ecl.import_ecl_grdecl(gfile, **kwargs))
    elif fformat == "bgrdecl":
        result.update(_grid_import_ecl.import_ecl_bgrdecl(gfile, **kwargs))
    elif fformat == "xtg":
        result.update(_grid_import_xtgcpgeom.import_xtgcpgeom(gfile, **kwargs))
    elif fformat == "hdf":
        result.update(_grid_import_xtgcpgeom.import_hdf5_cpgeom(gfile, **kwargs))
    else:
        raise ValueError(f"Invalid file format: {fformat}")

    if gfile.memstream:
        result["name"] = "unknown"
    else:
        result["name"] = gfile.file.stem

    return result
