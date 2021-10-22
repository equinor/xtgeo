# -*- coding: utf-8 -*-
"""Grid import functions for various formats."""

import os

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

    if fformat is None:
        fformat = "guess"

    # note .grid is currently disabled; need to work at C backend
    fflist = set(
        [
            "egrid",
            "fegrid",
            "grdecl",
            "bgrdecl",
            "roff",
            "eclipserun",
            "guess",
            "hdf",
            "xtgf",
        ]
    )
    if fformat not in fflist:
        raise ValueError(
            "Invalid fformat: <{}>, options are {}".format(fformat, fflist)
        )

    # work on file extension
    _, fext = gfile.splitext(lower=True)

    if fformat == "guess":
        logger.info("Format is <guess>")
        if fext and fext in fflist:
            fformat = fext
        elif fext and fext not in fflist:
            fformat = "roff"  # try to assume binary ROFF...

    logger.info("File name to be used is %s", gfile.name)

    test_gfile = gfile.name
    if fformat == "eclipserun":
        test_gfile = gfile.name + ".EGRID"

    if os.path.isfile(test_gfile):
        logger.info("File %s exists OK", test_gfile)
    else:
        raise OSError("No such file: {}".format(test_gfile))

    if fformat == "roff":
        result.update(_grid_import_roff.import_roff(gfile, **kwargs))
    elif fformat in ["egrid", "fegrid"]:
        result.update(
            _grid_import_ecl.import_ecl_egrid(gfile, fileformat=fformat, **kwargs)
        )
    elif fformat == "grdecl":
        result.update(_grid_import_ecl.import_ecl_grdecl(gfile, **kwargs))
    elif fformat == "bgrdecl":
        result.update(_grid_import_ecl.import_ecl_bgrdecl(gfile, **kwargs))
    elif fformat == "xtgf":
        result.update(_grid_import_xtgcpgeom.import_xtgcpgeom(gfile, **kwargs))
    elif fformat == "hdf":
        result.update(_grid_import_xtgcpgeom.import_hdf5_cpgeom(gfile, **kwargs))
    else:
        raise ValueError("Invalid file format")

    result["name"] = gfile.file.stem

    return result
