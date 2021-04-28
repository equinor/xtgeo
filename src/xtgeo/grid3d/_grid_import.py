# -*- coding: utf-8 -*-
"""Grid import functions for various formats."""

import os
import xtgeo
from xtgeo.common import XTGeoDialog

from xtgeo.grid3d import _grid_import_roff, _grid_import_xtgcpgeom
from xtgeo.grid3d import _grid_import_ecl


xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


def from_file(
    gfile,
    fformat=None,
    initprops=None,
    restartprops=None,
    restartdates=None,
):  # pylint: disable=too-many-branches
    """Import grid geometry from file, and makes an instance of this class."""
    if not isinstance(gfile, xtgeo._XTGeoFile):
        raise RuntimeError("Error gfile must be a _XTGeoFile instance")
    args = {}
    args["filesrc"] = gfile.name

    if fformat is None:
        fformat = "guess"

    # note .grid is currently disabled; need to work at C backend
    fflist = {"egrid", "grdecl", "bgrdecl", "roff", "eclipserun", "guess", "xtgf"}
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
        args.update(_grid_import_roff.import_roff(gfile))

    elif fformat == "egrid":
        args.update(_grid_import_ecl.import_ecl_egrid(gfile))

    elif fformat == "eclipserun":
        args.update(
            _grid_import_ecl.import_ecl_run(
                gfile.name,
                initprops=initprops,
                restartprops=restartprops,
                restartdates=restartdates,
            )
        )
    elif fformat == "grdecl":
        args.update(_grid_import_ecl.import_ecl_grdecl(gfile))
    elif fformat == "bgrdecl":
        args.update(_grid_import_ecl.import_ecl_bgrdecl(gfile))
    elif fformat == "xtgf":
        args.update(_grid_import_xtgcpgeom.import_xtgcpgeom(gfile))
    else:
        raise ValueError("Invalid file format")

    args["name"] = gfile.file.stem

    return args
