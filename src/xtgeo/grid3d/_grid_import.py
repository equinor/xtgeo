# -*- coding: utf-8 -*-
"""Grid import functions for various formats"""

from __future__ import print_function, absolute_import

import os

from xtgeo.common import XTGeoDialog

from xtgeo.grid3d import _grid_import_roff
from xtgeo.grid3d import _grid_import_ecl


xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


def from_file(
    self, gfile, fformat=None, initprops=None, restartprops=None, restartdates=None
):

    """Import grid geometry from file, and makes an instance of this class."""

    # pylint: disable=too-many-branches

    self._filesrc = gfile

    if fformat is None:
        fformat = "guess"

    # note .grid is currently disabled; need to work at C backend
    fflist = set(["egrid", "grdecl", "bgrdecl", "roff", "eclipserun", "guess"])
    if fformat not in fflist:
        raise ValueError(
            "Invalid fformat: <{}>, options are {}".format(fformat, fflist)
        )

    # work on file extension
    _froot, fext = os.path.splitext(gfile)
    fext = fext.replace(".", "")
    fext = fext.lower()

    if fformat == "guess":
        logger.info("Format is <guess>")
        fflist = ["egrid", "grdecl", "bgrdecl", "roff", "eclipserun"]
        if fext and fext in fflist:
            fformat = fext
        elif fext and fext not in fflist:
            fformat = "roff"  # try to assume binary ROFF...

    logger.info("File name to be used is %s", gfile)

    test_gfile = gfile
    if fformat == "eclipserun":
        test_gfile = gfile + ".EGRID"

    if os.path.isfile(test_gfile):
        logger.info("File %s exists OK", test_gfile)
    else:
        raise IOError("No such file: {}".format(test_gfile))

    if fformat == "roff":
        _grid_import_roff.import_roff(self, gfile)
    elif fformat == "egrid":
        _grid_import_ecl.import_ecl_egrid(self, gfile)
    elif fformat == "eclipserun":
        _grid_import_ecl.import_ecl_run(
            self,
            gfile,
            initprops=initprops,
            restartprops=restartprops,
            restartdates=restartdates,
        )
    elif fformat == "grdecl":
        _grid_import_ecl.import_ecl_grdecl(self, gfile)
    elif fformat == "bgrdecl":
        _grid_import_ecl.import_ecl_bgrdecl(self, gfile)
    else:
        raise SystemExit("Invalid file format")

    return self
