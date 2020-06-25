# -*- coding: utf-8 -*-
"""Grid import functions for various formats"""

from __future__ import print_function, absolute_import

import os

import xtgeo
from xtgeo.common import XTGeoDialog

from xtgeo.grid3d import _grid_import_roff
from xtgeo.grid3d import _grid_import_ecl


xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


def from_file(
    self,
    gfile,
    fformat=None,
    initprops=None,
    restartprops=None,
    restartdates=None,
    _xtgformat=1,
):
    """Import grid geometry from file, and makes an instance of this class."""

    # pylint: disable=too-many-branches

    if not isinstance(gfile, xtgeo._XTGeoFile):
        raise RuntimeError("Error gfile must be a _XTGeoFile instance")

    self._filesrc = gfile.name

    if fformat is None:
        fformat = "guess"

    # note .grid is currently disabled; need to work at C backend
    fflist = set(["egrid", "grdecl", "bgrdecl", "roff", "eclipserun", "guess"])
    if fformat not in fflist:
        raise ValueError(
            "Invalid fformat: <{}>, options are {}".format(fformat, fflist)
        )

    # work on file extension
    _froot, fext = gfile.splitext(lower=True)

    if fformat == "guess":
        logger.info("Format is <guess>")
        fflist = ["egrid", "grdecl", "bgrdecl", "roff", "eclipserun"]
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
        _grid_import_roff.import_roff(
            self, gfile, xtgformat=_xtgformat
        )  # xtgformat is a developer switch

    elif fformat == "egrid":
        _grid_import_ecl.import_ecl_egrid(self, gfile)

    elif fformat == "eclipserun":
        _grid_import_ecl.import_ecl_run(
            self,
            gfile.name,
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

    self.name = gfile.file.stem

    return self
