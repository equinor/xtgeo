"""Importing grid props from GRDECL, ascii or binary"""

from __future__ import print_function, absolute_import

import re
import os
from tempfile import mkstemp

import numpy as np
import numpy.ma as ma

import xtgeo
import xtgeo.cxtgeo._cxtgeo as _cxtgeo

from . import _grid_eclbin_record as _eclbin
from . import _grid3d_utils as utils

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)


def import_bgrdecl_prop(self, pfile, name="unknown", grid=None):
    """Import prop for binary files with GRDECL layout.

    Args:
        pfile (_XTgeoCFile): xtgeo file instance
        name (str, optional): Name of parameter. Defaults to "unknown".
        grid (Grid(), optional): XTGeo Grid instance. Defaults to None.

    Raises:
        xtgeo.KeywordNotFoundError: Cannot find property...
    """

    pfile.get_cfhandle()

    # scan file for properties; these have similar binary format as e.g. EGRID
    logger.info("Make kwlist by scanning")
    kwlist = utils.scan_keywords(
        pfile, fformat="xecl", maxkeys=1000, dataframe=False, dates=False
    )
    bpos = {}
    bpos[name] = -1

    for kwitem in kwlist:
        kwname, kwtype, kwlen, kwbyte = kwitem
        logger.info("KWITEM: %s", kwitem)
        if name == kwname:
            bpos[name] = kwbyte
            break

    if bpos[name] == -1:
        raise xtgeo.KeywordNotFoundError(
            "Cannot find property name {} in file {}".format(name, pfile.name)
        )
    self._ncol = grid.ncol
    self._nrow = grid.nrow
    self._nlay = grid.nlay

    values = _eclbin.eclbin_record(pfile, kwname, kwlen, kwtype, kwbyte)
    if kwtype == "INTE":
        self._isdiscrete = True
        # make the code list
        uniq = np.unique(values).tolist()
        codes = dict(zip(uniq, uniq))
        codes = {key: str(val) for key, val in codes.items()}  # val: strings
        self.codes = codes

    else:
        self._isdiscrete = False
        values = values.astype(np.float64)  # cast REAL (float32) to float64
        self.codes = {}

    # property arrays from binary GRDECL will be for all cells, but they
    # are in Fortran order, so need to convert...

    actnum = grid.get_actnum().values
    allvalues = values.reshape(self.dimensions, order="F")
    allvalues = np.asanyarray(allvalues, order="C")
    allvalues = ma.masked_where(actnum < 1, allvalues)
    self.values = allvalues
    self._name = name

    pfile.cfclose()


def import_grdecl_prop(self, pfile, name="unknown", grid=None):
    """Read a GRDECL ASCII property record"""

    if grid is None:
        raise ValueError("A grid instance is required as argument")

    self._ncol = grid.ncol
    self._nrow = grid.nrow
    self._nlay = grid.nlay
    self._name = name
    self._filesrc = pfile
    actnumv = grid.get_actnum().values

    # This requires that the Python part clean up comments
    # etc, and make a tmp file.
    fds, tmpfile = mkstemp(prefix="tmpxtgeo")
    os.close(fds)

    with open(pfile.name, "r") as oldfile, open(tmpfile, "w") as newfile:
        for line in oldfile:
            if not (re.search(r"^--", line) or re.search(r"^\s+$", line)):
                newfile.write(line)

    # now read the property
    nlen = self._ncol * self._nrow * self._nlay
    ier, values = _cxtgeo.grd3d_import_grdecl_prop(
        tmpfile, self._ncol, self._nrow, self._nlay, name, nlen, 0,
    )

    os.remove(tmpfile)

    if ier != 0:
        raise xtgeo.KeywordNotFoundError(
            "Cannot import {}, not present in file {}?".format(name, pfile)
        )

    self.values = values.reshape(self.dimensions)
    self.values = ma.masked_equal(self.values, actnumv == 0)
