# coding: utf-8
"""Private module, Grid Import private functions for ROFF format"""

from __future__ import print_function, absolute_import

# from collections import OrderedDict
# import json

import numpy as np

import xtgeo.cxtgeo._cxtgeo as _cxtgeo
import xtgeo

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)


def import_xtgeo(self, gfile):
    """Import xtgeo format (in prep)"""

    fhandle = gfile.get_cfhandle()

    # scan dimensions:
    ier, ncol, nrow, nlay, _metadata = _cxtgeo.grdcp3d_import_xtgeo_grid(
        0,
        np.zeros((0), dtype=np.float64),
        np.zeros((0), dtype=np.float32),
        np.zeros((0), dtype=np.int32),
        fhandle,
    )

    if ier != 0:
        raise ValueError("Cannot import, error code is: {}".format(ier))

    self._ncol = ncol
    self._nrow = nrow
    self._nlay = nlay

    nncol = ncol + 1
    nnrow = nrow + 1
    nnlay = nlay + 1

    self._coordsv = np.ones((nncol, nnrow, 6), dtype=np.float64)
    self._zcornsv = np.zeros((nncol, nnrow, nnlay, 4), dtype=np.float32)
    self._actnumsv = np.zeros((ncol, nrow, nlay), dtype=np.int32)

    # read data
    ier, ncol, nrow, nlay, _metadata = _cxtgeo.grdcp3d_import_xtgeo_grid(
        1,
        self._coordsv,
        self._zcornsv,
        self._actnumsv,
        fhandle,
    )

    if ier != 0:
        raise ValueError("Error in reading file, error code is: {}".format(ier))

    # themeta = json.loads(metadata, object_pairs_hook=OrderedDict)
    self._subgrids = None  # tmp

    gfile.cfclose()
