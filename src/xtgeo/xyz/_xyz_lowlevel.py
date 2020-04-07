# -*- coding: utf-8 -*-
"""Private low level routines (SWIG vs C)"""

from __future__ import print_function, absolute_import

import numpy as np
from xtgeo.common import XTGeoDialog
import xtgeo.cxtgeo._cxtgeo as _cxtgeo

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


def convert_np_carr_int(xyz, np_array):  # pragma: no cover
    """Convert numpy 1D array to C array, assuming int type."""

    # The numpy is always a double (float64), so need to convert first
    # xyz is the general object

    carr = _cxtgeo.new_intarray(xyz.nrow)

    np_array = np_array.astype(np.int32)

    _cxtgeo.swig_numpy_to_carr_i1d(np_array, carr)

    return carr


def convert_np_carr_double(xyz, np_array):  # pragma: no cover
    """Convert numpy 1D array to C array, assuming double type."""

    carr = _cxtgeo.new_doublearray(xyz.nrow)

    _cxtgeo.swig_numpy_to_carr_1d(np_array, carr)

    return carr


def convert_carr_double_np(xyz, carray, nlen=None):  # pragma: no cover
    """Convert a C array to numpy, assuming double type."""

    if nlen is None:
        nlen = len(xyz._df.index)

    nparray = _cxtgeo.swig_carr_to_numpy_1d(nlen, carray)

    return nparray
