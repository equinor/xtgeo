"""GridProperty (not GridProperies) low level functions"""


import numpy as np
import numpy.ma as ma

import xtgeo
import xtgeo.cxtgeo._cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


def f2c_order(obj, values1d):
    """Convert values1d from Fortran to C order, obj can be a Grid() or GridProperty()
    instance
    """
    val = np.reshape(values1d, (obj.ncol, obj.nrow, obj.nlay), order="F")
    val = np.asanyarray(val, order="C")
    val = val.ravel(order="K")
    return val


def c2f_order(obj, values1d):
    """Convert values1d from C to F order, obj can be a Grid() or GridProperty()
    instance
    """
    val = np.reshape(values1d, (obj.ncol, obj.nrow, obj.nlay), order="C")
    val = np.asanyarray(val, order="F")
    val = val.ravel(order="K")
    return val


def update_values_from_carray(self, carray, dtype, delete=False):
    """Transfer values from SWIG 1D carray to numpy, 3D array"""

    logger.debug("Update numpy from C array values")

    nv = self.ntotal

    self._isdiscrete = False

    if dtype == np.float64:
        logger.info("Entering conversion to numpy (float64) ...")
        values1d = _cxtgeo.swig_carr_to_numpy_1d(nv, carray)
    else:
        logger.info("Entering conversion to numpy (int32) ...")
        values1d = _cxtgeo.swig_carr_to_numpy_i1d(nv, carray)
        self._isdiscrete = True

    values = np.reshape(values1d, (self._ncol, self._nrow, self._nlay), order="F")

    # make into C order as this is standard Python order...
    values = np.asanyarray(values, order="C")

    # make it float64 or whatever(?) and mask it
    self._values = values
    self.mask_undef()

    # optionally delete the C array if needed
    if delete:
        carray = delete_carray(self, carray)


def update_carray(self, undef=None, discrete=None, dtype=None, order="F"):
    """Copy (update) values from numpy to SWIG, 1D array, returns a pointer
    to SWIG C array. If discrete is defined as True or False, force
    the SWIG array to be of that kind.

    Note that dtype will "override" current datatype if set. The resulting
    carray will be in Fortran order, unless order is specified as 'C'
    """

    dstatus = self._isdiscrete
    if discrete is not None:
        dstatus = bool(discrete)

    if undef is None:
        undef = xtgeo.UNDEF
        if dstatus:
            undef = xtgeo.UNDEF_INT

    logger.debug("Entering conversion from numpy to C array ...")

    values = self._values.copy()

    if not dtype:
        if dstatus:
            values = values.astype(np.int32)
        else:
            values = values.astype(np.float64)
    else:
        values = values.astype(dtype)

    values = ma.filled(values, undef)

    if order == "F":
        values = np.asfortranarray(values)
        values1d = np.ravel(values, order="K")

    if values1d.dtype == "float64" and dstatus and not dtype:
        values1d = values1d.astype("int32")
        logger.debug("Casting has been done")

    if values1d.dtype == "float64":
        logger.debug("Convert to carray (double)")
        carray = _cxtgeo.new_doublearray(self.ntotal)
        _cxtgeo.swig_numpy_to_carr_1d(values1d, carray)
    elif values1d.dtype == "float32":
        logger.debug("Convert to carray (float)")
        carray = _cxtgeo.new_floatarray(self.ntotal)
        _cxtgeo.swig_numpy_to_carr_f1d(values1d, carray)
    elif values1d.dtype == "int32":
        logger.debug("Convert to carray (int32)")
        carray = _cxtgeo.new_intarray(self.ntotal)
        _cxtgeo.swig_numpy_to_carr_i1d(values1d, carray)
    else:
        raise RuntimeError("Unsupported dtype, probable bug in {}".format(__name__))
    return carray


def delete_carray(self, carray):
    """Delete carray SWIG C pointer, return carray as None"""

    logger.debug("Enter delete carray values method for %d", id(self))
    if carray is None:
        return None

    if "int" in str(carray):
        _cxtgeo.delete_intarray(carray)
        carray = None
    elif "float" in str(carray):
        _cxtgeo.delete_floatarray(carray)
        carray = None
    elif "double" in str(carray):
        _cxtgeo.delete_doublearray(carray)
        carray = None
    else:
        raise RuntimeError("BUG?")

    return carray


def check_shape_ok(self, values):
    """Check if chape of values is OK"""
    (ncol, nrow, nlay) = values.shape
    if ncol != self._ncol or nrow != self._nrow or nlay != self._nlay:
        logger.error(
            "Wrong shape: Dimens of values %s %s %s" "vs %s %s %s",
            ncol,
            nrow,
            nlay,
            self._ncol,
            self._nrow,
            self._nlay,
        )
        return False
    return True
