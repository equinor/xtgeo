"""GridProperty (not GridProperies) low level functions"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.ma as ma

from xtgeo import _cxtgeo
from xtgeo.common import null_logger
from xtgeo.common.constants import UNDEF, UNDEF_INT

logger = null_logger(__name__)

if TYPE_CHECKING:
    from ctypes import Array as cArray

    from numpy.typing import DTypeLike

    from xtgeo.grid3d import Grid, GridProperty


def f2c_order(obj: Grid | GridProperty, values1d: np.ndarray) -> np.ndarray:
    """Convert values1d from Fortran to C order, obj can be a Grid() or GridProperty()
    instance
    """
    val = np.reshape(values1d, (obj.ncol, obj.nrow, obj.nlay), order="F")
    val = np.asanyarray(val, order="C")
    return val.ravel()


def c2f_order(obj: Grid | GridProperty, values1d: np.ndarray) -> np.ndarray:
    """Convert values1d from C to F order, obj can be a Grid() or GridProperty()
    instance
    """
    val = np.reshape(values1d, (obj.ncol, obj.nrow, obj.nlay), order="C")
    val = np.asanyarray(val, order="F")
    return val.ravel(order="F")


def update_values_from_carray(
    self: GridProperty,
    carray: cArray,
    dtype: DTypeLike,
    delete: bool = False,
) -> None:
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
    self.values = values  # type: ignore
    self.mask_undef()

    # optionally delete the C array if needed
    if delete:
        delete_carray(self, carray)


def update_carray(
    self: GridProperty,
    undef: int | float | None = None,
    discrete: bool | None = None,
    dtype: DTypeLike = None,
    order: Literal["C", "F", "A", "K"] = "F",
) -> cArray:
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
        undef = UNDEF
        if dstatus:
            undef = UNDEF_INT

    logger.debug("Entering conversion from numpy to C array ...")

    values = self.values.copy()

    if not dtype:
        values = values.astype(np.int32) if dstatus else values.astype(np.float64)
    else:
        values = values.astype(dtype)

    values = ma.filled(values, undef)
    values = np.asfortranarray(values)

    if order == "F":
        values = np.asfortranarray(values)

    values1d = np.ravel(values, order=order)

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
        raise RuntimeError(f"Unsupported dtype, probable bug in {__name__}")
    return carray


def delete_carray(self: GridProperty, carray: cArray) -> None:
    """Delete carray SWIG C pointer, return carray as None"""

    logger.debug("Enter delete carray values method for %d", id(self))
    if carray is None:
        return

    if "int" in str(carray):
        _cxtgeo.delete_intarray(carray)
        return
    if "float" in str(carray):
        _cxtgeo.delete_floatarray(carray)
        return
    if "double" in str(carray):
        _cxtgeo.delete_doublearray(carray)
        return

    raise RuntimeError("BUG?")


def check_shape_ok(self: GridProperty, values: np.ndarray) -> bool:
    """Check if chape of values is OK"""
    if values.shape == (self._ncol, self._nrow, self._nlay):
        return True
    logger.error(
        "Wrong shape: Dimens of values %s %s %s" "vs %s %s %s",
        *values.shape,
        self._ncol,
        self._nrow,
        self._nlay,
    )
    return False
