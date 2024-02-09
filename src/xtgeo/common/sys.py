"""Module for basic XTGeo interaction with OS/system files and folders."""

from __future__ import annotations

import hashlib
import io
import pathlib
from types import BuiltinFunctionType
from typing import TYPE_CHECKING, Literal

import numpy as np

from xtgeo import _cxtgeo
from xtgeo.io._file import FileWrapper

from ._xyz_enum import _AttrType
from .log import null_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt
    import pandas as pd

logger = null_logger(__name__)


def npfromfile(
    fname: str | pathlib.Path | io.BytesIO | io.StringIO,
    dtype: npt.DTypeLike = np.float32,
    count: int = 1,
    offset: int = 0,
    mmap: bool = False,
) -> np.ndarray:
    """Wrapper round np.fromfile to be compatible with older np versions."""
    try:
        if mmap and not isinstance(fname, (io.BytesIO, io.StringIO)):
            vals = np.memmap(
                fname, dtype=dtype, shape=(count,), mode="r", offset=offset
            )
        else:
            vals = np.fromfile(fname, dtype=dtype, count=count, offset=offset)
    except TypeError as err:
        # offset keyword requires numpy >= 1.17, need this for backward compat.:
        if "'offset' is an invalid" not in str(err):
            raise
        if not isinstance(fname, (str, pathlib.Path)):
            raise
        with open(fname, "rb") as buffer:
            buffer.seek(offset)
            vals = np.fromfile(buffer, dtype=dtype, count=count)
    return vals


def check_folder(
    fname: str | pathlib.Path | io.BytesIO | io.StringIO,
    raiseerror: type[Exception] | None = None,
) -> bool:
    """General function to check folder."""
    _nn = FileWrapper(fname)
    status = _nn.check_folder(raiseerror=raiseerror)
    del _nn
    return status


def generic_hash(
    gid: str, hashmethod: Literal["md5", "sha256", "blake2d"] | Callable = "md5"
) -> str:
    """Return a unique hash ID for current instance.

    This hash can e.g. be used to compare two instances for equality.

    Args:
        gid: Any string as signature, e.g. cumulative attributes of an instance.
        hashmethod: Supported methods are "md5", "sha256", "blake2b"
            or use a full function signature e.g. hashlib.sha128.
            Defaults to md5.

    Returns:
        Hash signature.

    Raises:
        KeyError: String in hashmethod has an invalid option

    .. versionadded:: 2.14

    """
    validmethods: dict[str, Callable] = {
        "md5": hashlib.md5,
        "sha256": hashlib.sha256,
        "blake2b": hashlib.blake2b,
    }

    if isinstance(hashmethod, str) and hashmethod in validmethods:
        mhash = validmethods[hashmethod]()
    elif isinstance(hashmethod, BuiltinFunctionType):
        mhash = hashmethod()
    else:
        raise ValueError(f"Invalid hash method provided: {hashmethod}")

    mhash.update(gid.encode())
    return mhash.hexdigest()


def inherit_docstring(inherit_from: Callable) -> Callable:
    def decorator_set_docstring(func: Callable) -> Callable:
        if func.__doc__ is None and inherit_from.__doc__ is not None:
            func.__doc__ = inherit_from.__doc__
        return func

    return decorator_set_docstring


# ----------------------------------------------------------------------------------
# Special methods for nerds, to be removed when not appplied any more
# ----------------------------------------------------------------------------------


def _convert_np_carr_int(length: int, np_array: np.ndarray) -> np.ndarray:
    """Convert numpy 1D array to C array, assuming int type.

    The numpy is always a double (float64), so need to convert first
    """
    carr = _cxtgeo.new_intarray(length)
    np_array = np_array.astype(np.int32)
    _cxtgeo.swig_numpy_to_carr_i1d(np_array, carr)
    return carr


def _convert_np_carr_double(length: int, np_array: np.ndarray) -> np.ndarray:
    """Convert numpy 1D array to C array, assuming double type."""
    carr = _cxtgeo.new_doublearray(length)
    _cxtgeo.swig_numpy_to_carr_1d(np_array, carr)
    return carr


def _convert_carr_double_np(
    length: int, carray: np.ndarray, nlen: int | None = None
) -> np.ndarray:
    """Convert a C array to numpy, assuming double type."""
    if nlen is None:
        nlen = length
    return _cxtgeo.swig_carr_to_numpy_1d(nlen, carray)


def _get_carray(
    dataframe: pd.DataFrame, attributes: _AttrType, attrname: str
) -> np.ndarray | None:
    """
    Returns the C array pointer (via SWIG) for a given attr.

    Type conversion is double if float64, int32 if DISC attr.
    Returns None if log does not exist.
    """
    np_array = None
    if attrname in dataframe:
        np_array = dataframe[attrname].values
    else:
        return None

    nlen = len(dataframe.index)
    if attributes[attrname] == _AttrType.DISC.value:
        carr = _convert_np_carr_int(nlen, np_array)
    else:
        carr = _convert_np_carr_double(nlen, np_array)
    return carr
