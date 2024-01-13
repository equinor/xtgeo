"""Auxillary functions for the well class

'self' is a Well() instance

"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from xtgeo.common import null_logger
from xtgeo.common._xyz_enum import _AttrName
from xtgeo.io._file import FileFormat, FileWrapper

from . import _well_io

logger = null_logger(__name__)


def _data_reader_factory(file_format: FileFormat):
    if file_format in (FileFormat.RMSWELL, FileFormat.IRAP_ASCII, FileFormat.UNKNOWN):
        return _well_io.import_rms_ascii
    if file_format == FileFormat.HDF:
        return _well_io.import_hdf5_well
    raise ValueError(
        f"Unknown file format {file_format}, supported formats are "
        "'rmswell', 'irap_ascii' and 'hdf'"
    )


def allow_deprecated_init(func: Callable):
    # This decorator is here to maintain backwards compatibility in the
    # construction of Well and should be deleted once the deprecation period
    # has expired, the construction will then follow the new pattern.
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not args and not kwargs:
            warnings.warn(
                "Initializing empty well is deprecated, please provide "
                "non-defaulted values, or use mywell = "
                "xtgeo.well_from_file('filename')",
                DeprecationWarning,
            )
            return func(
                self,
                *([0.0] * 3),
                "",
                pd.DataFrame(
                    {
                        _AttrName.XNAME.value: [],
                        _AttrName.YNAME.value: [],
                        _AttrName.ZNAME.value: [],
                    }
                ),
            )

        # Checking if we are doing an initialization from file and raise a
        # deprecation warning if we are.
        if "wfile" in kwargs or (
            len(args) >= 1 and isinstance(args[0], (str, Path, FileWrapper))
        ):
            warnings.warn(
                "Initializing directly from file name is deprecated and will be "
                "removed in xtgeo version 4.0. Use: "
                "mywell = xtgeo.well_from_file('filename') instead",
                DeprecationWarning,
            )
            if len(args) >= 1:
                wfile = args[0]
                args = args[1:]
            else:
                wfile = kwargs.pop("wfile", None)
            if len(args) >= 1:
                fformat = args[0]
                args = args[1:]
            else:
                fformat = kwargs.pop("fformat", None)

            mfile = FileWrapper(wfile)
            fmt = mfile.fileformat(fformat)
            kwargs = _data_reader_factory(fmt)(mfile, *args, **kwargs)
            kwargs["filesrc"] = mfile.file
            return func(self, **kwargs)
        return func(self, *args, **kwargs)

    return wrapper
