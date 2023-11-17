"""Auxillary functions for the well class

'self' is a Well() instance

"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from pathlib import Path

import pandas as pd

import xtgeo
from xtgeo.common import null_logger
from xtgeo.common._xyz_enum import _AttrName

from . import _well_io

logger = null_logger(__name__)


def _data_reader_factory(file_format: str | None = None):
    if file_format in ["rmswell", "irap_ascii", None]:
        return _well_io.import_rms_ascii
    if file_format == "hdf":
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
            len(args) >= 1 and isinstance(args[0], (str, Path, xtgeo._XTGeoFile))
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

            mfile = xtgeo._XTGeoFile(wfile)
            if fformat is None or fformat == "guess":
                fformat = mfile.detect_fformat()
            else:
                fformat = mfile.generic_format_by_proposal(fformat)
            kwargs = _data_reader_factory(fformat)(mfile, *args, **kwargs)
            kwargs["filesrc"] = mfile.file
            return func(self, **kwargs)
        return func(self, *args, **kwargs)

    return wrapper
