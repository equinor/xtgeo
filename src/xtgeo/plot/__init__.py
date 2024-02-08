"""The XTGeo plot package"""

import warnings

from .grid3d_slice import Grid3DSlice
from .xsection import XSection
from .xtmap import Map

warnings.warn(
    "xtgeo.plot is deprecated and will be removed in xtgeo 4.0. "
    "This functionality now lives in the `xtgeoviz` package.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "Grid3DSlice",
    "XSection",
    "Map",
]
