"""The XTGeo plot package"""
# flake8: noqa

import warnings

from xtgeo.plot.grid3d_slice import Grid3DSlice
from xtgeo.plot.xsection import XSection
from xtgeo.plot.xtmap import Map

warnings.warn(
    "xtgeo.plot is deprecated and will be removed in xtgeo 4.0. "
    "This functionality now lives in the `xtgeoviz` package.",
    DeprecationWarning,
    stacklevel=2,
)
