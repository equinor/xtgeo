"""Importing grid props from ROFF, binary"""


import numpy as np
import xtgeo

from ._roff_parameter import RoffParameter

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)


def import_roff(self, pfile, name):
    """Import ROFF format"""
    roff_name = name
    if name == "unknown":
        roff_name = None
    roff_param = RoffParameter.from_file(pfile._file, roff_name)
    self._codes = roff_param.xtgeo_codes()
    self._name = roff_param.name
    self._ncol = int(roff_param.nx)
    self._nrow = int(roff_param.ny)
    self._nlay = int(roff_param.nz)
    self._isdiscrete = roff_param.is_discrete
    self._undef = xtgeo.UNDEF_INT if self._isdiscrete else xtgeo.UNDEF
    self._values = roff_param.xtgeo_values()
    self.dtype = self._values.dtype

    roff_val = roff_param.values
    if isinstance(roff_val, bytes) or np.issubdtype(roff_val.dtype, np.uint8):
        self._roxar_dtype = np.uint8
    elif np.issubdtype(roff_val.dtype, np.integer):
        self._roxar_dtype = np.uint16
    elif np.issubdtype(roff_val.dtype, np.floating):
        self._roxar_dtype = np.float32
    else:
        raise ValueError(f"Could not deduce roxar type of {roff_val.dtype}")
