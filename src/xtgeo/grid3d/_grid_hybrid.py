from __future__ import print_function, absolute_import

import numpy as np


import xtgeo
import xtgeo.cxtgeo._cxtgeo as _cxtgeo

xtg = xtgeo.XTGeoDialog()
logger = xtg.functionlogger(__name__)


def make_hybridgrid(
    self,
    nhdiv=10,
    toplevel=1000.0,
    bottomlevel=1100.0,
    region=None,
    region_number=None,
):

    """Make hybrid grid."""

    newnlay = self.nlay * 2 + nhdiv
    newnzcorn = self.ncol * self.nrow * (newnlay + 1) * 4
    newnactnum = self.ncol * self.nrow * newnlay

    # initialize
    hyb_zcornsv = np.zeros(newnzcorn, dtype=np.float64)
    hyb_actnumsv = np.zeros(newnactnum, dtype=np.int32)

    if region is None:
        region_number = -1
        rvalues = np.ones(1, dtype=np.int32)
    else:
        rvalues = np.ma.filled(region.values, fill_value=xtgeo.UNDEF_INT)
        rvalues = rvalues.ravel()

    _cxtgeo.grd3d_convert_hybrid(
        self.ncol,
        self.nrow,
        self.nlay,
        self._coordsv,
        self._zcornsv,
        self._actnumsv,
        newnlay,
        hyb_zcornsv,
        hyb_actnumsv,
        toplevel,
        bottomlevel,
        nhdiv,
        rvalues,
        region_number,
    )

    del rvalues

    self._nlay = newnlay
    self._zcornsv = hyb_zcornsv
    self._actnumsv = hyb_actnumsv
