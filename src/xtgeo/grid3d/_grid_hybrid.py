from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from xtgeo import _cxtgeo
from xtgeo.common import null_logger
from xtgeo.common.constants import UNDEF_INT

logger = null_logger(__name__)

if TYPE_CHECKING:
    from xtgeo.grid3d import Grid, GridProperty


def make_hybridgrid(
    self: Grid,
    nhdiv: int = 10,
    toplevel: float = 1000.0,
    bottomlevel: float = 1100.0,
    region: GridProperty | None = None,
    region_number: int | None = None,
) -> None:
    """Make hybrid grid."""
    self._set_xtgformat1()

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
        rvalues = np.ma.filled(region.values, fill_value=UNDEF_INT)
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

    # when a hybridgrid is made, the current subrid settings lose relevance, hence
    # it is forced set to None
    self.subgrids = None

    self._nlay = newnlay
    self._zcornsv = hyb_zcornsv
    self._actnumsv = hyb_actnumsv
