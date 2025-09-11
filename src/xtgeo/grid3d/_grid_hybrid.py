from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import xtgeo._internal as _internal  # type: ignore
from xtgeo.common import null_logger

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
    self._set_xtgformat2()

    newnlay = self.nlay * 2 + nhdiv

    grid3d_cpp = _internal.grid3d.Grid(self)
    region_array = (
        region.values.astype(np.int32)
        if region
        else np.empty((0, 0, 0), dtype=np.int32)
    )
    hyb_zcornsv, hyb_actnumsv = grid3d_cpp.convert_to_hybrid_grid(
        toplevel,
        bottomlevel,
        nhdiv,
        region_array,
        int(region_number) if region_number is not None else -1,
    )

    # when a hybridgrid is made, the current subrid settings lose relevance, hence
    # it is forced set to None
    self.subgrids = None

    # update the grid in place
    self._nlay = newnlay
    self._zcornsv = hyb_zcornsv
    self._actnumsv = hyb_actnumsv.astype(np.int32)
