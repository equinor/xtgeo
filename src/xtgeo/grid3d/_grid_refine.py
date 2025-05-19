"""Private module for refinement of a grid."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import xtgeo._internal as _internal  # type: ignore
from xtgeo.common import XTGeoDialog, null_logger

xtg = XTGeoDialog()
logger = null_logger(__name__)

if TYPE_CHECKING:
    from xtgeo.grid3d import Grid, GridProperty


def refine_vertically(
    self: Grid,
    rfactor: int | dict[int, int],
    zoneprop: GridProperty | None = None,
) -> Grid:
    """Refine vertically, proportionally.

    See details in caller.
    """
    self._xtgformat1()
    self.make_zconsistent()

    rfactord = {}

    # case 1 rfactor as scalar value.
    if isinstance(rfactor, int):
        if self.subgrids:
            subgrids = self.get_subgrids()
            for i, _ in enumerate(self.subgrids.keys()):
                rfactord[i + 1] = rfactor
        else:
            rfactord[0] = rfactor
            subgrids = {}
            subgrids[1] = self.nlay

    # case 2 rfactor is a dict
    else:
        rfactord = dict(sorted(rfactor.items()))  # redefined to ordered
        # 2a: zoneprop is present
        if zoneprop is not None:
            oldsubgrids = None
            if self.subgrids:
                oldsubgrids = self.get_subgrids()

            subgrids = self.subgrids_from_zoneprop(zoneprop)

            if oldsubgrids and subgrids.values() != oldsubgrids.values():
                xtg.warn("ISSUES!!!")

        # 2b: zoneprop is not present
        elif zoneprop is None and self.subgrids:
            subgrids = self.get_subgrids()

        elif zoneprop is None and not self.subgrids:
            raise ValueError(
                "You gave in a dict, but no zoneprops and "
                "subgrids are not preesent in the grid"
            )
        else:
            raise ValueError("Some major unexpected issue in routine...")

    if len(subgrids) != len(rfactord):
        raise RuntimeError("Subgrids and refinements: different definition!")

    self.set_subgrids(subgrids)

    # Now, based on dict, give a value per subgrid for key, val in rfactor
    newsubgrids = {}
    newnlay = 0
    for (_x, rfi), (snam, sran) in zip(rfactord.items(), subgrids.items()):
        newsubgrids[snam] = sran * rfi
        newnlay += newsubgrids[snam]

    logger.debug("New layers: %s", newnlay)

    refine_factors = []

    for (_, rfi), (_, arr) in zip(rfactord.items(), self.subgrids.items()):
        for _ in range(len(arr)):
            refine_factors.append(rfi)

    self._xtgformat2()
    grid_cpp = _internal.grid3d.Grid(self)
    refine_factors = np.array(refine_factors, dtype=np.int8)
    ref_zcornsv, ref_actnumsv = grid_cpp.refine_vertically(refine_factors)

    # update instance:
    self._nlay = newnlay
    self._zcornsv = ref_zcornsv
    self._actnumsv = ref_actnumsv

    if self.subgrids is None or len(self.subgrids) <= 1:
        self.subgrids = None
    else:
        self.set_subgrids(newsubgrids)

    return self
