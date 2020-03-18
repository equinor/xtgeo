# -*- coding: utf-8 -*-
"""Private module for refinement of a grid"""
from __future__ import print_function, absolute_import

from collections import OrderedDict
import numpy as np

from xtgeo.common import XTGeoDialog
import xtgeo.cxtgeo._cxtgeo as _cxtgeo

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

# pylint: disable=too-many-branches
# pylint: disable=too-many-statements


def refine_vertically(self, rfactor, zoneprop=None):
    """Refine vertically, proportionally

    See details in caller.
    """
    rfactord = OrderedDict()

    # case 1 rfactor as scalar value.
    if isinstance(rfactor, int):
        if self.subgrids:
            subgrids = self.get_subgrids()
            for i, _ in enumerate(self.subgrids.keys()):
                rfactord[i + 1] = rfactor
        else:
            rfactord[0] = rfactor
            subgrids = OrderedDict()
            subgrids[1] = self.nlay

    # case 2 rfactor is a dict
    else:
        rfactord = OrderedDict(sorted(rfactor.items()))  # redefined to ordered
        # 2a: zoneprop is present
        if zoneprop is not None:
            oldsubgrids = None
            if self.subgrids:
                oldsubgrids = self.get_subgrids()

            subgrids = self.subgrids_from_zoneprop(zoneprop)

            if oldsubgrids:
                if subgrids.values() != oldsubgrids.values():
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
    newsubgrids = OrderedDict()
    newnlay = 0
    for (_x, rfi), (snam, sran) in zip(rfactord.items(), subgrids.items()):
        newsubgrids[snam] = sran * rfi
        newnlay += newsubgrids[snam]

    logger.debug("New layers: %s", newnlay)

    # refinefactors is an array with length nlay; has N refinements per single K layer
    refinefactors = _cxtgeo.new_intarray(self.nlay)

    totvector = []

    for (_tmp1, rfi), (_tmp2, arr) in zip(rfactord.items(), self.subgrids.items()):
        for _elem in range(len(arr)):
            totvector.append(rfi)

    for inn, rfi in enumerate(totvector):
        _cxtgeo.intarray_setitem(refinefactors, inn, rfi)

    ref_zcornsv = np.zeros(self.ncol * self.nrow * (newnlay + 1) * 4, dtype=np.float64)
    ref_actnumsv = np.zeros(self.ncol * self.nrow * newnlay, dtype=np.int32)

    ier = _cxtgeo.grd3d_refine_vert(
        self.ncol,
        self.nrow,
        self.nlay,
        self._zcornsv,
        self._actnumsv,
        newnlay,
        ref_zcornsv,
        ref_actnumsv,
        refinefactors,
    )

    if ier != 0:
        raise RuntimeError(
            "An error occured in the C routine "
            "grd3d_refine_vert, code {}".format(ier)
        )

    # update instance:
    self._nlay = newnlay
    self._zcornsv = ref_zcornsv
    self._actnumsv = ref_actnumsv

    if self.subgrids is None or len(self.subgrids) <= 1:
        self.subgrids = None
    else:
        self.set_subgrids(newsubgrids)

    return self
