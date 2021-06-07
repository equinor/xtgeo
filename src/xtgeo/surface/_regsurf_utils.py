"""RegularSurface utilities"""


import numpy as np

import xtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.common.calc import _swap_axes

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)
# pylint: disable=protected-access


def swapaxes(self):
    """Swap the axes columns vs rows, keep origin. Will change yflip."""
    self._rotation, self._yflip, swapped_values = _swap_axes(
        self._rotation,
        self._yflip,
        values=self.values.filled(xtgeo.UNDEF),
    )
    self._ncol, self._nrow = self._nrow, self._ncol
    self._xinc, self._yinc = self._yinc, self._xinc
    self.values = swapped_values["values"]


def autocrop(self):
    """Crop surface by looking at undefined areas, update instance"""

    minvalue = self.values.min()

    if np.isnan(minvalue):
        return

    arrx, arry = np.ma.where(self.values >= minvalue)

    imin = int(arrx.min())
    imax = int(arrx.max())

    jmin = int(arry.min())
    jmax = int(arry.max())

    xori, yori, _dummy = self.get_xy_value_from_ij(imin + 1, jmin + 1)

    ncol = imax - imin + 1
    nrow = jmax - jmin + 1

    self._values = self.values[imin : imax + 1, jmin : jmax + 1]
    self._ilines = self.ilines[imin : imax + 1]
    self._xlines = self.xlines[jmin : jmax + 1]
    self._ncol = ncol
    self._nrow = nrow
    self._xori = xori
    self._yori = yori
