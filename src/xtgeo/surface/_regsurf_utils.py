"""RegularSurface utilities"""


import numpy as np

import xtgeo
import xtgeo.cxtgeo._cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)
# pylint: disable=protected-access


def swapaxes(self):
    """Swap the axes columns vs rows, keep origin. Will change yflip."""

    ncol = _cxtgeo.new_intpointer()
    nrow = _cxtgeo.new_intpointer()
    yflip = _cxtgeo.new_intpointer()
    xinc = _cxtgeo.new_doublepointer()
    yinc = _cxtgeo.new_doublepointer()
    rota = _cxtgeo.new_doublepointer()

    _cxtgeo.intpointer_assign(ncol, self._ncol)
    _cxtgeo.intpointer_assign(nrow, self._nrow)
    _cxtgeo.intpointer_assign(yflip, self._yflip)

    _cxtgeo.doublepointer_assign(xinc, self._xinc)
    _cxtgeo.doublepointer_assign(yinc, self._yinc)
    _cxtgeo.doublepointer_assign(rota, self._rotation)

    val = self.get_values1d(fill_value=xtgeo.UNDEF)

    ier = _cxtgeo.surf_swapaxes(
        ncol, nrow, yflip, self.xori, xinc, self.yori, yinc, rota, val, 0
    )
    if ier != 0:
        raise RuntimeError(
            "Unspecied runtime error from {}: Code: {}".format(__name__, ier)
        )

    self._ncol = _cxtgeo.intpointer_value(ncol)
    self._nrow = _cxtgeo.intpointer_value(nrow)
    self._yflip = _cxtgeo.intpointer_value(yflip)

    self._xinc = _cxtgeo.doublepointer_value(xinc)
    self._yinc = _cxtgeo.doublepointer_value(yinc)
    self._rotation = _cxtgeo.doublepointer_value(rota)

    ilines = self._xlines.copy()
    xlines = self._ilines.copy()

    self._ilines = ilines
    self._xlines = xlines

    self.values = val  # reshaping and masking is done in self.values


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
