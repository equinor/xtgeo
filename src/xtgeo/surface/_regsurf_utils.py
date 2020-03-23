"""RegularSurface utilities"""

import xtgeo
import xtgeo.cxtgeo._cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

#


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
