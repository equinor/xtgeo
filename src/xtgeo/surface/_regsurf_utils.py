"""RegularSurface utilities (basic low level)"""

import xtgeo.cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

_cxtgeo.xtg_verbose_file("NONE")

XTGDEBUG = xtg.get_syslevel()

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

    val = self.get_values1d(fill_value=self.undef)

    ier = _cxtgeo.surf_swapaxes(
        ncol, nrow, yflip, self.xori, xinc, self.yori, yinc, rota, val, 0, XTGDEBUG
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


# =========================================================================
# METHODS BELOW SHALL BE DEPRECATED!!
# Helper methods, for internal usage
# -------------------------------------------------------------------------
# copy self (update) values from SWIG carray to numpy, 1D array


# def _update_values(self):
#     nnum = self._ncol * self._nrow

#     if self._cvalues is None and self._values is not None:
#         return

#     elif self._cvalues is None and self._values is None:
#         logger.critical('_cvalues and _values is None in '
#                         '_update_values. STOP')
#         sys.exit(9)

#     xvv = _cxtgeo.swig_carr_to_numpy_1d(nnum, self._cvalues)

#     xvv = np.reshape(xvv, (self._ncol, self._nrow), order='F')

#     # make it masked
#     xvv = ma.masked_greater(xvv, self._undef_limit)

#     self._values = xvv

#     self._delete_cvalues()

# # copy (update) values from numpy to SWIG, 1D array


# def _update_cvalues(self):
#     logger.debug('Enter update cvalues method...')
#     nnum = self._ncol * self._nrow

#     if self._values is None and self._cvalues is not None:
#         logger.debug('CVALUES unchanged')
#         return

#     elif self._cvalues is None and self._values is None:
#         logger.critical('_cvalues and _values is None in '
#                         '_update_cvalues. STOP')
#         sys.exit(9)

#     elif self._cvalues is not None and self._values is None:
#         logger.critical('_cvalues and _values are both present in '
#                         '_update_cvalues. STOP')
#         sys.exit(9)

#     # make a 1D F order numpy array, and update C array
#     xvv = ma.filled(self._values, self._undef)
#     xvv = np.reshape(xvv, -1, order='F')

#     self._cvalues = _cxtgeo.new_doublearray(nnum)

#     _cxtgeo.swig_numpy_to_carr_1d(xvv, self._cvalues)
#     logger.debug('Enter method... DONE')

#     self._values = None


# def _delete_cvalues(self):
#     logger.debug('Enter delete cvalues values method...')

#     if self._cvalues is not None:
#         _cxtgeo.delete_doublearray(self._cvalues)

#     self._cvalues = None
#     logger.debug('Enter method... DONE')

# # check i
# f values shape is OK (return True or False)


# def _check_shape_ok(self, values):

#     if not values.flags['F_CONTIGUOUS']:
#         logger.error('Wrong order; shall be Fortran (Flags: {}'
#                      .format(values.flags))
#         return False

#     (ncol, nrow) = values.shape
#     if ncol != self._ncol or nrow != self._nrow:
#         logger.error('Wrong shape: Dimens of values {} {} vs {} {}'
#                      .format(ncol, nrow, self._ncol, self._nrow))
#         return False
#     return True


# def _convert_np_carr_double(self, np_array, nlen):
#     """Convert numpy 1D array to C array, assuming double type"""
#     carr = _cxtgeo.new_doublearray(nlen)

#     _cxtgeo.swig_numpy_to_carr_1d(np_array, carr)

#     return carr


# def _convert_carr_double_np(self, carray, nlen=None):
#     """Convert a C array to numpy, assuming double type."""
#     if nlen is None:
#         nlen = len(self._df.index)

#     nparray = _cxtgeo.swig_carr_to_numpy_1d(nlen, carray)

#     return nparray
