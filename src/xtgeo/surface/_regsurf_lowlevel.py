"""RegularSurface utilities (low level)"""

import xtgeo.cxtgeo._cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

#


# ======================================================================================
# Helper methods, for internal usage


def get_carr_double(self):
    """Return the SWIG Carray object"""

    carr = _cxtgeo.new_doublearray(self.ncol * self.nrow)

    _cxtgeo.swig_numpy_to_carr_1d(self.get_values1d(), carr)

    return carr


# ======================================================================================
# METHODS BELOW SHALL BE DEPRECATED!!
# Helper methods, for internal usage
# --------------------------------------------------------------------------------------
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
#     xvv = ma.masked_greater(xvv, xtgeo.UNDEF_LIMIT)

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
#     xvv = ma.filled(self._values, xtgeo.UNDEF)
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


# def _convert_carr_double_np(self, carray, nlen=None):
#     """Convert a C array to numpy, assuming double type."""
#     if nlen is None:
#         nlen = len(self._df.index)

#     nparray = _cxtgeo.swig_carr_to_numpy_1d(nlen, carray)

#     return nparray
