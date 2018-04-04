"""GridProperty (not GridProperies) low level functions"""

from __future__ import print_function, absolute_import

import numpy as np
import numpy.ma as ma

import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

_cxtgeo.xtg_verbose_file('NONE')
xtg_verbose_level = xtg.get_syslevel()


def update_values_from_carray(self, carray, dtype, delete=False):
    """Transfer values from SWIG 1D carray to numpy, 3D array"""

    logger.debug('Update numpy from C array values')

    n = self.ntotal
    logger.info('N is {}'.format(n))
    logger.info('Name is {}'.format(self._name))

    self._isdiscrete = False

    if dtype == np.float64:
        logger.info('Entering conversion to numpy (float64) ...')
        values1d = _cxtgeo.swig_carr_to_numpy_1d(n, carray)
    else:
        logger.info('Entering conversion to numpy (int32) ...')
        values1d = _cxtgeo.swig_carr_to_numpy_i1d(n, carray)
        self._isdiscrete = True

    logger.debug('Values1D min: {}'.format(values1d.min()))
    logger.debug('Values1D max: {}'.format(values1d.max()))
    logger.debug('Values1D avg: {}'.format(values1d.mean()))
    values = np.reshape(values1d, (self._ncol, self._nrow, self._nlay),
                        order='F')

    # make into C order as this is standard Python order...
    values = np.asanyarray(values, order='C')

    # make it float64 or whatever(?) and mask it
    self._values = values
    self.mask_undef()

    # optionally delete the C array if needed
    if delete:
        carray = delete_carray(self, carray)


def update_carray(self, undef=None):
    """Copy (update) values from numpy to SWIG, 1D array, returns a pointer
    to SWIG C array"""

    if undef is None:
        undef = self._undef
        if self._isdiscrete:
            undef = self._undef_i

    logger.debug('Entering conversion from numpy to C array ...')

    values = self._values.copy()
    values = ma.filled(values, undef)

    values = np.asfortranarray(values)
    values1d = np.ravel(values, order='K')

    if values1d.dtype == 'float64' and self._isdiscrete:
        values1d = values1d.astype('int32')
        logger.debug('Casting has been done')

    if self._isdiscrete is False:
        logger.debug('Convert to carray (double)')
        carray = _cxtgeo.new_doublearray(self.ntotal)
        _cxtgeo.swig_numpy_to_carr_1d(values1d, carray)
    else:
        carray = _cxtgeo.new_intarray(self.ntotal)
        _cxtgeo.swig_numpy_to_carr_i1d(values1d, carray)

    return carray


def delete_carray(self, carray):
    """Delete carray SWIG C pointer, return carray as None"""
    logger.debug('Enter delete carray values method...')
    if carray is None:
        return None

    if self._isdiscrete:
        _cxtgeo.delete_intarray(carray)
        carray = None
        return carray

    else:
        _cxtgeo.delete_doublearray(carray)
        carray = None
        return carray


def check_shape_ok(self, values):
    """Check if chape of values is OK"""
    (ncol, nrow, nlay) = values.shape
    if ncol != self._ncol or nrow != self._nrow or nlay != self._nlay:
        logger.error('Wrong shape: Dimens of values {} {} {}'
                     'vs {} {} {}'
                     .format(ncol, nrow, nlay,
                             self._ncol, self._nrow, self._nlay))
        return False
    return True
