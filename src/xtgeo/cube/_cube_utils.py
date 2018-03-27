"""Cube utilities (basic low level)"""
import sys
import logging
import numpy as np

import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_cxtgeo.xtg_verbose_file('NONE')

xtg = XTGeoDialog()
xtg_verbose_level = xtg.get_syslevel()


def swapaxes(self):
    """Swap the axes inline vs xline, keep origin."""

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

    values1d = self.values.reshape(-1)

    ier = _cxtgeo.cube_swapaxes(ncol, nrow, self.nlay, yflip,
                                self.xori, xinc,
                                self.yori, yinc, rota, values1d,
                                0, xtg_verbose_level)
    if ier != 0:
        raise Exception

    self._ncol = _cxtgeo.intpointer_value(ncol)
    self._nrow = _cxtgeo.intpointer_value(nrow)
    self._yflip = _cxtgeo.intpointer_value(yflip)

    self._xinc = _cxtgeo.doublepointer_value(xinc)
    self._yinc = _cxtgeo.doublepointer_value(yinc)
    self._rotation = _cxtgeo.doublepointer_value(rota)


# copy (update) values from SWIG carray to numpy, 3D array, Fortran order
# to be DEPRECATED
def update_values(cube):

    if cube._cvalues is None and cube._values is None:
        logger.critical('Something is wrong. STOP!')
        sys.exit(9)

    elif cube._cvalues is None:
        return cube._values, None

    logger.debug('Updating numpy values...')
    ncrl = cube._ncol * cube._nrow * cube._nlay
    xv = _cxtgeo.swig_carr_to_numpy_f1d(ncrl, cube._cvalues)

    xv = np.reshape(xv, (cube._ncol, cube._nrow, cube._nlay), order='F')

    logger.debug('Updating numpy values... done')

    xtype = xv.dtype
    logger.info('VALUES of type {}'.format(xtype))

    # free the C values (to save memory)
    _cxtgeo.delete_floatarray(cube._cvalues)

    return xv, None


# copy (update) values from numpy to SWIG, 1D array
# TO BE DEPRECATED
def update_cvalues(cube):
    logger.debug('Enter update cvalues method...')
    n = cube._ncol * cube._nrow * cube._nlay

    if cube._values is None and cube._cvalues is not None:
        logger.debug('CVALUES unchanged')
        return None, cube._cvalues

    elif cube._cvalues is None and cube._values is None:
        logger.critical('_cvalues and _values is None in '
                        '_update_cvalues. STOP')
        sys.exit(9)

    elif cube._cvalues is not None and cube._values is None:
        logger.critical('_cvalues and _values are both present in '
                        '_update_cvalues. STOP')
        sys.exit(9)

    # make a 1D F order numpy array, and update C array
    xv = cube._values.copy()
    xv = np.reshape(xv, -1, order='F')

    xcv = _cxtgeo.new_floatarray(n)

    # convert...
    _cxtgeo.swig_numpy_to_carr_f1d(xv, xcv)
    logger.debug('Enter method... DONE')

    return None, xcv
