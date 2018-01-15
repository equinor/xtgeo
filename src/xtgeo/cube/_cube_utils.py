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

# Low Level methods


# copy (update) values from SWIG carray to numpy, 3D array, Fortran order
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
