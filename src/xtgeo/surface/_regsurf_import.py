"""Import RegularSurface data."""
import numpy as np
import numpy.ma as ma

import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

xtg_verbose_level = xtg.get_syslevel()
_cxtgeo.xtg_verbose_file('NONE')


def import_irap_binary(self, mfile):
    # using swig type mapping

    logger.debug('Enter function...')
    # need to call the C function...
    _cxtgeo.xtg_verbose_file('NONE')

    xtg_verbose_level = xtg.get_syslevel()

    if xtg_verbose_level < 0:
        xtg_verbose_level = 0

    # read with mode 0, to get mx my
    xlist = _cxtgeo.surf_import_irap_bin(mfile, 0, 1, 0, xtg_verbose_level)

    nval = xlist[1] * xlist[2]  # mx * my
    xlist = _cxtgeo.surf_import_irap_bin(mfile, 1, nval, 0, xtg_verbose_level)

    ier, ncol, nrow, ndef, xori, yori, xinc, yinc, rot, val = xlist

    if ier != 0:
        raise RuntimeError('Problem in {}, code {}'.format(__name__, ier))

    val = np.reshape(val, (ncol, nrow), order='C')

    val = ma.masked_greater(val, _cxtgeo.UNDEF_LIMIT)

    if np.isnan(val).any():
        logger.info('NaN values are found, will mask...')
        val = ma.masked_invalid(val)

    self._ncol = ncol
    self._nrow = nrow
    self._xori = xori
    self._yori = yori
    self._xinc = xinc
    self._yinc = yinc
    self._rotation = rot
    self._values = val
    self._filesrc = mfile


def import_irap_ascii(self, mfile):
    # version using swig type mapping

    logger.debug('Enter function...')
    # need to call the C function...
    _cxtgeo.xtg_verbose_file('NONE')

    xtg_verbose_level = xtg.get_syslevel()

    if xtg_verbose_level < 0:
        xtg_verbose_level = 0

    # read with mode 0, scan to get mx my
    xlist = _cxtgeo.surf_import_irap_ascii(mfile, 0, 1, 0, xtg_verbose_level)

    nv = xlist[1] * xlist[2]  # mx * my
    xlist = _cxtgeo.surf_import_irap_ascii(mfile, 1, nv, 0, xtg_verbose_level)

    ier, ncol, nrow, ndef, xori, yori, xinc, yinc, rot, val = xlist

    if ier != 0:
        raise RuntimeError('Problem in {}, code {}'.format(__name__, ier))

    val = np.reshape(val, (ncol, nrow), order='C')

    val = ma.masked_greater(val, _cxtgeo.UNDEF_LIMIT)

    if np.isnan(val).any():
        logger.info('NaN values are found, will mask...')
        val = ma.masked_invalid(val)

    self._ncol = ncol
    self._nrow = nrow
    self._xori = xori
    self._yori = yori
    self._xinc = xinc
    self._yinc = yinc
    self._rotation = rot
    self._values = val
    self._filesrc = mfile
